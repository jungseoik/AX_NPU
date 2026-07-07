"""
PE-Core-L14-336 비전인코더 NPU 추론.

  - MXQInferenceFull  : image->embedding 전부 NPU (CPU head 없음). **권장.** 배치는 1모델+멀티스레드 sync로 코어 활용.
                        attn_pool의 QKᵀ outlier만 16bit로 올린 full MXQ(--qk16)로 cos 0.99.
  - MXQInferenceHybrid: NPU trunk + CPU attn_pool/proj. **레거시.** QKᵀ 16bit 해결 전 방식.

배경: attn_pool(577토큰->1토큰 cross-attention pooling)을 그냥 INT8로 양자화하면 QKᵀ matmul의
outlier 때문에 깨졌었다(full-NPU cos 0.46). Mobilint 해결책(그 score matmul만 16bit)으로
full 모델도 NPU에서 cos 0.99 달성 → CPU head 우회가 불필요해졌다.
(reports/vendor/mobilint_resolution_attn_pool.md)

공통 인터페이스(TRTInference 호환): model(image) -> (B,1024).
입력: torch.Tensor (B,3,336,336)/(3,336,336) 또는 numpy. 전처리(preprocess_image)는 호출측.
요구: qbruntime + NPU(/dev/aries0). 모델 코드는 vendored(pe_npu/pe_vendor)라 외부 레포 불필요.
"""
from __future__ import annotations

import os

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

import qbruntime

from .pe_model import IMAGE_SIZE, load_pe

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FEAT_MXQ = os.path.join(HERE, "out", "pe_feat.mxq")
DEFAULT_FULL_MXQ = os.path.join(HERE, "out", "pe_full.mxq")


class MXQInferenceFull:
    """full NPU 추론: image -> embedding 전부 NPU 한 번에 (CPU head 없음).

    attn_pool의 QK^T outlier로 INT8서 깨지던 head를, 그 score MatMul만 16bit로 올린
    full MXQ(컴파일 시 --qk16)로 NPU에서 직접 돌린다. 원본 pth 대비 cos ≈ 0.99.
    (Mobilint 해결책. reports/vendor/mobilint_resolution_attn_pool.md)

    hybrid(MXQInferenceHybrid) 대비:
      - CPU attn_pool/proj 불필요 → 고채널 CPU 병목 제거, torch/pe_vendor 의존 없음.
      - 입력/출력 인터페이스 동일: model(image (B,3,336,336)) -> (B,1024).
    """

    def __init__(self, full_mxq_path: str = DEFAULT_FULL_MXQ, device_id: int = 0,
                 num_threads: int = 8):
        self.acc = qbruntime.Accelerator(device_id)
        self.model = qbruntime.Model(full_mxq_path)
        self.model.launch(self.acc)
        self.image_size = IMAGE_SIZE
        # 배치 추론 동시성: 1모델 + num_threads개 스레드로 동기 infer를 동시 호출한다.
        # 런타임이 동시 호출을 카드 코어에 안전 분배 → single 모드서 8코어 활용, 출력 정확(cos 1.0).
        # (주의: qbruntime async(infer_async) multi-in-flight는 출력버퍼 충돌로 깨져 쓰지 않는다.
        #  multi-model 인스턴스는 이 방식과 처리량 동일하고 메모리만 낭비라 쓰지 않는다.)
        self.num_threads = max(1, int(num_threads))
        self._pool = None
        if self.num_threads > 1:
            from concurrent.futures import ThreadPoolExecutor
            self._pool = ThreadPoolExecutor(max_workers=self.num_threads)

    @classmethod
    def from_hf(cls, repo_id: str = None, device_id: int = 0, revision: str = None,
                scheme: str = "single", num_threads: int = 8):
        """HF에서 미리 컴파일된 full MXQ를 받아 추론기 구성 (qbruntime만 필요).

        scheme: 코어모드 (single|multi|global4|global8). HF `<scheme>/pe_full.mxq`.
                single=throughput(멀티스레드), global8=단건 latency 최소.
        num_threads: 배치 추론 시 동시 동기호출 스레드 수(카드 코어수=8 권장, single 모드서 8코어 활용).
        """
        from . import assets
        repo = repo_id or assets.HF_REPO
        full = assets.ensure_full_mxq(repo_id=repo, revision=revision, scheme=scheme)
        return cls(full_mxq_path=full, device_id=device_id, num_threads=num_threads)

    @classmethod
    def load(cls, scheme: str = "single", local_mxq: str = None, repo_id: str = None,
             revision: str = None, device_id: int = 0, num_threads: int = 8):
        """**기본 로더**: 로컬 mxq 있으면 사용 → 없으면 HF에서 다운로드 → 그래도 없으면 컴파일 안내.

        생성자(로컬만)와 from_hf(HF만)를 합친 편의 진입점 (YOLONPU.load와 대칭).
        """
        from . import assets
        cand = local_mxq or DEFAULT_FULL_MXQ
        if cand and os.path.exists(cand):
            mxq = cand
        else:
            try:
                mxq = assets.ensure_full_mxq(repo_id=repo_id or assets.HF_REPO,
                                             revision=revision, scheme=scheme)
            except Exception as e:
                raise FileNotFoundError(
                    f"HF에 {scheme}/pe_full.mxq가 없습니다 ({type(e).__name__}). 직접 컴파일하세요:\n"
                    f"  python -m pe_npu.compile --mode compile --save {DEFAULT_FULL_MXQ} "
                    f"--qk16 --scheme {scheme} --calib-data-path <calib_hwc> --device cpu"
                ) from e
        return cls(full_mxq_path=mxq, device_id=device_id, num_threads=num_threads)

    def __call__(self, *args, **kwargs):
        return self.infer(*args, **kwargs)

    def _infer_one(self, chw):
        hwc = np.ascontiguousarray(chw.transpose(1, 2, 0))          # NPU 입력 = HWC
        o = self.model.infer(hwc)                                   # image -> embedding (NPU 전부, 동기)
        o = o[0] if isinstance(o, (list, tuple)) else o
        return np.asarray(o).reshape(-1).astype(np.float32)

    def infer(self, image):
        return_torch = _HAS_TORCH and isinstance(image, torch.Tensor)
        arr = image.detach().cpu().numpy() if return_torch else np.asarray(image)
        arr = arr.astype(np.float32, copy=False)
        if arr.ndim == 3:
            arr = arr[None]
        B = arr.shape[0]
        if self._pool is not None and B > 1:
            # 1모델 + 스레드풀 동기 호출 (순서 보존). single 모드서 8코어 동시 활용.
            embs = list(self._pool.map(self._infer_one, [arr[i] for i in range(B)]))
        else:
            embs = [self._infer_one(arr[i]) for i in range(B)]
        out = np.stack(embs, axis=0)                                # (B,1024)
        if return_torch:
            t = torch.from_numpy(out)
            return t.to(image.device) if image.is_cuda else t
        return out

    def __del__(self):
        try:
            if getattr(self, "_pool", None) is not None:
                self._pool.shutdown(wait=False)
        except Exception:
            pass


class MXQInferenceHybrid:
    """[레거시] NPU trunk(feat MXQ) + CPU pool/proj head. TRTInference 호환.

    QKᵀ 16bit 해결(MXQInferenceFull) 이전 방식. CPU attn_pool이 고채널에서 병목이라
    신규 코드는 MXQInferenceFull 사용 권장. 하위호환/비교용으로 유지.

    pool head(CPU)는 두 가지 소스 중 하나로 복원한다:
      - pool_head_path 지정 → 추출된 가중치(pe_pool_head.pt)만 로드. PE 전체 가중치
        (HF 2GB+) 다운로드 없이 동작 = 옵션 B(가져와 쓰기).
      - None(기본) → load_pe로 원본 PE를 로드해 attn_pool/proj 사용 = 옵션 A(직접 컴파일).
    """

    def __init__(self, feat_mxq_path: str = DEFAULT_FEAT_MXQ, model_name: str = "PE-Core-L14-336",
                 device_id: int = 0, pool_head_path: str = None):
        if not _HAS_TORCH:
            raise RuntimeError("torch 필요 (CPU pool head 연산)")

        # --- NPU: forward_features MXQ ---
        self.acc = qbruntime.Accelerator(device_id)
        self.model = qbruntime.Model(feat_mxq_path)
        self.model.launch(self.acc)

        # --- CPU: pool head (attn_pool + proj) ---
        if pool_head_path:
            # 옵션 B: 구조만(가중치 X) 만들고 추출본 로드 → HF 전체 가중치 다운로드 회피
            from .pe_vendor import pe as pe_mod
            ckpt = torch.load(pool_head_path, map_location="cpu")
            skel = pe_mod.CLIP.from_config(
                ckpt.get("model_name", model_name), device="cpu", load_default_weights=False
            ).float().eval()
            skel.visual.attn_pool.load_state_dict(ckpt["attn_pool"])
            if ckpt.get("proj") is not None and skel.visual.proj is not None:
                with torch.no_grad():
                    skel.visual.proj.copy_(ckpt["proj"])
            self.visual = skel.visual
        else:
            # 옵션 A: 원본(미패치) PE 전체에서 attn_pool/proj 사용
            self.visual = load_pe(model_name=model_name, mode="clip", patch=False).visual
        self.image_size = IMAGE_SIZE

    @classmethod
    def from_hf(cls, repo_id: str = None, model_name: str = "PE-Core-L14-336",
                device_id: int = 0, revision: str = None):
        """옵션 B: HF에서 미리 컴파일된 MXQ + pool head를 받아 추론기를 구성.

        qbcompiler / 원본 PE 전체 가중치 없이 동작한다. repo_id 미지정 시 기본 PIA-SPACE-LAB/MXQ_NPU.
        """
        from . import assets
        repo = repo_id or assets.HF_REPO
        feat = assets.ensure_feat_mxq(repo_id=repo, revision=revision)
        pool = assets.ensure_pool_head(repo_id=repo, revision=revision)
        return cls(feat_mxq_path=feat, model_name=model_name, device_id=device_id, pool_head_path=pool)

    def __call__(self, *args, **kwargs):
        return self.infer(*args, **kwargs)

    def infer(self, image):
        return_torch = _HAS_TORCH and isinstance(image, torch.Tensor)
        arr = image.detach().cpu().numpy() if return_torch else np.asarray(image)
        arr = arr.astype(np.float32, copy=False)
        if arr.ndim == 3:
            arr = arr[None]
        B = arr.shape[0]

        embs = []
        for i in range(B):
            chw = arr[i]                                         # (3,336,336)
            hwc = np.ascontiguousarray(chw.transpose(1, 2, 0))   # NPU 입력 = HWC
            feo = self.model.infer(hwc)                          # NPU trunk
            feo = feo[0] if isinstance(feo, (list, tuple)) else feo
            feat = torch.from_numpy(np.asarray(feo).reshape(1, 577, 1024).astype(np.float32))
            with torch.no_grad():
                pooled = self.visual._pool(feat)                 # CPU attn_pool
                if self.visual.proj_dim is not None:
                    pooled = pooled @ self.visual.proj           # CPU proj
            embs.append(pooled.reshape(-1).numpy())

        out = np.stack(embs, axis=0)                             # (B,1024)
        if return_torch:
            t = torch.from_numpy(out)
            return t.to(image.device) if image.is_cuda else t
        return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-mxq", default=DEFAULT_FEAT_MXQ)
    ap.add_argument("--batch", type=int, default=2)
    args = ap.parse_args()
    m = MXQInferenceHybrid(args.feat_mxq)
    dummy = np.random.randn(args.batch, 3, IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)
    out = m.infer(dummy)
    print("input:", dummy.shape, "-> output:", out.shape)
