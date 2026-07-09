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

import glob
import os
from concurrent.futures import ThreadPoolExecutor

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

# 코어모드별 '카드당 동시 슬롯' = 8코어 / (추론당 코어수). 스레드풀 크기 = 슬롯 × 카드.
CORE_MODE_SLOTS = {"single": 8, "global4": 2, "global8": 1, "multi": 1}


def detect_npu_devices() -> list[int]:
    """장착된 NPU device id(`/dev/ariesN`) 목록을 정렬해 반환. 없으면 빈 리스트."""
    ids = []
    for p in glob.glob("/dev/aries*"):
        s = os.path.basename(p)[len("aries"):]
        if s.isdigit():
            ids.append(int(s))
    return sorted(ids)


def _slots_for_core_mode(core_mode) -> int:
    name = str(core_mode).split(".")[-1].lower()   # CoreMode.Global4 -> "global4"
    return CORE_MODE_SLOTS.get(name, 1)


class MXQInferenceFull:
    """full NPU 추론: image -> embedding 전부 NPU (CPU head 없음). **단일/멀티카드 통합.**

    attn_pool QK^T outlier로 INT8서 깨지던 head를 그 score MatMul만 16bit로(--qk16) 올려
    NPU에서 직접 돌린다. 원본 pth 대비 cos ≈ 0.99. (reports/vendor/mobilint_resolution_attn_pool.md)

    카드 선택:
      - device_ids=None(기본) → 단일 카드(device_id).       예) MXQInferenceFull()            # aries0
      - device_ids=[0,1]      → 지정 카드들.                예) MXQInferenceFull(device_ids=[0,1])
      - device_ids="auto"     → 장착된 NPU 전부.            예) MXQInferenceFull(device_ids="auto")

    동시성(배치 자동 확장): 카드당 1모델 launch + **스레드풀(슬롯×카드)로 동기 infer 동시 호출**.
    슬롯 = 코어모드별 카드당 동시 추론 수(single=8, global4=2, global8=1, multi=1) — 컴파일된 모드를
    `get_core_mode()`로 읽어 자동 결정(slots_per_card로 override). infer에 B장을 주면 카드에
    라운드로빈 분산되어 알아서 병렬 처리된다(출력 순서 보존, cos 1.0).
    (async(infer_async) multi-in-flight는 출력 깨져 안 씀 — 동기+멀티스레드로 동시성 확보.)
    """

    def __init__(self, full_mxq_path: str = DEFAULT_FULL_MXQ, device_id: int = 0,
                 device_ids=None, slots_per_card: int = None, num_threads: int = None):
        if device_ids == "auto":
            ids = detect_npu_devices()
            if not ids:
                raise RuntimeError("NPU 디바이스(/dev/aries*)를 찾지 못했습니다.")
        elif device_ids is None:
            ids = [device_id]
        else:
            ids = list(device_ids)
        self.device_ids = ids

        # 카드당 Model 1개 launch
        self.accs, self.models = [], []
        for d in ids:
            acc = qbruntime.Accelerator(d)
            m = qbruntime.Model(full_mxq_path)
            m.launch(acc)
            self.accs.append(acc)
            self.models.append(m)
        self.n = len(self.models)
        self.model = self.models[0]          # 하위호환(단일 참조)
        self.image_size = IMAGE_SIZE

        # 슬롯: slots_per_card > num_threads(하위호환 별칭) > 코어모드 자동
        try:
            self.core_mode = str(self.models[0].get_core_mode())
        except Exception:
            self.core_mode = "single"
        if slots_per_card and slots_per_card > 0:
            slots = slots_per_card
        elif num_threads and num_threads > 0:
            slots = num_threads
        else:
            slots = _slots_for_core_mode(self.core_mode)
        self.slots_per_card = slots
        self.W = slots * self.n              # 스레드풀 크기 = 슬롯 × 카드
        self._pool = ThreadPoolExecutor(max_workers=self.W) if self.W > 1 else None

    @classmethod
    def from_hf(cls, repo_id: str = None, device_id: int = 0, device_ids=None,
                revision: str = None, scheme: str = "global4", slots_per_card: int = None):
        """HF에서 미리 컴파일된 full MXQ(`<scheme>/pe_full.mxq`)를 받아 추론기 구성 (qbruntime만 필요).

        scheme: 코어모드 (single|multi|global4|global8). 기본 **global4**(다채널 균형).
        device_ids: None(단일)|리스트|"auto"(전체).
        """
        from . import assets
        repo = repo_id or assets.HF_REPO
        full = assets.ensure_full_mxq(repo_id=repo, revision=revision, scheme=scheme)
        return cls(full_mxq_path=full, device_id=device_id, device_ids=device_ids,
                   slots_per_card=slots_per_card)

    @classmethod
    def load(cls, scheme: str = "global4", local_mxq: str = None, repo_id: str = None,
             revision: str = None, device_id: int = 0, device_ids=None, slots_per_card: int = None):
        """**기본 로더**: local_mxq 지정 시 사용 → 없으면 HF `<scheme>/pe_full.mxq` → 없으면 컴파일 안내.
        기본 scheme=**global4**. 단일/멀티카드(device_ids)는 그대로 전달. (YOLONPU.load와 동일 규칙)"""
        from . import assets
        if local_mxq and os.path.exists(local_mxq):
            mxq = local_mxq
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
        return cls(full_mxq_path=mxq, device_id=device_id, device_ids=device_ids,
                   slots_per_card=slots_per_card)

    def __len__(self):
        return self.n

    def __call__(self, *args, **kwargs):
        return self.infer(*args, **kwargs)

    def _infer_one(self, model, chw):
        hwc = np.ascontiguousarray(chw.transpose(1, 2, 0))          # NPU 입력 = HWC
        o = model.infer(hwc)                                        # image -> embedding (동기, NPU 전부)
        o = o[0] if isinstance(o, (list, tuple)) else o
        return np.asarray(o).reshape(-1).astype(np.float32)

    def infer(self, image):
        """image: (B,3,336,336)/(3,336,336) numpy/torch -> (B,1024).
        B장을 카드에 라운드로빈(i%n), 카드당 슬롯만큼 동시 infer로 배치 자동 병렬(순서 보존)."""
        return_torch = _HAS_TORCH and isinstance(image, torch.Tensor)
        arr = image.detach().cpu().numpy() if return_torch else np.asarray(image)
        arr = arr.astype(np.float32, copy=False)
        if arr.ndim == 3:
            arr = arr[None]
        B = arr.shape[0]
        if B == 0:
            empty = np.empty((0, 1024), dtype=np.float32)
            return torch.from_numpy(empty) if return_torch else empty

        out = [None] * B
        if self._pool is None or B == 1:
            for i in range(B):
                out[i] = self._infer_one(self.models[i % self.n], arr[i])
        else:
            def work(i):
                out[i] = self._infer_one(self.models[i % self.n], arr[i])   # 이미지 i -> 카드 i%n
            list(self._pool.map(work, range(B)))

        emb = np.stack(out, axis=0)                                 # (B,1024)
        if return_torch:
            t = torch.from_numpy(emb)
            return t.to(image.device) if image.is_cuda else t
        return emb

    def dispose(self):
        if getattr(self, "_pool", None) is not None:
            try:
                self._pool.shutdown(wait=False)
            except Exception:
                pass
            self._pool = None
        for m in getattr(self, "models", []):
            try:
                m.dispose()
            except Exception:
                pass

    def __del__(self):
        self.dispose()


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
