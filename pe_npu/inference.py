"""
MXQInferenceHybrid — PE-Core-L14-336 비전인코더 NPU 추론 (hybrid).

구조 (정확도 0.997 검증, reports/SOLUTION_single_io_compile.md):
  image -> [24 transformer blocks = NPU INT8 (feat MXQ)] -> 토큰피쳐 (1,577,1024)
        -> [attn_pool + proj = CPU float (vendored pe_vendor 가중치)] -> 임베딩 (1,1024)

이유: attn_pool(577토큰->1토큰 cross-attention pooling)은 NPU INT8 양자화에서 구조적으로
깨진다(full-NPU cos 0.46). 이 작은 head(전체 연산의 0.8%)만 CPU float로 돌린다.

TRTInference(trt_load.py)와 동일 인터페이스: model(image) -> (B,1024).
입력: torch.Tensor (B,3,336,336)/(3,336,336) 또는 numpy. 전처리(preprocess_image)는 호출측.

요구: qbruntime + NPU(/dev/aries0), feat MXQ. 모델 코드는 vendored(pe_npu/pe_vendor)라 외부 레포 불필요.
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


class MXQInferenceHybrid:
    """NPU trunk(feat MXQ) + CPU pool/proj head. TRTInference 호환.

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
