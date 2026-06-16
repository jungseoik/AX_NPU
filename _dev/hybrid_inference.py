"""
MXQInferenceHybrid — PE-Core-L14-336 비전인코더 NPU 추론 (hybrid).

구조 (정확도 0.9972 검증, reports/SOLUTION_single_io_compile.md):
  image → [24 transformer blocks = NPU INT8 (feat MXQ)] → 토큰피쳐 (1,577,1024)
        → [attn_pool + proj = CPU float (perception_models 가중치)] → 임베딩 (1,1024)

이유: attn_pool(577토큰→1토큰 cross-attention pooling)은 Mobilint NPU의 INT8 양자화에서
구조적으로 깨진다(full-NPU cos 0.46). Model Zoo/SDK 전체에 attention pooling을 NPU에서
정확도 유지한 선례가 없어, 이 작은 head(전체 연산의 0.8%)만 CPU float로 돌린다.

TRTInference(trt_load.py)와 동일 인터페이스: model(image_cuda) → (B,1024) 호환.
입력: torch.Tensor (B,3,336,336) 또는 (3,336,336) / numpy. 전처리(resize 336 + normalize 0.5)는 호출측.

요구: qbruntime + NPU(/dev/aries0), perception_models(Product-AI-mono), feat MXQ.
"""
from __future__ import annotations

import os
import sys

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

import qbruntime

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FEAT_MXQ = os.path.normpath(os.path.join(HERE, "..", "pe_onnx_export", "out", "pe_feat.mxq"))
PE_EXPORT_DIR = os.path.normpath(os.path.join(HERE, "..", "pe_onnx_export"))
IMAGE_SIZE = 336


class MXQInferenceHybrid:
    """NPU trunk(feat MXQ) + CPU pool/proj head. TRTInference 호환."""

    def __init__(self, feat_mxq_path: str = DEFAULT_FEAT_MXQ, model_name: str = "PE-Core-L14-336",
                 device_id: int = 0):
        if not _HAS_TORCH:
            raise RuntimeError("torch 필요 (CPU pool head 연산)")

        # --- NPU: forward_features MXQ ---
        self.acc = qbruntime.Accelerator(device_id)
        self.model = qbruntime.Model(feat_mxq_path)
        self.model.launch(self.acc)

        # --- CPU: pool head (attn_pool + proj) 가중치 로드 ---
        if PE_EXPORT_DIR not in sys.path:
            sys.path.insert(0, PE_EXPORT_DIR)
        from export_pe_onnx import _add_packages_to_path
        _add_packages_to_path()
        from pia.ai.tasks.T2VRet.models.PE.perception_models.pe_core.vision_encoder import pe as pe_mod

        clip = pe_mod.CLIP.from_config(model_name, device="cpu", load_default_weights=True).float().eval()
        self.visual = clip.visual          # _pool(attn_pool) + proj 사용 (trunk는 NPU가 대체)
        self.image_size = IMAGE_SIZE

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
            chw = arr[i]                               # (3,336,336)
            hwc = np.ascontiguousarray(chw.transpose(1, 2, 0))  # NPU 입력 = HWC
            feo = self.model.infer(hwc)                # NPU trunk
            feo = feo[0] if isinstance(feo, (list, tuple)) else feo
            feat = torch.from_numpy(np.asarray(feo).reshape(1, 577, 1024).astype(np.float32))
            with torch.no_grad():
                pooled = self.visual._pool(feat)       # CPU attn_pool
                if self.visual.proj_dim is not None:
                    pooled = pooled @ self.visual.proj  # CPU proj
            embs.append(pooled.reshape(-1).numpy())

        out = np.stack(embs, axis=0)                   # (B,1024)
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
    print("input:", dummy.shape, "→ output:", out.shape)
