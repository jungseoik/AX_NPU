"""
[레거시 — hybrid 전용] pool head(attn_pool + proj) 가중치만 추출해 pe_pool_head.pt로 저장.

현재 권장 경로는 full NPU(`MXQInferenceFull`, attn_pool도 NPU)라 이 파일이 필요 없다.
이 스크립트는 레거시 hybrid(`MXQInferenceHybrid`, CPU pool head) 호환용으로만 유지된다.

hybrid 구조상 NPU trunk(MXQ)는 weight를 자체 포함하지만, CPU에서 도는 attn_pool+proj는
별도 가중치가 필요해, 원본 PE 전체 가중치(2GB+) 없이 CPU pool head를 복원하려고 그 부분만
떼어 저장(약 55MB)한다.

사용:
  python -m pe_npu.export_pool_head --out ./pe_npu/out/pe_pool_head.pt
"""
from __future__ import annotations

import argparse

import torch

from .pe_model import MODEL_NAME, load_pe


def export_pool_head(out_path: str = "pe_npu/out/pe_pool_head.pt", model_name: str = MODEL_NAME):
    """원본 PE에서 attn_pool state_dict + proj 텐서를 추출해 저장."""
    visual = load_pe(model_name=model_name, mode="clip", patch=False).visual
    payload = {
        "model_name": model_name,
        "attn_pool": visual.attn_pool.state_dict(),
        "proj": (visual.proj.detach().clone() if visual.proj is not None else None),
        "proj_dim": visual.proj_dim,
        "pool_type": visual.pool_type,
    }
    torch.save(payload, out_path)
    print(f"[OK] pool head 저장 -> {out_path}  (pool_type={visual.pool_type}, proj_dim={visual.proj_dim})")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="pe_npu/out/pe_pool_head.pt")
    ap.add_argument("--model-name", default=MODEL_NAME)
    args = ap.parse_args()
    export_pool_head(args.out, args.model_name)
