"""
미리 컴파일된 NPU 자산(MXQ trunk + pool head)을 HuggingFace Hub에서 받아오는 헬퍼.

옵션 B(가져와 쓰기): qbcompiler/원본 PE 전체 가중치 없이, 미리 컴파일해 올려둔 산출물만
받아 바로 추론한다. MXQ는 aries2 아키텍처 바이너리라 어디서 컴파일하든 동일하므로
한 번 만들어 공유하면 된다.

HF repo 구조 (기본 PIA-SPACE-LAB/MXQ_NPU):
  pe_full.mxq        # full NPU (trunk+attn_pool, QK^T 16bit, image->embedding). 권장
  pe_feat.mxq        # NPU trunk만 (INT8) — hybrid(+CPU pool head)용. 레거시
  pe_pool_head.pt    # hybrid용 CPU pool head 가중치 (attn_pool + proj, 약 55MB)
"""
from __future__ import annotations

import os

HF_REPO = "PIA-SPACE-LAB/MXQ_NPU"
FULL_MXQ = "pe_full.mxq"
FEAT_MXQ = "pe_feat.mxq"
POOL_HEAD = "pe_pool_head.pt"


def download_asset(filename: str, repo_id: str = HF_REPO, revision: str = None):
    """HF Hub에서 파일 1개를 받아 로컬 캐시 경로를 반환."""
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)


def ensure_full_mxq(path: str = None, repo_id: str = HF_REPO, revision: str = None):
    """로컬 path가 있으면 그대로, 없으면 HF에서 full MXQ(image->embedding)를 받아 경로 반환."""
    if path and os.path.exists(path):
        return path
    return download_asset(FULL_MXQ, repo_id, revision)


def ensure_feat_mxq(path: str = None, repo_id: str = HF_REPO, revision: str = None):
    """로컬 path가 있으면 그대로, 없으면 HF에서 trunk MXQ를 받아 경로 반환 (hybrid 레거시)."""
    if path and os.path.exists(path):
        return path
    return download_asset(FEAT_MXQ, repo_id, revision)


def ensure_pool_head(path: str = None, repo_id: str = HF_REPO, revision: str = None):
    """로컬 path가 있으면 그대로, 없으면 HF에서 pool head를 받아 경로 반환."""
    if path and os.path.exists(path):
        return path
    return download_asset(POOL_HEAD, repo_id, revision)
