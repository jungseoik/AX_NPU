"""
미리 컴파일된 NPU 자산(MXQ trunk + pool head)을 HuggingFace Hub에서 받아오는 헬퍼.

옵션 B(가져와 쓰기): qbcompiler/원본 PE 전체 가중치 없이, 미리 컴파일해 올려둔 산출물만
받아 바로 추론한다. MXQ는 aries2 아키텍처 바이너리라 어디서 컴파일하든 동일하므로
한 번 만들어 공유하면 된다.

HF repo 구조 (기본 PIA-SPACE-LAB/MXQ_NPU):
  README.md              # 모델카드
  single/pe_full.mxq     # full NPU (trunk+attn_pool, QK^T 16bit, image->embedding) — 코어모드별 폴더
  multi/pe_full.mxq      #   각 폴더에 pe_full.mxq + CALIBRATION.md
  global4/pe_full.mxq    #   ensure_full_mxq(scheme=...)로 선택
  global8/pe_full.mxq
  pe_feat.mxq            # [레거시] NPU trunk만 (INT8) — hybrid(+CPU pool head)용
  pe_pool_head.pt        # [레거시] hybrid용 CPU pool head 가중치 (attn_pool + proj, 약 55MB)
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


# 양자화 스킴 — HF repo의 `<quant>/<scheme>/pe_full.mxq` 폴더명.
#   W8A16       weight 8bit / activation 16bit(output·ffn) — **기본값, 기존 배포본과 동일**
#   W4A16       weight 4bit(value만 8) / activation 동일 — 크기 -42%, 처리량 +43%, 정확도 하락
#   W4A8_L5A16  W4 + activation 8bit, outlier 상위 5개 텐서만 16bit — 속도 우선
DEFAULT_QUANT = "W8A16"
QUANT_SCHEMES = ("W8A16", "W4A16", "W4A8_L5A16")


def ensure_full_mxq(path: str = None, repo_id: str = HF_REPO, revision: str = None,
                    scheme: str = "single", quant: str = DEFAULT_QUANT):
    """로컬 path가 있으면 그대로, 없으면 HF에서 full MXQ(image->embedding)를 받아 경로 반환.

    scheme: 코어모드 (single|multi|global4|global8)
    quant : 양자화 스킴 (W8A16 기본 | W4A16 | W4A8_L5A16)

    HF 경로는 `<quant>/<scheme>/pe_full.mxq`. 기본 스킴은 예전 경로(`<scheme>/pe_full.mxq`)로
    자동 폴백하여 구버전 배포본과도 호환된다.
    """
    if path and os.path.exists(path):
        return path
    if quant not in QUANT_SCHEMES:
        raise ValueError(f"알 수 없는 양자화 스킴 {quant!r} — 사용 가능: {list(QUANT_SCHEMES)}")
    try:
        return download_asset(f"{quant}/{scheme}/{FULL_MXQ}", repo_id, revision)
    except Exception:
        if quant != DEFAULT_QUANT:
            raise
        # 구조 개편 이전 배포본(최상위 <scheme>/pe_full.mxq)
        return download_asset(f"{scheme}/{FULL_MXQ}", repo_id, revision)


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
