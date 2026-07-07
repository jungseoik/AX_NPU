"""
미리 컴파일된 YOLO NPU MXQ를 HuggingFace Hub에서 받아오는 헬퍼 (PE assets와 대칭).

HF repo 구조 (PE와 같은 repo `PIA-SPACE-LAB/MXQ_NPU` 안 `yolo/` 하위):
  yolo/
    yolo11n/<scheme>/yolo11n.mxq + CALIBRATION.md   # scheme = single/multi/global4/global8
    yolo11n/yolo11n.onnx                             # fp32 원본(재검증/재컴파일용)
    yolo11m/<scheme>/yolo11m.mxq + ...
    yolo11l/<scheme>/yolo11l.mxq + ...

모델(yolo11n/m/l)·코어모드(scheme)를 골라 당겨 쓴다:
  YOLONPU.from_hf(model="yolo11m", scheme="single")           # 단일 카드
  YOLONPU.from_hf(model="yolo11m", scheme="single", device_ids="auto")   # 멀티카드
"""
from __future__ import annotations

import os

HF_REPO = "PIA-SPACE-LAB/MXQ_NPU"
YOLO_PREFIX = "yolo"


def ensure_yolo_mxq(model: str = "yolo11m", scheme: str = "single", path: str = None,
                    repo_id: str = HF_REPO, revision: str = None):
    """로컬 path가 있으면 그대로, 없으면 HF `yolo/<model>/<scheme>/<model>.mxq`를 받아 경로 반환."""
    if path and os.path.exists(path):
        return path
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo_id, revision=revision,
                           filename=f"{YOLO_PREFIX}/{model}/{scheme}/{model}.mxq")


def ensure_yolo_onnx(model: str = "yolo11m", path: str = None,
                     repo_id: str = HF_REPO, revision: str = None):
    """fp32 ONNX(`yolo/<model>/<model>.onnx`)를 받아 경로 반환 (재검증/재컴파일용)."""
    if path and os.path.exists(path):
        return path
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo_id, revision=revision,
                           filename=f"{YOLO_PREFIX}/{model}/{model}.onnx")
