"""TTA_인증용 라벨(json, 초 단위) → 프레임 레벨 이진 라벨.

확정된 평가 프로토콜:
  - 음성은 **전체 200영상 교차**. 어떤 영상의 json 에 카테고리 c 이벤트가 없으면
    그 영상의 모든 프레임은 c 에 대해 음성이다(intrusion 영상 포함).
  - 영상 파일명이 폴더 간 중복되므로(176 unique / 200 files) 키는 "<폴더>/<파일명>".
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

CATEGORIES = ("falldown", "fire", "smoke")
FOLDERS = ("falldown", "fire", "intrusion", "smoke")


def video_key(folder: str, path: str) -> str:
    return f"{folder}/{os.path.splitext(os.path.basename(path))[0]}"


def load_labels(root: str, categories=CATEGORIES, folders=FOLDERS) -> dict:
    """{key: {"events": {cat: [(t0,t1), ...]}, "folder": ...}}"""
    out = {}
    for f in folders:
        for jp in sorted(glob.glob(os.path.join(root, f, "*.json"))):
            d = json.load(open(jp, encoding="utf-8"))
            ev = {c: [] for c in categories}
            for e in d.get("events", []):
                c = e["category"]
                if c in ev:
                    t0, t1 = float(e["timestamp"][0]), float(e["timestamp"][1])
                    ev[c].append((min(t0, t1), max(t0, t1)))
            out[video_key(f, jp)] = {"events": ev, "folder": f}
    return out


def frame_labels(events: dict, frames: np.ndarray, fps: float,
                 categories=CATEGORIES) -> np.ndarray:
    """(T, C) bool. frames 는 프레임 인덱스, GT timestamp 는 초 → fps 로 변환."""
    lab = np.zeros((len(frames), len(categories)), bool)
    t = frames / float(fps)
    for ci, c in enumerate(categories):
        for t0, t1 in events.get(c, []):
            lab[:, ci] |= (t >= t0) & (t <= t1)
    return lab
