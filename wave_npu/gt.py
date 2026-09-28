"""평가 데이터셋 라벨 → 프레임 레벨 이진 라벨. **두 가지 라벨 형식을 모두 읽는다.**

| 데이터셋 | 라벨 | 형식 |
|---|---|---|
| `TTA_인증용` | `<이름>.json` | `events[].timestamp [시작초, 끝초]` → fps 로 프레임 변환 |
| `TTA_clip`   | `<이름>.csv`  | `clip_frame,<event>` 프레임 단위 0/1 (변환 없음) |

CSV 쪽이 원본 그대로다 — `TTA_clip` 은 fps 가 7~50 으로 섞여 있어 초 단위로 왕복하면
경계에서 오차가 난다. 새 데이터는 CSV 형식으로 통일해 들여온다(`wave_npu.import_clips`).

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


def load_frame_csv(path: str) -> tuple:
    """`clip_frame,<event>` CSV → (이벤트 컬럼명, 0/1 np.ndarray)."""
    import csv as _csv
    rows = list(_csv.DictReader(open(path, encoding="utf-8-sig")))
    if not rows:
        return None, np.zeros(0, bool)
    col = [c for c in rows[0] if c != "clip_frame"][0]
    return col, np.asarray([int(r[col]) for r in rows], bool)


def load_labels_csv(root: str, categories, folders) -> dict:
    """프레임 CSV 형식(`TTA_clip`) → {key: {"frames": bool 배열, "folder": ...}}.

    폴더명 = 카테고리명이다(항목당 1폴더). 그 폴더의 CSV 가 그 카테고리의 정답이고,
    다른 카테고리에 대해서는 전부 음성으로 본다(항목별 독립 이진 시험).
    """
    out = {}
    for f in folders:
        for cp in sorted(glob.glob(os.path.join(root, f, "*.csv"))):
            _, v = load_frame_csv(cp)
            key = video_key(f, cp)
            out[key] = {"frames": {f: v}, "folder": f, "n": len(v)}
    return out


def frame_labels_csv(entry: dict, n: int, categories) -> np.ndarray:
    """(T, C) bool. 해당 폴더의 카테고리만 CSV 값을 쓰고 나머지는 음성."""
    lab = np.zeros((n, len(categories)), bool)
    for ci, c in enumerate(categories):
        v = entry["frames"].get(c)
        if v is None:
            continue
        lab[:min(n, len(v)), ci] = v[:n]
    return lab


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
