"""캐시된 임베딩 + GT 를 하나의 작업 배열로 조립."""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np

from .gt import CATEGORIES, FOLDERS, frame_labels, load_labels, video_key


@dataclass
class Dataset:
    emb: np.ndarray      # (N,1024) float32, L2 정규화
    lab: np.ndarray      # (N,C) bool
    vid: np.ndarray      # (N,) int32  비디오 인덱스
    frame: np.ndarray    # (N,) int32  원본 프레임 인덱스
    keys: list           # 비디오별 "<folder>/<name>"
    folders: np.ndarray  # (V,) 비디오별 폴더명
    categories: tuple

    @property
    def n_videos(self):
        return len(self.keys)

    def subset_videos(self, vid_mask: np.ndarray) -> "Dataset":
        keep = vid_mask[self.vid]
        old2new = -np.ones(len(self.keys), np.int32)
        old2new[np.where(vid_mask)[0]] = np.arange(vid_mask.sum())
        return Dataset(self.emb[keep], self.lab[keep], old2new[self.vid[keep]],
                       self.frame[keep], [k for k, m in zip(self.keys, vid_mask) if m],
                       self.folders[vid_mask], self.categories)


def load(emb_dir="wave_npu/cache/emb_tta", root="eval/datasets/TTA_인증용",
         fps_sample: float = 0.0, categories=CATEGORIES, folders=FOLDERS) -> Dataset:
    """fps_sample>0 이면 그 fps 로 균등 서브샘플(재디코딩 없음), 0 이면 전 프레임."""
    labels = load_labels(root, categories, folders)
    E, L, V, F, keys, folder_list = [], [], [], [], [], []
    for f in folders:
        for p in sorted(glob.glob(os.path.join(emb_dir, f, "*.npz"))):
            key = f"{f}/{os.path.splitext(os.path.basename(p))[0]}"
            if key not in labels:
                raise KeyError(f"라벨 없음: {key}")
            z = np.load(p, allow_pickle=True)
            emb = z["emb"].astype(np.float32)
            fr = z["frames"].astype(np.int32)
            fps = float(z["fps"])
            if fps_sample > 0:
                n = max(1, int(round(len(fr) / fps * fps_sample)))
                sel = np.floor(np.arange(n) * len(fr) / n).astype(int)
                emb, fr = emb[sel], fr[sel]
            E.append(emb)
            L.append(frame_labels(labels[key]["events"], fr, fps, categories))
            V.append(np.full(len(fr), len(keys), np.int32))
            F.append(fr)
            keys.append(key)
            folder_list.append(f)
    if not keys:
        raise FileNotFoundError(f"임베딩 캐시가 비었습니다: {emb_dir}")
    return Dataset(np.concatenate(E), np.concatenate(L), np.concatenate(V),
                   np.concatenate(F), keys, np.asarray(folder_list), tuple(categories))


def load_text(path="wave_npu/cache/text_feats.npz"):
    z = np.load(path, allow_pickle=True)
    return z["emb"].astype(np.float32), z["cls"].astype(np.int32), list(z["prompt"])


def load_person(person_dir="wave_npu/cache/person_tta", root="eval/datasets/TTA_인증용",
                categories=("intrusion",), folders=None, signal="max_conf"):
    """YOLO11 사람검출 캐시 → Dataset 과 같은 축(프레임 정렬)의 신호/라벨.

    반환: (score (N,), lab (N,C) bool, vid (N,), keys, folders(V,))
    """
    import glob
    from .gt import frame_labels, load_labels
    folders = folders or ("falldown", "fire", "intrusion", "smoke")
    labels = load_labels(root, categories, folders)
    S, L, V, keys, fl = [], [], [], [], []
    for f in folders:
        for p in sorted(glob.glob(os.path.join(person_dir, f, "*.npz"))):
            key = f"{f}/{os.path.splitext(os.path.basename(p))[0]}"
            z = np.load(p)
            S.append(z[signal].astype(np.float32))
            L.append(frame_labels(labels[key]["events"], z["frames"], float(z["fps"]), categories))
            V.append(np.full(len(z["frames"]), len(keys), np.int32))
            keys.append(key)
            fl.append(f)
    if not keys:
        raise FileNotFoundError(f"사람검출 캐시가 비었습니다: {person_dir}")
    return (np.concatenate(S), np.concatenate(L), np.concatenate(V), keys, np.asarray(fl))
