"""YOLO11(NPU)로 프레임별 사람 존재 신호 추출 — intrusion 계열 카테고리용.

intrusion 은 "사람이 등장하면 이벤트"이므로 CLIP 유사도가 아니라 **검출기**가 맞는 도구다.
PE 임베딩 캐시와 같은 방식으로 프레임별 신호를 한 번만 뽑아 캐시한다.

출력 npz: { frames, n_person, max_conf, max_area, sum_area, fps, n_frames }
"""
from __future__ import annotations

import argparse
import os
import time

import cv2
import numpy as np

PERSON_CLS = 0  # COCO


def sample_indices(n_all: int, fps_video: float, fps_sample: float):
    if fps_sample <= 0:
        return np.arange(n_all)
    k = max(1, int(round(n_all / fps_video * fps_sample)))
    return np.floor(np.arange(k) * n_all / k).astype(int)


def decode_bgr(path: str, keep) -> tuple:
    cap = cv2.VideoCapture(path)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    n_all = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    keep = set(int(x) for x in keep)
    frames, idxs, i = [], [], -1
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        i += 1
        if i in keep:
            frames.append(bgr)
            idxs.append(i)
    cap.release()
    return frames, np.asarray(idxs, np.int32), fps, n_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="eval/datasets/TTA_인증용")
    ap.add_argument("--out", default="wave_npu/cache/person_tta")
    ap.add_argument("--model", default="yolo11m")
    ap.add_argument("--scheme", default="single")
    ap.add_argument("--device-ids", default="1,3,4,5")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--conf", type=float, default=0.10, help="낮게 잡고 나중에 스윕")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    from yolo_npu.detect import YOLONPU
    from .embed import tta_videos

    det = YOLONPU.load(a.model, a.scheme,
                       device_ids=[int(x) for x in a.device_ids.split(",")],
                       conf_thres=a.conf)
    vids = tta_videos(a.root)
    if a.limit:
        vids = vids[:a.limit]
    t0, n = time.time(), 0
    for i, (cat, path) in enumerate(vids, 1):
        dst = os.path.join(a.out, cat, os.path.splitext(os.path.basename(path))[0] + ".npz")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst):
            continue
        cap = cv2.VideoCapture(path)
        n_all = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_v = float(cap.get(cv2.CAP_PROP_FPS)); cap.release()
        keep = sample_indices(n_all, fps_v, a.fps)
        frames, idxs, fps, n_all = decode_bgr(path, keep)
        if not frames:
            continue
        dets = det.detect_batch(frames)
        H, W = frames[0].shape[:2]
        npn, mc, ma, sa = [], [], [], []
        for d in dets:
            p = [x for x in d if int(x[5]) == PERSON_CLS]
            npn.append(len(p))
            mc.append(max((x[4] for x in p), default=0.0))
            areas = [max(0.0, (x[2] - x[0])) * max(0.0, (x[3] - x[1])) / (W * H) for x in p]
            ma.append(max(areas, default=0.0))
            sa.append(float(sum(areas)))
        np.savez_compressed(dst, frames=idxs, n_person=np.asarray(npn, np.int32),
                            max_conf=np.asarray(mc, np.float32), max_area=np.asarray(ma, np.float32),
                            sum_area=np.asarray(sa, np.float32),
                            fps=np.float32(fps), n_frames=np.int32(n_all))
        n += len(frames)
        el = time.time() - t0
        print(f"[person] {i}/{len(vids)} {cat}/{os.path.basename(path)} T={len(frames)} "
              f"| {n / el:.1f} f/s | ETA {el / i * (len(vids) - i) / 60:.1f}m", flush=True)
    print(f"[person] done {n} frames in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
