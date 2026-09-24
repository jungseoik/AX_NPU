"""비디오 → 프레임 → PE-Core 임베딩(NPU) 캐시.

전처리는 pe_npu.preprocess / PIA_Wave encoders._preprocess_crops 와 **비트 동일**하게 맞춘다:
  uint8 CHW → TF.resize(336, bilinear, antialias) → [0,1] → normalize(0.5, 0.5)

전 프레임(24fps)을 한 번 뽑아두면 이후 2/3/6fps 샘플링은 재디코딩 없이 서브샘플로 끝난다.
출력: <out>/<category>/<video>.npz  { emb: (T,1024) float16 L2정규화, frames: (T,) int32, fps, n_frames }
"""
from __future__ import annotations

import argparse
import os
import queue
import threading
import time

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

IMAGE_SIZE = 336
MEAN = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
STD = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)


def decode_video(path: str, stride: int = 1, keep=None):
    """비디오 → (uint8 (T,3,336,336), frame_indices, fps, n_frames).

    resize 는 uint8 텐서 위에서 수행한다(참조 전처리와 동일 순서).
    keep 을 주면 그 프레임 인덱스만 전처리한다(느린 CPU 레퍼런스용).
    """
    cap = cv2.VideoCapture(path)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames, idxs = [], []
    i = -1
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        i += 1
        if keep is not None:
            if i not in keep:
                continue
        elif i % stride:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()          # uint8 (3,H,W)
        t = TF.resize(t, [IMAGE_SIZE, IMAGE_SIZE],
                      interpolation=InterpolationMode.BILINEAR, antialias=True)
        frames.append(t)
        idxs.append(i)
    cap.release()
    if not frames:
        return None
    return torch.stack(frames).numpy(), np.asarray(idxs, np.int32), fps, n_frames


def to_model_input(u8: np.ndarray) -> np.ndarray:
    """uint8 (B,3,336,336) → float32 정규화 (B,3,336,336)."""
    t = torch.from_numpy(u8).float().div_(255.0)
    t = (t - MEAN) / STD
    return t.numpy()


def embed_dataset(videos, out_dir, device_ids, scheme="single", quant=None,
                  batch=64, stride=1, decode_workers=6, overwrite=False):
    from pe_npu.inference import MXQInferenceFull

    todo = []
    for cat, path in videos:
        dst = os.path.join(out_dir, cat, os.path.splitext(os.path.basename(path))[0] + ".npz")
        if overwrite or not os.path.exists(dst):
            todo.append((cat, path, dst))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
    print(f"[embed] {len(todo)}/{len(videos)} videos to process "
          f"(cards={device_ids} scheme={scheme} batch={batch} stride={stride})", flush=True)
    if not todo:
        return

    model = MXQInferenceFull.from_hf(scheme=scheme, device_ids=list(device_ids), quant=quant)
    print(f"[embed] NPU ready: {len(model)} cards, core_mode={model.core_mode}, pool={model.W}", flush=True)

    q: "queue.Queue" = queue.Queue(maxsize=4)

    def producer():
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=decode_workers) as ex:
            for item, fut in [(t, ex.submit(decode_video, t[1], stride)) for t in todo]:
                q.put((item, fut))
        q.put(None)

    threading.Thread(target=producer, daemon=True).start()

    t_start = time.time()
    done = n_emb = 0
    while True:
        got = q.get()
        if got is None:
            break
        (cat, path, dst), fut = got
        res = fut.result()
        if res is None:
            print(f"[embed] SKIP (no frames) {path}", flush=True)
            continue
        u8, idxs, fps, n_frames = res
        embs = []
        for s in range(0, len(u8), batch):
            x = to_model_input(u8[s:s + batch])
            embs.append(model.infer(x))
        emb = np.concatenate(embs, 0).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=-1, keepdims=True) + 1e-12
        np.savez_compressed(dst, emb=emb.astype(np.float16), frames=idxs,
                            fps=np.float32(fps), n_frames=np.int32(n_frames))
        done += 1
        n_emb += len(emb)
        el = time.time() - t_start
        print(f"[embed] {done}/{len(todo)} {cat}/{os.path.basename(path)} "
              f"T={len(emb)} | {n_emb} emb | {n_emb / el:.1f} emb/s | ETA {el / done * (len(todo) - done) / 60:.1f}m",
              flush=True)
    model.dispose()
    print(f"[embed] done: {done} videos, {n_emb} embeddings, {(time.time() - t_start) / 60:.1f} min", flush=True)


def tta_videos(root: str, categories=("falldown", "fire", "intrusion", "smoke")):
    import glob
    out = []
    for c in categories:
        out += [(c, p) for p in sorted(glob.glob(os.path.join(root, c, "*.mp4")))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="eval/datasets/TTA_인증용")
    ap.add_argument("--out", default="out/wave/emb_tta")
    ap.add_argument("--device-ids", default="1,3,4,5")
    ap.add_argument("--scheme", default="single")
    ap.add_argument("--quant", default=None)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--stride", type=int, default=1, help="프레임 스트라이드(1=전 프레임)")
    ap.add_argument("--decode-workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    vids = tta_videos(a.root)
    if a.limit:
        vids = vids[:a.limit]
    embed_dataset(vids, a.out, [int(x) for x in a.device_ids.split(",")],
                  scheme=a.scheme, quant=a.quant, batch=a.batch, stride=a.stride,
                  decode_workers=a.decode_workers, overwrite=a.overwrite)


if __name__ == "__main__":
    main()
