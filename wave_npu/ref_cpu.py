"""fp32 CPU 레퍼런스 임베딩 — NPU(INT8 양자화 MXQ)가 정확도를 얼마나 깎는지 재기 위한 대조군.

원본 PIA_Wave 는 GPU/TensorRT fp16 이지만 이 서버엔 GPU 가 없으므로, 같은 가중치의
PE-Core vision tower 를 CPU fp32 로 돌린 값을 '양자화 전' 기준으로 삼는다.
캐시 포맷은 wave_npu.embed 와 동일해서 같은 평가 코드를 그대로 쓸 수 있다.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from .embed import decode_video, to_model_input, tta_videos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="eval/datasets/TTA_인증용")
    ap.add_argument("--out", default="out/wave/emb_tta_fp32")
    ap.add_argument("--fps", type=float, default=2.0, help="레퍼런스는 서브샘플만 (CPU가 느림)")
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    from pe_npu.pe_model import load_pe
    model = load_pe(mode="full", patch=False).eval()

    vids = tta_videos(a.root)
    if a.limit:
        vids = vids[:a.limit]
    t0, n = time.time(), 0
    for i, (cat, path) in enumerate(vids, 1):
        dst = os.path.join(a.out, cat, os.path.splitext(os.path.basename(path))[0] + ".npz")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst):
            continue
        # data.load 의 서브샘플과 **완전히 같은 프레임**을 골라야 NPU 와 1:1 비교가 된다.
        import cv2
        cap = cv2.VideoCapture(path)
        n_all = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
        fps_v = 24.0
        k = max(1, int(round(n_all / fps_v * a.fps))) if a.fps > 0 else n_all
        keep = set(np.floor(np.arange(k) * n_all / k).astype(int).tolist())
        res = decode_video(path, keep=keep)
        if res is None:
            continue
        u8, idxs, fps, n_frames = res
        out = []
        with torch.inference_mode():
            for s in range(0, len(u8), a.batch):
                x = torch.from_numpy(to_model_input(u8[s:s + a.batch]))
                out.append(model(x).float().numpy())
        emb = np.concatenate(out, 0)
        emb /= np.linalg.norm(emb, axis=-1, keepdims=True) + 1e-12
        np.savez_compressed(dst, emb=emb.astype(np.float16), frames=idxs,
                            fps=np.float32(fps), n_frames=np.int32(n_frames))
        n += len(emb)
        el = time.time() - t0
        print(f"[ref] {i}/{len(vids)} {cat}/{os.path.basename(path)} T={len(emb)} | "
              f"{n / el:.1f} emb/s | ETA {el / i * (len(vids) - i) / 60:.1f}m", flush=True)
    print(f"[ref] done {n} embeddings in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
