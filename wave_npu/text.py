"""프롬프트 CSV → PE-Core text tower 임베딩 (CPU).

vision 은 NPU(MXQ)지만 text tower 는 추론 경로가 아니다 — 프롬프트당 1회만 뽑아
캐시하면 이후 전부 numpy 연산이라 CPU 로 충분하다.

입력 CSV 컬럼: ID, class, prompt
출력 npz: { emb:(N,1024) float32 L2정규화, cls:(N,) int32, prompt:(N,) str }
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

PIA_WAVE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "third_party", "PIA_Wave")


def load_prompts(csv_path: str):
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    prompts = [r["prompt"].strip() for r in rows]
    cls = np.asarray([int(r["class"]) for r in rows], np.int32)
    return prompts, cls


def encode_text(prompts, model_name="PE-Core-L14-336", batch=256):
    if PIA_WAVE not in sys.path:
        sys.path.insert(0, PIA_WAVE)
    import core.vision_encoder.pe as pe
    import core.vision_encoder.transforms as transforms

    model = pe.CLIP.from_config(model_name, pretrained=True).eval()
    tok = transforms.get_text_tokenizer(model.context_length)

    out, t0 = [], time.time()
    with torch.inference_mode():
        for s in range(0, len(prompts), batch):
            f = model(text=tok(prompts[s:s + batch]))[1].float()
            f = f / f.norm(dim=-1, keepdim=True)
            out.append(f.numpy())
            n = min(s + batch, len(prompts))
            el = time.time() - t0
            print(f"[text] {n}/{len(prompts)} | {n / el:.0f} p/s | "
                  f"ETA {el / n * (len(prompts) - n) / 60:.1f}m", flush=True)
    return np.concatenate(out, 0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="third_party/PIA_Wave/data/text_features_tuningfree_v2_soil최종.csv")
    ap.add_argument("--out", default="wave_npu/cache/text_feats.npz")
    ap.add_argument("--batch", type=int, default=256)
    a = ap.parse_args()

    prompts, cls = load_prompts(a.csv)
    print(f"[text] {len(prompts)} prompts, classes={np.bincount(cls).tolist()}", flush=True)
    emb = encode_text(prompts, batch=a.batch)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(a.out, emb=emb, cls=cls, prompt=np.asarray(prompts, dtype=object))
    print(f"[text] saved {a.out} {emb.shape}", flush=True)


if __name__ == "__main__":
    main()
