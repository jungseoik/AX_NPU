#!/usr/bin/env python3
"""같은 300장을 PE 1차(제품 경로)로 추론한다.

제품(`npu_pe_2stage`)과 동일하게:
  · 전처리  preprocess.preprocess_image (cv2 336 + 0.5 정규화)
  · 추론    MXQInferenceFull (image -> 1024 임베딩, 전부 NPU)
  · 판정    event._decide_categories_by_distribution (METHOD=dist 기본)
            = 카테고리별 유사도 히스토그램(80bin)과 normal 히스토그램의 IoU 가
              임계값(0.3) 미만이면 그 카테고리를 후보로 본다
"""
import json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, "/home/gpuadmin/AX_NPU")
from pe_npu.inference import MXQInferenceFull
import cv2, torch

# 제품(npu_pe_2stage/preprocess.py)과 동일: cv2 INTER_LINEAR 336 + mean/std 0.5
MEAN = STD = 0.5


def preprocess(imgs):
    arr = np.stack([cv2.resize(im, (336, 336), interpolation=cv2.INTER_LINEAR)
                    for im in imgs])                       # (B,336,336,3) uint8
    x = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().float().div_(255.0)
    return x.sub_(MEAN).div_(STD).numpy()                  # (B,3,336,336)

CARD = int(sys.argv[1]) if len(sys.argv) > 1 else 3
TXT = "/home/gpuadmin/AX_NPU/third_party/Product-AI-mono/assets/model/text_features.json"
IOU_BINS, IOU_THR, NORMAL = 80, 0.3, 0
CAT_ID = {"falldown": 1, "fire": 2, "smoke": 3}


def hist_iou(a, b):
    if a is None or b is None:
        return 0.0
    u = float(np.maximum(a, b).sum())
    return float(np.minimum(a, b).sum()) / u if u > 0 else 0.0


def decide(sim, masks):
    by = {c: sim[m] for c, m in masks.items() if m.sum() > 0}
    flat = np.concatenate(list(by.values()))
    lo, hi = float(flat.min()), float(flat.max())
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    edges = np.linspace(lo, hi, IOU_BINS + 1)
    h = {}
    for c, s in by.items():
        cnt, _ = np.histogram(s, bins=edges)
        t = cnt.sum()
        h[c] = cnt / t if t > 0 else cnt.astype(float)
    ious = {c: hist_iou(h.get(NORMAL), h[c]) for c in h if c != NORMAL}
    return {c for c, v in ious.items() if v < IOU_THR}, ious


def main():
    idx = json.loads(Path("/tmp/ax_eval/index.json").read_text())
    print(f"텍스트 특징 로드 ... ({TXT})", flush=True)
    items = json.load(open(TXT))
    vecs = np.asarray([it["feature"] for it in items], dtype=np.float32)
    cls = np.asarray([it["class"] for it in items])
    masks = {c: (cls == c) for c in np.unique(cls)}
    print(f"  {vecs.shape} / 클래스 {sorted(masks)}", flush=True)

    print(f"PE 모델 로드 (카드 {CARD}) ...", flush=True)
    model = MXQInferenceFull.from_hf(scheme="single", device_id=CARD)

    imgs = [cv2.cvtColor(cv2.imread(r["path"]), cv2.COLOR_BGR2RGB) for r in idx]
    t0 = time.perf_counter()
    batch = preprocess(imgs)
    emb = model(batch)
    el = time.perf_counter() - t0
    print(f"  임베딩 {emb.shape}  {el:.1f}s ({el/len(idx)*1000:.0f}ms/장)", flush=True)
    model.dispose()

    out = []
    for r, e in zip(idx, np.asarray(emb)):
        sim = e @ vecs.T
        preds, ious = decide(sim, masks)
        cid = CAT_ID[r["category"]]
        out.append({**r, "answer": "yes" if cid in preds else "no",
                    "predicts": sorted(int(x) for x in preds),
                    "iou": round(float(ious.get(cid, -1)), 4)})
    Path("/tmp/ax_eval/raw_pe.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"저장 /tmp/ax_eval/raw_pe.json  ({len(out)}건)")
    print("=== PE_DONE ===")


if __name__ == "__main__":
    main()
