"""NPU(MXQ INT8 계열) vs CPU fp32 레퍼런스 — 양자화가 이 과제의 F1 을 깎는가.

같은 프레임 집합에 대해 임베딩 코사인 유사도와, **동일한 판정 설정에서의 프레임 F1**을 나란히 낸다.
(fp32 캐시는 CPU 가 느려 0.5fps 로만 뽑으므로, NPU 쪽도 같은 프레임으로 맞춰 비교한다.)
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from . import data
from .evaluate import evaluate, table
from .score import category_scores, sims_matrix, smooth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npu", default="wave_npu/cache/emb_tta")
    ap.add_argument("--fp32", default="wave_npu/cache/emb_tta_fp32")
    ap.add_argument("--root", default="eval/datasets/TTA_인증용")
    ap.add_argument("--text", default="wave_npu/cache/text_feats.npz")
    ap.add_argument("--mask", default="wave_npu/artifacts/mask_mean.npz")
    ap.add_argument("--combo", default="wave_npu/artifacts/combo_24fps.json")
    ap.add_argument("--fps", type=float, default=0.5)
    a = ap.parse_args()

    ref = data.load(a.fp32, a.root, fps_sample=0.0)          # fp32 캐시는 이미 서브샘플
    npu = data.load(a.npu, a.root, fps_sample=a.fps)
    assert npu.keys == ref.keys, "비디오 집합 불일치"
    if not np.array_equal(npu.frame, ref.frame):
        raise SystemExit(f"프레임 인덱스 불일치: npu {npu.frame[:8]} vs fp32 {ref.frame[:8]}")
    print(f"[cmp] videos={len(ref.keys)} frames={len(ref.emb)} (동일 프레임 확인)")

    cos = np.sum(npu.emb * ref.emb, 1)
    print(f"[cmp] 임베딩 cos(NPU, fp32): mean={cos.mean():.4f} min={cos.min():.4f} "
          f"p1={np.quantile(cos, 0.01):.4f} p50={np.median(cos):.4f}")

    temb, cls, _ = data.load_text(a.text)
    mask = np.load(a.mask, allow_pickle=True)["mask"]
    combo = json.load(open(a.combo))
    te, cl = temb[mask], cls[mask]

    out = {}
    for name, ds in (("NPU (MXQ W8A16)", npu), ("CPU fp32", ref)):
        sims = sims_matrix(ds.emb, te)
        res = {}
        for i, c in enumerate(ds.categories):
            raw = category_scores(sims, cl, ds.categories, rule=combo[c]["rule"])
            # 0.5fps 라 24fps 기준 창을 그대로 못 쓴다 → 창은 1(평활 없음)로 통일해 비교
            res[c] = evaluate(raw[:, [i]], ds.lab[:, [i]], (c,))[c]
        res["macro_f1"] = float(np.mean([res[c]["f1"] for c in ds.categories]))
        out[name] = res
        print("\n" + table(res, ds.categories, f"=== {name} (0.5fps, 평활 없음, 동일 프롬프트·규칙) ==="))
    d = out["NPU (MXQ W8A16)"]["macro_f1"] - out["CPU fp32"]["macro_f1"]
    print(f"\n[cmp] macro F1 차이 (NPU − fp32) = {d:+.4f}")


if __name__ == "__main__":
    main()
