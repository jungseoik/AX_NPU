#!/usr/bin/env python3
"""같은 설정 3회 반복으로 실행 간 변동폭을 재고, 크기 차이가 그 밖인지 본다."""
import json, statistics as st
from pathlib import Path

MODES = ["224x224", "256x192"]
CATS = ["fire", "smoke", "falldown"]
NAME = {"bs32": "VLM bs32", "ft": "VLM ft"}
runs = {m: [json.loads(Path(f"/tmp/ax_eval/rep_{m}_{r}.json").read_text()) for r in (1, 2, 3)]
        for m in MODES}


def met(rs):
    tp = sum(1 for r in rs if r["truth"] == "yes" and r["answer"] == "yes")
    fp = sum(1 for r in rs if r["truth"] == "no" and r["answer"] == "yes")
    fn = sum(1 for r in rs if r["truth"] == "yes" and r["answer"] != "yes")
    tn = len(rs) - tp - fp - fn
    p = tp / (tp + fp) if tp + fp else 0
    rc = tp / (tp + fn) if tp + fn else 0
    return tp, fp, (tp + tn) / len(rs) if rs else 0, 2 * p * rc / (p + rc) if p + rc else 0


def cell(vals):
    lo, hi = min(vals), max(vals)
    return f"{st.median(vals):.0f}" + (f" ({lo}~{hi})" if lo != hi else "")


print("■ 3회 반복 — 중앙값 (최소~최대)\n")
print(f"{'카테고리':<10}{'서버':<11}{'크기':<10}{'정탐':>12}{'오탐':>12}{'F1 중앙':>9}{'F1 폭':>8}")
print("-" * 76)
summary = {}
for cat in CATS + [None]:
    for t in ("bs32", "ft"):
        for m in MODES:
            vs = [met(r[t] if cat is None else [x for x in r[t] if x["category"] == cat])
                  for r in runs[m]]
            tps = [v[0] for v in vs]; fps = [v[1] for v in vs]; f1s = [v[3] for v in vs]
            summary[(cat, t, m)] = (tps, fps, f1s)
            print(f"{(cat or '전체'):<10}{NAME[t]:<11}{m:<10}{cell(tps):>12}{cell(fps):>12}"
                  f"{st.median(f1s):>9.3f}{max(f1s)-min(f1s):>8.3f}")
        print()

print("\n■ 크기 차이가 실행 변동폭을 넘는가 (전체 기준)\n")
for t in ("bs32", "ft"):
    a_tp, a_fp, a_f1 = summary[(None, t, "224x224")]
    b_tp, b_fp, b_f1 = summary[(None, t, "256x192")]
    print(f"  {NAME[t]}")
    for nm, A, B in (("정탐", a_tp, b_tp), ("오탐", a_fp, b_fp)):
        sep = "겹치지 않음 → 유의" if (min(B) > max(A) or max(B) < min(A)) else "범위 겹침 → 잡음 가능"
        print(f"    {nm}: 224 {min(A)}~{max(A)} / 256 {min(B)}~{max(B)}   {sep}")
    sep = "겹치지 않음 → 유의" if (min(b_f1) > max(a_f1) or max(b_f1) < min(a_f1)) else "범위 겹침 → 잡음 가능"
    print(f"    F1 : 224 {min(a_f1):.3f}~{max(a_f1):.3f} / 256 {min(b_f1):.3f}~{max(b_f1):.3f}   {sep}")

print("\n■ 같은 설정 안에서 답이 바뀐 건수 (회차 간, 300장 중)")
for m in MODES:
    for t in ("bs32", "ft"):
        d = [sum(1 for x, y in zip(runs[m][i][t], runs[m][j][t]) if x["answer"] != y["answer"])
             for i, j in ((0, 1), (0, 2), (1, 2))]
        print(f"  {m} {NAME[t]:<11}{d}")
