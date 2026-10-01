#!/usr/bin/env python3
"""224x224 vs 256x192 — 같은 300장, 같은 프롬프트. 알람/오탐 개수 기준."""
import json
from pathlib import Path

RUNS = {"224x224": "raw_224x224.json", "256x192": "raw_256x192.json"}
raw = {k: json.loads(Path(f"/tmp/ax_eval/{v}").read_text()) for k, v in RUNS.items()}
NAME = {"bs32": "VLM bs32", "ft": "VLM ft"}
CATS = ["fire", "smoke", "falldown"]


def met(rs):
    tp = sum(1 for r in rs if r["truth"] == "yes" and r["answer"] == "yes")
    fp = sum(1 for r in rs if r["truth"] == "no" and r["answer"] == "yes")
    fn = sum(1 for r in rs if r["truth"] == "yes" and r["answer"] != "yes")
    tn = len(rs) - tp - fp - fn
    p = tp / (tp + fp) if tp + fp else 0
    rc = tp / (tp + fn) if tp + fn else 0
    return dict(alarm=tp + fp, tp=tp, fp=fp, fn=fn, acc=(tp + tn) / len(rs) if rs else 0,
                prec=p, rec=rc, f1=2 * p * rc / (p + rc) if p + rc else 0)


H = (f"{'카테고리':<10}{'서버':<11}{'크기':<10}{'알람':>6}{'정탐':>6}{'오탐':>6}{'미탐':>6}"
     f"{'정확도':>9}{'F1':>8}")
print(H); print("-" * 80)
rep = {}
for cat in CATS + [None]:
    for t in ("bs32", "ft"):
        for sz in RUNS:
            rs = raw[sz][t] if cat is None else [r for r in raw[sz][t] if r["category"] == cat]
            m = met(rs); rep[f"{cat or 'ALL'}/{t}/{sz}"] = m
            print(f"{(cat or '전체'):<10}{NAME[t]:<11}{sz:<10}{m['alarm']:>6}{m['tp']:>6}"
                  f"{m['fp']:>6}{m['fn']:>6}{m['acc']:>9.3f}{m['f1']:>8.3f}")
        print()
print("-" * 80)
print("\n■ 크기 바꿨을 때 변화 (256x192 − 224x224)\n")
print(f"{'카테고리':<10}{'서버':<11}{'정탐':>8}{'오탐':>8}{'F1':>10}")
print("-" * 48)
for cat in CATS + [None]:
    for t in ("bs32", "ft"):
        a = rep[f"{cat or 'ALL'}/{t}/224x224"]; b = rep[f"{cat or 'ALL'}/{t}/256x192"]
        print(f"{(cat or '전체'):<10}{NAME[t]:<11}{b['tp']-a['tp']:>+8}{b['fp']-a['fp']:>+8}"
              f"{b['f1']-a['f1']:>+10.3f}")
    print()
print("\n■ 답이 바뀐 건수 (300장 중)")
for t in ("bs32", "ft"):
    d = sum(1 for x, y in zip(raw["224x224"][t], raw["256x192"][t]) if x["answer"] != y["answer"])
    print(f"  {NAME[t]:<11}{d:>4}건")
Path("/tmp/ax_eval/report_size.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
