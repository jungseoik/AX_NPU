#!/usr/bin/env python3
"""세 경로 + 2단계 직렬을 같은 300장으로 채점. 알람/오탐 개수 기준 칼럼 포함."""
import json
from collections import Counter
from pathlib import Path

raw = json.loads(Path("/tmp/ax_eval/raw.json").read_text())
raw["pe"] = json.loads(Path("/tmp/ax_eval/raw_pe.json").read_text())
NAME = {"pe": "PE 1차", "bs32": "VLM bs32", "ft": "VLM ft"}
CATS = ["fire", "smoke", "falldown"]


def metrics(rs):
    tp = sum(1 for r in rs if r["truth"] == "yes" and r["answer"] == "yes")
    fp = sum(1 for r in rs if r["truth"] == "no" and r["answer"] == "yes")
    fn = sum(1 for r in rs if r["truth"] == "yes" and r["answer"] != "yes")
    tn = sum(1 for r in rs if r["truth"] == "no" and r["answer"] != "yes")
    n = len(rs)
    p = tp / (tp + fp) if tp + fp else 0
    rc = tp / (tp + fn) if tp + fn else 0
    return dict(n=n, tp=tp, fp=fp, fn=fn, tn=tn, alarm=tp + fp, acc=(tp + tn) / n if n else 0,
                prec=p, rec=rc, f1=2 * p * rc / (p + rc) if p + rc else 0,
                fp_rate=fp / (tp + fp) if tp + fp else 0, fpr_neg=fp / (fp + tn) if fp + tn else 0)


HDR = (f"{'카테고리':<10}{'경로':<16}{'알람':>6}{'오탐':>6}{'미탐':>6}{'오탐률':>8}"
       f"{'정확도':>9}{'정밀도':>9}{'재현율':>9}{'F1':>8}")


def row(label, tag, m):
    return (f"{label:<10}{tag:<16}{m['alarm']:>6}{m['fp']:>6}{m['fn']:>6}{m['fp_rate']:>8.3f}"
            f"{m['acc']:>9.3f}{m['prec']:>9.3f}{m['rec']:>9.3f}{m['f1']:>8.3f}")


rep = {}
print("■ 단독 (카테고리당 100장 = 양성 50 + 음성 50)\n")
print(HDR); print("-" * 94)
for cat in CATS:
    for t in ("pe", "bs32", "ft"):
        m = metrics([r for r in raw[t] if r["category"] == cat]); rep[f"{cat}/{t}"] = m
        print(row(cat, NAME[t], m))
    print()
print("-" * 94)
for t in ("pe", "bs32", "ft"):
    m = metrics(raw[t]); rep[f"ALL/{t}"] = m
    print(row("전체", NAME[t], m))

print("\n\n■ 2단계 직렬 (PE 1차 yes → VLM 2차 검증, 둘 다 yes 여야 알람)\n")
print(HDR); print("-" * 94)
pe = {(r["category"], r["clip"]): r for r in raw["pe"]}
for vlm in ("bs32", "ft"):
    comb = [{**r, "answer": "yes" if (pe[(r["category"], r["clip"])]["answer"] == "yes"
                                      and r["answer"] == "yes") else "no"} for r in raw[vlm]]
    for cat in CATS:
        m = metrics([r for r in comb if r["category"] == cat]); rep[f"{cat}/pe+{vlm}"] = m
        print(row(cat, "PE + " + NAME[vlm], m))
    m = metrics(comb); rep[f"ALL/pe+{vlm}"] = m
    print(row("전체", "PE + " + NAME[vlm], m)); print()

print("\n■ 오탐 알람 개수만 (음성 150장 중 몇 번 울렸나)\n")
print(f"{'경로':<16}{'fire':>8}{'smoke':>8}{'falldown':>10}{'합계':>8}{'음성대비':>10}")
print("-" * 62)
rows = [("pe", raw["pe"]), ("bs32", raw["bs32"]), ("ft", raw["ft"])]
for vlm in ("bs32", "ft"):
    comb = [{**r, "answer": "yes" if (pe[(r["category"], r["clip"])]["answer"] == "yes"
                                      and r["answer"] == "yes") else "no"} for r in raw[vlm]]
    rows.append((f"pe+{vlm}", comb))
for tag, rs in rows:
    per = [metrics([r for r in rs if r["category"] == c])["fp"] for c in CATS]
    tot = metrics(rs)
    label = NAME.get(tag) or ("PE + " + NAME[tag.split("+")[1]])
    print(f"{label:<16}{per[0]:>8}{per[1]:>8}{per[2]:>10}{tot['fp']:>8}{tot['fpr_neg']:>9.1%}")
Path("/tmp/ax_eval/report_all.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
