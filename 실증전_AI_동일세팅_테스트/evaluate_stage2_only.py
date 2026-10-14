# -*- coding: utf-8 -*-
"""2차 단독 프레임 F1 + 2단계 파이프라인과의 비교.

세 가지를 나란히 둔다. 어디서 성능이 깎이는지 분리해서 보려는 것이다.
  1차 단독        PE + 알람 FSM
  1차+2차         운영 파이프라인(알람 발화 시점 1장만 2차가 본다)
  2차 단독        1초마다 전 프레임을 2차가 본다 — 1차·FSM 없음
"""
from __future__ import annotations

import json
import os

import settings as S


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return p, r, (2 * p * r / (p + r) * 100 if (p + r) else 0.0)


def episodes(a):
    out, s = [], None
    for i, v in enumerate(a):
        if v and s is None:
            s = i
        elif not v and s is not None:
            out.append((s, i - 1)); s = None
    if s is not None:
        out.append((s, len(a) - 1))
    return out


def pipeline_rows(cat):
    """1차 단독 / 1차+2차 의 (gt, 1차, 2단계) 행."""
    s1 = json.load(open(os.path.join(S.OUT, "stage1.json"), encoding="utf-8"))
    verd = {r["evidence"]: r for r in json.load(
        open(os.path.join(S.OUT, "stage2.json"), encoding="utf-8"))}
    skip = S.EVAL_EXCLUDE.get(cat, ())
    rows = []
    for c in s1["clips"]:
        if c["cat_dir"] in skip:
            continue
        act, gt = c["active"][cat], c["gt"][cat]
        fin = [0] * len(act)
        for a in c["alarms"]:
            if a["cat"] != cat or verd.get(a["evidence"], {}).get("label") != "yes":
                continue
            for s_, e_ in episodes(act):
                if s_ <= a["idx"] <= e_:
                    for i in range(s_, e_ + 1):
                        fin[i] = 1
        rows.append((gt, act, fin))
    return rows


def score(rows, col):
    tp = fp = fn = tn = 0
    for r in rows:
        for g, p in zip(r[0], r[col]):
            tp += g == 1 and p == 1
            fp += g == 0 and p == 1
            fn += g == 1 and p == 0
            tn += g == 0 and p == 0
    return prf(tp, fp, fn), tp, fp, fn, tn


def main():
    s2 = json.load(open(os.path.join(S.OUT, "stage2_only.json"), encoding="utf-8"))
    m = s2["설정"]
    print("=" * 92)
    print("2차 단독 평가 — 1차·알람 FSM 없이 1초 간격 전 프레임 VQA")
    print("=" * 92)
    print(f"  입력 {m['fps']}fps · {m['size']} · 호출 {m['호출수']:,}건 · 오류 {m['오류']}")
    print(f"  문안 variant={m['variant']}")
    print("  프레임 예측 = 그 1초 구간을 2차 판정으로 통째로 채움\n")

    hdr = (f"{'카테고리':10} {'방식':12} {'TP':>6} {'FP':>6} {'FN':>6} {'TN':>7} "
           f"{'정밀도':>9} {'재현율':>9} {'F1':>8}")
    print(hdr); print("-" * 92)
    macro = {}
    for cat in S.CATEGORIES:
        rows_p = pipeline_rows(cat)
        rows_2 = [(c["gt"][cat], None, c["pred"][cat])
                  for c in s2["clips"] if cat in c["gt"]]
        res = [("1차 단독", score(rows_p, 1)),
               ("1차+2차", score(rows_p, 2)),
               ("2차 단독", score(rows_2, 2))]
        n = sum(len(r[0]) for r in rows_2)
        note = f"  (평가 {len(rows_2)}건 {n:,}프레임"
        note += f", {'/'.join(S.EVAL_EXCLUDE[cat])} 폴더 제외)" if cat in S.EVAL_EXCLUDE else ")"
        print(f"{cat}{note}")
        for name, ((p, r, f), tp, fp, fn, tn) in res:
            print(f"{'':10} {name:12} {tp:6} {fp:6} {fn:6} {tn:7} "
                  f"{p*100:8.2f}% {r*100:8.2f}% {f:7.2f}")
            macro.setdefault(name, []).append(f)
        print("-" * 92)
    print(f"{'매크로 평균':10}")
    for k, v in macro.items():
        print(f"{'':10} {k:12} {'':6} {'':6} {'':6} {'':7} {'':9} {'':9} "
              f"{sum(v)/len(v):7.2f}")

    json.dump({k: sum(v) / len(v) for k, v in macro.items()},
              open(os.path.join(S.OUT, "f1_compare.json"), "w"),
              ensure_ascii=False, indent=1)
    print(f"\n-> {S.OUT}/f1_compare.json")


if __name__ == "__main__":
    main()
