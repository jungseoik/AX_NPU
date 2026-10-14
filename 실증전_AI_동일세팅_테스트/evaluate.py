# -*- coding: utf-8 -*-
"""프레임 레벨 F1 산출 — 1차 단독 vs 1차+2차(운영 2단계).

프레임 예측의 정의 = **알람이 떠 있는 프레임**(이벤트 상태 1=시작 / 2=지속).
  1차 단독 : 1차 알람 상태 그대로
  2단계    : 1차 알람 구간 중 **2차가 통과(yes)시킨 에피소드만** 양성으로 남긴다.
             2차가 기각하면 그 에피소드 전체가 음성이 된다(운영에서 알람 미생성).

★ 교차 음성: 각 카테고리를 150건 전체에 대해 평가한다. 해당 카테고리 폴더 50건이
  양성 후보, 나머지 100건은 음성이다. 폴더 안에서만 재면 TTA_인증용에 정상 영상이
  0건이라 "무조건 이벤트"가 만점을 받는다.
"""
from __future__ import annotations

import json
import os
import sys

import settings as S


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) * 100 if (p + r) else 0.0
    return p, r, f


def episodes(active):
    """연속 1 구간 -> [(시작idx, 끝idx)] (끝 포함)."""
    out, s = [], None
    for i, v in enumerate(active):
        if v and s is None:
            s = i
        elif not v and s is not None:
            out.append((s, i - 1)); s = None
    if s is not None:
        out.append((s, len(active) - 1))
    return out


def main():
    s1 = json.load(open(os.path.join(S.OUT, "stage1.json"), encoding="utf-8"))
    s2p = os.path.join(S.OUT, "stage2.json")
    verdict = {}
    if os.path.exists(s2p):
        for r in json.load(open(s2p, encoding="utf-8")):
            verdict[r["evidence"]] = r
    else:
        print("  [!] stage2.json 없음 — 1차 단독 결과만 산출합니다\n")

    clips = s1["clips"]
    m = s1["설정"]
    print("=" * 84)
    print("실증 전 AI 동일세팅 테스트 — 프레임 레벨 F1")
    print("=" * 84)
    print(f"  1차  PE {m['quant']}/{m['tuning']}/{m['scheme']}  core={m['core_mode']}  "
          f"NPU {m['npu_cards']}장")
    print(f"       판정 {m['method']} top{m['top_candidate']}  프롬프트 {m['n_prompts']:,}개  "
          f"알람 {m['queue_size']}창중{m['alarm_duration']}  {m['fps']}fps")
    print(f"  2차  ft bs32 VQA  {S.VQA_SIZE[0]}x{S.VQA_SIZE[1]}  {S.VQA_FRAMES}프레임")
    nfr = sum(c["n"] for c in clips)
    print(f"  대상 {len(clips)}건 / {nfr:,}프레임 (교차 음성: 카테고리별 전체 {len(clips)}건 평가)\n")

    agg = {}
    for cat in S.CATEGORIES:
        # 에피소드별 2차 판정을 프레임 마스크로 펼친다
        skip = S.EVAL_EXCLUDE.get(cat, ())
        rows = []
        for c in clips:
            if c["cat_dir"] in skip:          # 분모에서 뺀다(음성으로 쓰지 않는다)
                continue
            act = c["active"][cat]
            gt = c["gt"][cat]
            stage2 = [0] * len(act)
            for a in c["alarms"]:
                if a["cat"] != cat:
                    continue
                v = verdict.get(a["evidence"])
                passed = (v is not None and v.get("label") == "yes")
                if not passed:
                    continue
                for s, e in episodes(act):
                    if s <= a["idx"] <= e:
                        for i in range(s, e + 1):
                            stage2[i] = 1
            rows.append((gt, act, stage2))

        out = {}
        for name, col in (("1차 단독", 1), ("1차+2차", 2)):
            tp = fp = fn = tn = 0
            for gt, act, st2 in rows:
                pr = act if col == 1 else st2
                for g, p in zip(gt, pr):
                    tp += g == 1 and p == 1
                    fp += g == 0 and p == 1
                    fn += g == 1 and p == 0
                    tn += g == 0 and p == 0
            out[name] = (prf(tp, fp, fn), tp, fp, fn, tn)
        agg[cat] = out
        agg[cat]["_범위"] = (len(rows), sum(len(r[0]) for r in rows), skip)

    hdr = f"{'카테고리':10} {'단계':9} {'TP':>6} {'FP':>6} {'FN':>6} {'TN':>7} {'정밀도':>8} {'재현율':>8} {'F1':>8}"
    print(hdr); print("-" * 84)
    for cat in S.CATEGORIES:
        nclip, nfrm, skip = agg[cat]["_범위"]
        note = f"  (평가 {nclip}건 {nfrm:,}프레임"
        note += f", {'/'.join(skip)} 폴더 제외)" if skip else ")"
        print(f"{cat}{note}")
        for name in ("1차 단독", "1차+2차"):
            (p, r, f), tp, fp, fn, tn = agg[cat][name]
            lead = ""
            print(f"{lead:10} {name:9} {tp:6} {fp:6} {fn:6} {tn:7} "
                  f"{p*100:7.2f}% {r*100:7.2f}% {f:7.2f}")
        print("-" * 84)

    macro = {n: sum(agg[c][n][0][2] for c in S.CATEGORIES) / len(S.CATEGORIES)
             for n in ("1차 단독", "1차+2차")}
    print()
    print(f"{'매크로 평균':10} {'1차 단독':9} {'':6} {'':6} {'':6} {'':7} {'':8} {'':8} "
          f"{macro['1차 단독']:7.2f}")
    print(f"{'':10} {'1차+2차':9} {'':6} {'':6} {'':6} {'':7} {'':8} {'':8} "
          f"{macro['1차+2차']:7.2f}")

    # 클립 단위 — 프레임 F1 이 알람 지연에 지배되므로, "맞는 영상에서 울렸는가"를 분리해 본다
    print("\n" + "=" * 84)
    print("참고: 클립 단위 (영상에서 한 번이라도 울렸는가)")
    print("=" * 84)
    print(f"{'카테고리':10} {'단계':9} {'TP':>5} {'FP':>5} {'FN':>5} {'정밀도':>9} {'재현율':>9} {'F1':>8}")
    for cat in S.CATEGORIES:
        skip = S.EVAL_EXCLUDE.get(cat, ())
        for name in ("1차 단독", "1차+2차"):
            tp = fp = fn = 0
            for c in clips:
                if c["cat_dir"] in skip:
                    continue
                g = sum(c["gt"][cat]) > 0
                if name == "1차 단독":
                    pr = sum(c["active"][cat]) > 0
                else:
                    pr = any(verdict.get(a["evidence"], {}).get("label") == "yes"
                             for a in c["alarms"] if a["cat"] == cat)
                tp += g and pr; fp += (not g) and pr; fn += g and (not pr)
            p_, r_, f_ = prf(tp, fp, fn)
            print(f"{cat if name=='1차 단독' else '':10} {name:9} {tp:5} {fp:5} {fn:5} "
                  f"{p_*100:8.2f}% {r_*100:8.2f}% {f_:7.2f}")

    # 2차가 실제로 무엇을 걸렀는지
    if verdict:
        print("\n" + "=" * 84)
        print("2차 검출기 차단 내역")
        print("=" * 84)
        print(f"{'카테고리':10} {'1차 알람':>9} {'2차 통과':>9} {'2차 기각':>9} {'차단율':>9}")
        for cat in S.CATEGORIES:
            sk = S.EVAL_EXCLUDE.get(cat, ())
            al = [a for c in clips if c["cat_dir"] not in sk
                  for a in c["alarms"] if a["cat"] == cat]
            ys = sum(1 for a in al if verdict.get(a["evidence"], {}).get("label") == "yes")
            n = len(al)
            print(f"{cat:10} {n:9} {ys:9} {n-ys:9} "
                  f"{(n-ys)/n*100 if n else 0:8.1f}%")

    json.dump({"설정": m, "결과": {c: {k: {"precision": v[0][0], "recall": v[0][1],
                                          "f1": v[0][2], "tp": v[1], "fp": v[2],
                                          "fn": v[3], "tn": v[4]}
                                     for k, v in agg[c].items() if k != "_범위"}
                                 for c in agg},
               "매크로F1": macro},
              open(os.path.join(S.OUT, "f1.json"), "w"), ensure_ascii=False, indent=1)
    print(f"\n-> {S.OUT}/f1.json")


if __name__ == "__main__":
    sys.exit(main())
