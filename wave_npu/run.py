"""실험 드라이버: baseline 스윕 → 프롬프트 최적화 → 최종 평가.

평가 프로토콜(확정): 프레임 레벨 per-category F1, 음성은 **전체 200영상 교차**,
임계값은 카테고리별 in-sample 최적(상한). 참고로 영상 단위 train/test 분리 수치도 낸다.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from . import data, optimize, prompts
from .evaluate import evaluate, table
from .score import RULES, calibrate, category_scores, sims_matrix, smooth

CSV = "third_party/PIA_Wave/data/text_features_tuningfree_v2_soil최종.csv"


def split_videos(ds, ratio=0.8, seed=42):
    """영상 단위 8:2 분할 — 같은 영상의 프레임이 train/test 에 섞이지 않게."""
    rng = np.random.default_rng(seed)
    v = np.arange(ds.n_videos)
    # 폴더별 층화 (카테고리 비율 유지)
    train = np.zeros(ds.n_videos, bool)
    for f in np.unique(ds.folders):
        idx = v[ds.folders == f]
        rng.shuffle(idx)
        train[idx[:int(round(len(idx) * ratio))]] = True
    return train


def cmd_baseline(a):
    ds = data.load(a.emb, a.root, fps_sample=a.fps)
    temb, cls, _ = data.load_text(a.text)
    print(f"[data] frames={len(ds.emb)} videos={ds.n_videos} prompts={len(temb)} "
          f"cats={ds.categories}", flush=True)
    t = time.time()
    sims = sims_matrix(ds.emb, temb)
    print(f"[data] sims {sims.shape} in {time.time() - t:.1f}s", flush=True)

    rows = []
    for rule in (a.rules.split(",") if a.rules else RULES):
        raw = category_scores(sims, cls, ds.categories, rule=rule)
        for cal in a.cals.split(","):
            cs = calibrate(raw, ds.vid, cal)
            for win in [int(w) for w in a.wins.split(",")]:
                s = smooth(cs, ds.vid, win)
                res = evaluate(s, ds.lab, ds.categories)
                rows.append((rule, win, cal, res))
                print(f"{rule:15s} cal={cal:6s} win={win:3d}  macro={res['macro_f1']:.4f}  " +
                      "  ".join(f"{c}={res[c]['f1']:.4f}" for c in ds.categories), flush=True)
    rows.sort(key=lambda r: -r[3]["macro_f1"])
    best = rows[0]
    print(f"\n[best] rule={best[0]} win={best[1]} cal={best[2]}")
    print(table(best[3], ds.categories))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"rule": best[0], "win": best[1], "cal": best[2],
               "all": [{"rule": r, "win": w, "cal": cal, "macro_f1": x["macro_f1"],
                        **{c: x[c] for c in ds.categories}} for r, w, cal, x in rows]},
              open(a.out, "w"), indent=1)
    print(f"[saved] {a.out}")


def cmd_optimize(a):
    ds = data.load(a.emb, a.root, fps_sample=a.fps)
    temb, cls, _ = data.load_text(a.text)
    space = prompts.build(CSV)
    sims = sims_matrix(ds.emb, temb)
    print(f"[data] frames={len(ds.emb)} prompts={len(temb)}", flush=True)

    if a.holdout:
        # 정직한 일반화 확인: 프롬프트 선택 자체를 train 영상에서만 하고 test 에서 잰다.
        tr = split_videos(ds)
        trf = tr[ds.vid]
        objt = optimize.Objective(sims[trf], cls, ds.lab[trf], ds.vid[trf], ds.categories,
                                  rule=a.rule, win=a.win)
        mtr, str_, _ = optimize.coordinate_ascent(objt, space, ds.categories,
                                                  rounds=a.rounds, min_keep=a.min_keep)
        print(f"\n[holdout] 선택된 프롬프트 {int(mtr.sum())}개\n" + optimize.describe(space, str_))
        rtr = evaluate(objt.scores(mtr), ds.lab[trf], ds.categories)
        thr = {c: rtr[c]["thr"] for c in ds.categories}
        obje = optimize.Objective(sims[~trf], cls, ds.lab[~trf], ds.vid[~trf], ds.categories,
                                  rule=a.rule, win=a.win)
        rte = evaluate(obje.scores(mtr), ds.lab[~trf], ds.categories, thresholds=thr)
        print("\n" + table(rtr, ds.categories, "=== train 에서 프롬프트·임계값 선택 ==="))
        print("\n" + table(rte, ds.categories, "=== test (한 번도 안 본 영상, 고정 임계값) ==="))
        return

    if a.per_category:
        # 카테고리마다 독립적으로 프롬프트를 고른다 (임계값·규칙이 이미 카테고리별이므로 정당).
        out = {}
        for c in ds.categories:
            rule = a.rule_map.get(c, a.rule)
            win = a.win_map.get(c, a.win)
            print(f"\n########## {c} (rule={rule} win={win}) ##########", flush=True)
            o = optimize.Objective(sims, cls, ds.lab, ds.vid, ds.categories,
                                   rule=rule, win=win, target=c)
            m, st, sc = optimize.coordinate_ascent(o, space, ds.categories,
                                                   rounds=a.rounds, min_keep=a.min_keep)
            r = evaluate(o.scores(m), ds.lab, ds.categories)[c]
            print(f"[{c}] F1={r['f1']:.4f} P={r['precision']:.4f} R={r['recall']:.4f} "
                  f"prompts={int(m.sum())}")
            print(optimize.describe(space, st))
            out[c] = {"mask": m, "rule": rule, "win": win, "f1": r["f1"]}
        macro = float(np.mean([v["f1"] for v in out.values()]))
        print(f"\n=== per-category 최적 (in-sample) macro_f1={macro:.4f} ===")
        for c in ds.categories:
            print(f"  {c:10s} {out[c]['rule']:15s} win={out[c]['win']:3d} "
                  f"prompts={int(out[c]['mask'].sum()):5d}  F1={out[c]['f1']:.4f}")
        np.savez_compressed(a.out, macro_f1=macro,
                            **{f"mask_{c}": out[c]["mask"] for c in ds.categories},
                            **{f"rule_{c}": out[c]["rule"] for c in ds.categories},
                            **{f"win_{c}": out[c]["win"] for c in ds.categories})
        print(f"[saved] {a.out}")
        return

    obj = optimize.Objective(sims, cls, ds.lab, ds.vid, ds.categories, rule=a.rule, win=a.win)
    mask, state, score = optimize.coordinate_ascent(obj, space, ds.categories,
                                                    rounds=a.rounds, min_keep=a.min_keep)

    res = evaluate(obj.scores(mask), ds.lab, ds.categories)
    print("\n" + table(res, ds.categories, f"=== optimized (rule={a.rule} win={a.win}, in-sample) ==="))
    print("\n" + optimize.describe(space, state))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez_compressed(a.out, mask=mask,
                        state=json.dumps({k: (sorted(v) if not isinstance(v, dict)
                                              else {str(c): sorted(s) for c, s in v.items()})
                                          for k, v in state.items()}),
                        rule=a.rule, win=a.win, macro_f1=score)
    print(f"[saved] {a.out}")


def cmd_final(a):
    ds = data.load(a.emb, a.root, fps_sample=a.fps)
    temb, cls, _ = data.load_text(a.text)
    z = np.load(a.mask, allow_pickle=True) if a.mask else None
    mask = z["mask"] if z is not None else None
    rule = a.rule or (str(z["rule"]) if z is not None else "iou_std")
    win = a.win if a.win else (int(z["win"]) if z is not None else 1)
    print(f"[final] rule={rule} win={win} prompts={int(mask.sum()) if mask is not None else len(temb)} "
          f"frames={len(ds.emb)} (fps_sample={a.fps or 'all'})", flush=True)

    if mask is not None:                      # 마스크된 프롬프트만으로 sims 계산
        temb, cls = temb[mask], cls[mask]
        mask = None
    sims = sims_matrix(ds.emb, temb)
    s = smooth(category_scores(sims, cls, ds.categories, rule=rule, mask=mask), ds.vid, win)

    res = evaluate(s, ds.lab, ds.categories)
    print("\n" + table(res, ds.categories, "=== in-sample (200영상 전체 튜닝·보고 — 확정 프로토콜) ==="))

    # 참고: 영상 단위 8:2 분할 — train 에서 임계값을 정하고 test 에서 측정
    tr = split_videos(ds)
    trf, tef = tr[ds.vid], ~tr[ds.vid]
    rtr = evaluate(s[trf], ds.lab[trf], ds.categories)
    thr = {c: rtr[c]["thr"] for c in ds.categories}
    rte = evaluate(s[tef], ds.lab[tef], ds.categories, thresholds=thr)
    print("\n" + table(rte, ds.categories, "=== 참고: held-out test (train 임계값 고정, 과적합 폭 확인) ==="))

    if a.out:
        json.dump({"rule": rule, "win": win,
                   "in_sample": {c: res[c] for c in ds.categories} | {"macro_f1": res["macro_f1"]},
                   "held_out": {c: rte[c] for c in ds.categories} | {"macro_f1": rte["macro_f1"]}},
                  open(a.out, "w"), indent=1)
        print(f"[saved] {a.out}")


def cmd_percat(a):
    """카테고리별 마스크·규칙·창(optimize --per-category 산출물)으로 최종 평가."""
    ds = data.load(a.emb, a.root, fps_sample=a.fps)
    temb, cls, _ = data.load_text(a.text)
    z = np.load(a.mask, allow_pickle=True)
    tr = split_videos(ds)
    trf = tr[ds.vid]
    ins, hel = {}, {}
    print(f"[percat] frames={len(ds.emb)} (fps_sample={a.fps or 'all'})")
    for i, c in enumerate(ds.categories):
        m = z[f"mask_{c}"]
        rule = a.rule_map.get(c, str(z[f"rule_{c}"]))
        win = a.win_map.get(c, int(z[f"win_{c}"]))
        te, cl = temb[m], cls[m]
        raw = category_scores(sims_matrix(ds.emb, te), cl, ds.categories, rule=rule)
        cand = [int(w) for w in a.wins.split(",")] if a.sweep_win else [win]
        best = (-1.0, None, None)
        for w in cand:
            sw = smooth(raw, ds.vid, w)[:, i]
            f = evaluate(sw[:, None], ds.lab[:, [i]], (c,))[c]["f1"]
            if f > best[0]:
                best = (f, w, sw)
        _, win, sc = best
        r = evaluate(sc[:, None], ds.lab[:, [i]], (c,))[c]
        ins[c] = dict(r, rule=rule, win=win, n_prompts=int(m.sum()))
        rtr = evaluate(sc[trf][:, None], ds.lab[trf][:, [i]], (c,))[c]
        hel[c] = evaluate(sc[~trf][:, None], ds.lab[~trf][:, [i]], (c,),
                          thresholds={c: rtr["thr"]})[c]
        print(f"  {c:10s} rule={rule:15s} win={win:3d} prompts={int(m.sum()):5d}  "
              f"in-sample F1={r['f1']:.4f}  held-out F1={hel[c]['f1']:.4f}")
    ins["macro_f1"] = float(np.mean([ins[c]["f1"] for c in ds.categories]))
    hel["macro_f1"] = float(np.mean([hel[c]["f1"] for c in ds.categories]))
    print("\n" + table(ins, ds.categories, "=== in-sample (확정 프로토콜) ==="))
    print("\n" + table(hel, ds.categories, "=== 참고: held-out (train 임계값 고정) ==="))
    if a.out:
        json.dump({"in_sample": ins, "held_out": hel}, open(a.out, "w"), indent=1)
        print(f"[saved] {a.out}")


def cmd_combo(a):
    """카테고리별로 규칙·평활창을 독립 선택 — 임계값이 이미 카테고리별이므로 정당하다.

    (fire 는 top-k 투표 계열, falldown/smoke 는 margin 계열이 강해 규칙을 하나로 묶으면 손해.)
    """
    ds = data.load(a.emb, a.root, fps_sample=a.fps)
    temb, cls, _ = data.load_text(a.text)
    mask = np.load(a.mask, allow_pickle=True)["mask"] if a.mask else None
    n_prompt = int(mask.sum()) if mask is not None else len(temb)
    if mask is not None:
        temb, cls = temb[mask], cls[mask]
        mask = None
    sims = sims_matrix(ds.emb, temb)
    wins = [int(w) for w in a.wins.split(",")]
    rules = a.rules.split(",") if a.rules else RULES

    best = {c: (-1, None, None, None) for c in ds.categories}
    for rule in rules:
        try:
            raw = category_scores(sims, cls, ds.categories, rule=rule, mask=mask)
        except ValueError as e:
            print(f"  skip {rule}: {e}")
            continue
        for win in wins:
            sc = smooth(raw, ds.vid, win)
            res = evaluate(sc, ds.lab, ds.categories)
            for i, c in enumerate(ds.categories):
                if res[c]["f1"] > best[c][0]:
                    best[c] = (res[c]["f1"], rule, win, res[c])
    print(f"\n=== 카테고리별 최적 규칙 (prompts={n_prompt}) ===")
    print(f"{'category':10s} {'rule':16s}{'win':>4s} {'F1':>8s} {'P':>8s} {'R':>8s}")
    for c in ds.categories:
        f, rule, win, r = best[c]
        print(f"{c:10s} {rule:16s}{win:>4d} {f:8.4f} {r['precision']:8.4f} {r['recall']:8.4f}")
    macro = float(np.mean([best[c][0] for c in ds.categories]))
    print(f"{'macro':10s} {'':20s} {macro:8.4f}")
    if a.out:
        json.dump({c: {"rule": best[c][1], "win": best[c][2], **best[c][3]}
                   for c in ds.categories} | {"macro_f1": macro}, open(a.out, "w"), indent=1)
        print(f"[saved] {a.out}")


def cmd_diagnose(a):
    """카테고리별 최적 임계값에서 어떤 영상이 틀리는지 — 오탐/미탐 상위 목록."""
    ds = data.load(a.emb, a.root, fps_sample=a.fps)
    temb, cls, _ = data.load_text(a.text)
    z = np.load(a.mask, allow_pickle=True) if a.mask else None
    mask = z["mask"] if z is not None else None
    rule = a.rule or (str(z["rule"]) if z is not None else "iou_std")
    win = a.win or (int(z["win"]) if z is not None else 1)
    if mask is not None:                      # 마스크된 프롬프트만으로 sims 계산
        temb, cls = temb[mask], cls[mask]
        mask = None
    sims = sims_matrix(ds.emb, temb)
    s = smooth(category_scores(sims, cls, ds.categories, rule=rule, mask=mask), ds.vid, win)
    res = evaluate(s, ds.lab, ds.categories)
    print(table(res, ds.categories, f"rule={rule} win={win}"))
    for i, c in enumerate(ds.categories):
        pred = s[:, i] >= res[c]["thr"]
        gt = ds.lab[:, i]
        print(f"\n--- {c}: 영상별 오차 (FP=오탐 프레임수, FN=미탐 프레임수) ---")
        rows = []
        for v in range(ds.n_videos):
            m = ds.vid == v
            fp = int(np.sum(pred[m] & ~gt[m])); fn = int(np.sum(~pred[m] & gt[m]))
            if fp or fn:
                rows.append((fp + fn, fp, fn, int(m.sum()), ds.keys[v]))
        rows.sort(reverse=True)
        for tot, fp, fn, n, k in rows[:a.top]:
            print(f"  {fp:4d}FP {fn:4d}FN / {n:4d}fr  {k}")
        print(f"  (오차 있는 영상 {len(rows)}/{ds.n_videos})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["baseline", "optimize", "final", "diagnose", "combo", "percat"])
    ap.add_argument("--emb", default="wave_npu/cache/emb_tta")
    ap.add_argument("--text", default="wave_npu/cache/text_feats.npz")
    ap.add_argument("--root", default="eval/datasets/TTA_인증용")
    ap.add_argument("--fps", type=float, default=2.0, help="0=전 프레임")
    ap.add_argument("--rules", default=None)
    ap.add_argument("--wins", default="1,3,5,7,9")
    ap.add_argument("--cals", default="none", help="비디오별 기준선 보정: none,median,min")
    ap.add_argument("--rule", default="iou_std")
    ap.add_argument("--win", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--per-category", action="store_true",
                    help="카테고리마다 프롬프트를 독립 선택 (그 카테고리 F1만 최대화)")
    ap.add_argument("--rule-map", default="", help="예: falldown=mean_margin,smoke=topmean_margin")
    ap.add_argument("--win-map", default="", help="예: falldown=25,smoke=25")
    ap.add_argument("--min-keep", type=int, default=1,
                    help="축마다 최소 유지 개수 — 선택이 1개로 붕괴하는 과적합 완화")
    ap.add_argument("--holdout", action="store_true",
                    help="프롬프트 선택을 train 영상에서만 하고 test 에서 평가(일반화 확인)")
    ap.add_argument("--mask", default=None)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--sweep-win", action="store_true", help="percat: --wins 로 카테고리별 창 재탐색")
    ap.add_argument("--out", default="wave_npu/artifacts/baseline.json")
    a = ap.parse_args()
    a.rule_map = dict(x.split("=") for x in a.rule_map.split(",") if x)
    a.win_map = {k: int(v) for k, v in (x.split("=") for x in a.win_map.split(",") if x)}
    {"baseline": cmd_baseline, "optimize": cmd_optimize, "final": cmd_final,
     "diagnose": cmd_diagnose, "combo": cmd_combo, "percat": cmd_percat}[a.cmd](a)


if __name__ == "__main__":
    main()
