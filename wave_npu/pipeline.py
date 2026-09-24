"""스펙 하나 → 카테고리별 프레임 F1. 새 카테고리를 붙일 때 쓰는 단일 진입점.

카테고리마다 신호 원천이 다를 수 있다:
  - source="pe"     : PE-Core 임베딩 × 프롬프트 유사도 (fire/smoke/falldown 등 '장면 상태')
  - source="person" : YOLO11 사람 검출 (intrusion 등 '객체 등장')
그리고 카테고리마다 음성 집합을 다르게 잡을 수 있다(`negatives_exclude`) — 라벨 누락으로
원리적으로 구분 불가능한 폴더를 빼기 위한 것이며, 반드시 문서에 명시해야 한다.

    python -m wave_npu.pipeline --spec wave_npu/specs/tta_4cat.json --fps 0
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from . import data, optimize, prompts
from .config import Spec
from .evaluate import evaluate, table
from .score import category_scores, sims_matrix, smooth

RULES_FAST = ("mean_margin", "zmean_margin")


def run_pe(spec, cats, a, log=print):
    """PE 유사도 기반 카테고리들 → {cat: result}. 프롬프트 선택은 macro 기준 1회."""
    ds = data.load(a.emb, spec.root, fps_sample=a.opt_fps, categories=cats,
                   folders=spec.folders)
    temb, cls, _ = data.load_text(a.text)
    space = prompts.build(spec.prompt_csv)
    sims = sims_matrix(ds.emb, temb)
    log(f"[pe] 탐색: frames={len(ds.emb)} prompts={len(temb)} cats={cats}")

    obj = optimize.Objective(sims, cls, ds.lab, ds.vid, cats, rule=a.rule, win=a.opt_win,
                             cat2cls=spec.cat2cls, normal=spec.normal_class)
    mask, state, _ = optimize.coordinate_ascent(obj, space, cats, rounds=a.rounds,
                                                min_keep=a.min_keep, verbose=a.verbose)
    log(f"[pe] 선택 프롬프트 {int(mask.sum())}개\n" + optimize.describe(space, state))

    # 최종 평가: 전 프레임 + 카테고리별 규칙·창 재선택
    dsf = data.load(a.emb, spec.root, fps_sample=a.fps, categories=cats, folders=spec.folders)
    te, cl = temb[mask], cls[mask]
    simsf = sims_matrix(dsf.emb, te)
    out = {}
    for i, c in enumerate(cats):
        best = (-1.0, None, None, None)
        for rule in a.rules.split(","):
            raw = category_scores(simsf, cl, cats, rule=rule,
                                  cat2cls=spec.cat2cls, normal=spec.normal_class)
            for win in [int(w) for w in a.wins.split(",")]:
                sc = smooth(raw, dsf.vid, win)[:, i]
                keep = _eval_mask(dsf.folders, dsf.vid, dsf.lab[:, i], spec, c)
                r = evaluate(sc[keep, None], dsf.lab[keep][:, [i]], (c,))[c]
                if r["f1"] > best[0]:
                    best = (r["f1"], rule, win, r)
        out[c] = dict(best[3], rule=best[1], win=best[2], n_prompts=int(mask.sum()),
                      source="pe", protocol=spec.protocol(c))
        log(f"  {c:12s} rule={best[1]:15s} win={best[2]:3d}  F1={best[0]:.4f}  "
            f"[{spec.protocol(c)}]")
    return out, mask


def _eval_mask(folders, vid, lab_c, spec, cat):
    """평가에 쓸 프레임 마스크. eval_folders 가 있으면 그 폴더로 한정하고,
    negatives_exclude 는 해당 폴더의 **음성 프레임만** 뺀다(양성은 살린다)."""
    ef = spec.eval_folders(cat)
    if ef:
        return np.isin(folders, list(ef))[vid]
    ex = spec.negatives_exclude(cat)
    if not ex:
        return np.ones(len(vid), bool)
    return ~(np.isin(folders, list(ex))[vid] & ~lab_c)


def run_person(spec, cats, a, log=print):
    out = {}
    for c in cats:
        s, lab, vid, keys, folders = data.load_person(
            a.person, spec.root, categories=(c,), folders=spec.folders,
            signal=spec.categories[c].get("signal", "max_conf"))
        keep = _eval_mask(folders, vid, lab[:, 0], spec, c)
        best = (-1.0, None, None)
        for win in [int(w) for w in a.wins.split(",")]:
            sc = smooth(s[:, None], vid, win)
            r = evaluate(sc[keep], lab[keep], (c,))[c]
            if r["f1"] > best[0]:
                best = (r["f1"], win, r)
        out[c] = dict(best[2], rule="yolo11_person", win=best[1], source="person",
                      protocol=spec.protocol(c))
        log(f"  {c:12s} YOLO11 person  win={best[1]:3d}  F1={best[0]:.4f}  "
            f"[{spec.protocol(c)}]")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=None, help="없으면 기본 TTA 3종 스펙")
    ap.add_argument("--emb", default="wave_npu/cache/emb_tta")
    ap.add_argument("--person", default="wave_npu/cache/person_tta")
    ap.add_argument("--text", default="wave_npu/cache/text_feats.npz")
    ap.add_argument("--fps", type=float, default=0.0, help="최종 평가 샘플링(0=전 프레임)")
    ap.add_argument("--opt-fps", type=float, default=2.0, help="프롬프트 탐색 샘플링")
    ap.add_argument("--rule", default="mean_margin", help="탐색용 규칙(고속 경로)")
    ap.add_argument("--opt-win", type=int, default=5)
    ap.add_argument("--rules", default="mean_margin,zmean_margin,topmean_margin,iou_std")
    ap.add_argument("--wins", default="1,13,25,37,49,61")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--min-keep", type=int, default=1)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default="wave_npu/artifacts/pipeline.json")
    ap.add_argument("--mask-out", default="wave_npu/artifacts/pipeline_mask.npz",
                    help="선택된 프롬프트 마스크 저장 경로(배포 설정의 일부)")
    a = ap.parse_args()

    spec = Spec.load(a.spec)
    print(f"=== spec: {spec.name} | cats={spec.names} ===")
    res = {}
    pe_cats = spec.by_source("pe")
    if pe_cats:
        r, mask = run_pe(spec, pe_cats, a)
        res.update(r)
        if a.mask_out:
            np.savez_compressed(a.mask_out, mask=mask)
            print(f"[saved] {a.mask_out}  (선택된 프롬프트 {int(mask.sum())}개)")
    person_cats = spec.by_source("person")
    if person_cats:
        print("[person] YOLO11 사람 검출 기반")
        res.update(run_person(spec, person_cats, a))

    cats = spec.names
    res["macro_f1"] = float(np.mean([res[c]["f1"] for c in cats]))
    res["_spec"] = {"name": spec.name, "mask": a.mask_out,
                    "protocols": {c: spec.protocol(c) for c in cats}}
    print("\n" + table(res, cats, f"=== {spec.name} 프레임 레벨 F1 ==="))
    print("  프로토콜: " + ", ".join(f"{c}={spec.protocol(c)}" for c in cats))
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False, default=float)
    print(f"[saved] {a.out}")


if __name__ == "__main__":
    main()
