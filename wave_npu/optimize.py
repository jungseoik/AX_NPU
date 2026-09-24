"""프롬프트 선택 최적화 (APO 이식 — 생성 없이 '선택'만).

APO-AI-GUI 의 아이디어(프롬프트 부분집합을 지워가며 지표를 올린다)를 이 파이프라인의
목적함수(프레임 레벨 macro F1)에 맞춰 옮긴 것. 단 무구조 16k 탐색 대신
prompts.PromptSpace 의 세 축(장면/인원상황/이벤트문구) 위에서 좌표상승법으로 탐색한다
— 탐색 공간이 25+15+42 로 줄어 빠르고 과적합도 덜하다.
"""
from __future__ import annotations

import numpy as np

from .evaluate import evaluate
from .score import FastMeanScorer, category_scores, smooth


class Objective:
    """마스크 → macro F1. sims 를 미리 계산해 두고 재사용한다."""

    FAST_RULES = ("mean_margin", "iou_std", "zmean_margin")

    def __init__(self, sims, cls, lab, vid, categories, rule="iou_std", win=1,
                 fast=True, target=None, **kw):
        self.target = target
        self.sims, self.cls, self.lab, self.vid = sims, cls, lab, vid
        self.categories, self.rule, self.win, self.kw = categories, rule, win, kw
        self.n_calls = 0
        self._fast = FastMeanScorer(sims, cls, categories) if (
            fast and rule in self.FAST_RULES and not kw) else None

    def scores(self, mask):
        if self._fast is not None:
            s = self._fast.scores(mask, self.rule)
        else:
            s = category_scores(self.sims, self.cls, self.categories,
                                rule=self.rule, mask=mask, **self.kw)
        return smooth(s, self.vid, self.win)

    def __call__(self, mask):
        """target 이 지정되면 그 카테고리의 F1 하나만 최대화한다(카테고리별 검출기)."""
        self.n_calls += 1
        try:
            res = evaluate(self.scores(mask), self.lab, self.categories)
        except ValueError:
            return -1.0, None
        if self.target is not None:
            return res[self.target]["f1"], res
        return res["macro_f1"], res


def greedy_axis(obj, space, state, axis, cls=None, verbose=True, min_keep=1):
    """한 축에 대해 backward 제거 + forward 추가를 둘 다 해보고 더 좋은 쪽을 채택."""
    def mk(sel):
        st = {k: (dict(v) if isinstance(v, dict) else set(v)) for k, v in state.items()}
        if axis == "phrases":
            st["phrases"][cls] = set(sel)
        else:
            st[axis] = set(sel)
        return space.mask(st["scenes"], st["occs"], st["phrases"])

    if axis == "phrases":
        universe = list(range(len(space.phrases[cls])))
        cur = sorted(state["phrases"][cls])
    else:
        universe = list(range(len(space.scenes if axis == "scenes" else space.occs)))
        cur = sorted(state[axis])

    base, _ = obj(mk(cur))

    # backward: 지웠을 때 가장 많이 오르는 원소를 하나씩 제거
    bw, bw_score = list(cur), base
    improved = True
    while improved and len(bw) > max(1, min_keep):
        improved = False
        best = (bw_score, None)
        for e in bw:
            s, _ = obj(mk([x for x in bw if x != e]))
            if s > best[0] + 1e-9:
                best = (s, e)
        if best[1] is not None:
            bw = [x for x in bw if x != best[1]]
            bw_score = best[0]
            improved = True

    # forward: 빈 집합에서 가장 좋은 원소를 하나씩 추가 (개선 없으면 중단)
    fw, fw_score = [], -1.0
    while len(fw) < len(universe):
        # min_keep 미만인 동안은 개선이 없어도 가장 좋은 원소를 계속 채운다
        forced = len(fw) < min_keep
        best_s, best_e = (-np.inf, None) if forced else (fw_score, None)
        for e in universe:
            if e in fw:
                continue
            s, _ = obj(mk(fw + [e]))
            if s > best_s + (0.0 if forced else 1e-9):
                best_s, best_e = s, e
        if best_e is None:
            break
        fw.append(best_e)
        fw_score = best_s

    sel, score = (bw, bw_score) if bw_score >= fw_score else (fw, fw_score)
    if verbose:
        name = f"{axis}[{cls}]" if axis == "phrases" else axis
        print(f"  {name:12s}: {base:.4f} -> {score:.4f}  (keep {len(sel)}/{len(universe)}, "
              f"bw={bw_score:.4f} fw={fw_score:.4f})", flush=True)
    return sel, score


def coordinate_ascent(obj, space, categories, rounds=2, verbose=True, min_keep=1):
    """세 축을 번갈아 최적화. 반환: (mask, state, score)"""
    from .score import CAT2CLS
    state = {
        "scenes": set(range(len(space.scenes))),
        "occs": set(range(len(space.occs))),
        "phrases": {c: set(range(len(v))) for c, v in space.phrases.items()},
    }
    score, _ = obj(space.mask(state["scenes"], state["occs"], state["phrases"]))
    if verbose:
        print(f"[opt] start macro_f1={score:.4f}", flush=True)

    axes = [("scenes", None), ("occs", None), ("phrases", 0)] + \
           [("phrases", CAT2CLS[c]) for c in categories]
    for r in range(rounds):
        if verbose:
            print(f"[opt] round {r + 1}/{rounds}", flush=True)
        for axis, c in axes:
            sel, s = greedy_axis(obj, space, state, axis, c, verbose, min_keep)
            if s > score:
                if axis == "phrases":
                    state["phrases"][c] = set(sel)
                else:
                    state[axis] = set(sel)
                score = s
    mask = space.mask(state["scenes"], state["occs"], state["phrases"])
    if verbose:
        print(f"[opt] final macro_f1={score:.4f}  prompts={int(mask.sum())}  "
              f"evals={obj.n_calls}", flush=True)
    return mask, state, score


def describe(space, state):
    out = ["scenes: " + ", ".join(space.scenes[i] for i in sorted(state["scenes"])),
           "occupancy: " + ", ".join(space.occs[i] for i in sorted(state["occs"]))]
    for c, sel in sorted(state["phrases"].items()):
        out.append(f"phrases[cls{c}]: " + ", ".join(space.phrases[c][i] for i in sorted(sel)))
    return "\n".join(out)
