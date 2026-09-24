"""프레임 레벨 per-category P/R/F1 + 임계값 스윕.

프로토콜(확정): 음성은 전체 200영상 교차 — 해당 카테고리 이벤트가 라벨되지 않은
영상의 모든 프레임은 그 카테고리에 대해 음성이다.
"""
from __future__ import annotations

import numpy as np


def prf1(pred: np.ndarray, gt: np.ndarray):
    tp = int(np.sum(pred & gt))
    fp = int(np.sum(pred & ~gt))
    fn = int(np.sum(~pred & gt))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f, tp, fp, fn


def best_threshold(score: np.ndarray, gt: np.ndarray, n_steps: int = 400):
    """F1 최대가 되는 임계값(score >= thr → 이벤트)을 분위수 격자에서 탐색."""
    qs = np.quantile(score, np.linspace(0, 1, n_steps))
    best = (-1.0, 0.0, 0.0, 0.0)
    for t in np.unique(qs):
        p, r, f, *_ = prf1(score >= t, gt)
        if f > best[0]:
            best = (f, float(t), p, r)
    f, t, p, r = best
    return {"f1": f, "thr": t, "precision": p, "recall": r}


def evaluate(scores: np.ndarray, lab: np.ndarray, categories, thresholds=None):
    """카테고리별 결과 dict. thresholds 가 있으면 고정 임계값 평가(out-of-sample 용)."""
    res = {}
    for i, c in enumerate(categories):
        gt = lab[:, i]
        if thresholds is None:
            r = best_threshold(scores[:, i], gt)
        else:
            p, rc, f, *_ = prf1(scores[:, i] >= thresholds[c], gt)
            r = {"f1": f, "thr": thresholds[c], "precision": p, "recall": rc}
        r["pos_rate"] = float(gt.mean())
        # 전부 양성으로 찍었을 때의 F1 — 지표가 자명하게 달성되는지 확인용
        r["trivial_f1"] = 2 * r["pos_rate"] / (1 + r["pos_rate"])
        res[c] = r
    res["macro_f1"] = float(np.mean([res[c]["f1"] for c in categories]))
    return res


def table(res, categories, title=""):
    lines = []
    if title:
        lines.append(title)
    lines.append(f"{'category':10s} {'F1':>7s} {'P':>7s} {'R':>7s} {'thr':>9s} {'pos%':>6s} {'trivF1':>7s}")
    for c in categories:
        r = res[c]
        lines.append(f"{c:10s} {r['f1']:7.4f} {r['precision']:7.4f} {r['recall']:7.4f} "
                     f"{r['thr']:9.4f} {100 * r['pos_rate']:6.1f} {r['trivial_f1']:7.3f}")
    lines.append(f"{'macro':10s} {res['macro_f1']:7.4f}")
    return "\n".join(lines)
