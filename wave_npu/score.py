"""프레임 임베딩 × 프롬프트 임베딩 → 카테고리별 점수 (결정 규칙 모음).

모든 규칙은 **높을수록 이벤트**가 되도록 부호를 맞춘다(원본 iou_std 는 낮을수록
이벤트라 음수를 취한다). 규칙·프롬프트 부분집합·평활 창을 바꿔가며 스윕하는 것이
점수를 올리는 주 수단이고, 전부 캐시된 임베딩 위의 numpy 연산이다.

프롬프트 클래스 = {0: normal, 1: falldown, 2: fire, 3: smoke}
"""
from __future__ import annotations

import numpy as np

NORMAL = 0
CAT2CLS = {"falldown": 1, "fire": 2, "smoke": 3}
RULES = ("iou_std", "mean_margin", "max_margin", "softmax", "topk_frac", "zmean_margin",
         "topmean_margin")


def sims_matrix(emb: np.ndarray, temb: np.ndarray, chunk: int = 8192) -> np.ndarray:
    """(N,1024) @ (P,1024).T → (N,P) float32. 청크로 나눠 메모리 스파이크를 막는다."""
    out = np.empty((len(emb), len(temb)), np.float32)
    for s in range(0, len(emb), chunk):
        np.dot(emb[s:s + chunk], temb.T, out=out[s:s + chunk])
    return out


def _class_stats(sims, cls, mask, classes):
    """{c: (mean, std, max)} — 각 (N,)."""
    st = {}
    for c in classes:
        idx = np.where((cls == c) & mask)[0]
        if idx.size == 0:
            raise ValueError(f"class {c} 프롬프트가 mask 로 전부 제거됨")
        s = sims[:, idx]
        st[c] = (s.mean(1), s.std(1), s.max(1))
    return st


def _std_iou(m_a, s_a, m_b, s_b):
    lo = np.maximum(m_a - s_a, m_b - s_b)
    hi = np.minimum(m_a + s_a, m_b + s_b)
    inter = np.maximum(0.0, hi - lo)
    union = 2 * s_a + 2 * s_b - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def _top_mean(s, frac):
    """행별 상위 frac 비율 열의 평균 — 관련 없는 프롬프트에 둔감한 집계."""
    k = max(1, int(round(s.shape[1] * frac)))
    if k >= s.shape[1]:
        return s.mean(1)
    part = np.partition(s, -k, axis=1)[:, -k:]
    return part.mean(1)


def category_scores(sims, cls, categories, rule="iou_std", mask=None,
                    topk=13, temp=0.01, frac=0.1, cat2cls=None, normal=None) -> np.ndarray:
    """(N, C) float32. 높을수록 이벤트.

    cat2cls/normal 을 주면 카테고리→프롬프트 클래스 매핑을 스펙에서 받는다
    (wave_npu.config.Spec). 미지정 시 모듈 기본값(falldown/fire/smoke).
    """
    if mask is None:
        mask = np.ones(len(cls), bool)
    cat2cls = cat2cls or CAT2CLS
    normal = NORMAL if normal is None else normal
    classes = [normal] + [cat2cls[c] for c in categories]

    if rule == "topk_frac":
        idx = np.where(mask)[0]
        s = sims[:, idx]
        top = np.argpartition(-s, topk, axis=1)[:, :topk]
        tc = cls[idx][top]                                   # (N,topk)
        return np.stack([(tc == cat2cls[c]).mean(1) for c in categories], 1).astype(np.float32)

    if rule == "topmean_margin":
        tm = {}
        for c in classes:
            idx = np.where((cls == c) & mask)[0]
            if idx.size == 0:
                raise ValueError(f"class {c} 프롬프트가 mask 로 전부 제거됨")
            tm[c] = _top_mean(sims[:, idx], frac)
        return np.stack([tm[cat2cls[c]] - tm[normal] for c in categories], 1).astype(np.float32)

    st = _class_stats(sims, cls, mask, classes)
    mn, sn, xn = st[normal]
    cols = []
    for c in categories:
        me, se, xe = st[cat2cls[c]]
        if rule == "iou_std":
            cols.append(-_std_iou(mn, sn, me, se))
        elif rule == "mean_margin":
            cols.append(me - mn)
        elif rule == "max_margin":
            cols.append(xe - xn)
        elif rule == "zmean_margin":
            # 프레임별 전체 프롬프트 평균/표준편차로 정규화 → 장면별 유사도 오프셋 제거
            allm = np.mean([st[k][0] for k in classes], 0)
            alls = np.std([st[k][0] for k in classes], 0) + 1e-6
            cols.append((me - allm) / alls - (mn - allm) / alls)
        elif rule == "softmax":
            logits = np.stack([st[k][2] for k in classes], 1) / temp
            p = np.exp(logits - logits.max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
            cols.append(p[:, classes.index(cat2cls[c])])
        else:
            raise ValueError(f"unknown rule {rule}")
    return np.stack(cols, 1).astype(np.float32)


def smooth(scores: np.ndarray, vid: np.ndarray, win: int) -> np.ndarray:
    """비디오 경계를 넘지 않는 중앙 이동평균 (win 프레임)."""
    if win <= 1:
        return scores
    out = np.empty_like(scores)
    k = np.ones(win, np.float32) / win
    for v in np.unique(vid):
        m = vid == v
        seg = scores[m]
        pad = win // 2
        padded = np.pad(seg, ((pad, win - 1 - pad), (0, 0)), mode="edge")
        for c in range(seg.shape[1]):
            out[np.where(m)[0], c] = np.convolve(padded[:, c], k, mode="valid")
    return out


def calibrate(scores: np.ndarray, vid: np.ndarray, mode: str = "none",
              q: float = 0.5) -> np.ndarray:
    """비디오별 기준선 보정 — 카메라/장면마다 다른 유사도 오프셋을 제거한다.

    단, 이 데이터셋의 smoke 영상은 이벤트가 영상의 84%를 덮어서(=거의 전 구간 양성)
    비디오 내부 분위수를 빼면 오히려 신호가 사라질 수 있다. 스윕으로 확인할 것.
    """
    if mode == "none":
        return scores
    out = scores.copy()
    for v in np.unique(vid):
        m = vid == v
        seg = scores[m]
        base = seg.min(0) if mode == "min" else np.quantile(seg, q, axis=0)
        out[m] = seg - base
    return out


class FastMeanScorer:
    """mean/std 기반 규칙 전용 고속 경로 — 탐색 루프용.

    마스크가 바뀔 때마다 sims 를 gather 하면(6000×16k float32 복사) 1.6s/eval 이라
    수천 번 도는 좌표상승법에 못 쓴다. 클래스 평균·분산은 마스크 벡터와의 **행렬-벡터 곱**으로
    구할 수 있으므로 sims 와 sims² 를 한 번만 만들어 두고 재사용한다 (≈30배).
    """

    def __init__(self, sims, cls, categories, cat2cls=None, normal=None):
        self.s = np.ascontiguousarray(sims)
        self.s2 = self.s * self.s
        self.cls = cls
        self.categories = categories
        self.cat2cls = cat2cls or CAT2CLS
        self.normal = NORMAL if normal is None else normal
        self.classes = [self.normal] + [self.cat2cls[c] for c in categories]

    def _stats(self, mask):
        st = {}
        for c in self.classes:
            w = ((self.cls == c) & mask).astype(np.float32)
            n = w.sum()
            if n == 0:
                raise ValueError(f"class {c} 프롬프트가 mask 로 전부 제거됨")
            m = self.s @ w / n
            v = np.maximum(self.s2 @ w / n - m * m, 0.0)
            st[c] = (m, np.sqrt(v))
        return st

    def scores(self, mask, rule="mean_margin"):
        st = self._stats(mask)
        mn, sn = st[self.normal]
        cols = []
        for c in self.categories:
            me, se = st[self.cat2cls[c]]
            if rule == "mean_margin":
                cols.append(me - mn)
            elif rule == "iou_std":
                cols.append(-_std_iou(mn, sn, me, se))
            elif rule == "zmean_margin":
                allm = np.mean([st[k][0] for k in self.classes], 0)
                alls = np.std([st[k][0] for k in self.classes], 0) + 1e-6
                cols.append((me - allm) / alls - (mn - allm) / alls)
            else:
                raise ValueError(f"FastMeanScorer 가 지원하지 않는 rule: {rule}")
        return np.stack(cols, 1).astype(np.float32)
