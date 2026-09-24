"""프롬프트 풀의 factorial 구조 파싱.

풀(16,125개)은 완전한 3축 조합이다:
  장면 seg0 (25)  ×  인원/상황 seg1 (15)  ×  이벤트 문구 seg2 (클래스별 22/8/6/6)
  (+ normal 에만 seg2 없는 2문장형 375개)

따라서 "어떤 프롬프트를 쓸까"는 16k 부분집합 탐색이 아니라 **세 축 위의 선택**으로
다루는 편이 탐색도 빠르고 과적합도 훨씬 덜하다.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass

import numpy as np


def _segs(p: str):
    return [x.strip() for x in p.strip().split(".") if x.strip()]


@dataclass
class PromptSpace:
    cls: np.ndarray        # (P,) 0..3
    scene: np.ndarray      # (P,) 장면 인덱스
    occ: np.ndarray        # (P,) 인원/상황 인덱스
    phrase: np.ndarray     # (P,) 이벤트 문구 인덱스 (클래스 내), 없으면 -1
    scenes: list
    occs: list
    phrases: dict          # {cls: [문구, ...]}
    prompts: list

    def mask(self, scenes=None, occs=None, phrases=None) -> np.ndarray:
        """선택된 축 값들로 프롬프트 마스크. None 이면 전체 허용.
        phrases: {cls: set(idx)}"""
        m = np.ones(len(self.cls), bool)
        if scenes is not None:
            m &= np.isin(self.scene, list(scenes))
        if occs is not None:
            m &= np.isin(self.occ, list(occs))
        if phrases:
            for c, keep in phrases.items():
                sel = self.cls == c
                m &= ~sel | np.isin(self.phrase, list(keep) + [-1])
        return m


def build(csv_path: str) -> PromptSpace:
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    prompts = [r["prompt"].strip() for r in rows]
    cls = np.asarray([int(r["class"]) for r in rows], np.int32)
    seg = [_segs(p) for p in prompts]

    scenes = sorted({s[0] for s in seg})
    occs = sorted({s[1] for s in seg if len(s) > 1})
    ph = {}
    for c in np.unique(cls):
        ph[int(c)] = sorted({s[2] for s, k in zip(seg, cls) if k == c and len(s) > 2})

    si = {v: i for i, v in enumerate(scenes)}
    oi = {v: i for i, v in enumerate(occs)}
    pi = {c: {v: i for i, v in enumerate(v_)} for c, v_ in ph.items()}

    scene = np.asarray([si[s[0]] for s in seg], np.int32)
    occ = np.asarray([oi[s[1]] if len(s) > 1 else -1 for s in seg], np.int32)
    phrase = np.asarray([pi[int(c)][s[2]] if len(s) > 2 else -1
                         for s, c in zip(seg, cls)], np.int32)
    return PromptSpace(cls, scene, occ, phrase, scenes, occs, ph, prompts)
