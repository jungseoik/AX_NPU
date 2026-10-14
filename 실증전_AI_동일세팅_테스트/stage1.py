# -*- coding: utf-8 -*-
"""1차 검출기 — 운영(macs-inference) 동일 경로로 PE NPU 추론 + 알람 판정.

★ 판정 로직을 베껴 쓰지 않는다. 운영 모듈을 **그대로 import** 한다.
   - 전처리  : pia_prod...pe_npu.parallel_preprocess.ParallelPreprocessor
   - 추론    : pia_prod...pe_npu.multi_npu.MultiNPUInferenceFull
   - 판정/알람: pia_prod...pe_npu.event.PENpuEventManager  (topk + 5창3 FSM)
   베끼면 미묘하게 달라지고, 달라진 걸 증명할 방법이 없다.

출력: out/stage1.json  +  evidence/<cat>/<clip>__t<sec>.jpg (알람 시작 프레임 원본)
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import torch

import settings as S

sys.path.insert(0, f"{S.MONO}/packages")

# 운영 기본값을 환경변수로 **명시 고정**한다(기본값에 의존하지 않고 눈에 보이게).
os.environ.setdefault("PE_NPU_METHOD", S.PE_METHOD)
os.environ.setdefault("PE_TOP_CANDIDATE", str(S.PE_TOP_CANDIDATE))
os.environ.setdefault("PE_QUEUE_SIZE", str(S.PE_QUEUE_SIZE))
os.environ.setdefault("PE_ALARM_DURATION_THRESHOLD", str(S.PE_ALARM_DURATION))

from pia_prod.AI.modules.pe_npu.config import (  # noqa: E402
    CATEGORY_EVENT_MAP, INDEX_MAPPING, TEMPORAL_SIZE,
)
from pia_prod.AI.modules.pe_npu.event import PENpuEventManager  # noqa: E402
from pia_prod.AI.modules.pe_npu.multi_npu import MultiNPUInferenceFull  # noqa: E402
from pia_prod.AI.modules.pe_npu.parallel_preprocess import ParallelPreprocessor  # noqa: E402
from pia_prod.AI.modules.pe_npu.prompts import load_text_feature  # noqa: E402

# user_param 규약: {"user_param": {"retEvent": [<카테고리 ret 문자열>, ...]}}
RET = {c: CATEGORY_EVENT_MAP[c][0] for c in S.CATEGORIES}   # falldown -> "falldown_ret"


def clip_list():
    """평가 대상 전체. **교차 음성**이 성립하도록 3개 카테고리 영상을 한 풀로 묶는다.

    TTA_인증용에는 정상(음성) 영상이 0건이다. 카테고리 폴더 안에서만 평가하면
    "무조건 이벤트"가 만점을 받아 수치가 무의미해진다. 그래서 각 카테고리를
    **150개 전체**에 대해 평가하고, 다른 카테고리 영상 100개가 음성 역할을 한다.
    """
    out = []
    for cat in S.CATEGORIES:
        d = os.path.join(S.DATASET, cat)
        for f in sorted(os.listdir(d)):
            if f.endswith(".mp4"):
                out.append({"cat_dir": cat, "name": f[:-4], "mp4": os.path.join(d, f),
                            "json": os.path.join(d, f[:-4] + ".json")})
    return out


def _decode(args):
    """프로세스 풀 워커. 디코드 + 336 리사이즈까지 여기서 끝낸다.

    ★ 리사이즈를 워커에서 하는 이유는 두 가지다.
      ① 전달 데이터가 1108x828 -> 336x336 로 줄어 IPC 가 가벼워진다(6GB -> 760MB).
      ② 수치가 바뀌지 않는다 — 운영 전처리의 기본 백엔드가 cv2 INTER_LINEAR 이고
         (PE_NPU_RESIZE 기본값), 이미 336x336 이면 preprocess 가 resize 를 건너뛴다.
         즉 "여기서 미리 하느냐 거기서 하느냐"의 차이뿐이다.
      색변환도 리사이즈도 채널별 선형연산이라 순서가 결과를 바꾸지 않는다.
    """
    path, fps, size = args
    frames, secs, dur = read_frames(path, fps)
    # PE 입력용 336x336
    small = [cv2.resize(f, (size, size), interpolation=cv2.INTER_LINEAR)
             if f.shape[:2] != (size, size) else f for f in frames]
    # ★ 증거 이미지는 **원본 해상도** 로 따로 들고 간다.
    #   운영은 2차에 원본 프레임을 넘긴다. PE 전처리용 336x336(종횡비 파괴)을
    #   그대로 2차에 주면 2차가 보는 그림이 운영과 달라진다 — 실제로 한 번 틀렸다.
    #   원본 배열을 그대로 반환하면 IPC 가 6GB 라 JPEG 로 눌러서 보낸다.
    jpg = []
    for f in frames:
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(f, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, 95])
        jpg.append(buf.tobytes() if ok else None)
    return small, jpg, secs, dur


def read_frames(path, fps):
    """운영 입력률(fps)로 균등 샘플. 각 1초 구간의 **중앙** 프레임을 집는다.

    ★ 색공간: 운영은 PE_NPU_BGR2RGB=false, 즉 "입력은 이미 RGB"가 전제다.
      cv2 는 BGR 로 주므로 여기서 RGB 로 바꿔 넣는다(안 바꾸면 R↔B 가 뒤집혀
      임베딩이 조용히 오염되고 에러 없이 정확도만 무너진다).
    """
    cap = cv2.VideoCapture(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = total / src_fps
    n = max(1, int(dur * fps))
    # 1초 구간의 중앙 프레임 번호. 탐색(seek)이 아니라 **순차 디코드**로 집는다
    # (15초 영상에서 seek 15회가 전체 순차 디코드보다 느리다).
    want = {min(total - 1, int(round(((i + 0.5) / fps) * src_fps))): i for i in range(n)}
    frames, secs = [None] * n, [(i + 0.5) / fps for i in range(n)]
    pos = 0
    while want:
        if not cap.grab():                    # 필요 없는 프레임은 retrieve 생략
            break
        if pos in want:
            ok, bgr = cap.retrieve()
            if ok:
                frames[want.pop(pos)] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            else:
                want.pop(pos)
        pos += 1
    cap.release()
    keep = [i for i, f in enumerate(frames) if f is not None]
    return [frames[i] for i in keep], [secs[i] for i in keep], dur


def gt_bins(label_path, cat, n, fps):
    """초 단위 라벨 -> 1/fps 구간 격자의 0/1. 구간이 조금이라도 겹치면 양성.

    ★ 반드시 events[].category 로 걸러야 한다. 라벨은 폴더명과 1:1 이 아니다 —
      fire 폴더 50건 **전부**가 smoke 이벤트를 함께 달고 있다(불이 나면 연기가 난다).
      안 거르면 fire 영상의 smoke 구간이 fire 양성으로 둔갑한다.
    """
    g = np.zeros(n, np.int8)
    ev = json.load(open(label_path, encoding="utf-8")).get("events", [])
    for e in ev:
        if e.get("category") != cat:
            continue
        s, t = e["timestamp"]
        for i in range(n):
            lo, hi = i / fps, (i + 1) / fps
            if min(hi, t) > max(lo, s):
                g[i] = 1
    return g.tolist()


def main():
    os.makedirs(S.OUT, exist_ok=True)
    for c in S.CATEGORIES:
        os.makedirs(os.path.join(S.EVIDENCE, c), exist_ok=True)

    print("=" * 78)
    print("1차 검출기 — 운영 동일 세팅")
    print("=" * 78)
    print(f"  MXQ        : {S.PE_MXQ}")
    print(f"               quant={S.PE_QUANT} tuning={S.PE_TUNING} scheme={S.PE_SCHEME}")
    print(f"  text feat  : {S.TEXT_FEATURES}")
    print(f"  판정       : method={S.PE_METHOD} top_candidate={S.PE_TOP_CANDIDATE}")
    print(f"  알람 FSM   : 최근 {S.PE_QUEUE_SIZE}창 중 {S.PE_ALARM_DURATION}회 이상 -> 발화")
    print(f"  입력       : {S.FPS} fps (운영 PENpuService fps)")
    assert os.path.exists(S.PE_MXQ), f"MXQ 없음: {S.PE_MXQ}"

    # ── 자산 로드 ────────────────────────────────────────────────────────────
    t0 = time.time()
    ids, cls_list, prompts, feats, weights = load_text_feature(S.TEXT_FEATURES, "cpu")
    print(f"\n  텍스트 임베딩 {feats.shape} ({time.time()-t0:.1f}s)")
    print(f"    class 분포: {dict(sorted(zip(*np.unique(cls_list, return_counts=True))))}")

    model = MultiNPUInferenceFull(S.PE_MXQ)
    pre = ParallelPreprocessor()
    print(f"  NPU {len(model)}장  core_mode={model.core_mode}  slots/card={model.slots_per_card}")

    txt = {"ids": ids, "vectors": feats, "class_list": cls_list,
           "prompt_list": prompts, "weight_list": weights}

    clips = clip_list()
    per_dir = {c: sum(1 for x in clips if x["cat_dir"] == c) for c in S.CATEGORIES}
    print(f"  평가 대상 {len(clips)}건  " +
          "  ".join(f"{c} {n}" for c, n in per_dir.items()) + "\n")

    # ── 디코딩 (프로세스 병렬) ────────────────────────────────────────────────
    # NPU 가 아니라 **영상 디코딩이 병목**이다(실측 NPU 사용률 2.7%).
    # 15초/24fps 영상에서 1fps 15장을 쓰려고 361프레임을 푸는 구조라 CPU 로 흩뿌린다.
    nw = min(S.DECODE_WORKERS, os.cpu_count() or 8, len(clips))
    print(f"  디코딩 {len(clips)}건 / 워커 {nw}개 ...", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=nw) as ex:
        decoded = list(ex.map(_decode,
                              [(c["mp4"], S.FPS, S.PE_IMG_SIZE[0]) for c in clips],
                              chunksize=1))
    print(f"  디코딩 완료 {time.time()-t0:.1f}s "
          f"({sum(len(d[0]) for d in decoded):,}프레임)\n", flush=True)

    results, t0, nfr = [], time.time(), 0
    for k, c in enumerate(clips, 1):
        frames, jpgs, secs, dur = decoded[k - 1]
        if not frames:
            print(f"  [!] 프레임 없음: {c['name']}")
            continue
        nfr += len(frames)

        # 카테고리별로 **독립 EventManager** — 운영은 스트림마다 매니저 상태가 독립이고,
        # 여기선 한 영상을 3개 카테고리로 각각 보므로 상태가 섞이면 안 된다.
        mgr = {}
        for cat in S.CATEGORIES:
            m = PENpuEventManager()
            m.prepare_vectors(txt)
            mgr[cat] = m

        sid = "s0"
        up = {cat: {"user_param": {"retEvent": [RET[cat]]}} for cat in S.CATEGORIES}
        queues = {cat: deque(maxlen=TEMPORAL_SIZE) for cat in S.CATEGORIES}

        # 1차 추론 — 영상 전체를 한 배치로 NPU 에 넣는다(카드 분산은 추론기가 한다)
        vecs = model(pre(frames))
        if isinstance(vecs, np.ndarray):
            vecs = torch.from_numpy(vecs)

        rec = {"name": c["name"], "cat_dir": c["cat_dir"], "dur": dur,
               "secs": secs, "n": len(frames),
               # 폴더와 무관하게 **그 영상의 라벨**에서 카테고리별로 뽑는다.
               # (fire 폴더 영상도 smoke 양성 구간을 실제로 갖고 있다)
               "gt": {cat: gt_bins(c["json"], cat, len(frames), S.FPS)
                      for cat in S.CATEGORIES},
               "active": {cat: [] for cat in S.CATEGORIES},
               # raw = 알람 FSM 이전, topk 가 그 프레임을 양성 후보로 봤는가(0/1).
               # active(FSM 통과) 와 나눠 봐야 "1차가 아예 못 봤다" 와
               # "봤는데 5창중3 을 못 채웠다" 가 구분된다.
               "raw": {cat: [] for cat in S.CATEGORIES},
               "alarms": []}

        for i, v in enumerate(vecs):
            for cat in S.CATEGORIES:
                q = queues[cat]
                while len(q) < TEMPORAL_SIZE:
                    q.append(v)
                q.append(v)
                alarms, _ = mgr[cat].update({sid: q}, [sid], [up[cat]])
                # duration_queue 의 마지막 원소 = 이번 프레임의 topk 판정(FSM 이전)
                rec["raw"][cat].append(int(mgr[cat].duration_queue[sid][RET[cat]][-1]))
                st = mgr[cat].event_status[sid][RET[cat]]
                rec["active"][cat].append(int(st in (1, 2)))   # 1=시작 2=지속
                if st == 1:                                     # 알람 발화 순간
                    # 폴더가 달라도 stem 이 같은 **다른 영상**이 14쌍 있다(md5 불일치).
                    # cat_dir 를 넣지 않으면 증거가 덮어써진다.
                    ev = f"{c['cat_dir']}__{c['name']}__{cat}__t{secs[i]:05.1f}.jpg"
                    # 원본 해상도 JPEG 를 그대로 쓴다(PE 전처리용 336 이 아니다)
                    with open(os.path.join(S.EVIDENCE, cat, ev), "wb") as fo:
                        fo.write(jpgs[i])
                    rec["alarms"].append({"cat": cat, "idx": i, "sec": secs[i],
                                          "evidence": ev})
        results.append(rec)

        if k % 10 == 0 or k == len(clips):
            el = time.time() - t0
            print(f"  [{k:3}/{len(clips)}] {nfr:5}프레임  {nfr/max(el,1e-9):5.1f} f/s  "
                  f"알람누적 {sum(len(r['alarms']) for r in results)}")

    meta = {"설정": {"mxq": S.PE_MXQ, "quant": S.PE_QUANT, "tuning": S.PE_TUNING,
                    "scheme": S.PE_SCHEME, "core_mode": model.core_mode,
                    "text_features": S.TEXT_FEATURES, "n_prompts": int(feats.shape[0]),
                    "method": S.PE_METHOD, "top_candidate": S.PE_TOP_CANDIDATE,
                    "queue_size": S.PE_QUEUE_SIZE, "alarm_duration": S.PE_ALARM_DURATION,
                    "fps": S.FPS, "npu_cards": len(model)},
            "clips": results}
    with open(os.path.join(S.OUT, "stage1.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)

    na = sum(len(r["alarms"]) for r in results)
    print(f"\n  완료: {len(results)}건 / {nfr:,}프레임 / {time.time()-t0:.1f}s")
    per_cat = {c: sum(1 for r in results for a in r["alarms"] if a["cat"] == c)
               for c in S.CATEGORIES}
    print(f"  1차 알람 발화 {na}건  " +
          "  ".join(f"{c} {n}" for c, n in per_cat.items()))
    print(f"  -> {S.OUT}/stage1.json, {S.EVIDENCE}/")


if __name__ == "__main__":
    main()
