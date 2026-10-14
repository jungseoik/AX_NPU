# -*- coding: utf-8 -*-
"""2차 단독 평가 — 1차·알람 FSM 없이 **2차 VQA 만** 전 프레임에 돌린다.

목적: 2차 검출기 자체의 능력을 1차 필터와 알람 지연에서 **분리해서** 본다.
  · 1차 파이프라인: 1차 후보 -> 알람 FSM(5창중3) -> 2차 1장 -> 최종
  · 여기(2차 단독): 1초마다 프레임 1장 -> 2차 -> 그 1초 구간을 통째로 1/0 으로 채움

알람 FSM 이 없으므로 3프레임 발화 지연이 사라진다. 1차 미탐도 영향을 주지 않는다.
남는 것은 **2차가 그 장면을 맞히는가** 뿐이다.

출력: out/stage2_only.json
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures as cf
import json
import os
import time

import cv2
import requests
import yaml

import settings as S

URL = os.environ.get("VQA_URL", "http://127.0.0.1:8100")


def clip_list():
    out = []
    for cat in S.CATEGORIES:
        d = os.path.join(S.DATASET, cat)
        for f in sorted(os.listdir(d)):
            if f.endswith(".mp4"):
                out.append({"cat_dir": cat, "name": f[:-4],
                            "mp4": os.path.join(d, f),
                            "json": os.path.join(d, f[:-4] + ".json")})
    return out


def sample(path, fps):
    """1/fps 구간의 중앙 프레임을 순차 디코드로 집어 JPEG 바이트로 돌려준다.

    서버가 256x192 로 리사이즈하므로 원본 해상도 그대로 보낸다(운영과 동일).
    """
    cap = cv2.VideoCapture(path)
    sf = cap.get(cv2.CAP_PROP_FPS) or 24.0
    tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = max(1, int(tot / sf * fps))
    want = {min(tot - 1, int(round(((i + .5) / fps) * sf))): i for i in range(n)}
    buf, secs, pos = [None] * n, [(i + .5) / fps for i in range(n)], 0
    while want:
        if not cap.grab():
            break
        if pos in want:
            ok, bgr = cap.retrieve()
            if ok:
                e, j = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                buf[want.pop(pos)] = j.tobytes() if e else None
            else:
                want.pop(pos)
        pos += 1
    cap.release()
    keep = [i for i, b in enumerate(buf) if b is not None]
    return [buf[i] for i in keep], [secs[i] for i in keep]


def gt_bins(label_path, cat, n, fps):
    """초 단위 라벨 -> 구간 격자 0/1. events[].category 로 반드시 거른다."""
    g = [0] * n
    for e in json.load(open(label_path, encoding="utf-8")).get("events", []):
        if e.get("category") != cat:
            continue
        s, t = e["timestamp"]
        for i in range(n):
            if min((i + 1) / fps, t) > max(i / fps, s):
                g[i] = 1
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=48)
    args = ap.parse_args()

    raw = yaml.safe_load(open(S.VQA_PROMPTS, encoding="utf-8"))
    alias = {}
    for k, v in raw.items():
        if isinstance(v, dict) and "text" in v:
            for al in [k] + list(v.get("aliases") or []):
                alias[al] = v["text"]
    prompts = {c: alias[f"{c}_ret"] for c in S.CATEGORIES}
    print(f"문안 출처: {S.VQA_PROMPTS}  (variant={S.VQA_PROMPT_VARIANT})")
    print("서버:", requests.get(f"{URL}/health", timeout=10).json())

    clips = clip_list()
    print(f"디코딩 {len(clips)}건 @ {S.FPS}fps ...", flush=True)
    t0 = time.time()
    with cf.ProcessPoolExecutor(max_workers=min(S.DECODE_WORKERS, os.cpu_count())) as ex:
        decoded = list(ex.map(sample, [c["mp4"] for c in clips],
                              [S.FPS] * len(clips), chunksize=1))
    print(f"  완료 {time.time()-t0:.1f}s  총 {sum(len(d[0]) for d in decoded):,}프레임\n")

    # 평가 범위 밖(화재 영상의 연기 판정)은 애초에 호출하지 않는다 — 비용 낭비다
    jobs = []
    for ci, (c, (frames, secs)) in enumerate(zip(clips, decoded)):
        for cat in S.CATEGORIES:
            if c["cat_dir"] in S.EVAL_EXCLUDE.get(cat, ()):
                continue
            for fi in range(len(frames)):
                jobs.append((ci, cat, fi))
    print(f"2차 호출 {len(jobs):,}건  동시 {args.workers}")

    sess = requests.Session()
    sess.mount("http://", requests.adapters.HTTPAdapter(
        pool_maxsize=args.workers * 2, pool_connections=args.workers * 2))

    def ask(job):
        ci, cat, fi = job
        b = base64.b64encode(decoded[ci][0][fi]).decode()
        body = {"model": "ft", "max_tokens": 1, "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}},
            {"type": "text", "text": prompts[cat]}]}]}
        r = sess.post(f"{URL}/v1/chat/completions", json=body, timeout=300)
        r.raise_for_status()
        return job, r.json()["choices"][0]["message"]["content"].strip().lower()

    pred = {}
    t0, err = time.time(), 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, fu in enumerate(cf.as_completed([ex.submit(ask, j) for j in jobs]), 1):
            try:
                job, lab = fu.result()
                pred[job] = 1 if lab == "yes" else 0
            except Exception:
                err += 1
            if i % 500 == 0 or i == len(jobs):
                el = time.time() - t0
                print(f"  {i:,}/{len(jobs):,}  {el/i*1000:.0f}ms/건  "
                      f"남은 {(len(jobs)-i)*el/i/60:.1f}분  오류 {err}", flush=True)

    out = []
    for ci, (c, (frames, secs)) in enumerate(zip(clips, decoded)):
        rec = {"name": c["name"], "cat_dir": c["cat_dir"], "n": len(frames), "secs": secs,
               "gt": {}, "pred": {}}
        for cat in S.CATEGORIES:
            if c["cat_dir"] in S.EVAL_EXCLUDE.get(cat, ()):
                continue
            rec["gt"][cat] = gt_bins(c["json"], cat, len(frames), S.FPS)
            rec["pred"][cat] = [pred.get((ci, cat, fi), 0) for fi in range(len(frames))]
        out.append(rec)

    meta = {"설정": {"방식": "2차 단독(1차·알람 FSM 없음)", "fps": S.FPS,
                    "size": f"{S.VQA_SIZE[0]}x{S.VQA_SIZE[1]}",
                    "문안": S.VQA_PROMPTS, "variant": S.VQA_PROMPT_VARIANT,
                    "호출수": len(jobs), "오류": err,
                    "소요초": round(time.time() - t0, 1)},
            "clips": out}
    p = os.path.join(S.OUT, "stage2_only.json")
    json.dump(meta, open(p, "w"), ensure_ascii=False)
    print(f"\n완료 {time.time()-t0:.1f}s  오류 {err}\n-> {p}")


if __name__ == "__main__":
    main()
