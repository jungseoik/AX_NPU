# -*- coding: utf-8 -*-
"""2차 검출기 호출 — 운영 bs32 검증서버의 **HTTP 계약 그대로** 쓴다.

직접 Classifier 를 잡지 않는다. 컨테이너를 운영과 똑같이 띄워 놓고
`POST /v1/chat/completions` 로 묻는다. 그래야 VLMPool 의 카드 선택(inflight 기준),
배치 묶기(최대 32), 세마포어(96), 프롬프트 처리까지 **운영 경로를 그대로 탄다.**

   참조: third_party/Product-AI-mono/packages/pia_2stage/validation_server_npu_bs32/README.md

이미지 크기는 서버의 NPU_PE_2STAGE_VLM_SIZE 가 정한다(기본 256x192 = 운영값).
클라이언트는 원본 jpg 를 그대로 보낸다 — 리사이즈는 서버가 한다.

문안도 서버 기준이다: 2stage_prompts 의 **ax_npu** variant(= 서버 DEFAULT_VARIANT).
카테고리는 폴더명이 아니라 YAML 의 `aliases`(`<cat>_ret`)로 찾는다 — 운영 판정도 그렇다.
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures as cf
import glob
import json
import os
import time

import requests
import yaml

import settings as S

URL = os.environ.get("VQA_URL", "http://127.0.0.1:8100")


def ask(sess, img_path, text, timeout=180):
    b64 = base64.b64encode(open(img_path, "rb").read()).decode()
    body = {"model": "ft", "max_tokens": 1, "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        {"type": "text", "text": text}]}]}
    r = sess.post(f"{URL}/v1/chat/completions", json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip().lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32,
                    help="동시 요청 수. 서버 세마포어가 96 이고 카드당 배치 32 라 "
                         "넉넉히 넣어야 배치가 성립한다(좁히면 1건씩 돈다)")
    a = ap.parse_args()

    raw = yaml.safe_load(open(S.VQA_PROMPTS, encoding="utf-8"))
    # alias -> text. 운영은 `falldown_ret` 같은 별칭으로 문안을 찾는다.
    alias = {}
    for key, v in raw.items():
        if not isinstance(v, dict) or "text" not in v:
            continue
        for al in [key] + list(v.get("aliases") or []):
            alias[al] = v["text"]
    missing = [f"{c}_ret" for c in S.CATEGORIES if f"{c}_ret" not in alias]
    if missing:
        raise KeyError(f"문안 별칭 누락: {missing} ({S.VQA_PROMPTS})")
    prompts = {c: {"text": alias[f"{c}_ret"]} for c in S.CATEGORIES}
    print(f"문안 출처: {S.VQA_PROMPTS}")
    print(f"  variant={S.VQA_PROMPT_VARIANT}  별칭 조회: " +
          ", ".join(f"{c}_ret({len(prompts[c]['text'])}자)" for c in S.CATEGORIES))
    jobs = [(cat, p) for cat in S.CATEGORIES
            for p in sorted(glob.glob(os.path.join(S.EVIDENCE, cat, "*.jpg")))]
    if not jobs:
        print("증거 이미지 0건"); return 1

    h = requests.get(f"{URL}/health", timeout=10).json()
    print(f"서버 {URL}  health={h}")
    print(f"총 {len(jobs)}건  동시 {a.workers}  (크기는 서버 NPU_PE_2STAGE_VLM_SIZE 가 결정)")

    sess = requests.Session()
    sess.mount("http://", requests.adapters.HTTPAdapter(pool_maxsize=a.workers * 2))
    rows, t0 = [], time.time()
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        fut = {ex.submit(ask, sess, p, prompts[cat]["text"]): (cat, p) for cat, p in jobs}
        for i, f in enumerate(cf.as_completed(fut), 1):
            cat, p = fut[f]
            try:
                rows.append({"cat": cat, "evidence": os.path.basename(p), "label": f.result()})
            except Exception as e:
                rows.append({"cat": cat, "evidence": os.path.basename(p),
                             "label": "ERROR", "error": f"{type(e).__name__}: {e}"})
            if i % 50 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}  {(time.time()-t0)/i*1000:.0f}ms/건")

    ok = [r for r in rows if r["label"] != "ERROR"]
    yes = sum(1 for r in ok if r["label"] == "yes")
    dt = time.time() - t0
    print(f"\n총 {len(rows)}건 / 통과(yes) {yes} / 기각(no) {len(ok)-yes} / "
          f"오류 {len(rows)-len(ok)} / {dt:.1f}s ({dt/len(rows)*1000:.0f}ms/건)")
    for cat in S.CATEGORIES:
        sub = [r for r in ok if r["cat"] == cat]
        y = sum(1 for r in sub if r["label"] == "yes")
        print(f"  {cat:9} {len(sub):4}건 -> 통과 {y:4} / 기각 {len(sub)-y:4}")
    for r in rows:
        if r["label"] == "ERROR":
            print(f"  [오류] {r['evidence']}: {r['error']}")

    rows.sort(key=lambda r: (S.CATEGORIES.index(r["cat"]), r["evidence"]))
    out = os.path.join(S.OUT, "stage2.json")
    json.dump(rows, open(out, "w"), ensure_ascii=False, indent=1)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
