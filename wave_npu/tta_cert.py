"""TTA 인증 시험 실행 스크립트 — 시험절차서 ①~⑦을 한 번에 수행하고 로그·원시자료를 남긴다.

    python -m wave_npu.tta_cert --spec wave_npu/specs/tta_4cat.json

시험 당일 TTA 시험원이 **옆에서 모니터로 함께 보는 것**을 전제로 설계했다:
진행 순서를 먼저 띄우고, 각 단계마다 무엇을 하는지와 진행률(tqdm)을 실시간으로 표시하며,
추론은 캐시가 아니라 **시험 장비의 NPU에서 실제로 수행**한다(--use-cache 는 개발용).

시험합의서 TTA-26-03361 기준:
  - 항목당 평가용 영상 100건(정상·이상·하드네거티브 포함), 12항목 합계 1,200건.
  - F1-score = 2·p·r/(p+r) × 100,  합격 기준 **≥ 90점**. 분모 0 이면 0점 처리.
  - 시험 중 모델·임계값·설정값 변경 금지 → 판정 설정은 실행 시점에 고정되어 로그에 박힌다.

판정 단위 **두 가지를 모두 산출**한다:
  - **클립(영상) 단위** — 시험규격의 판정 기준. 정답 라벨링 시트의 "<이벤트> 발생 여부"와 대조.
    영상 내 점수가 임계값 이상인 구간이 연속 K 프레임 이상이면 "발생"으로 판정.
  - **프레임 단위** — 참고 지표. 이벤트 구간을 시간축에서 얼마나 정확히 짚는지 본다.
    (클립 판정만 보면 "영상 어딘가에서 한 번 맞혔다"와 "구간을 정확히 짚었다"가 구분되지 않는다.)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
from tqdm import tqdm

from . import data
from .config import Spec
from .gt import frame_labels, load_labels

LINE = "=" * 82
SUB = "-" * 82

TID = {"falldown": ("T1", "쓰러짐 탐지"), "fire": ("T2", "화재 탐지"),
       "smoke": ("T3", "연기 탐지"), "intrusion": ("T4", "침입 탐지"),
       "loitering": ("T5", "배회 탐지"), "violence": ("T6", "폭력 탐지"),
       "smoking": ("T7", "흡연 탐지"), "crowd": ("T8", "군중 혼잡 탐지"),
       "elevator_falldown": ("T9", "엘리베이터 내 쓰러짐 탐지"),
       "escalator_falldown": ("T10", "에스컬레이터 내 쓰러짐 탐지"),
       "climbing": ("T11", "월담 탐지"), "tailgating": ("T12", "따라 들어가기 탐지")}


def log(msg=""):
    print(msg, flush=True)


def _sh(cmd, default="확인 불가"):
    try:
        o = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        return o.stdout.strip() or default
    except Exception:
        return default


def md5(path: str, chunk=1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ 진행 순서
def step0_plan(spec: Spec, a):
    log(LINE)
    log("  TTA-26-03361  NPU 기반 이상상황 12종 탐지 AI 영상분석 시스템 MACS (v1.0)")
    log("  인증 시험 실행 스크립트 — 시험절차서 ①~⑦")
    log(LINE)
    log(f"  시작 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}    "
        f"데이터셋 {spec.name}    시험항목 {len(spec.names)}개")
    log()
    log("  [ 진행 순서 ]")
    log("    ① 시험 환경·시험대상 식별정보 기록 (장비/OS/NPU 버전, 모델 md5 체크섬)")
    log("    ② 평가용 데이터셋·정답 라벨 식별정보 기록 (건수/양성·음성/해상도 구성)")
    log("    ③ 평가용 영상 일괄 추론  ← NPU 실행, 진행률 표시")
    log("    ④ 정답 라벨과 대조하여 TP·FP·FN 집계  (클립 단위 = 판정 / 프레임 단위 = 참고)")
    log("    ⑤ Precision·Recall·F1-score 산출")
    log("    ⑥ 항목별 90점 기준 적합 여부 판정")
    log("    ⑦ 최종 요약 및 제출 자료 저장")
    log()


# ---------------------------------------------------------------- ① 환경 기록
def step1_environment(models: dict) -> dict:
    log(LINE)
    log("① 시험 환경 및 시험대상 식별정보")
    log(LINE)
    env = {"시험일시": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    log(f"  시험일시            : {env['시험일시']}")

    env["장비모델"] = _sh("cat /sys/class/dmi/id/product_name 2>/dev/null")
    env["OS"] = _sh("lsb_release -ds 2>/dev/null || head -1 /etc/os-release")
    env["커널"] = platform.release()
    env["아키텍처"] = platform.machine()
    env["CPU"] = _sh("lscpu | grep 'Model name' | head -1 | cut -d: -f2- | xargs")
    env["CPU스레드"] = _sh("nproc")
    env["MEM"] = _sh("free -g | awk 'NR==2{print $2\" GB\"}'")
    log(f"  장비                : {env['장비모델']}")
    log(f"  OS / 커널           : {env['OS']} / {env['커널']} ({env['아키텍처']})")
    log(f"  CPU / MEM           : {env['CPU']} ({env['CPU스레드']} threads) / {env['MEM']}")

    status = _sh("mobilint-cli status 2>/dev/null")
    m = re.search(r"Aries:\s*([\d.]+)", status)
    env["NPU드라이버"] = m.group(1) if m else "확인 불가"
    fw = sorted(set(re.findall(r"^\|\s*\d+\s+\d+ C\s+([\d.]+)", status, re.M)))
    env["NPU펌웨어"] = ", ".join(fw) if fw else "확인 불가"
    try:
        import qbruntime
        env["NPU런타임"] = str(qbruntime.__version__)
    except Exception:
        env["NPU런타임"] = "확인 불가"
    env["NPU장치"] = sorted(int(d[5:]) for d in os.listdir("/dev") if d.startswith("aries"))
    log(f"  NPU 드라이버/런타임 : {env['NPU드라이버']} / {env['NPU런타임']}   펌웨어 {env['NPU펌웨어']}")
    log(f"  NPU 인식 장치       : {len(env['NPU장치'])}기  "
        f"/dev/aries{{{','.join(map(str, env['NPU장치']))}}}")

    env["python"] = platform.python_version()
    for mod in ("torch", "torchvision", "numpy", "cv2"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            env[mod] = "미설치"
    log(f"  python/torch/tv     : {env['python']} / {env['torch']} / {env['torchvision']}")
    log(f"  numpy / opencv      : {env['numpy']} / {env['cv2']}")

    log(SUB)
    log("  시험대상 AI 모델 — md5 체크섬 (컴파일 로그의 해시값과 대조)")
    env["모델"] = {}
    for name, path in models.items():
        if path and os.path.exists(path):
            info = {"경로": os.path.abspath(path), "md5": md5(path),
                    "크기bytes": os.path.getsize(path)}
            log(f"    {name:16s} md5={info['md5']}  {info['크기bytes'] / 1e6:8.2f} MB")
            log(f"    {'':16s} └ {path}")
        else:
            info = {"경로": path, "md5": "파일 없음", "크기bytes": 0}
            log(f"    {name:16s} !! 파일 없음: {path}")
        env["모델"][name] = info
    log()
    return env


# ------------------------------------------------- ② 데이터셋·정답 라벨 식별
def step2_dataset(spec: Spec, videos: dict, labels: dict) -> dict:
    log(LINE)
    log("② 평가용 데이터셋 및 정답 라벨 식별정보")
    log(LINE)
    log(f"  데이터셋            : {spec.name}   ({spec.root})")
    log(f"  판정 단위           : 클립(영상) 1건 = 이벤트 '발생 여부' 이진 판정  [시험규격]")
    log(f"                        + 프레임 단위 지표 병행 산출  [참고]")
    log(SUB)
    log(f"  {'항목':>4s} {'카테고리':16s} {'건수':>5s} {'양성':>5s} {'음성':>5s} {'프레임':>8s}  해상도 구성")
    info = {}
    for cat in spec.names:
        rows = labels[cat]
        pos = sum(r["gt"] for r in rows)
        res = {}
        for r in rows:
            res[r["해상도"]] = res.get(r["해상도"], 0) + 1
        nfr = sum(r["프레임수"] for r in rows)
        tid = TID.get(cat, ("T?", cat))[0]
        log(f"  {tid:>4s} {cat:16s} {len(rows):5d} {pos:5d} {len(rows) - pos:5d} {nfr:8,d}  "
            + ", ".join(f"{k}×{v}" for k, v in sorted(res.items(), key=lambda x: -x[1])))
        info[cat] = {"건수": len(rows), "양성": pos, "음성": len(rows) - pos,
                     "총프레임": nfr, "해상도": res}
        if len(rows) < 100:
            log(f"       ※ 평가 건수 {len(rows)} < 100 — 실제 건수를 성적서에 명시(시험합의서 예외사항)")
        if len(rows) - pos == 0:
            log(f"       ※ 음성 0건 — 이 항목은 오검출(FP)이 발생할 수 없어 Precision 이 항상 1")
    log()
    return info


# ------------------------------------------------------------------ 판정 계산
def clip_predict(scores: np.ndarray, thr: float, k: int):
    """연속 k 프레임 이상 thr 초과 → '발생'. (판정, 최장연속, 최대점수)"""
    run = best = 0
    for o in (scores >= thr):
        run = run + 1 if o else 0
        if run > best:
            best = run
    return bool(best >= k), int(best), float(scores.max() if len(scores) else 0.0)


def prf(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if (tp + fp) else 0.0          # 분모 0 → 0 처리
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) * 100 if (p + r) else 0.0
    return p, r, f


# ----------------------------------------------------------- ③ 일괄 추론 (NPU)
class Engines:
    """시험에 쓰는 NPU 모델들. 필요한 것만 올린다."""

    def __init__(self, spec, a):
        self.pe = self.yolo = None
        self.need_pe = bool([c for c in spec.names
                             if spec.categories[c].get("source", "pe") == "pe"])
        self.need_yolo = bool(spec.by_source("person"))
        devs = [int(x) for x in a.device_ids.split(",")] if a.device_ids else None

        if self.need_pe:
            log(f"  · PE-Core MXQ 로드 (scheme={a.scheme}, device={devs or [0]}) ...")
            from pe_npu.inference import MXQInferenceFull
            self.pe = (MXQInferenceFull(full_mxq_path=a.mxq, device_ids=devs)
                       if a.mxq and os.path.exists(a.mxq)
                       else MXQInferenceFull.from_hf(scheme=a.scheme, device_ids=devs))
            log(f"    → {len(self.pe)}카드, core_mode={self.pe.core_mode}, 스레드풀 {self.pe.W}")
        if self.need_yolo:
            log(f"  · YOLO11 MXQ 로드 ({a.yolo_model}) ...")
            from yolo_npu.detect import YOLONPU
            self.yolo = YOLONPU.load(a.yolo_model, a.scheme, local_mxq=a.yolo_mxq,
                                     device_ids=devs, conf_thres=0.10)
            log(f"    → {len(self.yolo)}카드")

    def dispose(self):
        for m in (self.pe, self.yolo):
            try:
                m.dispose()
            except Exception:
                pass


def _decode(path, folder, need_pe, need_yolo):
    """영상 1건 디코딩 + PE 입력 리사이즈. NPU 대기 뒤에 숨기려고 스레드로 미리 돌린다."""
    import cv2
    import torch
    import torchvision.transforms.functional as TF
    from torchvision.transforms import InterpolationMode
    from .embed import IMAGE_SIZE

    cap = cv2.VideoCapture(path)
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 24.0
    w, h = int(cap.get(3)), int(cap.get(4))
    small, full, idxs, i = [], [], [], -1
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        i += 1
        if need_pe:
            t = torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
            small.append(TF.resize(t.contiguous(), [IMAGE_SIZE, IMAGE_SIZE],
                                   interpolation=InterpolationMode.BILINEAR, antialias=True))
        if need_yolo:
            full.append(bgr)
        idxs.append(i)
    cap.release()
    if not idxs:
        return None
    return {"파일": f"{folder}/{os.path.splitext(os.path.basename(path))[0]}", "폴더": folder,
            "frames": np.asarray(idxs, np.int32), "fps": fps, "해상도": f"{w}x{h}",
            "_small": torch.stack(small).numpy() if need_pe else None, "_full": full}


def infer_videos(spec, a, eng: Engines):
    """평가용 영상을 1회 일괄 추론. 디코딩은 스레드풀로 선행, 추론은 NPU에서 순차 수행."""
    from concurrent.futures import ThreadPoolExecutor
    from .embed import to_model_input

    vids = []
    for f in spec.folders:
        d = os.path.join(spec.root, f)
        if not os.path.isdir(d):
            continue
        fs = [(f, os.path.join(d, x)) for x in sorted(os.listdir(d)) if x.endswith(".mp4")]
        vids += fs[:a.limit] if a.limit else fs
    log(f"  · 평가 대상 영상 {len(vids)}건 ({', '.join(spec.folders)})")
    log(f"  · 디코딩 스레드 {a.decode_workers} / NPU 배치 {a.batch}")
    log(SUB)

    out = {}
    nfr = 0
    t0 = time.time()
    bar = tqdm(total=len(vids), desc="  ③ NPU 일괄 추론", unit="건", ncols=82,
               bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                          "[{elapsed}<{remaining}{postfix}]")
    # 선행 디코딩은 **동시 N건으로 제한**한다. 원본 해상도 프레임을 통째로 들고 있어
    # (1080p 15초 ≈ 1.8 GB) 무제한으로 앞서 나가면 메모리가 먼저 터진다.
    inflight = max(2, a.decode_workers)
    with ThreadPoolExecutor(max_workers=a.decode_workers) as ex:
        pending, nxt = [], 0
        while nxt < len(vids) and len(pending) < inflight:
            f, p_ = vids[nxt]
            pending.append(ex.submit(_decode, p_, f, eng.need_pe, eng.need_yolo))
            nxt += 1
        while pending:
            rec = pending.pop(0).result()
            if nxt < len(vids):
                f, p_ = vids[nxt]
                pending.append(ex.submit(_decode, p_, f, eng.need_pe, eng.need_yolo))
                nxt += 1
            if rec is None:
                bar.update(1)
                continue
            if eng.need_pe:
                u8 = rec.pop("_small")
                e = np.concatenate([eng.pe.infer(to_model_input(u8[s:s + a.batch]))
                                    for s in range(0, len(u8), a.batch)], 0).astype(np.float32)
                rec["pe_emb"] = e / (np.linalg.norm(e, axis=-1, keepdims=True) + 1e-12)
                del u8
            else:
                rec.pop("_small", None)
            full = rec.pop("_full")
            if eng.need_yolo:
                det = eng.yolo.detect_batch(full)
                rec["person"] = np.asarray(
                    [max((x[4] for x in d if int(x[5]) == 0), default=0.0) for d in det],
                    np.float32)
            del full
            out.setdefault(rec["폴더"], []).append(rec)
            nfr += len(rec["frames"])
            bar.update(1)
            bar.set_postfix_str(f"{nfr:,}프레임 {nfr / max(time.time() - t0, 1e-9):.0f} f/s")
    bar.close()
    dt = time.time() - t0
    log(f"  · 추론 완료: {len(vids)}건 / {nfr:,}프레임 / {dt:.1f}s ({nfr / max(dt, 1e-9):.1f} frame/s)")
    log()
    return out


def build_scores(spec, a, raw: dict, cfgs: dict):
    """추론 산출물 → 카테고리별 프레임 점수열 + 정답(프레임/클립)."""
    from .score import category_scores, sims_matrix, smooth
    labels_gt = load_labels(spec.root, tuple(spec.names), spec.folders)
    allrecs = [r for v in raw.values() for r in v]

    pe_cats = [c for c in spec.names if spec.categories[c].get("source", "pe") == "pe"]
    scores = {c: {} for c in spec.names}
    if pe_cats:
        temb, cls, _ = data.load_text(a.text)
        mask = np.load(a.mask, allow_pickle=True)["mask"]
        te, cl = temb[mask], cls[mask]
        emb = np.concatenate([r["pe_emb"] for r in allrecs], 0)
        vid = np.concatenate([np.full(len(r["frames"]), i, np.int32)
                              for i, r in enumerate(allrecs)])
        S = sims_matrix(emb, te)
        for i, c in enumerate(pe_cats):
            raw_s = category_scores(S, cl, tuple(pe_cats), rule=cfgs[c]["rule"],
                                    cat2cls=spec.cat2cls, normal=spec.normal_class)
            sm = smooth(raw_s, vid, cfgs[c]["win"])[:, i]
            for j, r in enumerate(allrecs):
                scores[c][r["파일"]] = sm[vid == j]
    for c in spec.by_source("person"):
        vid = np.concatenate([np.full(len(r["frames"]), i, np.int32)
                              for i, r in enumerate(allrecs)])
        s = np.concatenate([r["person"] for r in allrecs])
        sm = smooth(s[:, None], vid, cfgs[c]["win"])[:, 0]
        for j, r in enumerate(allrecs):
            scores[c][r["파일"]] = sm[vid == j]

    rows = {}
    for c in spec.names:
        ef = spec.eval_folders(c)
        rc = []
        for r in allrecs:
            if ef and r["폴더"] not in ef:
                continue
            fl = frame_labels(labels_gt[r["파일"]]["events"], r["frames"], r["fps"], (c,))[:, 0]
            rc.append({"파일": r["파일"], "해상도": r["해상도"], "프레임수": len(r["frames"]),
                       "scores": scores[c][r["파일"]], "frame_gt": fl, "gt": bool(fl.any())})
        rows[c] = rc
    return rows


# ------------------------------------------------------- ④⑤⑥ 집계·산출·판정
def run_item(tid, cat, rows, cfg, outdir) -> dict:
    log(LINE)
    log(f"{tid}. {cfg['항목명']}  —  평가 {len(rows)}건 "
        f"(양성 {sum(r['gt'] for r in rows)} / 음성 {sum(not r['gt'] for r in rows)})")
    log(LINE)
    log(f"  판정 설정 [시험 중 변경 금지]  신호={cfg['rule']}  평활창={cfg['win']}f  "
        f"임계값={cfg['thr']:+.6f}  연속={cfg['k']}f")
    log(SUB)

    ctp = cfp = cfn = ctn = 0
    ftp = ffp = ffn = 0
    recs, errs = [], []
    for i, r in enumerate(tqdm(rows, desc=f"  ④ {tid} 정답 대조", unit="건", ncols=82,
                               bar_format="{desc}: {percentage:3.0f}%|{bar}| "
                                          "{n_fmt}/{total_fmt} [{elapsed}<{remaining}]"), 1):
        pred, runlen, peak = clip_predict(r["scores"], cfg["thr"], cfg["k"])
        gt = r["gt"]
        res = "TP" if (pred and gt) else "FP" if pred else "FN" if gt else "TN"
        ctp += res == "TP"; cfp += res == "FP"; cfn += res == "FN"; ctn += res == "TN"

        fp_ = r["scores"] >= cfg["thr"]; fg = r["frame_gt"]
        ftp += int(np.sum(fp_ & fg)); ffp += int(np.sum(fp_ & ~fg)); ffn += int(np.sum(~fp_ & fg))

        recs.append({"no": i, "파일명": r["파일"], "해상도": r["해상도"], "프레임수": r["프레임수"],
                     "정답(클립)": int(gt), "판정(클립)": int(pred), "결과": res,
                     "최대점수": round(peak, 6), "최장연속프레임": runlen,
                     "정답프레임수": int(fg.sum()), "판정프레임수": int(fp_.sum()),
                     "프레임TP": int(np.sum(fp_ & fg))})
        if res in ("FP", "FN"):
            errs.append(f"    [{res}] {r['파일']:52s} peak={peak:+.4f}  최장연속={runlen:3d}f")

    for e in errs:
        log(e)
    if not errs:
        log("    (오검출·미검출 없음)")

    csv_path = os.path.join(outdir, f"{tid}_{cat}_건별결과.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader(); w.writerows(recs)

    cp, cr, cf = prf(ctp, cfp, cfn)
    fpp, frr, ff = prf(ftp, ffp, ffn)
    passed = cf >= 90.0
    log(SUB)
    log(f"  ③ 건별 추론 결과 저장 : {csv_path}")
    log(f"  ④ 집계 [클립 = 판정]  : TP={ctp}  FP={cfp}  FN={cfn}   (TN={ctn})")
    log(f"     집계 [프레임 = 참고]: TP={ftp:,}  FP={ffp:,}  FN={ffn:,}")
    log(f"  ⑤ 클립  Precision={ctp}/({ctp}+{cfp})={cp:.4f}   Recall={ctp}/({ctp}+{cfn})={cr:.4f}")
    log(f"          F1-score = 2×{cp:.4f}×{cr:.4f}/({cp:.4f}+{cr:.4f})×100 = "
        f"**{cf:.2f} 점**")
    log(f"     프레임 Precision={fpp:.4f}  Recall={frr:.4f}  F1-score = {ff:.2f} 점  (참고)")
    log(f"  ⑥ 판정                : 클립 F1 {cf:.2f}점 {'≥' if passed else '<'} 90점  →  "
        f"{'■ 적합 (PASS)' if passed else '□ 부적합 (FAIL)'}")
    log()
    return {"시험항목ID": tid, "항목명": cfg["항목명"], "평가건수": len(rows),
            "양성": sum(r["gt"] for r in rows), "음성": sum(not r["gt"] for r in rows),
            "클립": {"TP": ctp, "FP": cfp, "FN": cfn, "TN": ctn,
                     "precision": round(cp, 6), "recall": round(cr, 6), "f1_score": round(cf, 2)},
            "프레임": {"TP": ftp, "FP": ffp, "FN": ffn,
                       "precision": round(fpp, 6), "recall": round(frr, 6),
                       "f1_score": round(ff, 2)},
            "기준": 90.0, "판정": "PASS" if passed else "FAIL",
            "설정": {k: cfg[k] for k in ("rule", "win", "thr", "k")}, "건별결과": csv_path}


# --------------------------------------------------------------------- 진입점
def main():
    ap = argparse.ArgumentParser(description="TTA-26-03361 인증 시험 실행")
    ap.add_argument("--spec", default="wave_npu/specs/tta_4cat.json")
    ap.add_argument("--config", default="wave_npu/specs/decision_4cat.json",
                    help="판정 설정(신호/평활창/임계값/연속프레임) — 시험 중 변경 금지")
    ap.add_argument("--mxq", default=None, help="PE-Core MXQ 경로 (미지정 시 HF)")
    ap.add_argument("--yolo-mxq", default=None)
    ap.add_argument("--yolo-model", default="yolo11m")
    ap.add_argument("--scheme", default="single")
    ap.add_argument("--device-ids", default=None, help="사용할 NPU. 시험은 1기 기준(예: 0)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--decode-workers", type=int, default=4,
                    help="영상 디코딩 선행 스레드 수 (= 동시 선행 건수 상한). "
                         "원본 프레임을 메모리에 들고 있어 무작정 키우면 안 된다")
    ap.add_argument("--text", default="wave_npu/cache/text_feats.npz")
    ap.add_argument("--mask", default="wave_npu/artifacts/pipeline_mask.npz")
    ap.add_argument("--out", default="wave_npu/artifacts/tta_run")
    ap.add_argument("--limit", type=int, default=0, help="개발용: 앞 N건만")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    spec = Spec.load(a.spec)
    cfgs = json.load(open(a.config, encoding="utf-8"))

    step0_plan(spec, a)
    env = step1_environment({"PE-Core MXQ": a.mxq, "YOLO11 MXQ": a.yolo_mxq,
                             "프롬프트 마스크": a.mask, "판정 설정": a.config})

    log(LINE)
    log("③ 평가용 영상 일괄 추론 (시험 장비 NPU)")
    log(LINE)
    eng = Engines(spec, a)
    if a.limit:
        log(f"  ※ 개발 모드: 앞 {a.limit}건만 처리")
    raw = infer_videos(spec, a, eng)
    eng.dispose()

    rows = build_scores(spec, a, raw, cfgs)
    step2_dataset(spec, raw, rows)

    results = []
    for cat in spec.names:
        tid, name = TID.get(cat, ("T?", cat))
        cfg = dict(cfgs[cat]); cfg["항목명"] = name; cfg.setdefault("k", 1)
        results.append(run_item(tid, cat, rows[cat], cfg, a.out))

    # ------------------------------------------------------------ ⑦ 최종 요약
    log(LINE)
    log("⑦ 최종 결과 요약 및 제출 자료")
    log(LINE)
    log(f"  {'ID':>4s} {'시험항목':24s} {'건수':>5s} | {'TP':>4s} {'FP':>4s} {'FN':>4s} "
        f"{'P':>7s} {'R':>7s} {'F1(점)':>8s} {'기준':>5s} {'판정':>6s} | {'프레임F1':>9s}")
    log(f"  {'':>4s} {'':24s} {'':>5s} | {'--- 클립 단위 (시험규격 판정) ---':^45s} | {'참고':>9s}")
    for r in results:
        c = r["클립"]
        log(f"  {r['시험항목ID']:>4s} {r['항목명']:24s} {r['평가건수']:5d} | {c['TP']:4d} {c['FP']:4d} "
            f"{c['FN']:4d} {c['precision']:7.4f} {c['recall']:7.4f} {c['f1_score']:8.2f} "
            f"{'≥90':>5s} {'PASS' if r['판정'] == 'PASS' else 'FAIL':>6s} | "
            f"{r['프레임']['f1_score']:9.2f}")
    npass = sum(r["판정"] == "PASS" for r in results)
    log(SUB)
    log(f"  적합 {npass} / {len(results)} 항목"
        + ("   →   전 항목 적합 (PASS)" if npass == len(results)
           else f"   →   부적합: {', '.join(r['시험항목ID'] for r in results if r['판정'] == 'FAIL')}"))

    report = {"시험신청번호": "TTA-26-03361", "시험일시": env["시험일시"],
              "환경": env, "결과": results}
    jp = os.path.join(a.out, "시험결과.json")
    json.dump(report, open(jp, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    sp = os.path.join(a.out, "집계표.csv")
    with open(sp, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["시험항목ID", "항목명", "평가건수", "양성", "음성",
                    "클립TP", "클립FP", "클립FN", "클립TN",
                    "클립Precision", "클립Recall", "클립F1(점)", "기준", "판정",
                    "프레임TP", "프레임FP", "프레임FN", "프레임F1(점,참고)"])
        for r in results:
            c, fr = r["클립"], r["프레임"]
            w.writerow([r["시험항목ID"], r["항목명"], r["평가건수"], r["양성"], r["음성"],
                        c["TP"], c["FP"], c["FN"], c["TN"], f"{c['precision']:.4f}",
                        f"{c['recall']:.4f}", f"{c['f1_score']:.2f}", "≥90", r["판정"],
                        fr["TP"], fr["FP"], fr["FN"], f"{fr['f1_score']:.2f}"])

    log(SUB)
    log("  제출 자료 (성적서 부속):")
    log(f"    · 집계 결과      : {sp}")
    log(f"    · 시험 결과 전문 : {jp}   (환경·모델 md5·판정 설정 포함)")
    for r in results:
        log(f"    · 건별 원시 결과 : {r['건별결과']}")
    log(f"  종료 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(LINE)
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
