"""NAS `tta_dataset`(클립 벤치마크) → 레포 표준 구조로 복사·정규화.

원본(NAS, 읽기전용)
    <항목>/<장소>/<subtype>/<이름>.mp4
    <항목>/<장소>/per_frame/<이름>.csv     ← 컬럼명이 subtype 마다 다름(하드네거티브는 sittingdown 등)
    gt_label.zip : gt_label/<항목>/<이름>.csv  ← **정규화된 정답지** (항목당 컬럼 1개)

레포(`eval/datasets/<name>/`) — 기존 `TTA_인증용` 컨벤션과 동일하게 평탄화
    <항목>/<이름>.mp4
    <항목>/<이름>.csv        ← 프레임 라벨 `clip_frame,<event>` (gt_label.zip 에서 가져옴)
    MANIFEST.csv             ← 항목·이름·subtype·양성여부·장소·fps·해상도·프레임수

**라벨은 gt_label.zip 을 정답으로 쓴다.** 폴더 안 per_frame CSV 는 subtype 별로 컬럼이 달라
(예: elv_violence 하드네거티브가 `sittingdown`) 그대로 읽으면 라벨이 어긋난다.

프레임 CSV 를 그대로 유지하는 이유: 이 데이터는 fps 가 7~50 으로 섞여 있어
초 단위 timestamp 로 바꾸면 경계에서 왕복 오차가 난다. 프레임 라벨이 원본 그대로다.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import shutil
import zipfile

import cv2
from tqdm import tqdm

NAS = "nas/200tb_home_sjpark/tta_dataset"
POS_SUBTYPE = {"falldown", "violence", "no_helmet"}    # 양성 subtype 폴더명


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=NAS)
    ap.add_argument("--dst", default="eval/datasets/TTA_clip")
    ap.add_argument("--items", default="elv_falldown,esc_falldown,elv_violence",
                    help="가져올 항목 (helmet 은 TTA 12종에 없어 기본 제외)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    items = a.items.split(",")

    # gt_label.zip → 임시 전개
    gtz = os.path.join(a.src, "gt_label.zip")
    gtdir = os.path.join(a.dst, "_gt_label")
    if not a.dry_run:
        os.makedirs(a.dst, exist_ok=True)
        with zipfile.ZipFile(gtz) as z:
            z.extractall(gtdir)
    gtroot = os.path.join(gtdir, "gt_label")

    rows = []
    for item in items:
        mp4s = sorted(glob.glob(os.path.join(a.src, item, "*", "*", "*.mp4")))
        if not mp4s:
            print(f"[!] {item}: mp4 없음 — 건너뜀")
            continue
        outdir = os.path.join(a.dst, item)
        if not a.dry_run:
            os.makedirs(outdir, exist_ok=True)
        miss = 0
        for p in tqdm(mp4s, desc=f"  {item:14s}", unit="건", ncols=78):
            parts = p.split(os.sep)
            place, subtype = parts[-3], parts[-2]
            stem = os.path.splitext(os.path.basename(p))[0]
            gt = os.path.join(gtroot, item, stem + ".csv")
            if not os.path.exists(gt):
                miss += 1
                continue
            cap = cv2.VideoCapture(p)
            fps, w, h = cap.get(cv2.CAP_PROP_FPS), int(cap.get(3)), int(cap.get(4))
            nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            lab = list(csv.DictReader(open(gt, encoding="utf-8")))
            col = [c for c in lab[0] if c != "clip_frame"][0]
            npos = sum(int(r[col]) for r in lab)
            rows.append({"item": item, "name": stem, "subtype": subtype, "place": place,
                         "is_pos": int(npos > 0), "event_col": col,
                         "fps": round(fps, 3), "width": w, "height": h,
                         "frames_video": nfr, "frames_label": len(lab), "event_frames": npos,
                         "src": os.path.relpath(p, a.src)})
            if not a.dry_run:
                shutil.copy2(p, os.path.join(outdir, stem + ".mp4"))
                shutil.copy2(gt, os.path.join(outdir, stem + ".csv"))
        if miss:
            print(f"[!] {item}: 정답지 없는 mp4 {miss}건")

    if a.dry_run:
        print(f"[dry-run] 복사 대상 {len(rows)}건")
        return
    shutil.rmtree(gtdir, ignore_errors=True)

    mf = os.path.join(a.dst, "MANIFEST.csv")
    with open(mf, "w", encoding="utf-8-sig", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w_.writeheader(); w_.writerows(rows)

    print(f"\n[완료] {a.dst}  총 {len(rows)}건")
    print(f"  {'항목':16s} {'건수':>5s} {'양성':>5s} {'음성':>5s}  subtype 구성")
    for item in items:
        r = [x for x in rows if x["item"] == item]
        if not r:
            continue
        st = {}
        for x in r:
            st[x["subtype"]] = st.get(x["subtype"], 0) + 1
        pos = sum(x["is_pos"] for x in r)
        print(f"  {item:16s} {len(r):5d} {pos:5d} {len(r) - pos:5d}  {st}")
    print(f"  답안지: {mf}")


if __name__ == "__main__":
    main()
