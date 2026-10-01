#!/usr/bin/env python3
"""TTA_4종_AX 에서 **이벤트 구간 가운데 프레임**을 뽑는다.

- pos 클립: 라벨(csv)의 1 구간 중 **가장 긴 연속 구간의 가운데** 프레임. 정답 = yes
- neg 클립: 이벤트가 없으므로 **클립 가운데** 프레임. 정답 = no
"""
import csv, json, sys
from pathlib import Path
import cv2

ROOT = Path("/home/gpuadmin/AX_NPU/eval/datasets/TTA_4종_AX")
OUT = Path("/tmp/ax_eval/frames")
TC = {"TC1_AX": "falldown", "TC2_AX": "fire", "TC3_AX": "smoke"}


def event_mid(csv_path: Path):
    """가장 긴 1-구간의 가운데 프레임 번호. 1 이 없으면 None."""
    rows = list(csv.reader(csv_path.open(encoding="utf-8-sig")))
    hdr, body = rows[0], rows[1:]
    col = 1 if len(hdr) > 1 else 0
    lab = [(int(r[0]), int(r[col])) for r in body if r and r[0].strip().isdigit()]
    best = cur = None
    for f, v in lab:
        if v == 1:
            cur = [f, f] if cur is None else [cur[0], f]
            if best is None or (cur[1] - cur[0]) > (best[1] - best[0]):
                best = list(cur)
        else:
            cur = None
    return None if best is None else (best[0] + best[1]) // 2, (len(lab) // 2)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    index = []
    for tc, cat in TC.items():
        d = ROOT / tc
        for mp4 in sorted(d.glob(f"{cat}_*.mp4")):
            stem = mp4.stem
            is_pos = "_pos_" in stem
            lab = d / f"{stem}.csv"
            mid = None
            if lab.exists():
                r = event_mid(lab)
                mid = r[0] if r else None
                fallback = r[1] if r else None
            else:
                fallback = None
            cap = cv2.VideoCapture(str(mp4))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            idx = mid if (is_pos and mid is not None) else (total // 2)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            cap.release()
            if not ok:
                print(f"  ❌ 읽기 실패 {stem} @{idx}", flush=True)
                continue
            fp = OUT / f"{cat}__{stem}__f{idx}.jpg"
            cv2.imwrite(str(fp), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            index.append({"category": cat, "clip": stem, "is_pos": is_pos,
                          "frame": idx, "total_frames": total,
                          "event_mid": mid, "path": str(fp),
                          "truth": "yes" if is_pos else "no"})
        print(f"  {tc}({cat}) 완료", flush=True)
    Path("/tmp/ax_eval/index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1))
    n = len(index)
    pos = sum(1 for r in index if r["is_pos"])
    print(f"총 {n}장 (pos {pos} / neg {n - pos})")
    for cat in set(r["category"] for r in index):
        rs = [r for r in index if r["category"] == cat]
        print(f"  {cat:9s} {len(rs):3d}장  pos {sum(1 for r in rs if r['is_pos'])}")
    print("=== EXTRACT_DONE ===")


if __name__ == "__main__":
    main()
