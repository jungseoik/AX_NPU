"""YOLO11 NPU 검출 + 경량 ByteTrack 트래킹 데모 (영상 → track ID 영상).
실행(기본, HF에서 자동): python demo_track_yolo11_npu.py --model yolo11m --video in.mp4 --classes 0
env: pe_npu_host (qbruntime + scipy + opencv)."""
import argparse, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import cv2
from yolo_npu import YOLONPU, ByteTrack, draw_tracks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11m", help="yolo11n/s/m/l/x")
    ap.add_argument("--scheme", default="single", help="single/multi/global4/global8")
    ap.add_argument("--mxq", default=None, help="로컬 mxq(지정 시 HF 대신 사용)")
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="track_out.mp4")
    ap.add_argument("--device-ids", default=None, help='예: "0" 또는 "0,1" 또는 "auto"')
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--classes", default=None, help="탐지 제한 클래스 id (쉼표, 예 0=person)")
    args = ap.parse_args()

    dids = args.device_ids
    if dids and dids != "auto":
        dids = [int(x) for x in dids.split(",")]
    det = YOLONPU.load(args.model, args.scheme, local_mxq=args.mxq,
                       device_ids=dids, conf_thres=args.conf)
    keep = set(int(c) for c in args.classes.split(",")) if args.classes else None

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W, H = int(cap.get(3)), int(cap.get(4))
    trk = ByteTrack(fps=fps)
    ByteTrack.reset_ids()
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    n, seen_ids, t0 = 0, set(), time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        boxes = det(frame)
        if keep:
            boxes = [b for b in boxes if b[5] in keep]
        tracks = trk.update(boxes)
        for *_, tid, _, _ in tracks:
            seen_ids.add(tid)
        writer.write(draw_tracks(frame, tracks, det.names))
        n += 1
    dt = time.perf_counter() - t0
    cap.release(); writer.release()
    print(f"{n}프레임 처리 {dt:.1f}s ({n/dt:.1f} fps), 누적 track ID {len(seen_ids)}개 → {args.out}")


if __name__ == "__main__":
    main()
