"""YOLO NPU 추론 데모 (스크립트). 이미지 → bbox → 저장.
실행: python demo_yolo.py --mxq yolo11m_single.mxq --image bus.jpg --out out.jpg
모델 바꾸려면 --mxq 만 변경 (11n/11m/11l …). env: pe_npu_host (qbruntime)."""
import argparse, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from yolo_npu import YOLONPU


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mxq", required=True, help="YOLO MXQ 경로 (모델 바꾸려면 이것만 변경)")
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="yolo_out.jpg")
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    args = ap.parse_args()

    det = YOLONPU(args.mxq, device_id=args.device_id, conf_thres=args.conf, iou_thres=args.iou)
    t0 = time.perf_counter()
    boxes = det(args.image)
    dt = (time.perf_counter() - t0) * 1000

    print(f"[{os.path.basename(args.mxq)}] {args.image} → 검출 {len(boxes)}개 ({dt:.1f} ms)")
    for x1, y1, x2, y2, cf, c in boxes:
        print(f"  {det.names[c]:12s} {cf:.2f}  ({int(x1)},{int(y1)})-({int(x2)},{int(y2)})")
    det.draw(args.image, boxes, args.out)
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
