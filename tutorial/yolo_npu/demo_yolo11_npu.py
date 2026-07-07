"""YOLO NPU 추론 데모 (스크립트). 이미지 → bbox → 저장.
실행(기본, HF에서 자동): python demo_yolo11_npu.py --model yolo11m --scheme single --image bus.jpg
       (로컬 mxq 쓰기):  python demo_yolo11_npu.py --mxq yolo_out/yolo11m_single.mxq --image bus.jpg
모델은 --model/--scheme 만 바꾸면 됨. env: pe_npu_host (qbruntime)."""
import argparse, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from yolo_npu import YOLONPU


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11m", help="yolo11n/s/m/l/x")
    ap.add_argument("--scheme", default="single", help="single/multi/global4/global8")
    ap.add_argument("--mxq", default=None, help="로컬 mxq 경로(지정 시 HF 대신 이걸 사용)")
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="yolo_out.jpg")
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    args = ap.parse_args()

    # 기본: HF 먼저 → 없으면 컴파일 안내. --mxq 주면 로컬 우선.
    det = YOLONPU.load(args.model, args.scheme, local_mxq=args.mxq,
                       device_id=args.device_id, conf_thres=args.conf, iou_thres=args.iou)
    tag = os.path.basename(args.mxq) if args.mxq else f"{args.model}/{args.scheme}"
    t0 = time.perf_counter()
    boxes = det(args.image)
    dt = (time.perf_counter() - t0) * 1000

    print(f"[{tag}] {args.image} → 검출 {len(boxes)}개 ({dt:.1f} ms)")
    for x1, y1, x2, y2, cf, c in boxes:
        print(f"  {det.names[c]:12s} {cf:.2f}  ({int(x1)},{int(y1)})-({int(x2)},{int(y2)})")
    det.draw(args.image, boxes, args.out)
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
