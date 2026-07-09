"""
미리 컴파일된 YOLO NPU MXQ를 HuggingFace Hub에 업로드 (PE 업로더와 대칭).

PE와 같은 repo(`PIA-SPACE-LAB/MXQ_NPU`) 안 `yolo/` 하위에 모델·코어모드별로 올린다:
  yolo/<model>/<scheme>/<model>.mxq + CALIBRATION.md   # scheme=single/multi/global4/global8
  yolo/<model>/<model>.onnx                            # fp32 원본(재검증/재컴파일용)

사전:
  - huggingface-cli login  (또는 HF_TOKEN)
  - scratch_yolo/final/ 에 <model>_<scheme>.mxq (12개) + <model>.onnx (3개) 준비
      python -m yolo_npu.compile --model yolo11m --schemes single,multi,global4,global8 \
        --calib <coco/val2017> --calib-num 200 --out <src-dir>

사용:
  python setup/upload_yolo_to_hf.py --src-dir scratch_yolo/final --calib-md scratch_yolo/final/CALIBRATION.md
  python setup/upload_yolo_to_hf.py --models yolo11m --schemes single global8   # 일부만
"""
import argparse
import os

DEFAULT_REPO = "PIA-SPACE-LAB/MXQ_NPU"
PREFIX = "yolo"
ALL_MODELS = ["yolo11n", "yolo11s", "yolo11m", "yolo11l"]
ALL_SCHEMES = ["single", "multi", "global4", "global8"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--src-dir", default="scratch_yolo/final",
                    help="<model>_<scheme>.mxq / <model>.onnx 들이 있는 디렉토리")
    ap.add_argument("--calib-md", default="scratch_yolo/final/CALIBRATION.md")
    ap.add_argument("--models", nargs="+", default=ALL_MODELS)
    ap.add_argument("--schemes", nargs="+", default=ALL_SCHEMES, choices=ALL_SCHEMES)
    ap.add_argument("--no-onnx", action="store_true", help="ONNX 업로드 생략")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = ap.parse_args()

    from huggingface_hub import HfApi
    api = HfApi(token=args.token)

    for model in args.models:
        for scheme in args.schemes:
            mxq = os.path.join(args.src_dir, f"{model}_{scheme}.mxq")
            if not os.path.exists(mxq):
                raise SystemExit(f"파일 없음: {mxq}\n  생성: python -m yolo_npu.compile "
                                 f"--model {model} --schemes {scheme} --calib <val2017> --calib-num 200 --out {args.src_dir}")
            dst = f"{PREFIX}/{model}/{scheme}/{model}.mxq"
            print(f"업로드 {mxq} ({os.path.getsize(mxq)/1e6:.0f}MB) -> {args.repo}/{dst}")
            api.upload_file(path_or_fileobj=mxq, path_in_repo=dst, repo_id=args.repo, repo_type="model")
            if os.path.exists(args.calib_md):
                api.upload_file(path_or_fileobj=args.calib_md,
                                path_in_repo=f"{PREFIX}/{model}/{scheme}/CALIBRATION.md",
                                repo_id=args.repo, repo_type="model")
        if not args.no_onnx:
            onnx = os.path.join(args.src_dir, f"{model}.onnx")
            if os.path.exists(onnx):
                print(f"업로드 {onnx} -> {args.repo}/{PREFIX}/{model}/{model}.onnx")
                api.upload_file(path_or_fileobj=onnx, path_in_repo=f"{PREFIX}/{model}/{model}.onnx",
                                repo_id=args.repo, repo_type="model")

    print(f"[OK] {args.repo}/{PREFIX}/ 업로드 완료. "
          f"사용: YOLONPU.from_hf(model='yolo11m', scheme='single')")


if __name__ == "__main__":
    main()
