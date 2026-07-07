"""
YOLO(11) → MXQ 컴파일 (Mobilint ARIES). ultralytics 모델 → ONNX → 4 코어모드 MXQ.
CLI: python -m yolo_npu.compile --model yolo11m --schemes single,global4 --calib <dir>

요구: qbcompiler + ultralytics + onnx (컴파일 전용 env, torch==2.7.1 매칭 — mmc ABI).
  conda create -n yolo_c python=3.10 && conda activate yolo_c
  pip install "torch==2.7.1" "torchvision==0.22.1" "numpy<2" ultralytics onnx onnxslim "onnxruntime>=1.19.2"
  pip install --no-deps download/qbcompiler-1.1.2+aries2-py3-none-any.whl

모델 변경은 --model만 바꾸면 됨: yolo11n / yolo11m / yolo11l / yolo11x / yolo11s.
calib(--calib) 미지정 시 random calib(정확도 무의미, latency 측정용). 실사용은 실이미지 calib 권장.
입력형식: letterbox 640 + RGB + /255 → (640,640,3) float32 (yolo_npu.detect.preprocess와 동일).
"""
from __future__ import annotations

import argparse
import glob
import os
import time

IMG_SIZE = 640
SCHEMES = ["single", "multi", "global4", "global8"]


def export_onnx(model_name: str, out_dir: str, imgsz: int = IMG_SIZE) -> str:
    """ultralytics에서 <model_name>.pt를 받아 ONNX로 export (없으면 자동 다운로드)."""
    from ultralytics import YOLO
    onnx = os.path.join(out_dir, f"{model_name}.onnx")
    if os.path.exists(onnx):
        return onnx
    m = YOLO(f"{model_name}.pt")
    path = m.export(format="onnx", imgsz=imgsz, opset=13, simplify=True, dynamic=False)
    os.replace(path, onnx)
    return onnx


def build_calib(calib_dir: str, out_dir: str, num: int = 64, imgsz: int = IMG_SIZE) -> str:
    """이미지 폴더 → YOLO 입력형식 npy(letterbox+RGB+/255) + npy_files.txt. 경로 반환."""
    import cv2
    import numpy as np
    from .detect import letterbox
    imgs = sorted(glob.glob(os.path.join(calib_dir, "**", "*.jpg"), recursive=True))[:num]
    if not imgs:
        raise FileNotFoundError(f"calib 이미지 없음: {calib_dir}/**/*.jpg")
    npy_dir = os.path.join(out_dir, "calib_npy")
    os.makedirs(npy_dir, exist_ok=True)
    paths = []
    for i, p in enumerate(imgs):
        im = cv2.imread(p)
        if im is None:
            continue
        lb, _, _ = letterbox(im, imgsz)
        rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
        x = np.ascontiguousarray(rgb.astype(np.float32) / 255.0)
        fp = os.path.abspath(os.path.join(npy_dir, f"c{i:04d}.npy"))
        np.save(fp, x); paths.append(fp)
    lst = os.path.join(npy_dir, "npy_files.txt")
    with open(lst, "w") as f:
        f.write("\n".join(paths) + "\n")
    print(f"[calib] {len(paths)}장 -> {lst}")
    return lst


def compile_model(model_name: str, out_dir: str, schemes=SCHEMES, calib_list: str = None,
                  device: str = "cpu", imgsz: int = IMG_SIZE):
    """<model_name>을 지정 코어모드들로 MXQ 컴파일. 반환: {scheme: mxq_path}."""
    os.makedirs(out_dir, exist_ok=True)
    onnx = export_onnx(model_name, out_dir, imgsz)
    print(f"[onnx] {onnx}")

    from qbcompiler import mxq_compile
    common = dict(model=onnx, backend="onnx", target_device="aries2",
                  yolo_decode_include=True, device=device)
    if calib_list:
        from qbcompiler import CalibrationConfig
        cc = CalibrationConfig(method=1, output=1, mode=1,
                               max_percentile=CalibrationConfig.MaxPercentile(percentile=0.9999, topk_ratio=0.01))
        calib_kw = dict(calib_data_path=calib_list, calibration_config=cc)
    else:
        calib_kw = dict(use_random_calib=True)

    out = {}
    for s in schemes:
        save = os.path.join(out_dir, f"{model_name}_{s}.mxq")
        t0 = time.perf_counter()
        mxq_compile(**common, inference_scheme=s, save_path=save, **calib_kw)
        sz = os.path.getsize(save) / 1e6
        print(f"[compile:{s}] OK {time.perf_counter()-t0:.0f}s -> {save} ({sz:.0f}MB)")
        out[s] = save
    return out


def main():
    ap = argparse.ArgumentParser(description="YOLO → MXQ (4 코어모드) 컴파일")
    ap.add_argument("--model", default="yolo11m", help="ultralytics 모델명 (yolo11n/s/m/l/x)")
    ap.add_argument("--out", default="./yolo_out", help="출력 디렉토리")
    ap.add_argument("--schemes", default="single,multi,global4,global8",
                    help="컴파일할 코어모드 (쉼표구분)")
    ap.add_argument("--calib", default=None,
                    help="calib 이미지 폴더 (미지정 시 random calib = latency용)")
    ap.add_argument("--calib-num", type=int, default=64)
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    args = ap.parse_args()

    calib_list = build_calib(args.calib, args.out, args.calib_num) if args.calib else None
    schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]
    compile_model(args.model, args.out, schemes=schemes, calib_list=calib_list, device=args.device)


if __name__ == "__main__":
    main()
