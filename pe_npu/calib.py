"""
PE vision encoder용 calibration 데이터 준비 (CLI: python -m pe_npu.calib).

COCO val2017 또는 ImageNet-1k 이미지를 PE 운영 전처리(preprocess_image)와 동일하게 가공해
(3,336,336) float32 .npy 로 저장하고 npy_files.txt 목록을 만든다. qbcompiler는 이 목록을
calib_data_path로 받아 activation 양자화 범위를 산출한다.

NPU 입력 레이아웃은 HWC라, torch backend 컴파일에는 to_hwc()로 HWC 변환한 calib을 쓴다.

사용:
    # COCO (이미지 폴더가 이미 있을 때) — CHW npy 생성
    python -m pe_npu.calib --dataset coco --src /path/to/val2017 --num 200 --out ./calib_coco
    # 이어서 HWC 변환 (NPU 입력 레이아웃)
    python -m pe_npu.calib --to-hwc --src ./calib_coco --out ./calib_coco_hwc
    # 한 번에: --hwc 플래그로 곧장 HWC npy 생성
    python -m pe_npu.calib --dataset coco --src /path/to/val2017 --num 200 --out ./calib_coco_hwc --hwc
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
from PIL import Image

from .preprocess import IMAGE_SIZE, preprocess_image


def iter_coco(src: str):
    """COCO val2017 이미지 폴더에서 jpg 경로를 순회."""
    if not src or not os.path.isdir(src):
        raise FileNotFoundError(
            f"COCO 이미지 폴더가 필요합니다 (--src). 다운로드:\n"
            f"  wget http://images.cocodataset.org/zips/val2017.zip && unzip val2017.zip"
        )
    paths = sorted(glob.glob(os.path.join(src, "*.jpg")))
    for p in paths:
        try:
            yield Image.open(p)
        except Exception:
            continue


def iter_imagenet():
    """HF에서 ImageNet-1k validation 스트리밍 (gated: hf auth login 필요)."""
    from datasets import load_dataset

    ds = load_dataset(
        "ILSVRC/imagenet-1k",
        data_files={"validation": "data/validation*.parquet"},
        split="validation",
        verification_mode="no_checks",
        streaming=True,
    )
    for sample in ds:
        yield sample["image"]


def _write_list(out_dir, saved_paths):
    list_path = os.path.join(out_dir, "npy_files.txt")
    with open(list_path, "w") as f:
        for p in saved_paths:
            f.write(p + "\n")
    return list_path


def prepare_calib(dataset: str, out: str, src: str = None, num: int = 1000, hwc: bool = False):
    """calibration npy + npy_files.txt 생성.

    dataset: 'coco' | 'imagenet'
    hwc    : True면 (336,336,3) HWC로 저장(NPU 입력 레이아웃), False면 (3,336,336) CHW.
    반환: 저장한 샘플 수.
    """
    os.makedirs(out, exist_ok=True)
    src_iter = iter_coco(src) if dataset == "coco" else iter_imagenet()

    n = 0
    saved_paths = []
    for img in src_iter:
        if n >= num:
            break
        try:
            arr = preprocess_image(img)  # (3,336,336)
        except Exception as e:
            print(f"  skip ({type(e).__name__}: {e})")
            continue
        if hwc:
            arr = np.ascontiguousarray(arr.transpose(1, 2, 0))  # (336,336,3)
        p = os.path.join(out, f"calib_{n:05d}.npy")
        np.save(p, arr)
        saved_paths.append(os.path.abspath(p))
        n += 1
        if n % 100 == 0:
            print(f"  {n}/{num}")

    list_path = _write_list(out, saved_paths)
    layout = "HWC (336,336,3)" if hwc else f"CHW (3,{IMAGE_SIZE},{IMAGE_SIZE})"
    print(f"[done] {n}개 calibration 샘플 저장 -> {out}  (layout: {layout})")
    print(f"       calib 목록 파일: {list_path}")
    if n == 0:
        raise SystemExit("저장된 샘플이 0개입니다. 입력 데이터 경로/접근 권한을 확인하세요.")
    return n


def to_hwc(src: str, out: str):
    """기존 CHW calib npy 디렉토리(src)를 HWC로 변환해 out에 저장 + npy_files.txt 갱신."""
    os.makedirs(out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(src, "calib_*.npy")))
    if not files:
        raise FileNotFoundError(f"CHW calib npy 없음: {src}/calib_*.npy")
    saved_paths = []
    for f in files:
        a = np.ascontiguousarray(np.load(f).transpose(1, 2, 0))  # (336,336,3)
        p = os.path.join(out, os.path.basename(f))
        np.save(p, a)
        saved_paths.append(os.path.abspath(p))
    list_path = _write_list(out, saved_paths)
    print(f"[done] {len(saved_paths)}개 HWC calib 생성 -> {out}")
    print(f"       calib 목록 파일: {list_path}")
    return len(saved_paths)


def main():
    ap = argparse.ArgumentParser(description="PE calibration 데이터 준비")
    ap.add_argument("--dataset", choices=["coco", "imagenet"],
                    help="calib 소스 데이터셋 (CHW/HWC npy 신규 생성)")
    ap.add_argument("--src", default=None,
                    help="dataset 사용 시: COCO val2017 이미지 폴더. --to-hwc 사용 시: CHW calib 디렉토리")
    ap.add_argument("--num", type=int, default=1000, help="추출할 이미지 수")
    ap.add_argument("--out", required=True, help="calib npy 저장 디렉토리")
    ap.add_argument("--hwc", action="store_true",
                    help="dataset에서 곧장 HWC(336,336,3) 레이아웃으로 저장 (NPU 입력)")
    ap.add_argument("--to-hwc", action="store_true",
                    help="--src의 기존 CHW calib을 HWC로 변환만 수행")
    args = ap.parse_args()

    if args.to_hwc:
        to_hwc(args.src, args.out)
    elif args.dataset:
        prepare_calib(args.dataset, args.out, src=args.src, num=args.num, hwc=args.hwc)
    else:
        ap.error("--dataset 또는 --to-hwc 중 하나가 필요합니다.")


if __name__ == "__main__":
    main()
