"""
PE vision encoder용 calibration 데이터 준비.

COCO val2017 또는 ImageNet-1k 이미지를 받아, perception_encoder의 실제 추론 전처리와
동일하게 가공한 뒤 (3,336,336) float32 .npy 로 저장한다. qbcompiler는 이 .npy 디렉토리를
calib_data_path로 받아 activation 양자화 범위를 산출한다.

전처리 (service.py의 preprocess_image 기준, 운영 추론과 일치시켜야 함):
  - RGB 변환
  - resize 336x336 (bilinear, antialias)
  - [0,1] 스케일 (/255)
  - normalize mean=std=0.5  (CLIP 방식)
  → 결과는 normalize 완료된 NCHW 텐서. (ONNX/MXQ 입력과 동일 형식)

데이터셋:
  - coco     : COCO val2017 (비gated, wget). 사람/실내외/사물 다양 → 감시 장면과 유사.
  - imagenet : ImageNet-1k val (HF gated, 토큰+약관동의 필요). 단일객체 중심.

사용:
    # COCO (이미지 폴더가 이미 있을 때)
    python prepare_calib.py --dataset coco --src /path/to/val2017 --num 1000 --out ./calib_coco
    # ImageNet (HF에서 직접)
    python prepare_calib.py --dataset imagenet --num 1000 --out ./calib_imagenet
"""
import argparse
import glob
import os

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import InterpolationMode

IMAGE_SIZE = 336
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]


def preprocess(img: Image.Image) -> np.ndarray:
    """service.preprocess_image와 동일한 파이프라인 → (3,336,336) float32."""
    img = img.convert("RGB")
    t = TF.pil_to_tensor(img)  # (3,H,W) uint8
    t = TF.resize(t, [IMAGE_SIZE, IMAGE_SIZE],
                  interpolation=InterpolationMode.BILINEAR, antialias=True)
    t = TF.convert_image_dtype(t, torch.float32)  # [0,1]
    t = TF.normalize(t, mean=MEAN, std=STD)
    return t.numpy().astype(np.float32)


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


def main():
    ap = argparse.ArgumentParser(description="PE calibration 데이터 준비")
    ap.add_argument("--dataset", required=True, choices=["coco", "imagenet"])
    ap.add_argument("--src", default=None, help="COCO val2017 이미지 폴더 (dataset=coco)")
    ap.add_argument("--num", type=int, default=1000, help="추출할 이미지 수")
    ap.add_argument("--out", required=True, help="calib npy 저장 디렉토리")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    src_iter = iter_coco(args.src) if args.dataset == "coco" else iter_imagenet()

    n = 0
    saved_paths = []
    for img in src_iter:
        if n >= args.num:
            break
        try:
            arr = preprocess(img)
        except Exception as e:
            print(f"  skip ({type(e).__name__}: {e})")
            continue
        p = os.path.join(args.out, f"calib_{n:05d}.npy")
        np.save(p, arr)
        saved_paths.append(os.path.abspath(p))
        n += 1
        if n % 100 == 0:
            print(f"  {n}/{args.num}")

    # ViT 등은 directory 방식 calibration이 막히므로(다중 컴포넌트), vlm vision과 동일하게
    # npy 경로 목록 파일(npy_files.txt)을 만들어 calib_data_path로 사용한다.
    list_path = os.path.join(args.out, "npy_files.txt")
    with open(list_path, "w") as f:
        for p in saved_paths:
            f.write(p + "\n")

    print(f"[done] {n}개 calibration 샘플 저장 → {args.out}  (shape per file: (3,{IMAGE_SIZE},{IMAGE_SIZE}))")
    print(f"       calib 목록 파일: {list_path}")
    if n == 0:
        raise SystemExit("저장된 샘플이 0개입니다. 입력 데이터 경로/접근 권한을 확인하세요.")


if __name__ == "__main__":
    main()
