"""
PE vision encoder 입력 전처리 (운영 service.preprocess_image와 동일).

RGB -> resize 336x336(bilinear, antialias) -> [0,1] 스케일(/255) -> normalize mean/std=0.5.
결과는 (3,336,336) float32 NCHW (ONNX/MXQ 입력과 동일 형식).
"""
from __future__ import annotations

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import InterpolationMode

IMAGE_SIZE = 336
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]


def preprocess_image(path_or_pil) -> np.ndarray:
    """이미지 경로 또는 PIL.Image -> (3,336,336) float32 (CLIP 전처리)."""
    img = path_or_pil
    if not isinstance(img, Image.Image):
        img = Image.open(img)
    img = img.convert("RGB")
    t = TF.pil_to_tensor(img)  # (3,H,W) uint8
    t = TF.resize(t, [IMAGE_SIZE, IMAGE_SIZE],
                  interpolation=InterpolationMode.BILINEAR, antialias=True)
    t = TF.convert_image_dtype(t, torch.float32)  # [0,1]
    t = TF.normalize(t, mean=MEAN, std=STD)
    return t.numpy().astype(np.float32)
