#!/usr/bin/env python3
"""CLIP ViT-L/14-336 vision encoder -> MXQ. W8A16.

78.20% top-1, cosine 0.995, 16.6 imgs/s.
(1000 ImageNet-1k val images, fp32 reference 78.00%; 8 Aries cores.)

    python compile_w8a16.py --calib-src /path/to/jpegs
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from transformers import CLIPVisionModelWithProjection

from qbcompiler import mxq_compile
from qbcompiler.configs import (BitConfig, CalibrationConfig,
                                PreprocessingConfig, Uint8InputConfig)

HERE = Path(__file__).resolve().parent
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", default="openai/clip-vit-large-patch14-336")
    ap.add_argument("--size", type=int, default=336)
    ap.add_argument("--target-device", default="aries-rb")
    ap.add_argument("--calib-src", type=Path, default=HERE / "calib_images",
                    help="directory of representative JPEGs (default: ./calib_images)")
    ap.add_argument("--calib-count", type=int, default=100)
    ap.add_argument("--inference-scheme", default="single")
    ap.add_argument("--device", default="gpu", help="calibration device")
    ap.add_argument("--onnx", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=None)
    a = ap.parse_args()
    tag = f"vit_l_14_{a.size}"
    a.onnx = a.onnx or HERE / f"{tag}_vision.onnx"
    a.output = a.output or HERE / f"{tag}_w8a16.mxq"
    a.calib = HERE / f"calib{a.size}_u8" / "model"
    return a


def export_onnx(a):
    """Vision tower + projection head. eager attention: SDPA will not lower."""
    model = CLIPVisionModelWithProjection.from_pretrained(
        a.model_id, attn_implementation="eager", torch_dtype=torch.float32).eval().cpu()

    class Tower(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, pixel_values):
            return self.m(pixel_values=pixel_values).image_embeds

    with torch.no_grad():
        torch.onnx.export(Tower(model).eval(), torch.randn(1, 3, a.size, a.size),
                          str(a.onnx), input_names=["pixel_values"],
                          output_names=["image_embeds"], opset_version=17,
                          do_constant_folding=True, dynamo=False)
    print(f"[onnx] {a.onnx.name} ({a.onnx.stat().st_size / 1e6:.0f} MB)")


def build_calib(a):
    """Raw uint8 [0,255] crops. NOT normalised floats — uint8Input builds
    calibrate in [0,255]; normalised input here silently costs ~23 points."""
    images = sorted(a.calib_src.glob("*.jpg"))[:a.calib_count]
    if len(images) < a.calib_count:
        raise SystemExit(f"need {a.calib_count} jpegs, found {len(images)} in "
                         f"{a.calib_src} — pass --calib-src")
    a.calib.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(images):
        rgb = cv2.cvtColor(cv2.imread(str(p), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        s = a.size / min(h, w)
        rgb = cv2.resize(rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_CUBIC)
        h, w = rgb.shape[:2]
        t, l = (h - a.size) // 2, (w - a.size) // 2
        np.save(a.calib / f"img_{i:05d}.npy",
                np.ascontiguousarray(rgb[t:t + a.size, l:l + a.size]))
    print(f"[calib] {a.calib_count} uint8 crops -> {a.calib}")


def main():
    a = parse_args()
    if not a.onnx.exists():
        export_onnx(a)
    if not a.calib.is_dir() or not any(a.calib.glob("*.npy")):
        build_calib(a)

    T = BitConfig.Transformer
    t0 = time.time()
    mxq_compile(
        model=str(a.onnx), target_device=a.target_device, backend="onnx",
        calib_data_path=str(a.calib), save_path=str(a.output),
        device=a.device,
        inference_scheme=a.inference_scheme,
        image_channels=3,
        calibration_config=CalibrationConfig(method=1, mode=0, output=0),
        bit_config=BitConfig(transformer=T(
            activation=T.Activation(query=8, key=8, value=8, head=8, output=16, ffn=16),
            weight=T.Weight(query=8, key=8, value=8, output=8, ffn=8, head=8))),
        uint8_input_config=Uint8InputConfig(apply=True, inputs=[], divisionFactor=255.0),
        preprocessing_config=PreprocessingConfig(
            apply=True, autoConvertFormat=True, inputConfigs={},
            pipeline=[{"op": "normalize", "mean": CLIP_MEAN, "std": CLIP_STD,
                       "scaleToUint8": True, "fuseIntoFirstLayer": True}]),
    )
    print(f"[mxq] {a.output.name} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
