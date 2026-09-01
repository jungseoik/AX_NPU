#!/usr/bin/env python3
"""CLIP ViT-L/14-336 vision encoder -> MXQ. W4A8 + OPTQ, 5 tensors kept at A16.

75.90% top-1, cosine 0.905, 28.4 imgs/s.
(1000 ImageNet-1k val images, fp32 reference 78.00%; 8 Aries cores.)

Without those 5 tensors held, all-8-bit activations score 21.9%. A16_TENSORS are
post-fusion mblt names, specific to this checkpoint; use select_a16.py to derive
them for another model.

    python compile_w4a8_l5a16.py --calib-src /path/to/jpegs
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
from qbcompiler.configs import (BitConfig, CalibrationConfig, OptqConfig,
                                PreprocessingConfig, SearchWeightScaleConfig,
                                Uint8InputConfig)

HERE = Path(__file__).resolve().parent
# (torch module, post-fusion mblt name). Run --profile to see the statistics
# these were chosen from; select_a16.py derives them for another checkpoint.
A16_TENSORS = [
    ("vision_model.encoder.layers.12.mlp.fc2", "add_76/reshape/quickgelu/conv2d"),
    ("vision_model.encoder.layers.11.mlp.fc2", "add_70/reshape/quickgelu/conv2d"),
    ("vision_model.encoder.layers.12.mlp.activation_fn", "add_76/reshape/quickgelu"),
    ("vision_model.encoder.layers.11.mlp.activation_fn", "add_70/reshape/quickgelu"),
    ("vision_model.encoder.layers.9.mlp.fc2", "add_58/reshape/quickgelu/conv2d"),
]

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
    ap.add_argument("--profile", action="store_true",
                    help="print the activation statistics A16_TENSORS was chosen from")
    ap.add_argument("--inference-scheme", default="single")
    ap.add_argument("--device", default="gpu", help="calibration device")
    ap.add_argument("--onnx", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=None)
    a = ap.parse_args()
    tag = f"vit_l_14_{a.size}"
    a.onnx = a.onnx or HERE / f"{tag}_vision.onnx"
    a.output = a.output or HERE / f"{tag}_w4a8_l5a16.mxq"
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


def crop(path, size):
    """Shortest edge to `size` (bicubic) + centre crop, HWC uint8."""
    rgb = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    s = size / min(h, w)
    rgb = cv2.resize(rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_CUBIC)
    h, w = rgb.shape[:2]
    t, l = (h - size) // 2, (w - size) // 2
    return np.ascontiguousarray(rgb[t:t + size, l:l + size])


def calib_images(a, count):
    images = sorted(a.calib_src.glob("*.jpg"))[:count]
    if len(images) < count:
        raise SystemExit(f"need {count} jpegs, found {len(images)} in "
                         f"{a.calib_src} — pass --calib-src")
    return images


def build_calib(a):
    """Raw uint8 [0,255] crops. NOT normalised floats — uint8Input builds
    calibrate in [0,255]; normalised input here silently costs ~23 points."""
    a.calib.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(calib_images(a, a.calib_count)):
        np.save(a.calib / f"img_{i:05d}.npy", crop(p, a.size))
    print(f"[calib] {a.calib_count} uint8 crops -> {a.calib}")


def profile_activations(a, n_images=16, top=12):
    """Show why A16_TENSORS is what it is.

    `ratio` = max / p99.9 over the tensor. An 8-bit symmetric scale is set by
    the max, so at 122x the step is ~1.0 while 99.9% of the values sit below
    1.05 — everything real collapses onto 0/+-1. Held tensors are marked [16].
    """
    tracked = ("mlp.fc1", "mlp.fc2", "mlp.activation_fn",
               "self_attn.out_proj", "layer_norm1", "layer_norm2")
    held = {t for t, _ in A16_TENSORS}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPVisionModelWithProjection.from_pretrained(
        a.model_id, attn_implementation="eager", torch_dtype=torch.float32).eval().to(dev)

    stats = {}

    def hook(name):
        def f(_m, _i, out):
            t = out[0] if isinstance(out, tuple) else out
            if not torch.is_tensor(t) or t.dim() < 2:
                return
            v = t.detach().abs().float().flatten()
            # torch.quantile caps well below 577*4096, so sample first
            idx = torch.randperm(v.numel(), device=v.device)[:200_000]
            s = stats.setdefault(name, [0.0, 0.0])
            s[0] = max(s[0], v.max().item())
            s[1] = max(s[1], torch.quantile(v[idx], 0.999).item())
        return f

    for name, mod in model.named_modules():
        if name.startswith("vision_model.encoder.layers.") and name.endswith(tracked):
            mod.register_forward_hook(hook(name))

    mean = np.array(CLIP_MEAN, np.float32)
    std = np.array(CLIP_STD, np.float32)
    imgs = np.stack([(crop(p, a.size).astype(np.float32) / 255.0 - mean) / std
                     for p in calib_images(a, n_images)])
    x = torch.from_numpy(imgs).permute(0, 3, 1, 2).to(dev)
    with torch.no_grad():
        for i in range(0, len(x), 4):
            model(pixel_values=x[i:i + 4])

    rows = sorted(((n, v[0], v[1], v[0] / max(v[1], 1e-9)) for n, v in stats.items()),
                  key=lambda r: -r[3])
    print(f"\n[profile] {len(rows)} encoder tensors over {n_images} images")
    print(f"{'tensor':<34}{'max':>9}{'p99.9':>9}{'ratio':>9}   16-bit")
    for name, mx, p, r in rows[:top]:
        print(f"{name.replace('vision_model.encoder.layers.', 'L'):<34}"
              f"{mx:9.1f}{p:9.2f}{r:8.1f}x   {'[16]' if name in held else ''}")
    ratios = [r for _, _, _, r in rows]
    print(f"median {np.median(ratios):.1f}x | above 20x: "
          f"{sum(1 for r in ratios if r > 20)} | held: {len(held)}")

    missing = held - set(stats)
    if missing:
        print(f"[profile] WARNING: not found in this model: {sorted(missing)}")
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()


def main():
    a = parse_args()
    if not a.onnx.exists():
        export_onnx(a)
    if not a.calib.is_dir() or not any(a.calib.glob("*.npy")):
        build_calib(a)
    if a.profile:
        profile_activations(a)

    T = BitConfig.Transformer
    t0 = time.time()
    mxq_compile(
        model=str(a.onnx), target_device=a.target_device, backend="onnx",
        calib_data_path=str(a.calib), save_path=str(a.output),
        device=a.device,
        inference_scheme=a.inference_scheme,
        image_channels=3,
        calibration_config=CalibrationConfig(method=1, mode=0, output=0),
        bit_config=BitConfig(
            transformer=T(
                activation=T.Activation(query=8, key=8, value=8, head=8,
                                        output=8, ffn=8),
                weight=T.Weight(query=4, key=4, value=8, output=4, ffn=4, head=4)),
            layerOverrides=BitConfig.LayerOverrides(
                activation16Bits=[n for _, n in A16_TENSORS])),
        optq_config=OptqConfig(apply=True, attributes=OptqConfig.Attributes(
            actOrder=True, blockSize=128, percDamp=0.01)),
        search_weight_scale_config=SearchWeightScaleConfig(
            apply=True, transformer=SearchWeightScaleConfig.Transformer(
                query=True, key=True, value=True, out=True, ffn=True)),
        uint8_input_config=Uint8InputConfig(apply=True, inputs=[], divisionFactor=255.0),
        preprocessing_config=PreprocessingConfig(
            apply=True, autoConvertFormat=True, inputConfigs={},
            pipeline=[{"op": "normalize", "mean": CLIP_MEAN, "std": CLIP_STD,
                       "scaleToUint8": True, "fuseIntoFirstLayer": True}]),
    )
    print(f"[mxq] {a.output.name} in {time.time() - t0:.0f}s "
          f"({len(A16_TENSORS)} tensors held at 16-bit)")


if __name__ == "__main__":
    main()
