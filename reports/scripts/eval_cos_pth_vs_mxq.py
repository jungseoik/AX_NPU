#!/usr/bin/env python3
"""Compare local full-NPU MXQ embeddings with unpatched PyTorch PE output."""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def cosine_summary(reference: np.ndarray, candidate: np.ndarray) -> dict:
    """Return per-image cosine statistics for equal-shaped 2-D embeddings."""
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.shape != candidate.shape:
        raise ValueError(
            f"shape mismatch: reference {reference.shape} != candidate {candidate.shape}"
        )
    if reference.ndim != 2:
        raise ValueError(f"expected 2-D embeddings, got shape {reference.shape}")
    if reference.shape[0] == 0:
        raise ValueError("cannot summarize zero embeddings")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(candidate)):
        raise ValueError("embeddings contain non-finite values")
    denominator = np.linalg.norm(reference, axis=1) * np.linalg.norm(candidate, axis=1)
    if not np.all(np.isfinite(denominator)):
        raise ValueError("cosine denominator contains non-finite values")
    if np.any(denominator <= 0):
        raise ValueError("cosine is undefined for zero-norm embeddings")
    cosines = np.sum(reference * candidate, axis=1) / denominator
    if not np.all(np.isfinite(cosines)):
        raise ValueError("cosine calculation produced non-finite values")
    return {
        "count": int(cosines.size),
        "mean_cos": float(np.mean(cosines)),
        "min_cos": float(np.min(cosines)),
        "max_cos": float(np.max(cosines)),
        "cosines": [float(value) for value in cosines],
    }


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def model_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("model must be LABEL=PATH")
    label, raw_path = value.split("=", 1)
    path = Path(raw_path).expanduser().resolve()
    if not label.strip():
        raise argparse.ArgumentTypeError("model label must not be empty")
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"MXQ file does not exist: {path}")
    return label.strip(), path


def collect_images(images_dir: Path, count: int) -> list[Path]:
    if not images_dir.is_dir():
        raise ValueError(f"image directory does not exist: {images_dir}")
    paths = sorted(
        path.resolve()
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if len(paths) < count:
        raise ValueError(f"need {count} distinct images, found {len(paths)} in {images_dir}")
    return paths[:count]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", type=model_spec, required=True,
                        metavar="LABEL=PATH")
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--num", type=positive_int, default=32)
    parser.add_argument("--pth-batch", type=positive_int, default=4)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args(argv)
    labels = [label for label, _ in args.model]
    if len(labels) != len(set(labels)):
        parser.error(f"duplicate model label(s): {labels}")
    return args


def to_embedding_array(output) -> np.ndarray:
    if isinstance(output, (tuple, list)):
        output = output[0]
    if hasattr(output, "detach"):
        output = output.detach().cpu().numpy()
    array = np.asarray(output, dtype=np.float32)
    return array.reshape(array.shape[0], -1)


def main(argv=None) -> int:
    args = parse_args(argv)

    import torch

    from pe_npu.pe_model import load_pe
    from pe_npu.preprocess import preprocess_image

    try:
        image_paths = collect_images(args.images_dir, args.num)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    inputs = np.stack([preprocess_image(path) for path in image_paths]).astype(np.float32)

    reference_model = load_pe(mode="full", patch=False).eval()
    reference_parts = []
    pth_started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(inputs), args.pth_batch):
            tensor = torch.from_numpy(inputs[start : start + args.pth_batch])
            reference_parts.append(to_embedding_array(reference_model(tensor)))
    pth_seconds = time.perf_counter() - pth_started
    reference = np.concatenate(reference_parts, axis=0)
    del reference_model

    # qbruntime is imported only for the actual evaluation, keeping the pure
    # cosine helper importable in compiler/test environments.
    from pe_npu.inference import MXQInferenceFull

    results = []
    for label, path in args.model:
        runtime = MXQInferenceFull(
            full_mxq_path=str(path),
            device_id=args.device_id,
            slots_per_card=1,
        )
        started = time.perf_counter()
        try:
            candidate = to_embedding_array(runtime.infer(inputs))
        finally:
            runtime.dispose()
        elapsed = time.perf_counter() - started
        summary = cosine_summary(reference, candidate)
        results.append(
            {
                "label": label,
                "mxq_path": str(path),
                "mxq_size_bytes": path.stat().st_size,
                "npu_seconds": elapsed,
                **summary,
            }
        )
        print(
            f"{label}: mean_cos={summary['mean_cos']:.8f} "
            f"min_cos={summary['min_cos']:.8f} max_cos={summary['max_cos']:.8f}"
        )

    payload = {
        "reference": "PE-Core-L14-336 load_pe(mode=full, patch=False)",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "images_dir": str(args.images_dir.resolve()),
        "image_count": len(image_paths),
        "image_paths": [str(path) for path in image_paths],
        "pth_batch": args.pth_batch,
        "pth_seconds": pth_seconds,
        "device_id": args.device_id,
        "results": results,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"JSON: {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
