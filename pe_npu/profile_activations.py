#!/usr/bin/env python3
"""Profile PE transformer activations on real images.

The attention implementation calls its output projection through a functional
path, so an ``attn.out_proj`` module hook is never fired.  Hooking the parent
``attn`` module records the post-projection tensor and closes that coverage gap.
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import sys
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BLOCK_RE = re.compile(r"^visual\.transformer\.resblocks\.(\d+)\.(.+)$")
DIRECT_SUFFIXES = {"mlp.c_fc", "mlp.gelu", "mlp.c_proj", "ln_1", "ln_2"}


def profile_label(module_name: str) -> str | None:
    """Map a PE module name to the stable label used in profile JSON."""
    match = BLOCK_RE.match(module_name)
    if match is None:
        return None
    block, suffix = match.groups()
    if suffix == "attn":
        return f"L{block}.attn.out_proj"
    if suffix in DIRECT_SUFFIXES:
        return f"L{block}.{suffix}"
    return None


def summarize_values(max_abs: float, sampled_abs: np.ndarray, seen: int) -> dict:
    """Summarize absolute activation values with the experiment's metric."""
    values = np.asarray(sampled_abs, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {
            "max": float(max_abs),
            "p99_9": None,
            "ratio": None,
            "seen": int(seen),
            "samples": 0,
        }
    p99_9 = float(np.quantile(values, 0.999))
    ratio = float(max_abs / p99_9) if p99_9 > 0 else None
    return {
        "max": float(max_abs),
        "p99_9": p99_9,
        "ratio": ratio,
        "seen": int(seen),
        "samples": int(values.size),
    }


def sample_indices(total: int, take: int, seed: int) -> np.ndarray:
    """Return a deterministic, unique random subset without stride aliasing."""
    if total < 1 or take < 1:
        raise ValueError("total and take must be >= 1")
    count = min(total, take)
    if count == total:
        return np.arange(total, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total, size=count, replace=False)).astype(
        np.int64,
        copy=False,
    )


@dataclass
class ActivationAccumulator:
    sample_limit: int
    seed: int = 0
    max_abs: float = 0.0
    seen: int = 0
    _chunks: list[np.ndarray] = field(default_factory=list)
    _stored: int = 0
    _calls: int = 0

    def update(self, output) -> None:
        import torch

        if isinstance(output, (tuple, list)):
            output = next((item for item in output if torch.is_tensor(item)), None)
        if not torch.is_tensor(output):
            return
        values = output.detach().float().abs().reshape(-1)
        count = values.numel()
        if count == 0:
            return
        self.seen += count
        self.max_abs = max(self.max_abs, float(values.max().cpu()))

        # Sample every invocation independently so all images and channel
        # residues contribute. A regular stride aliases badly with widths
        # such as 1024/4096 and biases the p99.9 estimate.
        indices = sample_indices(count, 4096, self.seed + self._calls)
        self._calls += 1
        index_tensor = torch.from_numpy(indices).to(values.device)
        sample = values.index_select(0, index_tensor).cpu().numpy().astype(
            np.float32,
            copy=False,
        )
        self._chunks.append(sample)
        self._stored += sample.size
        if self._stored > self.sample_limit * 2:
            merged = np.concatenate(self._chunks)
            indices = np.linspace(0, merged.size - 1, self.sample_limit, dtype=np.int64)
            self._chunks = [merged[indices]]
            self._stored = self.sample_limit

    def summary(self) -> dict:
        if not self._chunks:
            values = np.empty(0, dtype=np.float32)
        else:
            values = np.concatenate(self._chunks)
            if values.size > self.sample_limit:
                indices = np.linspace(0, values.size - 1, self.sample_limit, dtype=np.int64)
                values = values[indices]
        return summarize_values(self.max_abs, values, self.seen)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


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
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--num", type=positive_int, default=32)
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument("--sample-limit", type=positive_int, default=50_000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--top", type=positive_int, default=20)
    parser.add_argument("--json-out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    import torch

    from pe_npu.pe_model import load_pe
    from pe_npu.preprocess import preprocess_image

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested, but CUDA is not available")
    try:
        image_paths = collect_images(args.images_dir, args.num)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    model = load_pe(mode="full", patch=False).to(args.device).eval()
    accumulators: dict[str, ActivationAccumulator] = {}
    handles = []
    for module_name, module in model.named_modules():
        label = profile_label(module_name)
        if label is None:
            continue
        accumulator = ActivationAccumulator(
            args.sample_limit,
            seed=zlib.crc32(label.encode("utf-8")),
        )
        accumulators[label] = accumulator

        def capture(_module, _inputs, output, target=accumulator):
            target.update(output)

        handles.append(module.register_forward_hook(capture))

    if not handles:
        raise RuntimeError("no target PE transformer modules were found")

    try:
        with torch.no_grad():
            for start in range(0, len(image_paths), args.batch_size):
                batch_paths = image_paths[start : start + args.batch_size]
                batch = np.stack([preprocess_image(path) for path in batch_paths])
                model(torch.from_numpy(batch).to(args.device))
                print(f"profiled {min(start + len(batch_paths), len(image_paths))}/{len(image_paths)}")
    finally:
        for handle in handles:
            handle.remove()

    rows = []
    missing = []
    for label, accumulator in sorted(accumulators.items()):
        row = {"label": label, **accumulator.summary()}
        rows.append(row)
        if row["seen"] == 0:
            missing.append(label)
    ranked = sorted(
        rows,
        key=lambda row: row["ratio"] if row["ratio"] is not None else -1.0,
        reverse=True,
    )
    payload = {
        "model": "PE-Core-L14-336",
        "patch": False,
        "device": args.device,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "images_dir": str(args.images_dir.resolve()),
        "image_count": len(image_paths),
        "image_paths": [str(path) for path in image_paths],
        "hook_count": len(handles),
        "observed_hook_count": len(rows) - len(missing),
        "missing_hooks": missing,
        "sample_limit_per_hook": args.sample_limit,
        "rows": ranked,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("label                         max       p99.9      ratio       seen")
    for row in ranked[: args.top]:
        ratio = "n/a" if row["ratio"] is None else f"{row['ratio']:.2f}x"
        p99_9 = "n/a" if row["p99_9"] is None else f"{row['p99_9']:.5g}"
        print(f"{row['label']:<28} {row['max']:>9.5g} {p99_9:>11} {ratio:>10} {row['seen']:>11}")
    print(
        f"hooks observed: {payload['observed_hook_count']}/{payload['hook_count']}; "
        f"JSON: {args.json_out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
