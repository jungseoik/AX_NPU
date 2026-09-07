#!/usr/bin/env python3
"""1카드, 1모델+멀티스레드 동기 infer 처리량과 출력 무결성을 측정한다.

새 양자화 산출물은 명시적으로 전달한다:

    python reports/scripts/bench_modes_threaded.py \
      --model single=out/pe_w4a16_single.mxq \
      --model global4=out/pe_w4a16_global4.mxq \
      --images-dir download/coco/val2017 --device 1 --batch 32 --threads 8 \
      --json-out out/npu-vit-quantization/w4a16/bench.json

인자 없는 기존 호출과 DEVICE BATCH THREADS positional 호출은 HF 모델/기존 영상
기준선을 위해 유지한다. 명시적 --model 사용 시에는 반드시 실제 이미지 폴더를 함께 준다.
"""
from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

LEGACY_MODES = ("single", "global4", "global8", "multi")
LEGACY_SCRATCHPAD = Path(
    "/tmp/claude-1000/-home-gpuadmin/057c9eaa-b86d-4a41-ab6f-9dc6babfd1fe/scratchpad"
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def collect_image_paths(images_dir: Path, batch: int) -> list[Path]:
    if not images_dir.is_dir():
        raise ValueError(f"image directory does not exist: {images_dir}")
    paths = sorted(
        {
            path.resolve()
            for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        }
    )
    if len(paths) < batch:
        raise ValueError(
            f"need at least {batch} distinct image(s), found {len(paths)} in {images_dir}"
        )
    return paths[:batch]


def parse_model_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"model must be LABEL=PATH, got: {spec}")
    label, raw_path = spec.split("=", 1)
    label = label.strip()
    path = Path(raw_path).expanduser().resolve()
    if not label:
        raise ValueError(f"model label is empty: {spec}")
    if not path.is_file():
        raise ValueError(f"MXQ file does not exist: {path}")
    return label, path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("legacy_device", nargs="?", type=int, help=argparse.SUPPRESS)
    parser.add_argument("legacy_batch", nargs="?", type=positive_int, help=argparse.SUPPRESS)
    parser.add_argument("legacy_threads", nargs="?", type=positive_int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="측정할 로컬 MXQ. 여러 번 지정 가능",
    )
    parser.add_argument("--images-dir", type=Path, help="서로 다른 실제 이미지가 든 폴더")
    parser.add_argument("--device", type=int, help="Aries device id")
    parser.add_argument("--batch", type=positive_int, help="측정 이미지 수")
    parser.add_argument("--threads", type=positive_int, help="동기 infer worker 수")
    parser.add_argument("--json-out", type=Path, help="원자료 JSON 출력 경로")
    args = parser.parse_args(argv)

    args.device = args.device if args.device is not None else (
        args.legacy_device if args.legacy_device is not None else 1
    )
    args.batch = args.batch or args.legacy_batch or 32
    args.threads = args.threads or args.legacy_threads or 8

    if args.model:
        if args.images_dir is None:
            parser.error("--images-dir is required with explicit --model")
        try:
            args.models = [parse_model_spec(spec) for spec in args.model]
            labels = [label for label, _ in args.models]
            if len(labels) != len(set(labels)):
                raise ValueError(f"duplicate model label(s): {labels}")
            args.image_paths = collect_image_paths(args.images_dir, args.batch)
        except ValueError as exc:
            parser.error(str(exc))
    else:
        args.models = None
        args.image_paths = None

    if args.json_out is None:
        args.json_out = (
            ROOT / "out" / "npu-vit-quantization" / "bench_modes_threaded.json"
            if args.models
            else LEGACY_SCRATCHPAD / "bench_modes_threaded.json"
        )
    return args


def load_images(paths: list[Path]) -> list[np.ndarray]:
    from PIL import Image
    from pe_npu.preprocess import preprocess_image

    images = []
    for path in paths:
        chw = preprocess_image(Image.open(path))
        images.append(np.ascontiguousarray(chw.transpose(1, 2, 0)))
    return images


def load_legacy_video_frames(batch: int) -> list[np.ndarray]:
    import cv2
    from PIL import Image
    from pe_npu.preprocess import preprocess_image

    videos = (
        "event_video.mp4",
        "pe_binary_elvfalldown_video_1.mp4",
        "pe_binary_esfalldown_video.mp4",
    )
    images = []
    for video in videos:
        capture = cv2.VideoCapture(str(LEGACY_SCRATCHPAD / video))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        for frame_number in np.linspace(5, max(6, total - 5), batch // 3 + 2).astype(int):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
            ok, frame = capture.read()
            if ok:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                chw = preprocess_image(Image.fromarray(rgb))
                images.append(np.ascontiguousarray(chw.transpose(1, 2, 0)))
        capture.release()
    if len(images) < batch:
        raise RuntimeError(
            f"legacy videos yielded {len(images)} images; need {batch} in {LEGACY_SCRATCHPAD}"
        )
    return images[:batch]


def resolve_legacy_models() -> list[tuple[str, Path]]:
    from huggingface_hub import hf_hub_download

    return [
        (
            mode,
            Path(hf_hub_download("PIA-SPACE-LAB/MXQ_NPU", f"{mode}/pe_full.mxq")),
        )
        for mode in LEGACY_MODES
    ]


def flatten_output(output) -> np.ndarray:
    if isinstance(output, (list, tuple)):
        output = output[0]
    return np.asarray(output).reshape(-1)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / (denominator + 1e-12))


def validate_cosines(label: str, cosines: list[float]) -> float:
    """Fail closed when threaded outputs are non-finite or differ from serial."""
    values = np.asarray(cosines, dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise RuntimeError(f"output integrity failed for {label}: non-finite cosine")
    min_cos = float(values.min())
    if min_cos < 0.9999:
        raise RuntimeError(
            f"output integrity failed for {label}: min_cos={min_cos:.6f} < 0.9999"
        )
    return min_cos


def benchmark_model(label: str, mxq_path: Path, images: list[np.ndarray],
                    device: int, threads: int) -> dict:
    import qbruntime

    model = qbruntime.Model(str(mxq_path))
    model.launch(qbruntime.Accelerator(device))
    try:
        for _ in range(2):
            model.infer(images[0])

        serial_started = time.perf_counter()
        reference = [flatten_output(model.infer(image)) for image in images]
        serial_seconds = time.perf_counter() - serial_started

        work_queue = queue.Queue()
        for index, image in enumerate(images):
            work_queue.put((index, image))
        results = {}
        errors = []
        lock = threading.Lock()

        def worker():
            while True:
                try:
                    index, image = work_queue.get_nowait()
                except queue.Empty:
                    return
                try:
                    value = flatten_output(model.infer(image))
                    with lock:
                        results[index] = value
                except BaseException as exc:
                    with lock:
                        errors.append(exc)
                    return

        workers = [threading.Thread(target=worker) for _ in range(threads)]
        started = time.perf_counter()
        for worker_thread in workers:
            worker_thread.start()
        for worker_thread in workers:
            worker_thread.join()
        wall_seconds = time.perf_counter() - started

        if errors:
            raise RuntimeError(f"threaded infer failed: {errors[0]}") from errors[0]
        if len(results) != len(images):
            raise RuntimeError(
                f"threaded infer returned {len(results)}/{len(images)} outputs"
            )
        cosines = [cosine(results[index], reference[index]) for index in range(len(images))]
        min_cos = validate_cosines(label, cosines)

        return {
            "label": label,
            "mxq_path": str(mxq_path),
            "mxq_size_bytes": mxq_path.stat().st_size,
            "wall_ms": round(wall_seconds * 1000, 3),
            "img_s": round(len(images) / wall_seconds, 4),
            "serial_ms_per_image": round(serial_seconds * 1000 / len(images), 3),
            "min_cos": round(min_cos, 8),
            "mean_cos": round(float(np.mean(cosines)), 8),
        }
    finally:
        if hasattr(model, "dispose"):
            model.dispose()


def main(argv=None) -> int:
    args = parse_args(argv)
    images = (
        load_images(args.image_paths)
        if args.image_paths is not None
        else load_legacy_video_frames(args.batch)
    )
    models = args.models or resolve_legacy_models()

    print(
        f"[setup] dev{args.device}, batch={args.batch}, "
        f"1 model + {args.threads} sync threads"
    )
    rows = {}
    for label, mxq_path in models:
        row = benchmark_model(label, mxq_path, images, args.device, args.threads)
        rows[label] = row
        print(
            f"{label:>12} | {row['img_s']:>8.2f} img/s | "
            f"{row['serial_ms_per_image']:>8.2f} ms | min_cos={row['min_cos']:.6f}"
        )

    payload = {
        "device": args.device,
        "batch": args.batch,
        "threads": args.threads,
        "images": [str(path) for path in (args.image_paths or [])],
        "models": rows,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[json] {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
