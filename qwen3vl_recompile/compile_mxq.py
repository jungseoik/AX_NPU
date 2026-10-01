#!/usr/bin/env python3
"""③ 우리 .mblt + 우리 캘리브 → .mxq  (벤더 레시피 그대로)

벤더 `scripts/compile.py` 와 **같은 설정**을 쓴다. 바뀌는 건 입력 경로뿐이다:
  - model        : 우리가 export 한 .mblt
  - calib_data_path : 우리가 생성한 캘리브 json
  - rotation.pth : 벤더 것 재사용 (하다마드 행렬이라 가중치 무관 — 직교오차 5.96e-08,
                   절댓값 고유 1개로 확인)
  - configs/*.json : 벤더 것 재사용

★ batch32 는 `split_legacy_deepstack.py` 가 필요 없다. 그건 벤더의 **레거시 포맷**
  (deepstack 이 1입력으로 뭉쳐 있던 옛 export) 전용 어댑터다. 신규 export 는 처음부터
  deepstack 이 분리돼 있다. 실제로 신규 mblt 에 돌리면 스크립트가 이렇게 거부한다:
     "This adapter is for the checked legacy format; re-export newer MBLT normally"
"""
from __future__ import annotations
import argparse, hashlib, importlib.metadata, json, os, sys, time
from pathlib import Path

VERSION = "1.3.4+pia.capacity3"
MMC_SHA256 = {"decoder": "627c35ec0971d764c71fc9a7bfba931dc70c19def58c01ef544100f472bc5760",
              "vision":  "0e84af14aa72222a92b873f373d288f14c9b00e46031f7d4bbdacca5fe7cbad1"}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--part", required=True, choices=["decoder-b1", "decoder-b32", "vision", "vision8192"])
    p.add_argument("--mblt-dir", type=Path, required=True, help="export_mblt.py 산출물")
    p.add_argument("--calib-dir", type=Path, required=True, help="make_calib.py 산출물")
    p.add_argument("--vendor", type=Path, required=True, help="벤더 패키지(configs/, inputs/rotation.pth)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--gpu", default="0")
    p.add_argument("--force", action="store_true",
                   help="기존 출력 디렉토리를 지우고 다시 만든다(실패 재시도용)")
    a = p.parse_args()

    if sys.version_info[:2] != (3, 10):
        p.error("CPython 3.10 이어야 한다 (벤더 네이티브 휠)")
    if importlib.metadata.version("qbcompiler") != VERSION:
        p.error(f"qbcompiler {VERSION} 휠이어야 한다")
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    import torch
    torch.set_num_threads(8)
    if not torch.__version__.startswith("2.7.1") or torch.version.cuda != "12.8":
        p.error("torch 2.7.1 / CUDA 12.8 이어야 한다")
    if not torch.cuda.is_available():
        p.error("CUDA GPU 가 필요하다 (벤더 레시피 하드 요구)")

    decoder = a.part.startswith("decoder")
    stage = "decoder" if decoder else "vision"
    os.environ["QBCOMPILER_PIA_NATIVE"] = stage          # ★ import 전에 설정, 프로세스당 1회
    import qbcompiler, qbcompiler.mmc as mmc
    h = hashlib.sha256()
    with Path(mmc.__file__).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""): h.update(b)
    if h.hexdigest() != MMC_SHA256[stage]:
        p.error("mmc 바이너리가 벤더 기록과 다르다 — SDK 빌드를 섞지 말 것")

    batch = 32 if a.part == "decoder-b32" else 1
    scheme = "global8" if batch == 32 else "all"
    mblt = a.mblt_dir / ("decoder.mblt" if decoder else "vision.mblt")
    calib = a.calib_dir / ("decoder.json" if decoder else "vision.json")
    rot = a.vendor / "inputs/rotation.pth"
    for f in (mblt, calib, rot):
        if not f.is_file(): p.error(f"없음: {f}")

    out = a.out / a.part
    if out.exists():
        if not a.force:
            p.error(f"{out} 가 이미 있다 — --force 를 주거나 새 디렉토리를 쓸 것")
        import shutil; shutil.rmtree(out)
    out.mkdir(parents=True)

    cfg = json.loads((a.vendor / "configs" / ("decoder.json" if decoder else "vision.json")).read_text())
    cfg = {k: v for k, v in cfg.items() if not k.startswith("$")}
    cfg.pop("testMode", None)
    cfg["optimizeOption"] = 1
    if not decoder:
        cfg["resourceManagement"]["useGPUOnlyForCalibration"] = (a.part == "vision")
    cfg["llm"]["attributes"]["runtime"]["batchSize"] = batch
    limit = 8192 if a.part == "vision8192" else 4096
    cfg["llm"]["attributes"].update(maxSequenceLength=limit, maxCacheLength=limit,
                                    maxDataLength=limit, maxCoreDataLength=128)
    tr = cfg["advancedQuantization"]["EquivalentTransformation"]
    tr["SpinR1" if decoder else "HeadOutChRotation"]["matrixPath"] = str(rot)
    recipe = out / "requested-config.json"
    recipe.write_text(json.dumps(cfg, indent=2) + "\n")

    os.chdir(out)
    t = time.perf_counter()
    qbcompiler.mxq_compile(model=str(mblt), target_device="aries-rb",
                           calib_data_path=str(calib), save_path=str(out / "model.mxq"),
                           compile_config=str(recipe), device="gpu", inference_scheme=scheme,
                           use_random_calib=False,
                           config_save_path=str(out / "resolved-config.json"))
    art = out / "model.mxq"
    hh = hashlib.sha256()
    with art.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""): hh.update(b)
    info = {"part": a.part, "qbcompiler": VERSION, "native_stage": stage,
            "batch_size": batch, "inference_scheme": scheme,
            "mxq_sha256": hh.hexdigest(), "bytes": art.stat().st_size,
            "minutes": round((time.perf_counter() - t) / 60, 1),
            "mblt": str(mblt), "calib": str(calib),
            "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
    (out / "build-info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
