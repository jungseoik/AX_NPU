"""PIA_Wave 스크립트를 NPU 인코더와 함께 실행하는 런처(원본 파일 수정 없음).

    python -m wave_npu.pia_wave 02_Inference --model_type npu -v <video_dir> -o results -c fire ...
    python -m wave_npu.pia_wave 03_Text_Extractor --model_type npu ...

첫 인자는 third_party/PIA_Wave/scripts/VERSION_1/ 의 스크립트 이름.
NPU 카드는 --npu-devices 로 지정(기본 auto). 나머지 인자는 원본 스크립트에 그대로 전달.
"""
from __future__ import annotations

import os
import runpy
import sys

from .encoder import PIA_WAVE, register


def main():
    argv = sys.argv[1:]
    if not argv:
        print(__doc__)
        return 1
    script = argv[0]
    rest = argv[1:]

    devs, scheme, quant = "auto", "single", None
    keep = []
    i = 0
    while i < len(rest):
        if rest[i] == "--npu-devices":
            devs = rest[i + 1]; i += 2
        elif rest[i] == "--npu-scheme":
            scheme = rest[i + 1]; i += 2
        elif rest[i] == "--npu-quant":
            quant = rest[i + 1]; i += 2
        else:
            keep.append(rest[i]); i += 1
    if devs != "auto":
        devs = [int(x) for x in devs.split(",")]

    register(device_ids=devs, scheme=scheme, quant=quant)
    path = os.path.join(PIA_WAVE, "scripts", "VERSION_1",
                        script if script.endswith(".py") else script + ".py")
    if not os.path.exists(path):
        print(f"스크립트 없음: {path}")
        return 1
    sys.argv = [path] + keep
    runpy.run_path(path, run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
