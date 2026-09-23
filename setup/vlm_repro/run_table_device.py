"""벤더 `scripts/run_table.py` 를 --device 지원으로 감싸 실행한다.

패키지 파일을 수정하지 않는다 — `scripts/verify_files.py` 무결성 검사를 통과한 상태를
그대로 유지해야 하기 때문이다(원본을 고치면 즉시 검출된다).

원본은 `Classifier(device=0)` 으로 0번 카드 고정이고 CLI 로 바꿀 수 없다. 이 서버는
0번을 운영 파드가 쓰고 있어 그대로는 못 돌린다. 여기서 `pia_vlm.runtime.Classifier` 의
기본 device 를 바꾼 뒤 원본 main() 을 호출한다.

    python3.10 setup/vlm_repro/run_table_device.py --device 7 --sizes 224x224 \
        --repetitions 2 --output results/smoke
"""
from __future__ import annotations
import argparse, functools, runpy, sys
from pathlib import Path

REPRO = Path(__file__).resolve().parent  # 컨테이너 안에서는 /repro 로 마운트된다


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--device", type=int, default=0,
                    help="NPU device index (/dev/ariesN). 원본 run_table.py 에는 없는 인자")
    ap.add_argument("--repro", default="/repro", help="재현 패키지 루트")
    known, rest = ap.parse_known_args()

    root = Path(known.repro)
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root))

    from pia_vlm import runtime as rt
    if known.device != 0:
        rt.Classifier.__init__ = functools.partialmethod(
            rt.Classifier.__init__, device=known.device)
        print(f"[wrap] Classifier device={known.device} 로 고정", flush=True)

    sys.argv = ["run_table.py"] + rest
    runpy.run_path(str(root / "scripts" / "run_table.py"), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
