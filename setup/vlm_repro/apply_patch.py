"""벤더 재현 패키지에 --device 인자를 추가한다(원본은 .orig 로 보존).

`run_table.py` 가 NPU 카드를 고를 수 없게 되어 있어(`Classifier(device=0)` 고정)
이 서버처럼 특정 카드만 쓸 수 있는 환경에서 필요하다.

    python setup/vlm_repro/apply_patch.py            # 적용
    python setup/vlm_repro/apply_patch.py --revert   # 되돌리기
"""
from __future__ import annotations
import argparse, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "download/vendor/customer-capacity-repro/scripts/run_table.py"

ADD_ARG = ("    p.add_argument('--threads', type=int, default=4)\n",
           "    p.add_argument('--threads', type=int, default=4)\n"
           "    p.add_argument('--device', type=int, default=0,\n"
           "                   help='NPU device index (/dev/ariesN)')\n")
PASS_ARG = ("                           threads=args.threads, max_patches=case['max_patches'],\n",
            "                           threads=args.threads, max_patches=case['max_patches'],\n"
            "                           device=args.device,\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true")
    a = ap.parse_args()
    orig = TARGET.with_suffix(".py.orig")

    if not TARGET.exists():
        print(f"대상 없음: {TARGET}"); return 1
    if a.revert:
        if orig.exists():
            shutil.copy2(orig, TARGET); print(f"되돌림 ← {orig.name}")
        else:
            print("백업 없음 — 되돌릴 것이 없다")
        return 0

    s = TARGET.read_text(encoding="utf-8")
    if "--device" in s:
        print("이미 적용됨"); return 0
    for old, new in (ADD_ARG, PASS_ARG):
        if old not in s:
            print(f"패치 지점 미발견:\n{old}"); return 1
        s = s.replace(old, new, 1)
    if not orig.exists():
        shutil.copy2(TARGET, orig); print(f"원본 보존 → {orig.name}")
    TARGET.write_text(s, encoding="utf-8")
    print("--device 인자 추가 완료")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
