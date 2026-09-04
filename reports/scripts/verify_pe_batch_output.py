"""PE full-NPU 배치 출력 무결성 검증 — 배치 N개 한 번에 vs 1장씩 N번.

`MXQInferenceFull.infer(batch)` 는 카드 라운드로빈 + 카드당 슬롯 스레드풀로 **동기** infer 를
동시에 돌린다(async 다건 제출이 아니다 — 그건 출력이 깨진다). 순서는 `out[i]` 로 보존한다.
이 스크립트는 그 결과가 1장씩 돌린 것과 **비트 수준으로 같은지** 확인한다.

    python reports/scripts/verify_pe_batch_output.py --device 7
    python reports/scripts/verify_pe_batch_output.py --device 7 --schemes single,multi,global4,global8

판정: 배치 출력 vs 단건 출력의 장별 cos 가 1.0 이어야 한다(부동소수 오차 허용 1e-6).
하나라도 어긋나면 exit 1.
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.environ.get("AX_NPU_ROOT", str(Path(__file__).resolve().parents[2])))
from pe_npu.inference import MXQInferenceFull      # noqa: E402
from pe_npu.preprocess import preprocess_image     # noqa: E402

REPO = "PIA-SPACE-LAB/MXQ_NPU"


def cos(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", default="W8A16")
    ap.add_argument("--tuning", default="sws_optq")
    ap.add_argument("--schemes", default="single,multi,global4,global8")
    ap.add_argument("--batches", default="1,4,8,12,16,20")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--coco-dir", type=Path, default=Path("download/coco/val2017"))
    ap.add_argument("--tol", type=float, default=1e-6, help="1-cos 허용 오차")
    a = ap.parse_args()

    schemes = [s.strip() for s in a.schemes.split(",") if s.strip()]
    batches = [int(b) for b in a.batches.split(",") if b.strip()]
    paths = sorted(a.coco_dir.glob("*.jpg"))[:max(batches)]
    batch = np.stack([preprocess_image(p) for p in paths])
    print(f"[plan] {a.quant}/{a.tuning} × {schemes} × 배치 {batches}, device={a.device}", flush=True)

    from huggingface_hub import hf_hub_download
    fails = 0
    for s in schemes:
        mxq = hf_hub_download(REPO, f"{a.quant}/{a.tuning}/{s}/pe_full.mxq",
                              token=os.environ.get("HF_TOKEN"))
        m = MXQInferenceFull(full_mxq_path=mxq, device_id=a.device)
        # 기준: 1장씩 따로 (배치 경로를 타지 않는다)
        ref = np.stack([m.infer(batch[i:i + 1])[0] for i in range(max(batches))])
        for B in batches:
            got = m.infer(batch[:B])
            if got.shape != (B, 1024):
                print(f"[FAIL] {s} B={B}: shape {got.shape}"); fails += 1; continue
            cl = [cos(ref[i], got[i]) for i in range(B)]
            worst = min(cl)
            exact = sum(np.array_equal(ref[i], got[i]) for i in range(B))
            ok = (1.0 - worst) <= a.tol
            print(f"[{'ok ' if ok else 'FAIL'}] {s:8s} B={B:2d}  최저 cos={worst:.8f}  "
                  f"비트동일 {exact}/{B}", flush=True)
            if not ok:
                fails += 1
                for i, c in enumerate(cl):
                    if (1.0 - c) > a.tol:
                        print(f"        └ 이미지 {i}: cos={c:.6f}")
        m.dispose()

    print()
    if fails:
        print(f"❌ 배치 출력 불일치 {fails}건 — 배치 경로에 문제가 있다")
        return 1
    print("✅ 전 조합에서 배치 출력이 단건 출력과 일치 — 배치 처리 안전")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
