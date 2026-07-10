"""YOLO11 전 사이즈(n/s/m/l) NPU 배치지연 실측.

두 가지를 측정:
  (A) 코어모드 × 배치(1~64), NPU 1장   — 사이즈별
  (B) 카드수(1~7) × 배치(1~64), global4 — 사이즈별

패턴: 카드당 1모델 + 멀티스레드 동기 infer(검증된 다채널 패턴).
순수 NPU 추론만(전/후처리 제외). median of REPEAT.
NPU latency는 입력값·calib 무관 → 랜덤 입력 사용.
"""
import sys, os, time, json, statistics, argparse
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import qbruntime

FINAL = "/home/gpuadmin/AX_NPU/scratch_yolo/final"
SIZES = ["yolo11n", "yolo11s", "yolo11m", "yolo11l"]
MODES = ["single", "multi", "global4", "global8"]
BATCHES = [1, 2, 4, 8, 16, 32, 64]
REPEAT = 3
THREADS = 8   # 카드당 스레드

X = np.ascontiguousarray(np.random.rand(640, 640, 3).astype(np.float32))


def _infer(m, x):
    o = m.infer(x)
    return o[0] if isinstance(o, (list, tuple)) else o


def bench(mxq, ncards):
    """mxq를 ncards장에 올리고 카드당 THREADS 스레드 동기 infer. 배치별 median ms 반환."""
    models = []
    for d in range(ncards):
        acc = qbruntime.Accelerator(d)
        m = qbruntime.Model(mxq)
        m.launch(acc)
        models.append(m)
    # warmup
    for m in models:
        for _ in range(5):
            _infer(m, X)
    pool = ThreadPoolExecutor(max_workers=THREADS * ncards)

    def run(i):
        return _infer(models[i % ncards], X)

    res = {}
    for b in BATCHES:
        times = []
        for _ in range(REPEAT):
            t = time.perf_counter()
            list(pool.map(run, range(b)))
            times.append((time.perf_counter() - t) * 1000)
        res[b] = round(statistics.median(times), 1)
    pool.shutdown(wait=True)
    for m in models:
        try:
            m.dispose()
        except Exception:
            pass
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", choices=["A", "B", "all"], default="all")
    ap.add_argument("--sizes", default=",".join(SIZES))
    ap.add_argument("--out", default="/home/gpuadmin/AX_NPU/reports/assets/yolo_all_sizes.json")
    a = ap.parse_args()
    sizes = a.sizes.split(",")
    result = {"batches": BATCHES, "threads": THREADS, "repeat": REPEAT, "A": {}, "B": {}}

    if a.section in ("A", "all"):
        print("=== (A) 코어모드 × 배치 (NPU 1장) ===", flush=True)
        for sz in sizes:
            result["A"][sz] = {}
            for mode in MODES:
                mxq = f"{FINAL}/{sz}_{mode}.mxq"
                if not os.path.exists(mxq):
                    print(f"  [skip] {mxq} 없음", flush=True); continue
                r = bench(mxq, 1)
                result["A"][sz][mode] = r
                print(f"  {sz:9} {mode:8} " + " ".join(f"{r[b]:7.1f}" for b in BATCHES), flush=True)
                json.dump(result, open(a.out, "w"), indent=2)

    if a.section in ("B", "all"):
        print("=== (B) 카드수 × 배치 (global4) ===", flush=True)
        ncards = len(__import__("glob").glob("/dev/aries*"))
        cardlist = [c for c in [1, 2, 3, 4, 5, 6, 7] if c <= ncards]
        result["B_cardlist"] = cardlist
        for sz in sizes:
            result["B"][sz] = {}
            mxq = f"{FINAL}/{sz}_global4.mxq"
            if not os.path.exists(mxq):
                print(f"  [skip] {mxq} 없음", flush=True); continue
            for k in cardlist:
                r = bench(mxq, k)
                result["B"][sz][k] = r
                print(f"  {sz:9} {k}card  " + " ".join(f"{r[b]:7.1f}" for b in BATCHES), flush=True)
                json.dump(result, open(a.out, "w"), indent=2)

    json.dump(result, open(a.out, "w"), indent=2)
    print(f"\n저장: {a.out}", flush=True)


if __name__ == "__main__":
    main()
