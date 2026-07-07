"""데모: NPU 병렬 추론 — 멀티코어(async) & 멀티카드(round-robin).

같은 배치(N장)를 3가지 방식으로 처리하며 처리량을 비교한다:
  1) sync     : 단일 카드, blocking infer 루프 (1코어 직렬에 가까움)
  2) async    : 단일 카드, infer_async + set_async_pipeline_enabled → 8코어 동시 활용
  3) multicard: 장착된 NPU 전 카드에 라운드로빈 분산 (async)

코어(한 장 안 8코어)는 async로 자동 병렬, 카드(여러 장)는 라운드로빈으로 직접 분산.
상세 실측: ../../reports/performance/NPU_multicard_62ch_benchmark.md

실행:
  conda activate pe_npu_host
  python demo_parallel.py --batch 32
"""
import argparse
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath("../.."))
import pe_npu
from pe_npu import assets
import qbruntime


def detect_npus():
    ids = []
    for p in glob.glob("/dev/aries*"):
        s = os.path.basename(p)[len("aries"):]
        if s.isdigit():
            ids.append(int(s))
    return sorted(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32, help="동시에 처리할 채널/이미지 수")
    ap.add_argument("--repeat", type=int, default=5)
    args = ap.parse_args()

    npus = detect_npus()
    if not npus:
        sys.exit("NPU(/dev/aries*) 없음")
    mxq = assets.ensure_full_mxq(scheme="single")  # HF full NPU MXQ (image→embedding, single 모드)
    print(f"장착 NPU: {npus} ({len(npus)}장) | 배치 {args.batch}장")

    # 입력 준비 (HWC float32) — 지연 측정용 동일 입력 재사용
    img = sorted(glob.glob("images/*.jpg"))
    chw = pe_npu.preprocess_image(img[0]) if img else np.random.randn(3, 336, 336).astype(np.float32)
    x_hwc = np.ascontiguousarray(np.asarray(chw).transpose(1, 2, 0).astype(np.float32))

    def median_ms(fn):
        ts = []
        for _ in range(args.repeat):
            t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1000)
        ts.sort(); return ts[len(ts) // 2]

    # 1) sync (단일 카드)
    m_sync = qbruntime.Model(mxq); m_sync.launch(qbruntime.Accelerator(npus[0]))
    m_sync.infer(x_hwc)  # warmup
    t_sync = median_ms(lambda: [m_sync.infer(x_hwc) for _ in range(args.batch)])
    print(f"1) sync (1카드 직렬)       : {t_sync:7.1f} ms  ({args.batch/(t_sync/1000):6.1f} img/s)")
    m_sync.dispose()

    # 2) async (단일 카드, 8코어)
    mc = qbruntime.ModelConfig(); mc.set_async_pipeline_enabled(True)
    m_async = qbruntime.Model(mxq, mc); m_async.launch(qbruntime.Accelerator(npus[0]))
    m_async.infer_async(x_hwc).get()  # warmup
    def run_async():
        futs = [m_async.infer_async(x_hwc) for _ in range(args.batch)]
        for f in futs: f.get()
    t_async = median_ms(run_async)
    print(f"2) async (1카드, 8코어)    : {t_async:7.1f} ms  ({args.batch/(t_async/1000):6.1f} img/s)  x{t_sync/t_async:.1f}")
    m_async.dispose()

    # 3) multicard (전 카드 라운드로빈, async)
    models = []
    for d in npus:
        cfg = qbruntime.ModelConfig(); cfg.set_async_pipeline_enabled(True)
        mm = qbruntime.Model(mxq, cfg); mm.launch(qbruntime.Accelerator(d)); models.append(mm)
    for mm in models: mm.infer_async(x_hwc).get()  # warmup
    def run_multi():
        futs = [models[i % len(models)].infer_async(x_hwc) for i in range(args.batch)]
        for f in futs: f.get()
    t_multi = median_ms(run_multi)
    print(f"3) multicard ({len(npus)}카드, async): {t_multi:7.1f} ms  ({args.batch/(t_multi/1000):6.1f} img/s)  x{t_sync/t_multi:.1f}")
    for mm in models: mm.dispose()

    print("\n핵심: 코어(8개)는 async가 자동 병렬, 카드(여러 장)는 라운드로빈으로 직접 분산해야 함.")


if __name__ == "__main__":
    main()
