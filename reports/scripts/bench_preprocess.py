"""전처리 병렬화 벤치마크 (단일 vs 스레드 vs 프로세스), 1->62채널 스윕.

전처리 = 4K 원본 -> resize336 + normalize0.5 -> HWC float32 (모델 입력).
NPU 추론과 달리 채널마다 완전 독립 + CPU-bound라 멀티프로세스로 코어 분산이 잘 먹힌다.
"""
import sys, os, time, json, statistics
import numpy as np
import cv2

sys.path.insert(0, "/home/gpuadmin/AX_NPU/Product-AI-mono/packages")
SP = "/tmp/claude-1000/-home-gpuadmin/057c9eaa-b86d-4a41-ab6f-9dc6babfd1fe/scratchpad"

_cap = cv2.VideoCapture(f"{SP}/event_video.mp4"); _ok, _f = _cap.read(); _cap.release()
RAW = cv2.cvtColor(_f, cv2.COLOR_BGR2RGB)   # HWC uint8 (4K)
H, W = RAW.shape[:2]

_PRE = None
def _winit(raw):
    global _RAW, _PRE
    import torch; torch.set_num_threads(1)
    _RAW = raw
    from pia_prod.AI.modules.pe_npu.preprocess import preprocess_image
    _PRE = preprocess_image
def _wjob(_):
    a = _PRE([_RAW]).numpy()[0]
    return np.ascontiguousarray(a.transpose(1, 2, 0))

if __name__ == "__main__":
    from pia_prod.AI.modules.pe_npu.preprocess import preprocess_image
    from concurrent.futures import ThreadPoolExecutor
    import multiprocessing as mp

    MAX_CH = 62; REPEAT = 3

    def to_hwc(batch):
        a = batch.numpy().astype(np.float32)
        return [np.ascontiguousarray(a[i].transpose(1, 2, 0)) for i in range(len(a))]
    def baseline(n):
        return to_hwc(preprocess_image([RAW] * n))

    proc = mp.Pool(MAX_CH, initializer=_winit, initargs=(RAW,))
    proc.map(_wjob, range(MAX_CH))                         # warm
    thr = ThreadPoolExecutor(max_workers=32)
    def threaded(n):
        return list(thr.map(lambda _: _winit_skip(), range(n)))
    def thread_job(_):
        a = preprocess_image([RAW]).numpy()[0]
        return np.ascontiguousarray(a.transpose(1, 2, 0))
    list(thr.map(thread_job, range(8)))                    # warm
    baseline(8)                                            # warm

    def med(fn, n):
        return statistics.median([(_t(fn, n)) for _ in range(REPEAT)])
    def _t(fn, n):
        t = time.perf_counter(); fn(n); return (time.perf_counter()-t)*1000

    print(f"[원본 {W}x{H}] 1..{MAX_CH}채널 전처리 (median, ms)")
    rows = []
    for n in range(1, MAX_CH+1):
        b = med(baseline, n)
        th = med(lambda k: list(thr.map(thread_job, range(k))), n)
        pr = med(lambda k: proc.map(_wjob, range(k)), n)
        rows.append({"channels": n, "baseline_ms": round(b,1), "threaded_ms": round(th,1),
                     "process_ms": round(pr,1), "speedup_proc": round(b/pr,2),
                     "proc_per_ch_ms": round(pr/n,2)})
        if n <= 4 or n % 7 == 0 or n in (16,32,56,62):
            print(f"  ch={n:2d}: baseline={b:7.1f}  thread={th:7.1f}  process={pr:7.1f}  (proc {b/pr:.1f}x, {pr/n:.1f}ms/ch)")

    proc.close(); proc.join(); thr.shutdown()
    json.dump(rows, open(f"{SP}/bench_preprocess.json","w"), indent=2)
    print(f"\n저장: bench_preprocess.json ({len(rows)}행)")
