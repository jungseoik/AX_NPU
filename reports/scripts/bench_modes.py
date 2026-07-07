"""코어모드(single/multi/global4/global8) × 다채널 지연 + 메모리 벤치 (7카드).

각 모드 MXQ를 7카드에 async 분산. 채널 N 증가 시 배치 지연 + NPU 메모리(per card) + 호스트 RSS 측정.
"""
import sys, os, time, json, statistics, subprocess, threading, glob
import numpy as np
import qbruntime

NUM_NPU = 7
CHANNELS = [1, 2, 4, 7, 8, 14, 28, 42, 56]
REPEAT = 5
SP = "/tmp/claude-1000/-home-gpuadmin/057c9eaa-b86d-4a41-ab6f-9dc6babfd1fe/scratchpad"
MXQ_DIR = f"{SP}/mxq_modes"
MODES = ["single", "multi", "global4", "global8"]

sys.path.insert(0, "/home/gpuadmin/AX_NPU")
from pe_npu import preprocess_image

chw = preprocess_image(glob.glob("/home/gpuadmin/AX_NPU/tutorial/pe_npu/images/*.jpg")[0]) \
      if glob.glob("/home/gpuadmin/AX_NPU/tutorial/pe_npu/images/*.jpg") \
      else np.random.randn(3, 336, 336).astype(np.float32)
X = np.ascontiguousarray(np.asarray(chw).transpose(1, 2, 0).astype(np.float32))

def npu_mem_max_mb():
    """mobilint-cli status에서 카드별 사용 메모리 최대값(MB)."""
    try:
        out = subprocess.run(["/usr/local/bin/mobilint-cli", "status"],
                             capture_output=True, text=True, timeout=8).stdout
        vals = [int(s.split("MB")[0]) for s in __import__("re").findall(r"(\d+)MB / 16384MB", out)]
        return max(vals) if vals else -1
    except Exception:
        return -1

def host_rss_mb():
    for line in open(f"/proc/{os.getpid()}/status"):
        if line.startswith("VmRSS"):
            return int(line.split()[1]) // 1024
    return -1

def bench_mode(mode):
    mxq = f"{MXQ_DIR}/pe_feat_{mode}.mxq"
    models = []
    for d in range(NUM_NPU):
        cfg = qbruntime.ModelConfig(); cfg.set_async_pipeline_enabled(True)
        m = qbruntime.Model(mxq, cfg); m.launch(qbruntime.Accelerator(d)); models.append(m)
    core_mode = str(models[0].get_core_mode()); ncore = len(models[0].get_target_cores())
    for m in models:  # warmup
        for _ in range(3): m.infer_async(X).get()
    mem_loaded = npu_mem_max_mb()

    def run_batch(n):
        t = time.perf_counter()
        futs = [models[i % NUM_NPU].infer_async(X) for i in range(n)]
        for f in futs: f.get()
        return (time.perf_counter() - t) * 1000

    lat = {}
    for n in CHANNELS:
        lat[n] = round(statistics.median([run_batch(n) for _ in range(REPEAT)]), 1)

    # 메모리 under load: 백그라운드 flood 중 NPU/호스트 메모리 샘플 → peak
    stop = {"v": False}; peak = {"npu": mem_loaded, "rss": host_rss_mb()}
    def flood():
        while not stop["v"]:
            futs = [models[i % NUM_NPU].infer_async(X) for i in range(56)]
            for f in futs: f.get()
    th = threading.Thread(target=flood, daemon=True); th.start()
    for _ in range(12):
        peak["npu"] = max(peak["npu"], npu_mem_max_mb())
        peak["rss"] = max(peak["rss"], host_rss_mb())
        time.sleep(0.25)
    stop["v"] = True; th.join(timeout=5)

    for m in models: m.dispose()
    return {"mode": mode, "core_mode": core_mode, "cores_per_npu": ncore,
            "npu_mem_loaded_mb": mem_loaded, "npu_mem_peak_mb": peak["npu"],
            "host_rss_mb": peak["rss"], "latency_ms": lat}

if __name__ == "__main__":
    results = []
    for mode in MODES:
        if not os.path.exists(f"{MXQ_DIR}/pe_feat_{mode}.mxq"):
            print(f"[skip] {mode}: MXQ 없음"); continue
        print(f"\n=== {mode} ===")
        r = bench_mode(mode); results.append(r)
        print(f"  core_mode={r['core_mode']} cores/npu={r['cores_per_npu']}")
        print(f"  NPU mem: loaded={r['npu_mem_loaded_mb']}MB/card, peak={r['npu_mem_peak_mb']}MB/card | host RSS={r['host_rss_mb']}MB")
        print(f"  latency(ms): " + " ".join(f"{n}ch={v}" for n, v in r['latency_ms'].items()))
    json.dump(results, open(f"{SP}/bench_modes.json", "w"), indent=2)
    print(f"\n저장: bench_modes.json")
