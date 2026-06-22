"""NPU 대수 스케일링: 고정 채널 부하를 1/2/4/7대로 처리할 때 순수추론 시간."""
import sys, time, json, statistics
import numpy as np, cv2
from huggingface_hub import hf_hub_download
import qbruntime

sys.path.insert(0, "/home/gpuadmin/AX_NPU/Product-AI-mono/packages")
from pia_prod.AI.modules.pe_npu.preprocess import preprocess_image
SP = "/tmp/claude-1000/-home-gpuadmin/057c9eaa-b86d-4a41-ab6f-9dc6babfd1fe/scratchpad"
MXQ = hf_hub_download("PIA-SPACE-LAB/MXQ_NPU", "pe_feat.mxq")

cap = cv2.VideoCapture(f"{SP}/event_video.mp4"); _, f = cap.read(); cap.release()
raw = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
LOADS = [8, 28, 56]   # 채널 부하
REPEAT = 5

# 7대 미리 launch
models = []
for d in range(7):
    mc = qbruntime.ModelConfig(); mc.set_async_pipeline_enabled(True)
    m = qbruntime.Model(MXQ, mc); m.launch(qbruntime.Accelerator(d)); models.append(m)

def prep(n):
    b = preprocess_image([raw]*n).numpy().astype(np.float32)
    return [np.ascontiguousarray(b[i].transpose(1,2,0)) for i in range(n)]

# warmup
w = prep(1)[0]
for m in models:
    for _ in range(3): m.infer_async(w).get()

def infer(xs, k):  # k = 사용 NPU 대수
    t = time.perf_counter()
    futs = [models[i % k].infer_async(xs[i]) for i in range(len(xs))]
    for fu in futs: fu.get()
    return (time.perf_counter()-t)*1000

print("NPU 대수 스케일링 (순수 추론 ms, median):")
print(f"{'채널':>5} | {'1대':>8} {'2대':>8} {'4대':>8} {'7대':>8} | 7대 speedup")
out = []
for n in LOADS:
    xs = prep(n)
    res = {}
    for k in [1,2,4,7]:
        res[k] = statistics.median([infer(xs,k) for _ in range(REPEAT)])
    sp = res[1]/res[7]
    out.append({"channels":n, **{f"npu{k}_ms":round(res[k],1) for k in res}, "speedup_7x":round(sp,2)})
    print(f"{n:>5} | {res[1]:8.1f} {res[2]:8.1f} {res[4]:8.1f} {res[7]:8.1f} | x{sp:.2f}")

for m in models: m.dispose()
json.dump(out, open(f"{SP}/bench_scaling.json","w"), indent=2)
print("저장: bench_scaling.json")
