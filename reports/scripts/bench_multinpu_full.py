"""[full NPU] 7-NPU 멀티채널 배치 지연시간 벤치 (전처리 vs 순수추론 분리).

bench_multinpu.py(hybrid trunk)의 full NPU 판. 동일 구조:
  [P] 전처리   : 원본 4K 이미지 N장 → 모델 입력(HWC float32). resize336+normalize (CPU)
  [I] 순수추론 : 모델 input → output. **full NPU MXQ(trunk+attn_pool, QKᵀ16bit) = image→embedding 전부 NPU**
N=1..62 스윕, 7대(각 8코어, Single+async) 라운드로빈.
"""
import sys, time, json, statistics
import numpy as np, cv2
from PIL import Image
from huggingface_hub import hf_hub_download
import qbruntime

NUM_NPU = 7; MAX_CH = 62; REPEAT = 5
sys.path.insert(0, "/home/gpuadmin/AX_NPU")
from pe_npu.preprocess import preprocess_image   # AX_NPU 자기완결 전처리

MXQ = hf_hub_download("PIA-SPACE-LAB/MXQ_NPU", "single/pe_full.mxq")   # full NPU, single 모드
SP = "/tmp/claude-1000/-home-gpuadmin/057c9eaa-b86d-4a41-ab6f-9dc6babfd1fe/scratchpad"

_cap = cv2.VideoCapture(f"{SP}/event_video.mp4"); _ok, _f = _cap.read(); _cap.release()
if not _ok: raise RuntimeError("event_video.mp4 읽기 실패")
RAW = Image.fromarray(cv2.cvtColor(_f, cv2.COLOR_BGR2RGB))   # 4K PIL
W, H = RAW.size

def preprocess_batch(n):
    return [np.ascontiguousarray(preprocess_image(RAW).transpose(1, 2, 0)) for _ in range(n)]  # N×HWC

print(f"[init] {NUM_NPU}대 NPU async launch... (원본 {W}x{H}, full MXQ)")
models = []
for d in range(NUM_NPU):
    mc = qbruntime.ModelConfig(); mc.set_async_pipeline_enabled(True)
    m = qbruntime.Model(MXQ, mc); m.launch(qbruntime.Accelerator(d)); models.append(m)
ncore = len(models[0].get_target_cores())
print(f"[init] core mode={models[0].get_core_mode()}, cores/NPU={ncore}, 총={NUM_NPU*ncore}코어")

warm = preprocess_batch(1)[0]
for m in models:
    for _ in range(3): m.infer_async(warm).get()

def time_preprocess(n):
    t = time.perf_counter(); xs = preprocess_batch(n); return (time.perf_counter()-t)*1000, xs
def time_infer(xs):
    n = len(xs); t = time.perf_counter()
    futs = [models[i % NUM_NPU].infer_async(xs[i]) for i in range(n)]
    for f in futs: f.get()
    return (time.perf_counter()-t)*1000

print(f"[sweep] 1..{MAX_CH}채널, 각 {REPEAT}회 median (P=전처리 / I=full추론)")
rows = []
for n in range(1, MAX_CH + 1):
    pre_s, inf_s = [], []
    for _ in range(REPEAT):
        p, xs = time_preprocess(n); pre_s.append(p); inf_s.append(time_infer(xs))
    pre = statistics.median(pre_s); inf = statistics.median(inf_s); tot = pre + inf
    rows.append({"channels": n, "preprocess_ms": round(pre,1), "infer_ms": round(inf,1),
                 "total_ms": round(tot,1), "pre_per_ch_ms": round(pre/n,2),
                 "infer_per_ch_ms": round(inf/n,2), "infer_img_per_s": round(n/(inf/1000),1)})
    if n <= 8 or n % 7 == 0 or n in (16,32,56,62):
        print(f"  ch={n:2d}: P={pre:6.1f}  I={inf:6.1f}  tot={tot:6.1f} | I/ch={inf/n:5.1f}  {n/(inf/1000):5.1f} img/s")

for m in models: m.dispose() if hasattr(m,"dispose") else None
json.dump(rows, open(f"{SP}/bench_multinpu_full.json","w"), indent=2)
print(f"\n저장: bench_multinpu_full.json ({len(rows)}행)")
