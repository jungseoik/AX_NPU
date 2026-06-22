"""7-NPU 멀티채널 배치 지연시간 벤치마크 (전처리 vs 순수추론 분리).

배치로 N채널(N장)이 동시에 들어올 때, 7대 NPU(각 8코어, Single+async)에 라운드로빈
분산하여 '전체 배치가 끝나는 데 걸리는 시간'을 측정. N=1..62 스윕.

지연시간을 2단계로 분리 측정:
  [P] 전처리   : 원본 이미지 N장 → 모델 입력(HWC float32). resize+normalize+layout (CPU)
  [I] 순수추론 : 모델 input → output (NPU trunk, INT8 feat MXQ). infer_async 7대 분산
(참고: hybrid의 CPU pool head는 별도, 장당 ~2ms.)
"""
import sys, time, json, statistics
import numpy as np
import cv2
from huggingface_hub import hf_hub_download
import qbruntime

NUM_NPU = 7
MAX_CH = 62
REPEAT = 5
sys.path.insert(0, "/home/gpuadmin/AX_NPU/Product-AI-mono/packages")
from pia_prod.AI.modules.pe_npu.preprocess import preprocess_image  # 모듈의 배치 전처리

MXQ = hf_hub_download("PIA-SPACE-LAB/MXQ_NPU", "pe_feat.mxq")
SP = "/tmp/claude-1000/-home-gpuadmin/057c9eaa-b86d-4a41-ab6f-9dc6babfd1fe/scratchpad"

# 원본 프레임 1장 (실제 채널 입력 가정 = 4K 영상 프레임). 전처리 비용은 입력 해상도에 의존.
_cap = cv2.VideoCapture(f"{SP}/event_video.mp4")
_ok, _f = _cap.read(); _cap.release()
if not _ok:
    raise RuntimeError("event_video.mp4 읽기 실패")
raw = cv2.cvtColor(_f, cv2.COLOR_BGR2RGB)  # HWC uint8 (4K)
H, W = raw.shape[:2]

def preprocess_batch(n):
    """원본 N장 -> 모델 입력 HWC float32 N개 (전처리 단계 전부 포함)."""
    batch = preprocess_image([raw] * n)            # (N,3,336,336) torch, resize+normalize
    arr = batch.numpy().astype(np.float32)
    return [np.ascontiguousarray(arr[i].transpose(1, 2, 0)) for i in range(n)]  # -> HWC

print(f"[init] {NUM_NPU}대 NPU async 모델 launch... (원본 {W}x{H})")
models = []
for d in range(NUM_NPU):
    mc = qbruntime.ModelConfig(); mc.set_async_pipeline_enabled(True)
    acc = qbruntime.Accelerator(d)
    m = qbruntime.Model(MXQ, mc); m.launch(acc)
    models.append(m)
ncore = len(models[0].get_target_cores())
print(f"[init] core mode={models[0].get_core_mode()}, cores/NPU={ncore}, 총={NUM_NPU*ncore}코어")

warm = preprocess_batch(1)[0]
for m in models:
    for _ in range(3):
        m.infer_async(warm).get()

def time_preprocess(n):
    t = time.perf_counter(); xs = preprocess_batch(n); return (time.perf_counter()-t)*1000, xs

def time_infer(xs):
    n = len(xs)
    t = time.perf_counter()
    futs = [models[i % NUM_NPU].infer_async(xs[i]) for i in range(n)]
    for f in futs: f.get()
    return (time.perf_counter()-t)*1000

print(f"[sweep] 1..{MAX_CH}채널, 각 {REPEAT}회 median (P=전처리 / I=추론)")
rows = []
for n in range(1, MAX_CH + 1):
    pre_s, inf_s = [], []
    for _ in range(REPEAT):
        p, xs = time_preprocess(n); pre_s.append(p)
        inf_s.append(time_infer(xs))
    pre = statistics.median(pre_s); inf = statistics.median(inf_s); tot = pre + inf
    rows.append({"channels": n,
                 "preprocess_ms": round(pre,1), "infer_ms": round(inf,1), "total_ms": round(tot,1),
                 "pre_per_ch_ms": round(pre/n,2), "infer_per_ch_ms": round(inf/n,2),
                 "infer_img_per_s": round(n/(inf/1000),1)})
    if n <= 8 or n % 7 == 0 or n in (16,32,56,62):
        print(f"  ch={n:2d}: P={pre:6.1f}ms  I={inf:6.1f}ms  tot={tot:6.1f}ms | I/ch={inf/n:5.1f}ms  {n/(inf/1000):5.1f} img/s")

for m in models: m.dispose()
OUT = f"{SP}/bench_results.json"
json.dump(rows, open(OUT,"w"), indent=2)
print(f"\n결과 저장: {OUT} ({len(rows)}행)")
