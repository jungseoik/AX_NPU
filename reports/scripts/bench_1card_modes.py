"""[full NPU] 1장 NPU, 코어모드 4종 × 1~16채널 순수추론 증가폭.

NPU 1장(8코어)에서 single/global4/global8/multi MXQ로 채널 1→16 스윕, 순수추론(I)만 측정.
코어모드별 카드내 스케줄링 차이(슬롯=8/이미지당코어)가 채널 증가에 따라 어떻게 드러나는지.

  single  : 코어1/이미지 → 8슬롯 (1~8ch 평탄, 9ch+ 2웨이브)
  global4 : 코어4/이미지 → 2슬롯
  global8 : 코어8/이미지 → 1슬롯 (채널수에 선형)
  multi   : 클러스터

실행: python bench_1card_modes.py [device_id]   (기본 0)
"""
import sys, time, json, statistics
import numpy as np, cv2
from PIL import Image
from huggingface_hub import hf_hub_download
import qbruntime

sys.path.insert(0, "/home/gpuadmin/AX_NPU")
from pe_npu.preprocess import preprocess_image

DEV = int(sys.argv[1]) if len(sys.argv) > 1 else 0
MODES = ["single", "multi", "global4", "global8"]
CH = list(range(1, 17))   # 1~16
REPEAT = 5
SP = "/tmp/claude-1000/-home-gpuadmin/057c9eaa-b86d-4a41-ab6f-9dc6babfd1fe/scratchpad"

# 입력 1장 전처리 → HWC 재사용 (순수추론만 측정하므로 동일 입력 반복)
cap = cv2.VideoCapture(f"{SP}/event_video.mp4"); _, f = cap.read(); cap.release()
x = np.ascontiguousarray(preprocess_image(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))).transpose(1, 2, 0))

results = {}
for mode in MODES:
    mxq = hf_hub_download("PIA-SPACE-LAB/MXQ_NPU", f"{mode}/pe_full.mxq")
    mc = qbruntime.ModelConfig(); mc.set_async_pipeline_enabled(True)
    m = qbruntime.Model(mxq, mc); m.launch(qbruntime.Accelerator(DEV))
    ncore = len(m.get_target_cores())
    for _ in range(3): m.infer_async(x).get()   # warmup
    row = []
    for n in CH:
        def run():
            futs = [m.infer_async(x) for _ in range(n)]
            for fu in futs: fu.get()
        t = statistics.median([(lambda t0: (run(), (time.perf_counter()-t0)*1000)[1])(time.perf_counter()) for _ in range(REPEAT)])
        row.append(round(t, 1))
    results[mode] = {"cores_per_infer": None, "ncore": ncore, "I_ms": row}
    print(f"[{mode}] cores/NPU={ncore}  I(ms) 1~16: {row}")
    m.dispose() if hasattr(m, "dispose") else None

json.dump(results, open(f"{SP}/bench_1card_modes.json", "w"), indent=2)
print("\n=== 표 (순수추론 ms, 1장 NPU) ===")
print("ch  | " + " | ".join(f"{mode:>8}" for mode in MODES))
for i, n in enumerate(CH):
    print(f"{n:3d} | " + " | ".join(f"{results[mode]['I_ms'][i]:8.1f}" for mode in MODES))
print("저장: bench_1card_modes.json")
