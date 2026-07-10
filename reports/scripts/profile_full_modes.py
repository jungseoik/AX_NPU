"""full NPU 단계별 채널 스윕 (애프터 벤치, pe_npu_host, 7카드 async).

기존 NPU_pe_pipeline_e2e_hybrid.md(hybrid: P→T(NPU trunk)→Pool(CPU attn_pool)→E)의 애프터판.
full NPU는 Pool 단계가 없다: [P]전처리(CPU) → [N]image→embedding(NPU 전부).

실행: python profile_full_modes.py <mode1>:<mxq1> <mode2>:<mxq2> ...
  예: python profile_full_modes.py single:full_proof/pe_full_single.mxq global8:full_proof/pe_full_global8.mxq
"""
import os, sys, time, statistics
import numpy as np, cv2
from PIL import Image
import qbruntime

ROOT = "/home/gpuadmin/AX_NPU"
SP = "/tmp/claude-1000/-home-gpuadmin/057c9eaa-b86d-4a41-ab6f-9dc6babfd1fe/scratchpad"
sys.path.insert(0, ROOT)
from pe_npu.preprocess import preprocess_image

NUM_NPU = 7; REPEAT = 5
CHANNELS = [1, 4, 7, 8, 16, 28, 42, 56]
VIDEOS = [f"{SP}/event_video.mp4", f"{SP}/pe_binary_elvfalldown_video_1.mp4", f"{SP}/pe_binary_esfalldown_video.mp4"]

specs = [a.split(":", 1) for a in sys.argv[1:]] or [["single", f"{ROOT}/scratchpad_repro/full_proof/pe_full_single.mxq"]]

def grab(n):
    pics = []
    for vp in VIDEOS:
        cap = cv2.VideoCapture(vp); tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        for f in np.linspace(5, max(6, tot-5), n//len(VIDEOS)+2).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f)); ok, fr = cap.read()
            if ok: pics.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
        cap.release()
    return pics[:n]
frames = grab(60)

def med(fn):
    ts = []
    for _ in range(REPEAT):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter()-t0)*1000)
    return statistics.median(ts)

for mode, mxq in specs:
    if not os.path.exists(mxq):
        print(f"[skip] {mode}: {mxq} 없음"); continue
    models = []
    for d in range(NUM_NPU):
        mc = qbruntime.ModelConfig(); mc.set_async_pipeline_enabled(True)
        m = qbruntime.Model(mxq, mc); m.launch(qbruntime.Accelerator(d)); models.append(m)
    # warmup
    w = np.ascontiguousarray(preprocess_image(frames[0]).transpose(1,2,0)); models[0].infer_async(w).get()
    print(f"\n##### {mode}  ({os.path.basename(mxq)}) #####")
    print(f"{'ch':>3} | {'P 전처리':>8} | {'N NPU(full)':>11} | {'e2e':>7} | {'e2e/ch':>7}")
    for n in CHANNELS:
        frs = [frames[i % len(frames)] for i in range(n)]
        hwcs = [np.ascontiguousarray(preprocess_image(f).transpose(1,2,0)) for f in frs]
        def f_pre():
            for f in frs: preprocess_image(f)
        tP = med(f_pre)
        def f_npu():
            futs = [models[i % NUM_NPU].infer_async(hwcs[i]) for i in range(n)]
            _ = [np.asarray(fu.get()).reshape(-1) for fu in futs]
        tN = med(f_npu)
        e2e = tP + tN
        print(f"{n:>3} | {tP:8.0f} | {tN:11.0f} | {e2e:7.0f} | {e2e/n:7.1f}")
    for m in models:
        m.dispose() if hasattr(m, "dispose") else None
print("\n[done] full NPU = P(전처리 CPU) + N(image→embedding NPU). Pool(CPU attn_pool) 단계 없음.")
