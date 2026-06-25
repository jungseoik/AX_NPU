"""full NPU vs hybrid 단계별/채널별 지연 비교 (pe_npu_host, 7카드 분산).

목적: CPU attn_pool 병목이 full NPU에서 사라지는지 실측.
- hybrid : [P]전처리(CPU) -> [T]trunk(NPU) -> [Pool]attn_pool+proj(CPU) -> 임베딩
- full   : [P]전처리(CPU) -> [N]image→embedding(NPU 전부) -> 임베딩   (Pool 없음)

실행: python bench_full_vs_hybrid.py
"""
import os, sys, time, statistics
import numpy as np, torch, cv2
from PIL import Image
from huggingface_hub import hf_hub_download
import qbruntime

ROOT = "/home/gpuadmin/AX_NPU"
SP = "/tmp/claude-1000/-home-gpuadmin/057c9eaa-b86d-4a41-ab6f-9dc6babfd1fe/scratchpad"
FULL_MXQ = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/scratchpad_repro/full_proof/pe_full_qk16.mxq"
sys.path.insert(0, ROOT)
from pe_npu.preprocess import preprocess_image
from pe_npu.pe_vendor import pe as pe_mod

NUM_NPU = 7; REPEAT = 5
CHANNELS = [1, 4, 7, 8, 16, 28, 42, 56]
VIDEOS = [f"{SP}/event_video.mp4", f"{SP}/pe_binary_elvfalldown_video_1.mp4", f"{SP}/pe_binary_esfalldown_video.mp4"]

feat_mxq = hf_hub_download("PIA-SPACE-LAB/MXQ_NPU", "pe_feat.mxq")
pool_pt = hf_hub_download("PIA-SPACE-LAB/MXQ_NPU", "pe_pool_head.pt")

# trunk(7카드) + full(7카드) — async pipeline (운영과 동일한 다카드 분산)
def launch(path):
    ms = []
    for d in range(NUM_NPU):
        mc = qbruntime.ModelConfig(); mc.set_async_pipeline_enabled(True)
        m = qbruntime.Model(path, mc); m.launch(qbruntime.Accelerator(d)); ms.append(m)
    return ms
trunks = launch(feat_mxq)
fulls = launch(FULL_MXQ)
# CPU pool head (hybrid용)
ck = torch.load(pool_pt, map_location="cpu")
skel = pe_mod.CLIP.from_config("PE-Core-L14-336", device="cpu", load_default_weights=False).float().eval()
skel.visual.attn_pool.load_state_dict(ck["attn_pool"])
with torch.no_grad(): skel.visual.proj.copy_(ck["proj"])
visual = skel.visual

# 프레임
def grab(n):
    pics = []
    for vp in VIDEOS:
        cap = cv2.VideoCapture(vp); tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        for f in np.linspace(5, max(6, tot-5), n//len(VIDEOS)+2).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f)); ok, fr = cap.read()
            if ok: pics.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
        cap.release()
    return pics[:n]
pool_frames = grab(60)

def med(fn): return statistics.median([(lambda t0: (fn(), (time.perf_counter()-t0)*1000)[1])(time.perf_counter()) for _ in range(REPEAT)])

# warmup
w = preprocess_image(pool_frames[0]); hwc = np.ascontiguousarray(w.transpose(1,2,0))
trunks[0].infer(hwc); fulls[0].infer(hwc)

print(f"full={os.path.basename(FULL_MXQ)}  (median of {REPEAT}, 7카드)\n")
print(f"{'ch':>3} | {'P':>6} | {'hyb T':>7} | {'hyb Pool':>8} | {'hyb e2e':>8} | {'full N':>7} | {'full e2e':>8} | {'개선':>6}")
rows = []
for n in CHANNELS:
    chws = [preprocess_image(pool_frames[i % len(pool_frames)]) for i in range(n)]
    hwcs = [np.ascontiguousarray(c.transpose(1,2,0)) for c in chws]
    def f_pre():
        for fr in [pool_frames[i % len(pool_frames)] for i in range(n)]: preprocess_image(fr)
    tP = med(f_pre)
    # hybrid trunk (async 분산: 전부 dispatch 후 get)
    feat_box = {}
    def f_trunk():
        futs = [trunks[i % NUM_NPU].infer_async(hwcs[i]) for i in range(n)]
        feat_box["f"] = [np.asarray(fu.get()).reshape(1,577,1024) for fu in futs]
    tT = med(f_trunk); feats = feat_box["f"]
    def f_pool():
        for ft in feats:
            t = torch.from_numpy(ft.astype(np.float32))
            with torch.no_grad():
                p = visual._pool(t); p = p @ visual.proj
    tPool = med(f_pool)
    # full NPU (async 분산)
    def f_full():
        futs = [fulls[i % NUM_NPU].infer_async(hwcs[i]) for i in range(n)]
        _ = [np.asarray(fu.get()).reshape(-1) for fu in futs]
    tN = med(f_full)
    hyb = tP + tT + tPool; full = tP + tN
    rows.append((n, tP, tT, tPool, hyb, tN, full))
    print(f"{n:>3} | {tP:6.0f} | {tT:7.0f} | {tPool:8.0f} | {hyb:8.0f} | {tN:7.0f} | {full:8.0f} | {100*(hyb-full)/hyb:5.0f}%")

import json
json.dump(rows, open(f"{ROOT}/scratchpad_repro/full_proof/bench.json","w"))
for m in trunks+fulls:
    m.dispose() if hasattr(m,"dispose") else None
print("\n[done] hyb=전처리+trunk+CPUpool / full=전처리+NPU. 개선=CPU pool 제거 효과")
