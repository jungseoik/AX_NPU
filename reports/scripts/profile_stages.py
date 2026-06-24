"""_detect 파이프라인 단계별 지연 + e2e 프로파일 (이미지 리스트 입력, N채널 스윕).

단계: [P]전처리 → [T]NPU trunk(7카드 분산 async) → [Pool]CPU pool head → [E]event/알람
실제 영상 프레임 N개를 동시 입력으로 주고 각 단계 median 측정.
"""
import sys, os, time, statistics
from collections import deque
import numpy as np, torch, cv2
from PIL import Image

sys.path.insert(0, "/home/gpuadmin/AX_NPU/Product-AI-mono/packages")
from pia_prod.AI.modules.pe_npu.engine.preprocess import preprocess_image, IMAGE_SIZE
from pia_prod.AI.modules.pe_npu.engine.pe_vendor import pe as pe_mod
from pia_prod.AI.modules.pe_npu.parallel_preprocess import ParallelPreprocessor
from pia_prod.AI.modules.pe_npu.event import PENpuEventManager
from pia_prod.AI.modules.pe_npu.prompts import load_text_feature
from pia_prod.AI.global_config import USER_PARAM_KEY, RET_EVENT_KEY
from huggingface_hub import hf_hub_download
import qbruntime

SP = "/tmp/claude-1000/-home-gpuadmin/057c9eaa-b86d-4a41-ab6f-9dc6babfd1fe/scratchpad"
NUM_NPU = 7; REPEAT = 5
CHANNELS = [1, 2, 4, 7, 8, 12, 16, 28, 42, 56]   # 최대 56 = 7카드 x 8코어

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--mxq", default="", help="trunk MXQ 경로 (빈값=HF single)")
_ap.add_argument("--label", default="single")
_args = _ap.parse_args()
print(f"[init] 모델/pool/text/event 로드 (trunk={_args.label})")
mxq = _args.mxq if _args.mxq else hf_hub_download("PIA-SPACE-LAB/MXQ_NPU", "pe_feat.mxq")
pool_path = hf_hub_download("PIA-SPACE-LAB/MXQ_NPU", "pe_pool_head.pt")
# trunk: 7카드 async
models = []
for d in range(NUM_NPU):
    mc = qbruntime.ModelConfig(); mc.set_async_pipeline_enabled(True)
    m = qbruntime.Model(mxq, mc); m.launch(qbruntime.Accelerator(d)); models.append(m)
# pool head (CPU)
ck = torch.load(pool_path, map_location="cpu")
skel = pe_mod.CLIP.from_config("PE-Core-L14-336", device="cpu", load_default_weights=False).float().eval()
skel.visual.attn_pool.load_state_dict(ck["attn_pool"])
with torch.no_grad(): skel.visual.proj.copy_(ck["proj"])
visual = skel.visual
# text + event
ID, cls, prm, tf = load_text_feature(hf_hub_download("PIA-SPACE-LAB/PE-Core-L14-336", "text_features.json"), "cpu")
mgr = PENpuEventManager(); mgr.prepare_vectors({"ids": ID, "class_list": cls, "vectors": tf, "prompt_list": prm})
pp = ParallelPreprocessor()  # thread

# 실제 프레임 N개 (event 영상에서 간격 추출)
cap = cv2.VideoCapture(f"{SP}/event_video.mp4")
frames = []
for fno in [60, 450, 690, 120, 405, 660, 800, 900]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fno); ok, fr = cap.read()
    if ok: frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
cap.release()

def _t(fn): t = time.perf_counter(); fn(); return (time.perf_counter()-t)*1000
def med(fn): return statistics.median([_t(fn) for _ in range(REPEAT)])

# warmup
warm = pp([frames[0]]); models[0].infer_async(np.ascontiguousarray(warm[0].numpy().transpose(1,2,0))).get()

print(f"[sweep] 단계별 지연(ms, median of {REPEAT}) — 7카드 분산\n")
print(f"{'ch':>3} | {'P 전처리':>9} | {'T NPU추론':>10} | {'Pool(CPU)':>10} | {'E event':>9} | {'e2e':>8} | {'e2e/ch':>7}")
rows = []
for n in CHANNELS:
    fr_list = [frames[i % len(frames)] for i in range(n)]
    sids = [f"{i}_c" for i in range(n)]
    ups = [{USER_PARAM_KEY: {RET_EVENT_KEY: ["fire_ret", "smoke_ret", "falldown_ret"]}} for _ in range(n)]

    # [P] 전처리
    def f_pre(): return pp(fr_list)
    hwc_box = {}
    def f_pre_store():
        b = pp(fr_list); hwc_box["x"] = [np.ascontiguousarray(b[i].numpy().transpose(1,2,0)) for i in range(n)]
    tP = med(f_pre_store)
    xs = hwc_box["x"]
    # [T] NPU trunk (7카드 async 분산)
    feat_box = {}
    def f_trunk():
        futs = [models[i % NUM_NPU].infer_async(xs[i]) for i in range(n)]
        feat_box["f"] = [np.asarray(fu.get()).reshape(1, 577, 1024) for fu in futs]
    tT = med(f_trunk)
    feats = feat_box["f"]
    # [Pool] CPU pool head
    emb_box = {}
    def f_pool():
        out = []
        for ft in feats:
            t = torch.from_numpy(ft.astype(np.float32))
            with torch.no_grad():
                p = visual._pool(t)
                if visual.proj_dim is not None: p = p @ visual.proj
            out.append(p.reshape(-1))
        emb_box["e"] = out
    tPool = med(f_pool)
    embs = emb_box["e"]
    # [E] event/알람
    def f_event():
        vis = {sid: deque([e], maxlen=1) for sid, e in zip(sids, embs)}
        mgr.update(vis, sids, ups)
    tE = med(f_event)
    e2e = tP + tT + tPool + tE
    rows.append((n, tP, tT, tPool, tE, e2e))
    print(f"{n:>3} | {tP:9.1f} | {tT:10.1f} | {tPool:10.1f} | {tE:9.1f} | {e2e:8.1f} | {e2e/n:7.1f}")

for m in models: m.dispose()
pp.shutdown()
