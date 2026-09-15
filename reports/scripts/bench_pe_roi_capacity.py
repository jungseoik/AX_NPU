"""현행 vs 수정안 ROI + 카드수(1/4/8) × 채널(→62) 수용량.

_detect 가 ROI -> 전처리 -> NPU 순차 실행이므로 e2e = R + P + I 로 합산(별도 e2e 실측으로 검증).
"""
import os, sys, time, json, statistics as st
import numpy as np, torch
sys.path.insert(0, os.environ.get("PIA_PACKAGES", "packages"))  # Product-AI-mono/packages
from pia.vision.roi.roi_manager import batch_crop_region
from pia_prod.AI.modules.pe_npu.roi_manager import PERoIManager
from pia_prod.AI.modules.pe_npu.parallel_preprocess import ParallelPreprocessor
from pia_prod.AI.modules.pe_npu.multi_npu import MultiNPUInferenceFull
from pia_prod.AI.modules.pe_npu.preprocess import RESIZE_BACKEND

MXQ = os.environ["MXQ"]
W, H = int(os.environ.get("FW", 1280)), int(os.environ.get("FH", 720))
POLY = os.environ.get("POLY", "0") == "1"
CHS = [int(x) for x in os.environ.get("CHS", "1,8,16,32,48,62").split(",")]
CARDS = [[0], [0,1,2,3], list(range(8))]
IT = int(os.environ.get("IT", "3"))
md = st.median

def params(n):
    ups = []
    for i in range(n):
        pts = []
        if POLY:
            cx, cy, rx, ry = W//2, H//2, int(W*0.35), int(H*0.35)
            for k in range(8):
                a = 2*np.pi*k/8
                pts += [int(cx+rx*np.cos(a)), int(cy+ry*np.sin(a))]
        ups.append({"user_param": {"cameraId": f"c{i}",
                    "retEvent": {"falldown_ret": {"roi": {"polygonCoordinates": pts}}}}})
    return ups

class LegacyROI(PERoIManager):
    """수정 전 동작: numpy -> torch 업로드 후 torch_crop_region."""
    def process_batches_with_roi(self, batches, user_params):
        super().process_batches_with_roi(batches[:1], user_params[:1])  # roi_dict 채우기
        regions = []
        for i, b in enumerate(batches):
            k = f"c{i}"
            if k not in self.roi_dict:
                super().process_batches_with_roi([b], [user_params[i]])
            regions.append(self.roi_dict[k]["expanded_roi"])
        dev = [torch.from_numpy(b) for b in batches]
        return batch_crop_region(dev, regions)

pre = ParallelPreprocessor()
rng = np.random.default_rng(0)
print(f"[cfg] {W}x{H} poly={int(POLY)} resize={RESIZE_BACKEND} workers={pre._nworkers} iters={IT}", flush=True)

stage = {}   # (variant, ch) -> (R, P)
for ch in CHS:
    frames = [rng.integers(0, 255, (H, W, 3), dtype=np.uint8) for _ in range(ch)]
    ups = params(ch)
    for name, mgr in (("fix", PERoIManager()), ("legacy", LegacyROI())):
        c = mgr.process_batches_with_roi(frames, ups); pre(c)          # warm
        R, P = [], []
        for _ in range(IT):
            t0 = time.perf_counter(); c = mgr.process_batches_with_roi(frames, ups)
            t1 = time.perf_counter(); x = pre(c)
            t2 = time.perf_counter()
            R.append((t1-t0)*1e3); P.append((t2-t1)*1e3)
        stage[(name, ch)] = (md(R), md(P))
        print(f"  stage {name:6s} ch={ch:2d}  ROI {md(R):8.1f}  전처리 {md(P):7.1f}", flush=True)
    del frames

infer = {}   # (ncards, ch) -> I
for cards in CARDS:
    m = MultiNPUInferenceFull(MXQ, device_ids=cards)
    for ch in CHS:
        x = torch.from_numpy(rng.standard_normal((ch, 3, 336, 336)).astype(np.float32))
        m(x)
        I = []
        for _ in range(IT):
            t0 = time.perf_counter(); m(x); I.append((time.perf_counter()-t0)*1e3)
        infer[(len(cards), ch)] = md(I)
        print(f"  infer cards={len(cards)} ch={ch:2d}  NPU {md(I):8.1f}", flush=True)
    m.dispose()

print("JSON " + json.dumps({"stage": {f"{k[0]}|{k[1]}": v for k, v in stage.items()},
                            "infer": {f"{k[0]}|{k[1]}": v for k, v in infer.items()},
                            "wh": [W, H], "poly": POLY}))
