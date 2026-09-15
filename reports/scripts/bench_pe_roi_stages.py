"""ROI 크롭 / resize 단계 단독 스윕 (채널 × 해상도 × 구현)."""
import os, sys, time, json, statistics as st
import numpy as np, torch
sys.path.insert(0, os.environ.get("PIA_PACKAGES", "packages"))  # Product-AI-mono/packages
from pia.vision.roi.roi_manager import batch_crop_region
from pia_prod.AI.modules.pe_npu.roi_manager import PERoIManager
from pia_prod.AI.modules.pe_npu.parallel_preprocess import ParallelPreprocessor
from pia_prod.AI.modules.pe_npu.preprocess import RESIZE_BACKEND
md = st.median
CHS = [1, 4, 8, 16, 24, 32, 48, 62]
IT = 3

class LegacyROI(PERoIManager):
    def process_batches_with_roi(self, batches, user_params):
        for i, b in enumerate(batches):
            if f"c{i}" not in self.roi_dict:
                super().process_batches_with_roi([b], [user_params[i]])
        regions = [self.roi_dict[f"c{i}"]["expanded_roi"] for i in range(len(batches))]
        return batch_crop_region([torch.from_numpy(b) for b in batches], regions)

def params(n, W, H, poly):
    ups = []
    for i in range(n):
        pts = []
        if poly:
            cx, cy, rx, ry = W//2, H//2, int(W*0.35), int(H*0.35)
            for k in range(8):
                a = 2*np.pi*k/8; pts += [int(cx+rx*np.cos(a)), int(cy+ry*np.sin(a))]
        ups.append({"user_param": {"cameraId": f"c{i}",
                    "retEvent": {"falldown_ret": {"roi": {"polygonCoordinates": pts}}}}})
    return ups

def t(fn, it=IT):
    fn()
    v = []
    for _ in range(it):
        t0 = time.perf_counter(); fn(); v.append((time.perf_counter()-t0)*1e3)
    return md(v)

pre = ParallelPreprocessor()
rng = np.random.default_rng(0)
print(f"### RESIZE_BACKEND={RESIZE_BACKEND} workers={pre._nworkers}")
out = {}
for (W, H) in [(1280, 720), (1920, 1080)]:
    for ch in CHS:
        frames = [rng.integers(0, 255, (H, W, 3), dtype=np.uint8) for _ in range(ch)]
        row = {}
        for poly in (False, True):
            ups = params(ch, W, H, poly)
            fx, lg = PERoIManager(), LegacyROI()
            row[f"roi_fix_p{int(poly)}"] = t(lambda: fx.process_batches_with_roi(frames, ups))
            row[f"roi_leg_p{int(poly)}"] = t(lambda: lg.process_batches_with_roi(frames, ups))
            crops = fx.process_batches_with_roi(frames, ups)
            row[f"resize_p{int(poly)}"] = t(lambda: pre(crops))
        out[f"{W}x{H}|{ch}"] = row
        print(f"{W}x{H} ch={ch:2d} | ROI 수정 {row['roi_fix_p0']:6.1f}/{row['roi_fix_p1']:7.1f} "
              f"현행 {row['roi_leg_p0']:7.1f}/{row['roi_leg_p1']:7.1f} | resize {row['resize_p0']:6.1f}/{row['resize_p1']:6.1f}"
              f"   (ROI없음/ROI있음)", flush=True)
        del frames
print("JSON " + json.dumps(out))
