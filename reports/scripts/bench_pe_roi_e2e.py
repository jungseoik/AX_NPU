"""① 측정 편차 원인(교차 실행 시 torch OMP 간섭) ② e2e 단일 타이머 최종 비교."""
import os, sys, time, statistics as st
from concurrent.futures import ThreadPoolExecutor
import numpy as np, torch, cv2
sys.path.insert(0, os.environ.get("PIA_PACKAGES", "packages"))  # Product-AI-mono/packages
from pia.vision.roi.roi_manager import batch_crop_region
from pia_prod.AI.modules.pe_npu.roi_manager import PERoIManager
from pia_prod.AI.modules.pe_npu.parallel_preprocess import ParallelPreprocessor
from pia_prod.AI.modules.pe_npu.multi_npu import MultiNPUInferenceFull
md = st.median; W, H, CH = 1280, 720, 62
MEAN = torch.tensor([0.5]*3).view(1,3,1,1); STD = torch.tensor([0.5]*3).view(1,3,1,1)

class LegacyROI(PERoIManager):
    def process_batches_with_roi(self, batches, user_params):
        for i, b in enumerate(batches):
            if f"c{i}" not in self.roi_dict:
                super().process_batches_with_roi([b], [user_params[i]])
        regions = [self.roi_dict[f"c{i}"]["expanded_roi"] for i in range(len(batches))]
        return batch_crop_region([torch.from_numpy(b) for b in batches], regions)

def params(n, poly):
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

class FusedPre:
    """스레드 cv2 resize(uint8) → 배치 1회 normalize."""
    def __init__(self, nw=32):
        self.pool = ThreadPoolExecutor(max_workers=nw)
    def __call__(self, crops):
        out = np.empty((len(crops), 336, 336, 3), dtype=np.uint8)
        def w(i):
            a = crops[i]
            out[i] = a if a.shape[:2] == (336, 336) else cv2.resize(a, (336, 336), interpolation=cv2.INTER_LINEAR)
        list(self.pool.map(w, range(len(crops))))
        t = torch.from_numpy(out).permute(0, 3, 1, 2).contiguous().float().div_(255)
        return t.sub_(MEAN).div_(STD)

def timed(fn, it=5):
    fn(); v = []
    for _ in range(it):
        s = time.perf_counter(); fn(); v.append((time.perf_counter()-s)*1e3)
    return md(v)

def main():
    MXQ = os.environ["MXQ"]
    rng = np.random.default_rng(0)
    frames = [rng.integers(0,255,(H,W,3),dtype=np.uint8) for _ in range(CH)]
    ups0 = params(CH, False)

    print("=== ① 측정 편차: legacy ROI 단독 vs 전처리와 교차 실행 (62ch, ROI 없음) ===")
    lg = LegacyROI(); pre = ParallelPreprocessor(workers=16)
    print(f"  ROI 단독 반복            : {timed(lambda: lg.process_batches_with_roi(frames, ups0)):8.1f} ms")
    def alt():
        c = lg.process_batches_with_roi(frames, ups0); pre(c)
    alt()
    R = []
    for _ in range(5):
        s = time.perf_counter(); c = lg.process_batches_with_roi(frames, ups0)
        R.append((time.perf_counter()-s)*1e3); pre(c)
    print(f"  ROI(전처리와 번갈아 실행): {md(R):8.1f} ms   ← 실서비스 패턴")
    print(f"  ROI 단독 재측정          : {timed(lambda: lg.process_batches_with_roi(frames, ups0)):8.1f} ms")

    print("\n=== ② e2e 단일 타이머 (8카드/4카드, 62ch) ===")
    fused = FusedPre(32)
    for poly in (False, True):
        ups = params(CH, poly)
        for ncards in (8, 4):
            m = MultiNPUInferenceFull(MXQ, device_ids=list(range(ncards)))
            lg2, fx = LegacyROI(), PERoIManager()
            a = timed(lambda: m(pre(lg2.process_batches_with_roi(frames, ups))), it=3)
            b = timed(lambda: m(pre(fx.process_batches_with_roi(frames, ups))), it=3)
            c = timed(lambda: m(fused(fx.process_batches_with_roi(frames, ups))), it=3)
            print(f"  ROI{'있음' if poly else '없음'} {ncards}카드 | 현행 {a:7.1f} | ROI수정 {b:7.1f} | ROI수정+전처리개선 {c:7.1f} ms", flush=True)
            m.dispose()

if __name__ == "__main__":
    main()
