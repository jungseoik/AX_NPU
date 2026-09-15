"""CPU 리소스 추가 투입 여지 + legacy ROI 곡선 재측정. (spawn 재귀 방지 위해 main 가드 필수)"""
import os, sys, time, threading, statistics as st
from concurrent.futures import ThreadPoolExecutor
import numpy as np, torch, cv2
sys.path.insert(0, os.environ.get("PIA_PACKAGES", "packages"))  # Product-AI-mono/packages
from pia.vision.roi.roi_manager import batch_crop_region
from pia_prod.AI.modules.pe_npu.roi_manager import PERoIManager
from pia_prod.AI.modules.pe_npu.parallel_preprocess import ParallelPreprocessor
from pia_prod.AI.modules.pe_npu.multi_npu import MultiNPUInferenceFull
md = st.median
W, H = 1280, 720

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

def timed(fn, it=5):
    fn()
    v, c0, t0 = [], time.process_time(), time.perf_counter()
    for _ in range(it):
        s = time.perf_counter(); fn(); v.append((time.perf_counter()-s)*1e3)
    return md(v), (time.process_time()-c0)/(time.perf_counter()-t0)

def main():
    MXQ = os.environ["MXQ"]
    CH = 62
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, (H, W, 3), dtype=np.uint8) for _ in range(CH)]

    print(f"=== [A] legacy ROI 곡선 재측정 (720p, 5회 median) ===", flush=True)
    for ch in (1, 4, 8, 16, 24, 32, 48, 62):
        f = frames[:ch]
        for poly in (False, True):
            ups = params(ch, poly); lg = LegacyROI()
            ms, cores = timed(lambda: lg.process_batches_with_roi(f, ups))
            print(f"  ch={ch:2d} ROI{'있음' if poly else '없음'}  {ms:8.1f}ms  ({ms/ch:5.1f}ms/frame, {cores:4.1f}코어)", flush=True)

    print(f"=== [B] CPU 단계(ROI+resize) 워커 스윕  {CH}ch (호스트 {os.cpu_count()} CPU) ===", flush=True)
    for poly in (False, True):
        ups = params(CH, poly)
        for nw in (1, 4, 8, 16, 24, 32, 48, 64):
            roi = PERoIManager()
            roi._pool = ThreadPoolExecutor(max_workers=nw) if nw > 1 else None
            pre = ParallelPreprocessor(mode="thread" if nw > 1 else "single", workers=nw)
            ms, cores = timed(lambda: pre(roi.process_batches_with_roi(frames, ups)))
            print(f"  ROI{'있음' if poly else '없음'} workers={nw:2d}  {ms:7.1f}ms  ({cores:4.1f}코어)", flush=True)
            pre.shutdown(); roi.shutdown()

    print("=== [C] cv2 내부 스레드 (workers=16, ROI 없음) ===", flush=True)
    ups = params(CH, False)
    for nt in (1, 2, 4):
        cv2.setNumThreads(nt)
        roi = PERoIManager(); pre = ParallelPreprocessor(workers=16)
        ms, cores = timed(lambda: pre(roi.process_batches_with_roi(frames, ups)))
        print(f"  setNumThreads={nt}  {ms:7.1f}ms ({cores:4.1f}코어)", flush=True)
        pre.shutdown(); roi.shutdown()
    cv2.setNumThreads(1)

    print("=== [D] 프로세스 풀 vs 스레드 (전처리만, ROI 없음) ===", flush=True)
    roi = PERoIManager(); crops = roi.process_batches_with_roi(frames, ups)
    for mode, nw in (("thread", 16), ("thread", 32), ("process", 16), ("process", 32)):
        pre = ParallelPreprocessor(mode=mode, workers=nw)
        ms, cores = timed(lambda: pre(crops), it=3)
        print(f"  {mode:8s} workers={nw:2d}  {ms:7.1f}ms ({cores:4.1f}코어)", flush=True)
        pre.shutdown()

    print("=== [E] NPU-CPU 파이프라이닝 (8카드, ROI 없음) ===", flush=True)
    m = MultiNPUInferenceFull(MXQ, device_ids=list(range(8)))
    roi = PERoIManager(); pre = ParallelPreprocessor(workers=16)
    ms, cores = timed(lambda: m(pre(roi.process_batches_with_roi(frames, ups))))
    print(f"  현재(순차)   {ms:7.1f}ms ({cores:4.1f}코어)", flush=True)
    box = {}
    def prep(): box["x"] = pre(roi.process_batches_with_roi(frames, ups))
    prep(); x = box["x"]
    n = 6; t0 = time.perf_counter()
    for _ in range(n):
        th = threading.Thread(target=prep); th.start()
        m(x); th.join(); x = box["x"]
    print(f"  파이프라인   {(time.perf_counter()-t0)/n*1e3:7.1f}ms", flush=True)
    m.dispose()

if __name__ == "__main__":
    main()
