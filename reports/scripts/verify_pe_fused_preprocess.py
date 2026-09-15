"""제안 A 검증: 출력 동등성 + 폴백 경로 + e2e."""
import os, sys, time, statistics as st
import numpy as np, torch, cv2
sys.path.insert(0, os.environ.get("PIA_PACKAGES", "packages"))
from pia_prod.AI.modules.pe_npu.parallel_preprocess import ParallelPreprocessor
from pia_prod.AI.modules.pe_npu.preprocess import preprocess_image, is_fusable
from pia_prod.AI.modules.pe_npu.roi_manager import PERoIManager
from pia_prod.AI.modules.pe_npu.multi_npu import MultiNPUInferenceFull
md = st.median; W, H = 1280, 720

def ups(n, poly=False):
    o = []
    for i in range(n):
        pts = []
        if poly:
            cx, cy, rx, ry = W//2, H//2, int(W*0.35), int(H*0.35)
            for k in range(8):
                a = 2*np.pi*k/8; pts += [int(cx+rx*np.cos(a)), int(cy+ry*np.sin(a))]
        o.append({"user_param": {"cameraId": f"c{i}", "retEvent": {"falldown_ret": {"roi": {"polygonCoordinates": pts}}}}})
    return o

def timed(fn, it=5):
    fn(); v = []
    for _ in range(it):
        s = time.perf_counter(); fn(); v.append((time.perf_counter()-s)*1e3)
    return md(v)

def main():
    rng = np.random.default_rng(0)
    for ch in (8, 62):
        frames = [rng.integers(0,255,(H,W,3),dtype=np.uint8) for _ in range(ch)]
        roi = PERoIManager(); crops = roi.process_batches_with_roi(frames, ups(ch))
        pre = ParallelPreprocessor()
        new = pre(crops)                                   # fused
        ref = torch.stack([preprocess_image([c])[0] for c in crops])   # 기존 장당 경로
        print(f"ch={ch:2d} fusable={is_fusable(crops)} shape={tuple(new.shape)} "
              f"최대오차={float((new-ref).abs().max()):.8f} 동일={torch.equal(new, ref)}")
    # 폴백 경로들
    small = [rng.integers(0,255,(H,W,3),dtype=np.uint8) for _ in range(3)]
    print("작은배치(3):", tuple(ParallelPreprocessor()(small).shape))
    gray = [rng.integers(0,255,(H,W,1),dtype=np.uint8) for _ in range(8)]
    print("1채널 fusable:", is_fusable(gray))
    t = [torch.randint(0,255,(H,W,3),dtype=torch.uint8) for _ in range(8)]
    print("torch 입력 fusable:", is_fusable(t), "->", tuple(ParallelPreprocessor()(t).shape))
    f32 = [rng.random((H,W,3)).astype(np.float32) for _ in range(8)]
    print("float 입력 fusable:", is_fusable(f32), "->", tuple(ParallelPreprocessor()(f32).shape))
    print("단일 텐서 입력:", tuple(ParallelPreprocessor()(torch.randint(0,255,(4,H,W,3),dtype=torch.uint8)).shape))

    # e2e
    MXQ = os.environ["MXQ"]
    frames = [rng.integers(0,255,(H,W,3),dtype=np.uint8) for _ in range(62)]
    for poly in (False, True):
        u = ups(62, poly)
        for nc in (8, 4):
            m = MultiNPUInferenceFull(MXQ, device_ids=list(range(nc)))
            roi = PERoIManager(); pre = ParallelPreprocessor()
            e = timed(lambda: m(pre(roi.process_batches_with_roi(frames, u))), it=3)
            print(f"e2e 62ch ROI{'있음' if poly else '없음'} {nc}카드: {e:7.1f} ms")
            m.dispose()

if __name__ == "__main__":
    main()
