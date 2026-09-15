"""NHWC 기본 on 검증: ROI 유무 × 폴백 경로 × 출력 동등성 × e2e."""
import os, sys, time, statistics as st
import numpy as np, torch
sys.path.insert(0, os.environ.get("PIA_PACKAGES", "packages"))
from pia_prod.AI.modules.pe_npu import config as cfg
from pia_prod.AI.modules.pe_npu.roi_manager import PERoIManager
from pia_prod.AI.modules.pe_npu.parallel_preprocess import ParallelPreprocessor
from pia_prod.AI.modules.pe_npu.multi_npu import MultiNPUInferenceFull
md = st.median; W, H = 1280, 720

def ups(n, poly):
    o = []
    for i in range(n):
        pts = []
        if poly:
            cx, cy, rx, ry = W//2, H//2, int(W*0.35), int(H*0.35)
            for k in range(8):
                a = 2*np.pi*k/8; pts += [int(cx+rx*np.cos(a)), int(cy+ry*np.sin(a))]
        o.append({"user_param":{"cameraId":f"c{i}","retEvent":{"falldown_ret":{"roi":{"polygonCoordinates":pts}}}}})
    return o

def timed(fn, it=5):
    fn(); v=[]
    for _ in range(it):
        s=time.perf_counter(); fn(); v.append((time.perf_counter()-s)*1e3)
    return md(v)

def main():
    print(f"[cfg] PE_NPU_OUTPUT_NHWC={cfg.PE_NPU_OUTPUT_NHWC}  RESIZE={cfg.PE_NPU_RESIZE}")
    rng = np.random.default_rng(0)
    m = MultiNPUInferenceFull(os.environ["MXQ"], device_ids=[0])

    print("\n① 레이아웃 × ROI 유무 — 전처리 출력 shape / 연속성 / 임베딩 동등성")
    for poly in (False, True):
        frames = [rng.integers(0,255,(H,W,3),dtype=np.uint8) for _ in range(8)]
        u = ups(8, poly)
        roi = PERoIManager(); crops = roi.process_batches_with_roi(frames, u)
        pn = ParallelPreprocessor(nhwc=True)(crops)     # 신규 기본
        pc = ParallelPreprocessor(nhwc=False)(crops)    # 기존 계약
        en, ec = np.asarray(m(pn)), np.asarray(m(pc))
        # crop이 원본 뷰인지(ROI 미설정) 확인 + 원본 오염 여부
        view = crops[0].base is not None or crops[0] is frames[0]
        before = frames[0].copy()
        _ = ParallelPreprocessor(nhwc=True)(crops)
        untouched = np.array_equal(before, frames[0])
        print(f"  ROI{'있음' if poly else '없음'}: NHWC{tuple(pn.shape)} contig={pn.is_contiguous()} / "
              f"NCHW{tuple(pc.shape)} | 임베딩 비트동일={np.array_equal(en, ec)} | "
              f"crop=원본뷰({view}) 원본무손상={untouched}")

    print("\n② 폴백 경로도 같은 추론기로 들어가는가 (NCHW로 나옴)")
    fr = [rng.integers(0,255,(H,W,3),dtype=np.uint8) for _ in range(8)]
    cases = {
        "torch 입력": [torch.from_numpy(f) for f in fr],
        "float 입력": [f.astype(np.float32)/255 for f in fr],
        "작은 배치(3)": fr[:3],
    }
    for name, imgs in cases.items():
        x = ParallelPreprocessor()(imgs)
        o = np.asarray(m(x))
        print(f"  {name:12s} -> 전처리 {tuple(x.shape)} / 추론 {o.shape} OK")
    z = np.asarray(m(torch.zeros(1,3,336,336)))   # service._init_default_values 의 zero-mask
    print(f"  zero-mask NCHW(1,3,336,336) -> {z.shape} OK (norm={np.linalg.norm(z):.3f})")
    m.dispose()

    print("\n③ e2e 62ch (ROI 없음 / 있음)")
    frames = [rng.integers(0,255,(H,W,3),dtype=np.uint8) for _ in range(62)]
    for poly in (False, True):
        u = ups(62, poly)
        for nc in (8, 4):
            mm = MultiNPUInferenceFull(os.environ["MXQ"], device_ids=list(range(nc)))
            roi = PERoIManager()
            for tag, nhwc in (("NCHW", False), ("NHWC", True)):
                pre = ParallelPreprocessor(nhwc=nhwc)
                t = timed(lambda: mm(pre(roi.process_batches_with_roi(frames, u))), it=3)
                print(f"  ROI{'있음' if poly else '없음'} {nc}카드 {tag}: {t:7.1f} ms", flush=True)
                pre.shutdown()
            mm.dispose()

if __name__ == "__main__":
    main()
