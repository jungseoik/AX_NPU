"""수정안 ROI 크롭이 현행(torch)과 동일한 결과를 내는지 + 최종 임베딩 cos 검증."""
import os, sys
import numpy as np, torch, cv2
sys.path.insert(0, os.environ.get("PIA_PACKAGES", "packages"))  # Product-AI-mono/packages
from pia.vision.roi.roi_manager import batch_crop_region
from pia_prod.AI.modules.pe_npu.roi_manager import PERoIManager
from pia_prod.AI.modules.pe_npu.parallel_preprocess import ParallelPreprocessor
from pia_prod.AI.modules.pe_npu.multi_npu import MultiNPUInferenceFull

MXQ = os.environ["MXQ"]

def params(n, W, H, poly):
    ups = []
    for i in range(n):
        if poly:
            cx, cy, rx, ry = W//2, H//2, int(W*0.35), int(H*0.35)
            pts = []
            for k in range(8):
                a = 2*np.pi*k/8
                pts += [int(cx+rx*np.cos(a)), int(cy+ry*np.sin(a))]
        else:
            pts = []
        ups.append({"user_param": {"cameraId": f"c{i}",
                    "retEvent": {"falldown_ret": {"roi": {"polygonCoordinates": pts}}}}})
    return ups

pre = ParallelPreprocessor()
m = MultiNPUInferenceFull(MXQ, device_ids=[0])
rng = np.random.default_rng(1)

for (W, H) in [(1280, 720), (1920, 1080)]:
    for poly in (False, True):
        n = 8
        frames = [rng.integers(0, 255, (H, W, 3), dtype=np.uint8) for _ in range(n)]
        roi = PERoIManager()
        new_crops = roi.process_batches_with_roi(frames, params(n, W, H, poly))
        # 현행(torch) 경로 재현
        roi2 = PERoIManager()
        _ = roi2.process_batches_with_roi(frames, params(n, W, H, poly))   # roi_dict 채우기
        regions = [roi2.roi_dict[f"c{i}"]["expanded_roi"] for i in range(n)]
        old_crops = batch_crop_region([torch.from_numpy(f) for f in frames], regions)

        # 픽셀 비교 (old: CHW tensor -> HWC numpy)
        diffs, shapes_ok = [], True
        for a, b in zip(new_crops, old_crops):
            bo = b.permute(1, 2, 0).numpy()
            if a.shape != bo.shape:
                shapes_ok = False; diffs.append(float("nan")); continue
            diffs.append(float((a.astype(np.int16) != bo.astype(np.int16)).mean()) * 100)

        e_new = np.asarray(m(pre(new_crops)))
        e_old = np.asarray(m(pre([b.permute(1, 2, 0).numpy() for b in old_crops])))
        cos = float(np.mean(np.sum(e_new*e_old, 1) /
                            (np.linalg.norm(e_new, axis=1)*np.linalg.norm(e_old, axis=1))))
        print(f"{W}x{H} poly={int(poly)} | shape일치={shapes_ok} 픽셀불일치={max(diffs):.4f}% "
              f"| 임베딩 cos(신 vs 현행)={cos:.6f}", flush=True)
m.dispose()
