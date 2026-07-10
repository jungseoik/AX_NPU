"""npu_intrusion 모듈 e2e 단계별 지연 벤치마크 (실측, 실영상 + 대구 ROI).

service._detect 파이프라인을 5단계로 분해해 각 단계 wall-clock을 재고,
실제 모듈 경로(detect_batch 스레드 병렬)의 e2e 총합도 함께 잰다.
  ① ROIcrop  : cv_crop_region(frame, expanded_roi)              [CPU]
  ② Pre      : letterbox 640 + BGR→RGB + /255                    [CPU]
  ③ Infer    : YOLONPU NPU 추론 (카드 라운드로빈 + 스레드 동기)   [NPU]
  ④ NMS      : conf 필터 + xyxy 역변환 + 클래스별 NMS             [CPU]
  ⑤ Alarm    : calc_intersect(bbox∩ROI 마름모4점) + duration 큐  [CPU]

채널(=스트림 프레임) N을 1→64 스윕(전 카드), 카드 1/2/4/7 스케일링(고정 부하).
입력 = kk_helmet_1/2(1920x1080) 실프레임, ROI = 대구 침입 폴리곤.
재현: conda activate pe_npu_host && python bench_npu_intrusion_e2e.py
"""
import sys, os, time, json, collections
import numpy as np
import cv2

sys.path.insert(0, "/home/gpuadmin/AX_NPU/Product-AI-mono/packages")
from pia_prod.AI.modules.npu_intrusion.detect import (
    YOLONPU, detect_npu_devices, preprocess, postprocess, IMG_SIZE,
)
from pia_prod.AI.modules.npu_intrusion.config import (
    NPU_INTRUSION_YOLO_MODEL, NPU_INTRUSION_SCHEME, TARGET_CLASSES,
    OD_CONFIDENCE_THRESHOLD, OD_NMS_THRESHOLD, ROI_EXPAND_RATIO,
    INTRUSION_QUEUE_SIZE, INTRUSION_ALARM_DURATION,
)
from pia.vision.roi.roi_manager import cv_crop_region
from pia.ai.tasks.OD.models.yolov8.coordinate_utils import calc_expand_coord
from pia.vision.postprocessing.bbox import calc_intersect

VIDEOS = ["/home/gpuadmin/AX_NPU/Product-AI-mono/assets/videos/kk_helmet_1.mp4",
          "/home/gpuadmin/AX_NPU/Product-AI-mono/assets/videos/kk_helmet_2.mp4"]
ROI = [[1166, 631], [1512, 560], [1810, 867], [1271, 1006], [1071, 833]]  # 대구 침입 ROI
FRAME_WH = (1920, 1080)
MAX_CH = 64
REPEAT = 5
WARMUP = 3
OUT_JSON = "/home/gpuadmin/AX_NPU/reports/assets/npu_intrusion_e2e_before.json"

# ROI 크롭/교집합 좌표 (roi_manager 재현: 캐시된 steady-state 값)
EXPANDED_ROI = calc_expand_coord(roi=ROI, frame_wh=FRAME_WH, expand_ratio=ROI_EXPAND_RATIO)
CROP_ORIGIN = np.array(EXPANDED_ROI)[0]
ROI_IN_CROP = (np.array(ROI) - CROP_ORIGIN).astype(np.int32)


def load_frames(n):
    """서로 다른 실프레임 n장 확보 (두 영상에서 균등 샘플, 부족하면 순환)."""
    frames = []
    for path in VIDEOS:
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total // (n // len(VIDEOS) + 2))
        idx = 0
        while len(frames) < n and idx < total:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, fr = cap.read()
            if ok:
                frames.append(fr)
            idx += step
        cap.release()
        if len(frames) >= n:
            break
    while len(frames) < n:            # 순환 채우기
        frames.append(frames[len(frames) % max(1, len(frames))].copy())
    return frames[:n]


def med(xs):
    return float(np.median(xs))


def bench_stage(det, frames):
    """한 배치(len=frames)에 대해 5단계 각각 REPEAT median(ms) + 실제 detect_batch e2e median(ms)."""
    n = len(frames)
    t_crop, t_pre, t_inf, t_nms, t_alarm, t_e2e = [], [], [], [], [], []
    pool = det._pool
    for _ in range(REPEAT):
        # ① ROIcrop
        s = time.perf_counter()
        crops = [cv_crop_region(f, EXPANDED_ROI) for f in frames]
        t_crop.append((time.perf_counter() - s) * 1000)
        # ② Pre
        s = time.perf_counter()
        pres = [preprocess(c, IMG_SIZE, True) for c in crops]
        t_pre.append((time.perf_counter() - s) * 1000)
        # ③ Infer (카드 라운드로빈 + 스레드풀 동기 infer = 검증된 다채널 패턴)
        xs = [p[0] for p in pres]
        s = time.perf_counter()
        outs = list(pool.map(lambda i: det._infer(det.models[i % det.n], xs[i]), range(n)))
        t_inf.append((time.perf_counter() - s) * 1000)
        # ④ NMS/후처리
        s = time.perf_counter()
        dets = [postprocess(outs[i], pres[i][1], pres[i][2],
                            OD_CONFIDENCE_THRESHOLD, OD_NMS_THRESHOLD, TARGET_CLASSES)
                for i in range(n)]
        t_nms.append((time.perf_counter() - s) * 1000)
        # ⑤ Alarm (calc_intersect + duration 큐 per stream)
        dqs = [collections.deque(maxlen=INTRUSION_QUEUE_SIZE) for _ in range(n)]
        s = time.perf_counter()
        for i in range(n):
            any_in = any(calc_intersect(d[:4], ROI_IN_CROP) for d in dets[i])
            dqs[i].append(1 if any_in else 0)
            _ = int(sum(dqs[i]) >= INTRUSION_ALARM_DURATION)
        t_alarm.append((time.perf_counter() - s) * 1000)
        # 실제 모듈 경로 e2e (crop → detect_batch(pre+infer+nms 스레드병렬) → alarm)
        s = time.perf_counter()
        crops2 = [cv_crop_region(f, EXPANDED_ROI) for f in frames]
        res = det.detect_batch(crops2)
        for i in range(n):
            any_in = any(calc_intersect(d[:4], ROI_IN_CROP) for d in res[i])
            dqs[i].append(1 if any_in else 0)
        t_e2e.append((time.perf_counter() - s) * 1000)
    return dict(crop=med(t_crop), pre=med(t_pre), infer=med(t_inf),
                nms=med(t_nms), alarm=med(t_alarm), e2e=med(t_e2e))


def main():
    cards_all = detect_npu_devices()
    print(f"[env] NPU cards={cards_all} model={NPU_INTRUSION_YOLO_MODEL} "
          f"scheme={NPU_INTRUSION_SCHEME} conf={OD_CONFIDENCE_THRESHOLD} iou={OD_NMS_THRESHOLD}")
    print(f"[roi] expanded_roi={EXPANDED_ROI.tolist()} crop_hw≈"
          f"{EXPANDED_ROI[:,1].max()-EXPANDED_ROI[:,1].min()}x{EXPANDED_ROI[:,0].max()-EXPANDED_ROI[:,0].min()}")
    frames_all = load_frames(MAX_CH)
    print(f"[frames] {len(frames_all)} distinct frames @ {frames_all[0].shape}")

    result = {"env": {"cards": cards_all, "model": NPU_INTRUSION_YOLO_MODEL,
                      "scheme": NPU_INTRUSION_SCHEME, "conf": OD_CONFIDENCE_THRESHOLD,
                      "iou": OD_NMS_THRESHOLD, "roi_expand": ROI_EXPAND_RATIO,
                      "queue": INTRUSION_QUEUE_SIZE, "alarm_dur": INTRUSION_ALARM_DURATION,
                      "expanded_roi": EXPANDED_ROI.tolist(), "frame_wh": FRAME_WH}}

    # ---- A. 채널 스윕 (전 카드) ----
    det = YOLONPU.load(model=NPU_INTRUSION_YOLO_MODEL, scheme=NPU_INTRUSION_SCHEME,
                       device_ids=cards_all, classes=TARGET_CLASSES, bgr2rgb=True,
                       conf_thres=OD_CONFIDENCE_THRESHOLD, iou_thres=OD_NMS_THRESHOLD)
    # warmup
    for _ in range(WARMUP):
        det.detect_batch([cv_crop_region(f, EXPANDED_ROI) for f in frames_all[:8]])
    CHS = list(range(1, 9)) + [10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64]
    sweep = []
    print(f"\n[A] 채널 스윕 (전 {len(cards_all)}카드)  ch | crop pre infer nms alarm | e2e-total | img/s")
    for n in CHS:
        r = bench_stage(det, frames_all[:n])
        imgs = n / (r["e2e"] / 1000.0)
        r.update(ch=n, imgs=imgs, stage_sum=r["crop"]+r["pre"]+r["infer"]+r["nms"]+r["alarm"])
        sweep.append(r)
        print(f"  {n:3d} | {r['crop']:6.1f} {r['pre']:6.1f} {r['infer']:6.1f} "
              f"{r['nms']:6.1f} {r['alarm']:6.2f} | {r['e2e']:7.1f} | {imgs:6.1f}")
    result["sweep"] = sweep
    det.dispose()

    # ---- B. 카드 스케일링 (고정 부하) ----
    LOADS = [8, 28, 56]
    NCARDS = [c for c in [1, 2, 4, 7] if c <= len(cards_all)]
    scaling = {}
    print(f"\n[B] 카드 스케일링 (e2e-total ms / infer ms)")
    for nc in NCARDS:
        det = YOLONPU.load(model=NPU_INTRUSION_YOLO_MODEL, scheme=NPU_INTRUSION_SCHEME,
                           device_ids=cards_all[:nc], classes=TARGET_CLASSES, bgr2rgb=True,
                           conf_thres=OD_CONFIDENCE_THRESHOLD, iou_thres=OD_NMS_THRESHOLD)
        for _ in range(WARMUP):
            det.detect_batch([cv_crop_region(f, EXPANDED_ROI) for f in frames_all[:8]])
        for load in LOADS:
            r = bench_stage(det, frames_all[:load])
            scaling[f"{nc}c_{load}"] = {"e2e": r["e2e"], "infer": r["infer"]}
            print(f"  cards={nc} load={load:3d}: e2e={r['e2e']:7.1f}  infer={r['infer']:6.1f}")
        det.dispose()
    result["scaling"] = scaling

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[done] → {OUT_JSON}")


if __name__ == "__main__":
    main()
