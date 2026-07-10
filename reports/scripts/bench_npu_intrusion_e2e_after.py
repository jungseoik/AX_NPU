"""npu_intrusion e2e 단계별 지연 벤치 — [최적화 후]. before(bench_npu_intrusion_e2e.py)와 동일 구조.

Tier 1 적용(Product-AI-mono npu_intrusion 모듈):
  ① 후처리 person-only fast-path (detect.postprocess: 단일클래스면 80-class argmax 스킵)
  ② ROI crop 마스킹 제거 → rect 슬라이스 + 스레드 (roi_manager.rect_crop + 풀)
  ③ 전처리/추론 풀 분리 (detect: _cpu_pool 16 for 전처리, _pool for infer, 후처리 순차) + cv2 threads=1
재현: conda activate pe_npu_host && python bench_npu_intrusion_e2e_after.py
"""
import sys, os, time, json, collections
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/home/gpuadmin/AX_NPU/Product-AI-mono/packages")
from pia_prod.AI.modules.npu_intrusion.detect import (
    YOLONPU, detect_npu_devices, preprocess, postprocess, IMG_SIZE)
from pia_prod.AI.modules.npu_intrusion.roi_manager import rect_crop     # [최적화] 마스킹 없는 rect crop
from pia_prod.AI.modules.npu_intrusion.config import (
    NPU_INTRUSION_YOLO_MODEL, NPU_INTRUSION_SCHEME, TARGET_CLASSES,
    OD_CONFIDENCE_THRESHOLD, OD_NMS_THRESHOLD, ROI_EXPAND_RATIO,
    INTRUSION_QUEUE_SIZE, INTRUSION_ALARM_DURATION, cpu_workers)
from pia.vision.roi.roi_manager import cv_crop_region                    # 정확성 대조용(전 방식)
from pia.ai.tasks.OD.models.yolov8.coordinate_utils import calc_expand_coord
from pia.vision.postprocessing.bbox import calc_intersect

VIDEOS = ["/home/gpuadmin/AX_NPU/Product-AI-mono/assets/videos/kk_helmet_1.mp4",
          "/home/gpuadmin/AX_NPU/Product-AI-mono/assets/videos/kk_helmet_2.mp4"]
ROI = [[1166, 631], [1512, 560], [1810, 867], [1271, 1006], [1071, 833]]
FRAME_WH = (1920, 1080)
MAX_CH = 64
REPEAT = 5
WARMUP = 3
OUT_JSON = "/home/gpuadmin/AX_NPU/reports/assets/npu_intrusion_e2e_after.json"

EXPANDED_ROI = calc_expand_coord(roi=ROI, frame_wh=FRAME_WH, expand_ratio=ROI_EXPAND_RATIO)
CROP_ORIGIN = np.array(EXPANDED_ROI)[0]
ROI_IN_CROP = (np.array(ROI) - CROP_ORIGIN).astype(np.int32)


def load_frames(n):
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
    while len(frames) < n:
        frames.append(frames[len(frames) % max(1, len(frames))].copy())
    return frames[:n]


def med(xs):
    return float(np.median(xs))


def bench_stage(det, frames, crop_pool):
    """최적화된 경로로 5단계 + 실제 e2e(threaded crop + detect_batch + alarm) 측정."""
    n = len(frames)
    t_crop, t_pre, t_inf, t_nms, t_alarm, t_e2e = [], [], [], [], [], []
    for _ in range(REPEAT):
        # ① ROIcrop (rect 슬라이스 + 스레드)
        s = time.perf_counter()
        crops = list(crop_pool.map(lambda f: rect_crop(f, EXPANDED_ROI), frames))
        t_crop.append((time.perf_counter() - s) * 1000)
        # ② Pre (CPU 전용 풀)
        s = time.perf_counter()
        pres = list(det._cpu_pool.map(lambda c: preprocess(c, IMG_SIZE, det.bgr2rgb), crops))
        t_pre.append((time.perf_counter() - s) * 1000)
        # ③ Infer
        xs = [p[0] for p in pres]
        s = time.perf_counter()
        outs = list(det._pool.map(lambda i: det._infer(det.models[i % det.n], xs[i]), range(n)))
        t_inf.append((time.perf_counter() - s) * 1000)
        # ④ NMS/후처리 (person-only fast-path, 순차)
        s = time.perf_counter()
        dets = [postprocess(outs[i], pres[i][1], pres[i][2],
                            OD_CONFIDENCE_THRESHOLD, OD_NMS_THRESHOLD, TARGET_CLASSES)
                for i in range(n)]
        t_nms.append((time.perf_counter() - s) * 1000)
        # ⑤ Alarm
        dqs = [collections.deque(maxlen=INTRUSION_QUEUE_SIZE) for _ in range(n)]
        s = time.perf_counter()
        for i in range(n):
            any_in = any(calc_intersect(d[:4], ROI_IN_CROP) for d in dets[i])
            dqs[i].append(1 if any_in else 0)
            _ = int(sum(dqs[i]) >= INTRUSION_ALARM_DURATION)
        t_alarm.append((time.perf_counter() - s) * 1000)
        # 실제 최적화 모듈 경로 e2e
        s = time.perf_counter()
        crops2 = list(crop_pool.map(lambda f: rect_crop(f, EXPANDED_ROI), frames))
        res = det.detect_batch(crops2)
        for i in range(n):
            any_in = any(calc_intersect(d[:4], ROI_IN_CROP) for d in res[i])
            dqs[i].append(1 if any_in else 0)
        t_e2e.append((time.perf_counter() - s) * 1000)
    return dict(crop=med(t_crop), pre=med(t_pre), infer=med(t_inf),
                nms=med(t_nms), alarm=med(t_alarm), e2e=med(t_e2e))


def correctness_check(det, frames):
    """[정확성] 최적화 경로(rect crop + fast-path postprocess) vs 전 방식(mask crop + 80-class)
    검출 결과가 동치인지 확인 — 침입 판정(calc_intersect)이 동일하게 나오는지."""
    same = 0
    for f in frames[:16]:
        # 최적화: rect crop
        crop_new = rect_crop(f, EXPANDED_ROI)
        det_new = det.detect_batch([crop_new])[0]
        # 전 방식: mask crop + 80-class 후처리
        crop_old = cv_crop_region(frame=f, region=EXPANDED_ROI)
        x, r, pad = preprocess(crop_old, IMG_SIZE, det.bgr2rgb)
        o = det._infer(det.models[0], x)
        det_old = postprocess(o, r, pad, OD_CONFIDENCE_THRESHOLD, OD_NMS_THRESHOLD, TARGET_CLASSES)
        alarm_new = any(calc_intersect(d[:4], ROI_IN_CROP) for d in det_new)
        alarm_old = any(calc_intersect(d[:4], ROI_IN_CROP) for d in det_old)
        same += int(alarm_new == alarm_old)
    return same, 16


def main():
    cards_all = detect_npu_devices()
    print(f"[env] cards={cards_all} model={NPU_INTRUSION_YOLO_MODEL} scheme={NPU_INTRUSION_SCHEME} "
          f"cpu_workers={cpu_workers()} host_cores={os.cpu_count()}")
    frames_all = load_frames(MAX_CH)
    print(f"[frames] {len(frames_all)} @ {frames_all[0].shape}")
    result = {"env": {"cards": cards_all, "model": NPU_INTRUSION_YOLO_MODEL,
                      "scheme": NPU_INTRUSION_SCHEME, "cpu_workers": cpu_workers(),
                      "host_cores": os.cpu_count()}}

    det = YOLONPU.load(model=NPU_INTRUSION_YOLO_MODEL, scheme=NPU_INTRUSION_SCHEME,
                       device_ids=cards_all, classes=TARGET_CLASSES, bgr2rgb=True,
                       conf_thres=OD_CONFIDENCE_THRESHOLD, iou_thres=OD_NMS_THRESHOLD)
    crop_pool = ThreadPoolExecutor(max_workers=cpu_workers())

    # 정확성
    same, tot = correctness_check(det, frames_all)
    print(f"[correctness] 침입판정 동치(최적화 vs 전방식): {same}/{tot}")
    result["correctness"] = {"same": same, "total": tot}

    for _ in range(WARMUP):
        det.detect_batch(list(crop_pool.map(lambda f: rect_crop(f, EXPANDED_ROI), frames_all[:8])))
    CHS = list(range(1, 9)) + [10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64]
    sweep = []
    print(f"\n[A] 채널 스윕 (전 {len(cards_all)}카드)  ch | crop pre infer nms alarm | e2e | img/s")
    for n in CHS:
        r = bench_stage(det, frames_all[:n], crop_pool)
        imgs = n / (r["e2e"] / 1000.0)
        r.update(ch=n, imgs=imgs, stage_sum=r["crop"]+r["pre"]+r["infer"]+r["nms"]+r["alarm"])
        sweep.append(r)
        print(f"  {n:3d} | {r['crop']:6.1f} {r['pre']:6.1f} {r['infer']:6.1f} "
              f"{r['nms']:6.1f} {r['alarm']:6.2f} | {r['e2e']:7.1f} | {imgs:6.1f}")
    result["sweep"] = sweep
    det.dispose()

    LOADS = [8, 28, 56]
    NCARDS = [c for c in [1, 2, 4, 7] if c <= len(cards_all)]
    scaling = {}
    print(f"\n[B] 카드 스케일링 (e2e / infer ms)")
    for nc in NCARDS:
        det = YOLONPU.load(model=NPU_INTRUSION_YOLO_MODEL, scheme=NPU_INTRUSION_SCHEME,
                           device_ids=cards_all[:nc], classes=TARGET_CLASSES, bgr2rgb=True,
                           conf_thres=OD_CONFIDENCE_THRESHOLD, iou_thres=OD_NMS_THRESHOLD)
        for _ in range(WARMUP):
            det.detect_batch(list(crop_pool.map(lambda f: rect_crop(f, EXPANDED_ROI), frames_all[:8])))
        for load in LOADS:
            r = bench_stage(det, frames_all[:load], crop_pool)
            scaling[f"{nc}c_{load}"] = {"e2e": r["e2e"], "infer": r["infer"]}
            print(f"  cards={nc} load={load:3d}: e2e={r['e2e']:7.1f}  infer={r['infer']:6.1f}")
        det.dispose()
    result["scaling"] = scaling
    crop_pool.shutdown()

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[done] → {OUT_JSON}")


if __name__ == "__main__":
    main()
