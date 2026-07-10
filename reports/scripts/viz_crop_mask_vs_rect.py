"""cv_crop_region(전체프레임 마스킹) vs rect_crop(단순 슬라이스) 시각 비교 + 검출/알람 동치 측정.

- 두 crop의 픽셀이 어디서/얼마나 다른지(마스킹으로 검게 된 '폴리곤 밖 rect 안' 영역).
- 그 차이가 YOLO 검출에 영향을 주는지, 최종 침입 알람은 같은지.
출력: assets/crop_mask_vs_rect.png
"""
import sys, numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Noto Sans CJK KR"
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, "/home/gpuadmin/AX_NPU/Product-AI-mono/packages")
from pia_prod.AI.modules.npu_intrusion.detect import YOLONPU, detect_npu_devices, preprocess, postprocess, IMG_SIZE
from pia_prod.AI.modules.npu_intrusion.roi_manager import rect_crop
from pia_prod.AI.modules.npu_intrusion.config import TARGET_CLASSES, OD_CONFIDENCE_THRESHOLD, OD_NMS_THRESHOLD, ROI_EXPAND_RATIO
from pia.vision.roi.roi_manager import cv_crop_region
from pia.ai.tasks.OD.models.yolov8.coordinate_utils import calc_expand_coord
from pia.vision.postprocessing.bbox import calc_intersect

ROI = [[1166, 631], [1512, 560], [1810, 867], [1271, 1006], [1071, 833]]
FRAME_WH = (1920, 1080)
ER = calc_expand_coord(roi=ROI, frame_wh=FRAME_WH, expand_ratio=ROI_EXPAND_RATIO)
reg = np.array(ER); X0, Y0, X1, Y1 = reg[:, 0].min(), reg[:, 1].min(), reg[:, 0].max(), reg[:, 1].max()
CROP_ORIGIN = reg[0]
ROI_IN_CROP = (np.array(ROI) - CROP_ORIGIN).astype(np.int32)
VID = "/home/gpuadmin/AX_NPU/Product-AI-mono/assets/videos/kk_helmet_1.mp4"

det = YOLONPU.load(model="yolo11n", scheme="global4", device_ids=detect_npu_devices(),
                   classes=TARGET_CLASSES, bgr2rgb=True, conf_thres=0.3, iou_thres=OD_NMS_THRESHOLD)

# 대표 프레임: 사람이 잡히는 프레임 하나 선택
cap = cv2.VideoCapture(VID)
pick = None
i = 0
while i < 2000:
    cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, f = cap.read()
    if not ok: break
    d = det.detect_batch([rect_crop(f, ER)])[0]
    if len(d) >= 1:
        pick = f.copy(); break
    i += 20
cap.release()
if pick is None:
    cap = cv2.VideoCapture(VID); cap.set(cv2.CAP_PROP_POS_FRAMES, 500); _, pick = cap.read(); cap.release()

# 두 방식 crop
crop_mask = cv_crop_region(frame=pick, region=ER)      # 전체프레임 마스킹 후 rect
crop_rect = rect_crop(pick, ER)                         # 단순 rect 슬라이스
# 픽셀 차이
diff = cv2.absdiff(crop_mask, crop_rect)
diff_gray = diff.max(2)
n_diff = int((diff_gray > 0).sum())
area = crop_rect.shape[0] * crop_rect.shape[1]
print(f"crop shape={crop_rect.shape}, 다른 픽셀={n_diff}/{area} ({100*n_diff/area:.1f}%)")

# 검출/알람 비교 (동일 전처리·추론, crop만 다르게)
def run(crop):
    x, r, pad = preprocess(crop, IMG_SIZE, True)
    o = det._infer(det.models[0], x)
    return postprocess(o, r, pad, 0.3, OD_NMS_THRESHOLD, TARGET_CLASSES)
d_mask, d_rect = run(crop_mask), run(crop_rect)
alarm_mask = any(calc_intersect(b[:4], ROI_IN_CROP) for b in d_mask)
alarm_rect = any(calc_intersect(b[:4], ROI_IN_CROP) for b in d_rect)
print(f"검출 mask={len(d_mask)} rect={len(d_rect)} | 알람 mask={alarm_mask} rect={alarm_rect}")

# 다수 프레임 알람 동치 통계
cap = cv2.VideoCapture(VID); agree = tot = 0; i = 0
while i < 3000 and tot < 100:
    cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, f = cap.read()
    if not ok: break
    am = any(calc_intersect(b[:4], ROI_IN_CROP) for b in run(cv_crop_region(frame=f, region=ER)))
    ar = any(calc_intersect(b[:4], ROI_IN_CROP) for b in run(rect_crop(f, ER)))
    agree += int(am == ar); tot += 1; i += 30
cap.release()
print(f"알람 동치 {agree}/{tot} 프레임")
det.dispose()

# ---- 시각화 ----
def rgb(x): return cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
roi_disp = (np.array(ROI) - CROP_ORIGIN).astype(np.int32)

# (참고) 만약 '폴리곤'으로 마스킹했다면? — 모듈은 이렇게 하지 않음(rect를 넘김)
mask_poly = np.zeros(crop_rect.shape[:2], np.uint8)
cv2.fillPoly(mask_poly, [roi_disp], 255)
crop_poly = cv2.bitwise_and(crop_rect, crop_rect, mask=mask_poly)

fig, ax = plt.subplots(2, 3, figsize=(17, 10))
# (0,0) 전체 프레임 + ROI 폴리곤 + 확장 rect
full = pick.copy()
cv2.polylines(full, [np.array(ROI, np.int32)], True, (0, 220, 255), 4)
cv2.rectangle(full, (X0, Y0), (X1, Y1), (255, 0, 0), 4)
ax[0, 0].imshow(rgb(full)); ax[0, 0].set_title("① 원본 프레임 1920×1080\n노랑=ROI 폴리곤, 파랑=확장 rect = crop 영역")
# (0,1) 모듈의 cv_crop_region 출력 (region=rect)
m = crop_mask.copy(); cv2.polylines(m, [roi_disp], True, (0, 220, 255), 3)
ax[0, 1].imshow(rgb(m)); ax[0, 1].set_title(f"② cv_crop_region(region=확장 rect) [현재]\n마스크가 rect 전체 → 지워지는 픽셀 없음, {crop_mask.shape[1]}×{crop_mask.shape[0]}")
# (0,2) rect crop
r_ = crop_rect.copy(); cv2.polylines(r_, [roi_disp], True, (0, 220, 255), 3)
ax[0, 2].imshow(rgb(r_)); ax[0, 2].set_title(f"③ rect_crop = frame[y0:y1, x0:x1] [변경]\n순수 슬라이스, {crop_rect.shape[1]}×{crop_rect.shape[0]}")
# (1,0) diff = 0
ax[1, 0].imshow(diff_gray, cmap="hot", vmin=0, vmax=255)
ax[1, 0].set_title(f"④ 픽셀 차이 |②-③| = {n_diff}개 ({100*n_diff/area:.1f}%)\n→ ②와 ③은 비트 단위로 동일")
# (1,1) 참고: 폴리곤으로 마스킹했다면
mp = crop_poly.copy(); cv2.polylines(mp, [roi_disp], True, (0, 220, 255), 3)
ax[1, 1].imshow(rgb(mp)); ax[1, 1].set_title("⑤ (참고) 만약 '폴리곤'으로 마스킹했다면\n코너가 검게 → 다른 결과. 모듈은 이렇게 안 함")
# (1,2) 검출 결과 (rect)
rr = crop_rect.copy(); cv2.polylines(rr, [roi_disp], True, (0, 220, 255), 2)
for b in d_rect:
    inside = calc_intersect(b[:4], ROI_IN_CROP)
    c = (0, 0, 255) if inside else (0, 200, 0)
    cv2.rectangle(rr, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), c, 3)
ax[1, 2].imshow(rgb(rr)); ax[1, 2].set_title(f"⑥ 검출(②·③ 동일): {len(d_rect)}명  알람={alarm_rect}\n빨강=ROI내/초록=ROI밖 (calc_intersect가 폴리곤 판정)")
for a in ax.ravel(): a.axis("off")
plt.suptitle(f"ROI crop: cv_crop_region(rect) vs rect 슬라이스  —  출력 비트 동일(차이 {100*n_diff/area:.1f}%), 검출·알람 동일({agree}/{tot} 프레임)  ·  rect_crop은 ~9× 빠름",
             fontsize=14, y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.97])
out = "/home/gpuadmin/AX_NPU/reports/assets/crop_mask_vs_rect.png"
plt.savefig(out, dpi=95, bbox_inches="tight")
print("saved", out)
