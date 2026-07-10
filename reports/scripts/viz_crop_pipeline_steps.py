"""기존(cv_crop_region) vs 변경(rect_crop) crop 프로세스를 '단계별로 각각' 펼쳐 비교.
윗줄=기존(4단계, 전체프레임 마스크+bitwise_and 낭비), 아랫줄=변경(1단계 슬라이스).
출력: assets/crop_pipeline_steps.png
"""
import sys, time, numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
plt.rcParams["font.family"] = "Noto Sans CJK KR"
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, "/home/gpuadmin/AX_NPU/Product-AI-mono/packages")
from pia.ai.tasks.OD.models.yolov8.coordinate_utils import calc_expand_coord

ROI = [[1166, 631], [1512, 560], [1810, 867], [1271, 1006], [1071, 833]]
FRAME_WH = (1920, 1080)
ER = calc_expand_coord(roi=ROI, frame_wh=FRAME_WH, expand_ratio=0.1)
reg = np.array(ER); X0, Y0, X1, Y1 = reg[:, 0].min(), reg[:, 1].min(), reg[:, 0].max(), reg[:, 1].max()
VID = "/home/gpuadmin/AX_NPU/Product-AI-mono/assets/videos/kk_helmet_1.mp4"

cap = cv2.VideoCapture(VID); cap.set(cv2.CAP_PROP_POS_FRAMES, 760); ok, frame = cap.read()
if not ok:
    cap.set(cv2.CAP_PROP_POS_FRAMES, 300); ok, frame = cap.read()
cap.release()


def timeit(fn, rep=50):
    for _ in range(5): fn()
    ts = []
    for _ in range(rep):
        s = time.perf_counter(); fn(); ts.append((time.perf_counter() - s) * 1000)
    return float(np.median(ts))


# ===== 기존(cv_crop_region) 내부 단계 재현 =====
def step_mask():
    m = np.zeros((1080, 1920), np.uint8)
    cv2.drawContours(m, [np.array(ER, np.int32)], -1, (255, 255, 255), -1, cv2.LINE_AA)
    return m
mask_full = step_mask()                                   # ② 전체프레임 마스크(rect 채움)
masked_full = cv2.bitwise_and(frame, frame, mask=mask_full)  # ③ bitwise_and (전체프레임)
crop_before = masked_full[Y0:Y1, X0:X1, :]                # ④ slice
# ===== 변경(rect_crop) =====
crop_after = np.ascontiguousarray(frame[Y0:Y1, X0:X1, :]) # slice 한 방

# 시간(단계별)
t_mask = timeit(step_mask)
t_and = timeit(lambda: cv2.bitwise_and(frame, frame, mask=mask_full))
t_slice = timeit(lambda: np.ascontiguousarray(frame[Y0:Y1, X0:X1, :]))
def _before_full():
    m = step_mask()
    return cv2.bitwise_and(frame, frame, mask=m)[Y0:Y1, X0:X1, :]
t_before = timeit(_before_full)
diff = int((cv2.absdiff(crop_before, crop_after).max(2) > 0).sum())
print(f"단계 시간(ms/frame): 마스크생성={t_mask:.2f}  bitwise_and={t_and:.2f}  slice={t_slice:.2f}")
print(f"기존 총={t_before:.2f}  변경 총={t_slice:.2f}  최종 crop 차이 픽셀={diff}")


def rgb(x): return cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
def frame_with_rect():
    f = frame.copy(); cv2.rectangle(f, (X0, Y0), (X1, Y1), (255, 0, 0), 5); return f

fig = plt.figure(figsize=(19, 8.6))
gs = GridSpec(2, 4, figure=fig, hspace=0.28, wspace=0.12)

# ---- 윗줄: 기존 cv_crop_region (4단계) ----
titles_b = [
    f"1) 원본 프레임 1920×1080\n(파랑=자를 rect)",
    f"2) 전체프레임 마스크 생성\ndrawContours로 rect 채움  [+{t_mask:.2f}ms]",
    f"3) bitwise_and (전체프레임)\nmask 밖=0  [+{t_and:.2f}ms]",
    f"4) slice [y0:y1, x0:x1]\n→ 최종 crop  [+{t_slice:.2f}ms]",
]
imgs_b = [rgb(frame_with_rect()), mask_full, rgb(masked_full), rgb(crop_before)]
for j, (im, t) in enumerate(zip(imgs_b, titles_b)):
    ax = fig.add_subplot(gs[0, j])
    ax.imshow(im, cmap=("gray" if j == 1 else None))
    ax.set_title(t, fontsize=10); ax.axis("off")
    if j in (1, 2):  # 낭비 단계 빨강 테두리
        for s in ax.spines.values(): s.set_visible(True); s.set_color("red"); s.set_linewidth(3)
        ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])

# ---- 아랫줄: 변경 rect_crop (1단계) ----
ax0 = fig.add_subplot(gs[1, 0]); ax0.imshow(rgb(frame_with_rect())); ax0.axis("off")
ax0.set_title("1) 원본 프레임 1920×1080\n(파랑=자를 rect)", fontsize=10)
ax1 = fig.add_subplot(gs[1, 1]); ax1.imshow(rgb(crop_after)); ax1.axis("off")
ax1.set_title(f"2) slice [y0:y1, x0:x1]\n→ 최종 crop  [+{t_slice:.2f}ms]  ★한 방", fontsize=10)
# 빈칸에 설명
axn = fig.add_subplot(gs[1, 2:]); axn.axis("off")
axn.text(0.02, 0.5,
         "변경(rect_crop): 마스크 생성·bitwise_and 단계가 통째로 없음.\n"
         f"→ 기존 {t_before:.2f}ms/frame  →  변경 {t_slice:.2f}ms/frame  (약 {t_before/t_slice:.0f}배 빠름)\n\n"
         f"그런데도 기존 4)와 변경 2)의 최종 crop은 픽셀 차이 = {diff}개 (비트 동일).\n"
         "이유: 기존이 마스크를 'rect'로 채워서(폴리곤 아님) 지워지는 픽셀이 없기 때문.",
         fontsize=13, va="center", ha="left")
for s in axn.spines.values(): s.set_visible(False)

fig.text(0.09, 0.955, "기존 (cv_crop_region) — 4단계", fontsize=14, fontweight="bold", color="#b00")
fig.text(0.09, 0.475, "변경 (rect_crop) — 1단계", fontsize=14, fontweight="bold", color="#06c")
plt.suptitle("ROI crop 프로세스 각각 펼쳐보기: 기존 4단계(마스크+and 낭비) vs 변경 1단계(슬라이스) — 최종 결과는 비트 동일",
             fontsize=15, y=1.0)
out = "/home/gpuadmin/AX_NPU/reports/assets/crop_pipeline_steps.png"
plt.savefig(out, dpi=95, bbox_inches="tight")
print("saved", out)
