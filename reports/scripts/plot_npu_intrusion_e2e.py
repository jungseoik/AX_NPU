"""npu_intrusion e2e 벤치 차트 (JSON → PNG). 좌: 단계 스택+e2e / 우: 카드 스케일링."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Noto Sans CJK KR"
plt.rcParams["axes.unicode_minus"] = False

J = json.load(open("/home/gpuadmin/AX_NPU/reports/assets/npu_intrusion_e2e_before.json"))
sw = J["sweep"]
ch = [r["ch"] for r in sw]
stages = ["crop", "pre", "infer", "nms", "alarm"]
labels = {"crop": "① ROIcrop (CPU)", "pre": "② Pre letterbox (CPU)",
          "infer": "③ NPU Infer", "nms": "④ NMS (CPU)", "alarm": "⑤ Alarm (CPU)"}
colors = {"crop": "#e15759", "pre": "#f28e2b", "infer": "#4e79a7",
          "nms": "#59a14f", "alarm": "#9c755f"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.6))

# 좌: 단계 스택 + 실제 e2e 라인
bottom = [0] * len(ch)
for s in stages:
    vals = [r[s] for r in sw]
    ax1.bar(range(len(ch)), vals, bottom=bottom, label=labels[s], color=colors[s], width=0.8)
    bottom = [b + v for b, v in zip(bottom, vals)]
ax1.plot(range(len(ch)), [r["e2e"] for r in sw], "k--o", ms=4, lw=1.8,
         label="실제 e2e (detect_batch 스레드병렬)")
ax1.set_xticks(range(len(ch)))
ax1.set_xticklabels(ch, fontsize=8)
ax1.set_xlabel("채널 수 (동시 스트림 프레임)")
ax1.set_ylabel("지연 (ms)")
ax1.set_title("npu_intrusion e2e 단계 분해 (7카드, yolo11n/global4, 1080p)\n"
              "순차합(스택) vs 실제 스레드병렬 e2e — NPU 추론은 최하단 얇은 층")
ax1.legend(fontsize=8, loc="upper left")
ax1.grid(axis="y", alpha=0.3)

# 우: 카드 스케일링 (load=56) — infer는 스케일, e2e는 평평
sc = J["scaling"]
cards = sorted({int(k.split("c_")[0]) for k in sc})
load = 56
e2e = [sc[f"{c}c_{load}"]["e2e"] for c in cards]
inf = [sc[f"{c}c_{load}"]["infer"] for c in cards]
ax2.plot(cards, e2e, "-o", color="#111", lw=2, label=f"e2e-total ({load}ch)")
ax2.plot(cards, inf, "-s", color="#4e79a7", lw=2, label=f"NPU infer ({load}ch)")
for c, y in zip(cards, e2e):
    ax2.annotate(f"{y:.0f}", (c, y), textcoords="offset points", xytext=(0, 8), fontsize=8)
for c, y in zip(cards, inf):
    ax2.annotate(f"{y:.0f}", (c, y), textcoords="offset points", xytext=(0, -14), fontsize=8, color="#4e79a7")
ax2.set_xlabel("NPU 카드 수")
ax2.set_ylabel("지연 (ms)")
ax2.set_xticks(cards)
ax2.set_title("카드 스케일링 (56채널 고정)\n추론은 카드에 비례↓, e2e는 CPU 병목이라 평평")
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)
ax2.set_ylim(0, max(e2e) * 1.25)

plt.tight_layout()
out = "/home/gpuadmin/AX_NPU/reports/assets/npu_intrusion_e2e_before.png"
plt.savefig(out, dpi=110, bbox_inches="tight")
print("saved", out)
