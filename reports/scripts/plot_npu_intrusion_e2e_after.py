"""npu_intrusion e2e 최적화 before vs after 비교 차트."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Noto Sans CJK KR"
plt.rcParams["axes.unicode_minus"] = False

B = json.load(open("/home/gpuadmin/AX_NPU/reports/assets/npu_intrusion_e2e_before.json"))
A = json.load(open("/home/gpuadmin/AX_NPU/reports/assets/npu_intrusion_e2e_after.json"))
bs = {r["ch"]: r for r in B["sweep"]}
as_ = {r["ch"]: r for r in A["sweep"]}
chs = sorted(set(bs) & set(as_))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.4))

# 1) e2e before vs after
ax1.plot(chs, [bs[c]["e2e"] for c in chs], "-o", color="#e15759", lw=2, ms=4, label="before")
ax1.plot(chs, [as_[c]["e2e"] for c in chs], "-o", color="#4e79a7", lw=2, ms=4, label="after (Tier1)")
ax1.set_xlabel("채널 수"); ax1.set_ylabel("e2e-total (ms)")
ax1.set_title("e2e 지연 before vs after\n64ch 424→121ms (3.5×)")
ax1.legend(); ax1.grid(alpha=0.3)

# 2) 64ch 단계별 그룹 바
stages = ["crop", "pre", "infer", "nms", "alarm"]
labels = ["①ROIcrop", "②Pre", "③Infer", "④NMS", "⑤Alarm"]
bv = [bs[64][s] for s in stages]; av = [as_[64][s] for s in stages]
x = range(len(stages)); w = 0.38
ax2.bar([i - w/2 for i in x], bv, w, color="#e15759", label="before")
ax2.bar([i + w/2 for i in x], av, w, color="#4e79a7", label="after")
for i, (b, a) in enumerate(zip(bv, av)):
    ax2.annotate(f"{b:.0f}", (i - w/2, b), ha="center", va="bottom", fontsize=8)
    ax2.annotate(f"{a:.0f}", (i + w/2, a), ha="center", va="bottom", fontsize=8, color="#4e79a7")
ax2.set_xticks(list(x)); ax2.set_xticklabels(labels, fontsize=9)
ax2.set_ylabel("지연 (ms)"); ax2.set_title("64채널 단계별 before vs after\ncrop·NMS·Pre 대폭↓ (NPU 추론은 동일)")
ax2.legend(); ax2.grid(axis="y", alpha=0.3)

# 3) 카드 스케일링 (56ch) before vs after
cards = sorted({int(k.split("c_")[0]) for k in A["scaling"]})
be = [B["scaling"][f"{c}c_56"]["e2e"] for c in cards]
ae = [A["scaling"][f"{c}c_56"]["e2e"] for c in cards]
ax3.plot(cards, be, "-o", color="#e15759", lw=2, label="before (평평)")
ax3.plot(cards, ae, "-o", color="#4e79a7", lw=2, label="after (스케일↑)")
for c, y in zip(cards, be): ax3.annotate(f"{y:.0f}", (c, y), textcoords="offset points", xytext=(0, 6), fontsize=8)
for c, y in zip(cards, ae): ax3.annotate(f"{y:.0f}", (c, y), textcoords="offset points", xytext=(0, -14), fontsize=8, color="#4e79a7")
ax3.set_xlabel("NPU 카드 수"); ax3.set_ylabel("e2e (ms)"); ax3.set_xticks(cards)
ax3.set_title("카드 스케일링 (56채널)\nCPU 병목 제거로 카드가 다시 먹힘")
ax3.legend(); ax3.grid(alpha=0.3); ax3.set_ylim(0, max(be) * 1.15)

plt.tight_layout()
out = "/home/gpuadmin/AX_NPU/reports/assets/npu_intrusion_e2e_after.png"
plt.savefig(out, dpi=110, bbox_inches="tight")
print("saved", out)
