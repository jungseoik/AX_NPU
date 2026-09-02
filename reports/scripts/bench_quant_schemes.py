"""양자화 스킴 × 코어모드 12조합 — 정확도(cos)와 1~20채널 배치 지연 측정.

원본 PyTorch 참조 임베딩은 한 번만 계산해 재사용한다(설정마다 다시 돌리면 CPU에서 매우 느리다).
"""
from __future__ import annotations
import json, os, statistics, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.environ.get("AX_NPU_ROOT", "/home/gpuadmin/AX_NPU"))
from pe_npu.inference import MXQInferenceFull      # noqa: E402
from pe_npu.pe_model import load_pe                # noqa: E402
from pe_npu.preprocess import preprocess_image     # noqa: E402

IMAGES = "/tmp/w4a16/coco/val2017"
MXQ_DIR = Path("/tmp/w4a16/mxq")
OUT = Path("/tmp/w4a16/bench_all.json")
REF_CACHE = Path("/tmp/w4a16/ref_emb.npy")
DEVICE = 7
CH = list(range(1, 21))
N_COS = 20
REPEAT = 3
MODES = ["single", "multi", "global4", "global8"]


def cos(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def hf_w8a16(mode: str) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download("PIA-SPACE-LAB/MXQ_NPU", f"{mode}/pe_full.mxq")


def main() -> int:
    paths = sorted(Path(IMAGES).glob("*.jpg"))[:max(max(CH), N_COS)]
    batch = np.stack([preprocess_image(p) for p in paths])

    if REF_CACHE.exists():
        emb_ref = np.load(REF_CACHE)
        print(f"[ref] 캐시 사용 {emb_ref.shape}", flush=True)
    else:
        ref = load_pe(model_name="PE-Core-L14-336", mode="full", patch=True).eval()
        t = time.monotonic()
        with torch.no_grad():
            emb_ref = np.stack([ref(torch.from_numpy(batch[i:i + 1])).cpu().numpy().reshape(-1)
                                for i in range(N_COS)])
        np.save(REF_CACHE, emb_ref)
        print(f"[ref] 원본 PyTorch 임베딩 {emb_ref.shape} ({time.monotonic()-t:.0f}s)", flush=True)

    results = []
    for quant in ["W8A16", "W4A16", "W4A8_L5A16"]:
        for mode in MODES:
            mxq = hf_w8a16(mode) if quant == "W8A16" else str(MXQ_DIR / f"{quant}_{mode}.mxq")
            if not os.path.exists(mxq):
                print(f"[skip] {quant}/{mode} — 파일 없음", flush=True)
                continue
            m = MXQInferenceFull(full_mxq_path=mxq, device_id=DEVICE)
            emb = m.infer(batch[:N_COS])
            cos_list = [cos(emb_ref[i], emb[i]) for i in range(N_COS)]

            m.infer(batch[:2])                                     # 워밍업
            rows = []
            for n in CH:
                ts = []
                for _ in range(REPEAT):
                    t = time.monotonic()
                    m.infer(batch[:n])
                    ts.append((time.monotonic() - t) * 1000)
                tot = statistics.median(ts)
                rows.append({"ch": n, "total_ms": round(tot, 1),
                             "per_img_ms": round(tot / n, 1), "img_s": round(n / (tot / 1000), 2)})
            m.dispose()

            rec = {"quant": quant, "mode": mode, "mxq": mxq,
                   "size_mb": round(os.path.getsize(mxq) / 1e6, 1),
                   "cos_mean": round(statistics.fmean(cos_list), 4),
                   "cos_min": round(min(cos_list), 4),
                   "ch1_ms": rows[0]["total_ms"], "ch20_ms": rows[-1]["total_ms"],
                   "ch20_img_s": rows[-1]["img_s"], "rows": rows}
            results.append(rec)
            print(f"[done] {quant:11s} {mode:8s} size={rec['size_mb']:6.1f}MB cos={rec['cos_mean']:.4f} "
                  f"1ch={rec['ch1_ms']:6.1f}ms 20ch={rec['ch20_ms']:7.1f}ms "
                  f"({rec['ch20_img_s']:.2f} img/s)", flush=True)

    OUT.write_text(json.dumps({"device": DEVICE, "repeat": REPEAT, "n_cos": N_COS,
                               "results": results}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[save] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
