"""HF 실험 경로 <quant>/<tuning>/<scheme>/pe_full.mxq 24종 NPU 검증 — cos + 채널 지연.

GPU 서버에서 컴파일만 하고 넘긴 매트릭스를 이 서버(NPU 보유)에서 정확도·속도로 채점한다.
참조 임베딩은 한 번만 계산해 재사용.
"""
from __future__ import annotations
import json, os, statistics, sys, time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download

sys.path.insert(0, os.environ.get("AX_NPU_ROOT", "/home/gpuadmin/AX_NPU"))
from pe_npu.inference import MXQInferenceFull      # noqa: E402
from pe_npu.pe_model import load_pe                # noqa: E402
from pe_npu.preprocess import preprocess_image     # noqa: E402

REPO = "PIA-SPACE-LAB/MXQ_NPU"
QUANTS = ["W8A16", "W4A16", "W4A8_L5A16"]
TUNINGS = ["none", "sws", "optq", "sws_optq"]
SCHEMES = ["single", "global4"]
CH = [1, 4, 8, 12, 16, 20]
N_COS, REPEAT, DEVICE = 20, 3, 7
OUT = Path("/tmp/w4a16/verify_matrix.json")
REF = Path("/tmp/w4a16/ref_emb.npy")


def cos(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> int:
    paths = sorted(Path("/tmp/w4a16/coco/val2017").glob("*.jpg"))[:max(max(CH), N_COS)]
    batch = np.stack([preprocess_image(p) for p in paths])
    if REF.exists():
        emb_ref = np.load(REF)
    else:
        ref = load_pe(model_name="PE-Core-L14-336", mode="full", patch=True).eval()
        with torch.no_grad():
            emb_ref = np.stack([ref(torch.from_numpy(batch[i:i+1])).cpu().numpy().reshape(-1)
                                for i in range(N_COS)])
        np.save(REF, emb_ref)
    print(f"[ref] {emb_ref.shape}", flush=True)

    results = []
    for q in QUANTS:
        for t in TUNINGS:
            for s in SCHEMES:
                rel = f"{q}/{t}/{s}/pe_full.mxq"
                try:
                    mxq = hf_hub_download(REPO, rel, token=os.environ.get("HF_TOKEN"))
                except Exception as e:
                    print(f"[skip] {rel}: {type(e).__name__}", flush=True); continue
                m = MXQInferenceFull(full_mxq_path=mxq, device_id=DEVICE)
                emb = m.infer(batch[:N_COS])
                cl = [cos(emb_ref[i], emb[i]) for i in range(N_COS)]
                m.infer(batch[:2])
                rows = {}
                for n in CH:
                    ts = []
                    for _ in range(REPEAT):
                        t0 = time.monotonic(); m.infer(batch[:n]); ts.append((time.monotonic()-t0)*1000)
                    rows[n] = round(statistics.median(ts), 1)
                m.dispose()
                rec = {"quant": q, "tuning": t, "scheme": s,
                       "size_mb": round(os.path.getsize(mxq)/1e6, 1),
                       "cos_mean": round(statistics.fmean(cl), 4), "cos_min": round(min(cl), 4),
                       "ms": rows, "img_s_20ch": round(20/(rows[20]/1000), 2)}
                results.append(rec)
                print(f"[done] {q:11s} {t:9s} {s:8s} cos={rec['cos_mean']:.4f} "
                      f"1ch={rows[1]:6.1f} 20ch={rows[20]:7.1f} ({rec['img_s_20ch']:5.2f} img/s)", flush=True)
                OUT.write_text(json.dumps({"results": results}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[save] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
