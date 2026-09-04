"""양자화×튜닝 매트릭스 NPU 검증 — cos + 채널 지연.

NPU 없는 컴파일 서버(GPU)에서 만든 mxq를 이 서버(NPU 보유)에서 정확도·속도로 채점한다.
참조 임베딩(원본 PyTorch)은 한 번 계산해 캐시하고 재사용한다.

두 가지 소스를 지원한다.

    # (권장) 로컬 디렉토리 — GPU 서버에서 scp/rsync로 받아온 mxq들
    python reports/scripts/verify_quant_tuning_matrix.py --src-dir out/matrix_120 --device 0

    # HF 실험 경로 <quant>/<tuning>/<scheme>/pe_full.mxq
    python reports/scripts/verify_quant_tuning_matrix.py --from-hf --device 0

로컬 디렉토리 모드는 파일명에서 조합을 읽는다(구분자 무관, 부분문자열 매칭):

    pe_W4A16_sws_optq_single.mxq  →  quant=W4A16  tuning=sws_optq  scheme=single

관련: reports/RUNBOOK_quant_matrix_120.md (재현 절차 + 기대값)
      reports/performance/NPU_pe_quant_tuning_compiler_version.md (1.1.2 vs 1.2.0)
"""
from __future__ import annotations
import argparse, json, os, statistics, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.environ.get("AX_NPU_ROOT", str(Path(__file__).resolve().parents[2])))
from pe_npu.inference import MXQInferenceFull      # noqa: E402
from pe_npu.pe_model import load_pe                # noqa: E402
from pe_npu.preprocess import preprocess_image     # noqa: E402

REPO = "PIA-SPACE-LAB/MXQ_NPU"
QUANTS = ["W8A16", "W4A16", "W4A8_L5A16"]
# tuning 이름은 긴 것부터 봐야 한다 — 'sws_optq'가 'sws'/'optq'로 잘못 잡히는 것을 막는다.
TUNINGS = ["sws_optq", "optq", "sws", "none"]
SCHEMES = ["global4", "global8", "single", "multi"]

#: 1.2.0 매트릭스 24종 NPU 실측(2026-09-04, GPU 컴파일 + 표준 이미지셋 download/coco/val2017 앞 20장).
#: 재컴파일 후 회귀 판정용. → reports/performance/NPU_pe_quant_tuning_matrix_120.md
BASELINE_120 = {
    ("W8A16", "none", "single"): 0.9937,
    ("W8A16", "none", "multi"): 0.9937,
    ("W8A16", "none", "global4"): 0.9937,
    ("W8A16", "none", "global8"): 0.9937,
    ("W8A16", "sws_optq", "single"): 0.9946,
    ("W8A16", "sws_optq", "multi"): 0.9946,
    ("W8A16", "sws_optq", "global4"): 0.9946,
    ("W8A16", "sws_optq", "global8"): 0.9946,
    ("W4A16", "none", "single"): 0.9175,
    ("W4A16", "none", "multi"): 0.9173,
    ("W4A16", "none", "global4"): 0.9175,
    ("W4A16", "none", "global8"): 0.9175,
    ("W4A16", "sws_optq", "single"): 0.9654,
    ("W4A16", "sws_optq", "multi"): 0.9654,
    ("W4A16", "sws_optq", "global4"): 0.9654,
    ("W4A16", "sws_optq", "global8"): 0.9654,
    ("W4A8_L5A16", "none", "single"): 0.8884,
    ("W4A8_L5A16", "none", "multi"): 0.8884,
    ("W4A8_L5A16", "none", "global4"): 0.8884,
    ("W4A8_L5A16", "none", "global8"): 0.8884,
    ("W4A8_L5A16", "sws_optq", "single"): 0.9420,
    ("W4A8_L5A16", "sws_optq", "multi"): 0.9420,
    ("W4A8_L5A16", "sws_optq", "global4"): 0.9420,
    ("W4A8_L5A16", "sws_optq", "global8"): 0.9420,
}
TOLERANCE = 0.005


def cos(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def _match(name: str, candidates: list[str]) -> str | None:
    """파일명에서 조합 이름을 찾는다(긴 이름 우선)."""
    low = name.lower()
    for c in candidates:
        if c.lower() in low:
            return c
    return None


def discover_local(src_dir: Path) -> list[tuple[str, str, str, Path]]:
    """디렉토리의 *.mxq 를 (quant, tuning, scheme, path)로 분류."""
    out = []
    for p in sorted(src_dir.rglob("*.mxq")):
        # 경로 전체를 보면 <quant>/<tuning>/<scheme>/pe_full.mxq 구조도 잡힌다.
        key = str(p.relative_to(src_dir))
        q, t, s = _match(key, QUANTS), _match(key, TUNINGS), _match(key, SCHEMES)
        if not (q and t and s):
            print(f"[skip] 조합 판별 실패: {key} (quant={q} tuning={t} scheme={s})")
            continue
        out.append((q, t, s, p))
    return out


def discover_hf(tunings, schemes) -> list[tuple[str, str, str, Path]]:
    from huggingface_hub import hf_hub_download
    out = []
    for q in QUANTS:
        for t in tunings:
            for s in schemes:
                rel = f"{q}/{t}/{s}/pe_full.mxq"
                try:
                    out.append((q, t, s, Path(hf_hub_download(REPO, rel,
                                                              token=os.environ.get("HF_TOKEN")))))
                except Exception as e:
                    print(f"[skip] {rel}: {type(e).__name__}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--src-dir", type=Path, help="mxq 가 있는 로컬 디렉토리(재귀 탐색)")
    src.add_argument("--from-hf", action="store_true",
                     help=f"HF {REPO} 의 <quant>/<tuning>/<scheme>/pe_full.mxq 에서 받는다")
    ap.add_argument("--device", type=int, default=0, help="NPU device id (/dev/ariesN)")
    ap.add_argument("--coco-dir", type=Path, default=None,
                    help="검증 이미지 폴더(jpg). 미지정 시 download/coco/val2017")
    ap.add_argument("--n-cos", type=int, default=20, help="cos 측정 장수")
    ap.add_argument("--channels", default="1,4,8,12,16,20",
                    help="지연 측정 채널(배치) 목록")
    ap.add_argument("--repeat", type=int, default=3, help="채널별 반복(median)")
    ap.add_argument("--tunings", default=None, help="--from-hf 에서만: 쉼표 구분")
    ap.add_argument("--schemes", default=None, help="--from-hf 에서만: 쉼표 구분")
    ap.add_argument("--out", type=Path, default=Path("reports/assets/verify_matrix.json"))
    ap.add_argument("--ref-cache", type=Path, default=Path("/tmp/pe_ref_emb.npz"),
                    help="참조 임베딩 캐시(.npz). 이미지 목록이 다르면 자동 재계산")
    a = ap.parse_args()

    ch = [int(x) for x in a.channels.split(",") if x.strip()]
    coco = a.coco_dir or Path("download/coco/val2017")
    paths = sorted(coco.glob("*.jpg"))[:max(max(ch), a.n_cos)]
    if len(paths) < max(max(ch), a.n_cos):
        raise SystemExit(f"이미지 부족: {coco} 에 {len(paths)}장 "
                         f"(필요 {max(max(ch), a.n_cos)}장) — --coco-dir 확인")
    batch = np.stack([preprocess_image(p) for p in paths])

    # 참조 임베딩은 "이 이미지들"에 대한 값이다. 캐시를 다른 --coco-dir 로 재사용하면
    # cos 가 조용히 무의미해진다(실제로 겪었다: W8A16 none 이 0.9937 대신 0.3396).
    # → 캐시에 이미지 이름 목록을 함께 저장하고, 어긋나면 재계산한다.
    ref_key = [p.name for p in paths[:a.n_cos]]
    emb_ref = None
    if a.ref_cache.exists():
        try:
            z = np.load(a.ref_cache, allow_pickle=True)
            if isinstance(z, np.ndarray):          # 구 포맷(.npy, 키 없음) — 신뢰할 수 없다
                print(f"[ref] 구 캐시({a.ref_cache})는 이미지 목록이 없어 재계산한다", flush=True)
            elif list(z["images"]) != ref_key:
                print(f"[ref] 캐시가 다른 이미지 세트 기준 → 재계산 "
                      f"(캐시 {list(z['images'])[0]}… vs 지금 {ref_key[0]}…)", flush=True)
            else:
                emb_ref = z["emb"]
                print(f"[ref] 캐시 재사용 {a.ref_cache} {emb_ref.shape}", flush=True)
        except Exception as e:
            print(f"[ref] 캐시 읽기 실패({type(e).__name__}) → 재계산", flush=True)
    if emb_ref is None:
        print("[ref] 원본 PyTorch 임베딩 계산 중(패치 모델은 batch=1 전용이라 1장씩)", flush=True)
        ref = load_pe(model_name="PE-Core-L14-336", mode="full", patch=True).eval()
        with torch.no_grad():
            emb_ref = np.stack([ref(torch.from_numpy(batch[i:i + 1])).cpu().numpy().reshape(-1)
                                for i in range(a.n_cos)])
        a.ref_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(a.ref_cache, emb=emb_ref, images=np.array(ref_key))
        print(f"[ref] {emb_ref.shape} → {a.ref_cache} (이미지 목록 동봉)", flush=True)

    if a.src_dir:
        jobs = discover_local(a.src_dir)
    else:
        jobs = discover_hf(
            [t.strip() for t in (a.tunings or ",".join(TUNINGS)).split(",") if t.strip()],
            [s.strip() for s in (a.schemes or ",".join(SCHEMES)).split(",") if s.strip()])
    if not jobs:
        raise SystemExit("검증할 mxq 가 없다")
    print(f"[plan] {len(jobs)}개 조합, device={a.device}", flush=True)

    results, drift = [], []
    for q, t, s, mxq in jobs:
        m = MXQInferenceFull(full_mxq_path=str(mxq), device_id=a.device)
        emb = m.infer(batch[:a.n_cos])
        cl = [cos(emb_ref[i], emb[i]) for i in range(a.n_cos)]
        m.infer(batch[:2])                      # warmup
        rows = {}
        for n in ch:
            ts = []
            for _ in range(a.repeat):
                t0 = time.monotonic(); m.infer(batch[:n]); ts.append((time.monotonic() - t0) * 1000)
            rows[n] = round(statistics.median(ts), 1)
        m.dispose()

        cm = round(statistics.fmean(cl), 4)
        rec = {"quant": q, "tuning": t, "scheme": s, "mxq": str(mxq),
               "size_mb": round(os.path.getsize(mxq) / 1e6, 1),
               "cos_mean": cm, "cos_min": round(min(cl), 4), "ms": rows,
               "img_s_max_ch": round(max(ch) / (rows[max(ch)] / 1000), 2)}

        flag = ""
        expected = BASELINE_120.get((q, t, s))
        if expected is not None:
            delta = cm - expected
            rec["baseline_120"] = expected
            rec["baseline_delta"] = round(delta, 4)
            if abs(delta) > TOLERANCE:
                flag = f"  ⚠ 기준 {expected:.4f} 대비 {delta:+.4f}"
                drift.append((q, t, s, expected, cm))
            else:
                flag = f"  ✓ 기준 {expected:.4f}"
        results.append(rec)
        print(f"[done] {q:11s} {t:9s} {s:8s} cos={cm:.4f} "
              f"1ch={rows[ch[0]]:6.1f} {max(ch)}ch={rows[max(ch)]:7.1f} "
              f"({rec['img_s_max_ch']:6.2f} img/s){flag}", flush=True)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"results": results}, ensure_ascii=False, indent=1),
                         encoding="utf-8")

    print(f"\n[save] {a.out}")
    print("\n| 양자화 | 튜닝 | 모드 | 크기 | cos | " + " | ".join(f"{n}ch" for n in ch) + " |")
    print("| --- | --- | --- | ---: | ---: | " + " | ".join("---:" for _ in ch) + " |")
    for r in results:
        print(f"| {r['quant']} | {r['tuning']} | {r['scheme']} | {r['size_mb']} | "
              f"{r['cos_mean']:.4f} | " + " | ".join(str(r['ms'][n]) for n in ch) + " |")

    if drift:
        print(f"\n⚠ 회귀 기준({TOLERANCE}) 이탈 {len(drift)}건 — 설정 차이를 확인할 것:")
        for q, t, s, exp, got in drift:
            print(f"   {q} {t} {s}: 기대 {exp:.4f} → 실측 {got:.4f}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
