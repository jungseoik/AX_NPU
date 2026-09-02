"""튜닝 매트릭스 MXQ 24종을 HF에 업로드 + 컴파일 환경/시간 문서 생성.

업로드 경로: PIA-SPACE-LAB/MXQ_NPU 의 `<quant>/<tuning>/<scheme>/pe_full.mxq`
  - 기존 배포 경로(`<quant>/<scheme>/pe_full.mxq`, 최상위 `<scheme>/`)는 건드리지 않는다.
  - NPU 검증(cos) 전 실험 자산 — 검증 후 승격은 upload_quant_schemes_to_hf.py 로.
문서: HF 루트 `TUNING_MATRIX.md` (컴파일 환경 + 24종 시간/크기/md5 표)

사용: export HF_TOKEN=... && python setup/upload_tuning_matrix_to_hf.py --src-dir pe_npu/out/matrix [--dry-run]
  --src-dir: pe_<quant>_<tuning>_<scheme>.mxq 들이 있는 디렉토리(잡 로그는 <src-dir>/logs/)
"""
import argparse, glob, hashlib, os, re, subprocess, tempfile, time

HERE = "pe_npu/out/matrix"  # 기본 산출물 위치 — main()에서 --src-dir로 덮어씀
REPO = "PIA-SPACE-LAB/MXQ_NPU"
QUANTS = ["W8A16", "W4A16", "W4A8_L5A16"]
TUNINGS = ["none", "sws", "optq", "sws_optq"]
SCHEMES = ["single", "global4"]

ENV_MD = """## 컴파일 환경 (2026-09-02, NPU 없는 GPU 서버)

| 항목 | 값 |
|---|---|
| Host | Ubuntu 24.04 (kernel 6.8), Intel Xeon 6530P 128스레드, RAM 251GB |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition 96GB × 2, driver 580.173.02 |
| Docker 이미지 | `mobilint/qbcompiler:1.1-cuda12.8.1-ubuntu22.04` (Ubuntu 22.04.5, Python 3.10.12) |
| qbcompiler | 1.1.2 (+aries2 whl, SDK 번들 1.0v) / torch 2.7.1+cu128 / numpy 1.26.0 |
| calib | COCO val2017 앞 200장(sorted), `pe_npu.preprocess.preprocess_image`(336², /255, mean·std 0.5) HWC float32 |
| CalibrationConfig | method=1, output=1(per-channel), mode=1, max_percentile(0.9999, topk 0.01) |
| 공통 | QKᵀ score MatMul 25개 16bit override(full 모드 필수), target aries2, `--device gpu` |
| 실행 | 컨테이너 4개 병렬(GPU0×1 — vLLM 88GB 상주와 공유, GPU1×3), 잡당 OMP 32스레드 |

### 조합 정의
- **W8A16**: act(q8,k8,v8,head8,out16,ffn16) / weight 전부 8 — 기존 배포본과 동일 비트
- **W4A16**: act 동일 / weight(q4,k4,v8,out4,ffn4,head4)
- **W4A8_L5A16**: act 전부 8 + outlier 상위 5텐서만 16bit(L3/L12/L10 mlp.c_proj, L12/L10 mlp.gelu) / weight W4와 동일
- **sws** = SearchWeightScale(apply, q/k/v/out/ffn 전부) · **optq** = OPTQ(actOrder, blockSize 128, percDamp 0.01)
- 스크립트: AX_NPU `reports/scripts/compile_quant_tuning_matrix.py` (Mobilint inquiry 04 예제 설정 준수)

### 주의
- **NPU 미검증** — cos/성능은 NPU 서버에서 `reports/scripts/bench_quant_schemes.py`로 검증 후 사용.
- **소요 시간은 설정 차이가 아니라 calibration 디바이스 차이** — GPU 정상 잡 5~8분(calib 3분),
  50~60분 잡은 컨테이너가 GPU 접근을 잃어 CPU 폴백(calib 47분). 산출물은 디바이스 무관 동일 로직.
"""


def md5(p, buf=1 << 20):
    h = hashlib.md5()
    with open(p, "rb") as f:
        while chunk := f.read(buf):
            h.update(chunk)
    return h.hexdigest()


def total_seconds(name):
    logf = os.path.join(HERE, "logs", f"{name}.log")
    # 초기 3건(matrix 이전 실행)은 로그 위치가 다르다 → 알려진 값 하드코딩
    known = {"pe_W4A16_sws_optq_single": 1286, "pe_W4A16_sws_optq_global4": 1277,
             "pe_W4A16_none_single": 3181}
    if name in known:
        return known[name]
    if os.path.exists(logf):
        m = re.findall(r"TOTAL (\d+)s", open(logf, errors="ignore").read())
        if m:
            return int(m[-1])
    return None


def main():
    global HERE
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", default=HERE)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    HERE = a.src_dir

    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])

    rows = []
    for q in QUANTS:
        for t in TUNINGS:
            for s in SCHEMES:
                name = f"pe_{q}_{t}_{s}"
                p = os.path.join(HERE, f"{name}.mxq")
                if not os.path.exists(p):
                    print(f"[없음] {name}")
                    continue
                sec = total_seconds(name)
                rows.append((q, t, s, p, os.path.getsize(p) / 1e6, sec, md5(p)))

    md = "# PE-Core-L14-336 — 양자화×튜닝 매트릭스 (GPU 컴파일 스윕)\n\n" + ENV_MD
    md += "\n## 산출물 (24종)\n\n| quant | tuning | scheme | 크기(MB) | 컴파일(s) | md5 | 경로 |\n|---|---|---|---:|---:|---|---|\n"
    for q, t, s, p, mb, sec, h in rows:
        dst = f"{q}/{t}/{s}/pe_full.mxq"
        md += f"| {q} | {t} | {s} | {mb:.1f} | {sec if sec else '-'} | `{h}` | `{dst}` |\n"
    md += ("\n## 사용 (검증용 수동 로드)\n```python\nfrom huggingface_hub import hf_hub_download\n"
           "p = hf_hub_download('PIA-SPACE-LAB/MXQ_NPU', 'W4A16/sws_optq/single/pe_full.mxq')\n"
           "from pe_npu.inference import MXQInferenceFull\nm = MXQInferenceFull(full_mxq_path=p, device_id=0)\n```\n")

    info_path = os.path.join(HERE, "TUNING_MATRIX.md")
    open(info_path, "w", encoding="utf-8").write(md)
    print(f"[md] {info_path} ({len(rows)}종)")

    if a.dry_run:
        print("[dry-run] 업로드 생략")
        return

    for q, t, s, p, mb, sec, h in rows:
        dst = f"{q}/{t}/{s}/pe_full.mxq"
        print(f"업로드 {os.path.basename(p)} ({mb:.1f}MB) -> {dst}", flush=True)
        api.upload_file(path_or_fileobj=p, path_in_repo=dst, repo_id=REPO, repo_type="model")
    api.upload_file(path_or_fileobj=info_path, path_in_repo="TUNING_MATRIX.md",
                    repo_id=REPO, repo_type="model")
    print(f"[OK] {len(rows)}개 + TUNING_MATRIX.md → {REPO}")


if __name__ == "__main__":
    main()
