"""튜닝 매트릭스 MXQ(qbcompiler 1.2.0 재컴파일본)를 HF에 업로드 + 컴파일 환경/시간 문서 생성.

업로드 경로: PIA-SPACE-LAB/MXQ_NPU 의 `<quant>/<tuning>/<scheme>/pe_full.mxq`
  - 기존 배포 경로(`<quant>/<scheme>/pe_full.mxq`, 최상위 `<scheme>/`)는 건드리지 않는다.
  - 1.1.2 산출물 24개는 2026-09-03 철회·삭제됨(OPTQ/SWS 구현 문제로 열화) — 이 스크립트가
    올리는 것은 **1.2.0 재컴파일본**이며, 루트 `TUNING_MATRIX.md`의 철회 공지를 새 표로 교체한다.
문서: HF 루트 `TUNING_MATRIX.md` (철회 이력 + 1.2.0 컴파일 환경 + 시간/크기/md5 표)

사용: export HF_TOKEN=... && python setup/upload_tuning_matrix_to_hf.py --src-dir out/matrix_120 [--dry-run]
  --src-dir: pe_<quant>_<tuning>_<scheme>.mxq 들이 있는 디렉토리(잡 로그는 <src-dir>/logs/)
"""
import argparse, hashlib, os, re

HERE = "out/matrix_120"  # 기본 산출물 위치 — main()에서 --src-dir로 덮어씀
REPO = "PIA-SPACE-LAB/MXQ_NPU"
QUANTS = ["W8A16", "W4A16", "W4A8_L5A16"]
TUNINGS = ["none", "sws_optq"]
SCHEMES = ["single", "multi", "global4", "global8"]

HEAD_MD = """# PE-Core-L14-336 — 양자화×튜닝 매트릭스 (qbcompiler **1.2.0** 재컴파일)

> **이력**: 2026-09-02 에 올렸던 qbcompiler **1.1.2** 산출물 24개는 해당 버전의 OPTQ/SearchWeightScale
> 구현 문제로 튜닝본이 열화돼 **2026-09-03 전부 철회·삭제**했다(같은 설정에서 1.2.0 은 부호가 반전 —
> 튜닝이 정확도를 올린다). 상세: AX_NPU `reports/performance/NPU_pe_quant_tuning_compiler_version.md`.
> 아래는 **1.2.0 으로 재컴파일한 매트릭스**다. 축도 바꿨다: 튜닝 {none, sws_optq} × 코어모드 4종 전부.

## 검증 상태

| 구분 | 상태 |
|---|---|
| 회귀 기준 4점(§아래 표 참조) | **1.2.0 NPU 실측 존재** — W8A16 sws_optq 0.9951 / W4A16 none 0.9126 / W4A16 sws_optq 0.9642 / W4A8_L5A16 sws_optq 0.8932 (single) |
| 나머지 20종 | **NPU cos 검증 대기** — `reports/scripts/verify_quant_tuning_matrix.py` 로 검증 후 사용 |
| 배포 경로 `<quant>/<scheme>/`, 루트 `<scheme>/` | 영향 없음(튜닝 없는 빌드, 계속 사용 가능) |

## 컴파일 환경 (2026-09-03, NPU 없는 GPU 서버)

| 항목 | 값 |
|---|---|
| Host | Ubuntu 24.04 (kernel 6.8), Intel Xeon 6530P 128스레드, RAM 251GB |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition 96GB × 2, driver 580.173.02 |
| Docker 이미지 | `mobilint/qbcompiler:1.2-cuda12.8.1-ubuntu22.04` (Ubuntu 22.04.5, Python 3.10.12) |
| 컴파일러 | **qbcompiler 1.2.0** (SDK 번들 **1.1v**) / torch 2.7.1+cu128 / numpy 1.26.0 |
| target_device | **aries-rb** (1.2.0 명명 — 자동 판별 `pe_npu/target_device.py`, 하드웨어는 Aries2) |
| calib | COCO val2017 앞 200장(sorted, 1.1.2 측정과 동일), `pe_npu.preprocess.preprocess_image`(336², /255, mean·std 0.5), HWC float32 |
| CalibrationConfig | method=1, output=1(per-channel), mode=1, max_percentile(0.9999, topk 0.01) |
| 공통 | QKᵀ score MatMul 25개 16bit override(full 모드 필수), `--device gpu` |
| 실행 | 잡마다 새 `--rm` 컨테이너 × 4워커 병렬(GPU 2장), 잡당 OMP 32스레드 — 조합당 5~13분 |

### 조합 정의
- **W8A16**: act(q8,k8,v8,head8,out16,ffn16) / weight 전부 8 — 기존 배포본과 동일 비트
- **W4A16**: act 동일 / weight(q4,k4,v8,out4,ffn4,head4)
- **W4A8_L5A16**: act 전부 8 + outlier 상위 5텐서만 16bit(L3/L12/L10 mlp.c_proj, L12/L10 mlp.gelu) / weight W4와 동일
- **sws_optq** = SearchWeightScale(apply, q/k/v/out/ffn 전부) + OPTQ(actOrder, blockSize 128, percDamp 0.01)
- 스크립트: AX_NPU `reports/scripts/compile_quant_tuning_matrix.py` · 절차: `reports/RUNBOOK_quant_matrix_120.md`
"""


def md5(p, buf=1 << 20):
    h = hashlib.md5()
    with open(p, "rb") as f:
        while chunk := f.read(buf):
            h.update(chunk)
    return h.hexdigest()


def total_seconds(name):
    logf = os.path.join(HERE, "logs", f"{name}.log")
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

    rows, missing = [], []
    for q in QUANTS:
        for t in TUNINGS:
            for s in SCHEMES:
                name = f"pe_{q}_{t}_{s}"
                p = os.path.join(HERE, f"{name}.mxq")
                if not os.path.exists(p):
                    missing.append(name)
                    continue
                rows.append((q, t, s, p, os.path.getsize(p) / 1e6, total_seconds(name), md5(p)))
    for name in missing:
        print(f"[없음] {name}")

    md = HEAD_MD
    md += f"\n## 산출물 ({len(rows)}종)\n\n| quant | tuning | scheme | 크기(MB) | 컴파일(s) | md5 | 경로 |\n|---|---|---|---:|---:|---|---|\n"
    for q, t, s, p, mb, sec, h in rows:
        dst = f"{q}/{t}/{s}/pe_full.mxq"
        md += f"| {q} | {t} | {s} | {mb:.1f} | {sec if sec else '-'} | `{h}` | `{dst}` |\n"
    md += ("\n## 사용 (검증용 수동 로드)\n```python\nfrom huggingface_hub import hf_hub_download\n"
           "p = hf_hub_download('PIA-SPACE-LAB/MXQ_NPU', 'W4A16/sws_optq/single/pe_full.mxq')\n"
           "from pe_npu.inference import MXQInferenceFull\nm = MXQInferenceFull(full_mxq_path=p, device_id=0)\n```\n"
           "\nNPU 검증(회귀 기준 자동 대조):\n```bash\npython reports/scripts/verify_quant_tuning_matrix.py "
           "--src-dir <mxq 디렉토리> --device 0\n```\n")

    info_path = os.path.join(HERE, "TUNING_MATRIX.md")
    open(info_path, "w", encoding="utf-8").write(md)
    print(f"[md] {info_path} ({len(rows)}종, 누락 {len(missing)})")

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
