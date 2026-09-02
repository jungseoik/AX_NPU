"""PE NPU 양자화 스킴별 MXQ를 HF에 업로드 — `<quant>/<scheme>/pe_full.mxq` 구조.

기존 배포본은 최상위 `<scheme>/pe_full.mxq` 였는데, 양자화 스킴이 늘어나면서 한 단계를 더 둔다.

  W8A16/{single,multi,global4,global8}/pe_full.mxq      기본값(기존 배포본과 동일 설정)
  W4A16/{...}                                           weight 4bit
  W4A8_L5A16/{...}                                      weight 4bit + activation 8bit(상위 5개만 16bit)

기존 최상위 경로는 **삭제하지 않는다** — 구버전 코드가 계속 동작하도록 두고,
새 코드는 `ensure_full_mxq(quant=...)`로 새 경로를 우선 조회한다.

사용:
  python setup/upload_quant_schemes_to_hf.py --quant W4A16 --src-dir /tmp/w4a16/mxq
  python setup/upload_quant_schemes_to_hf.py --quant W8A16 --from-existing   # 기존 최상위 → W8A16/ 복제
"""
import argparse, os, tempfile

DEFAULT_REPO = "PIA-SPACE-LAB/MXQ_NPU"
MODES = ["single", "multi", "global4", "global8"]
QUANTS = ["W8A16", "W4A16", "W4A8_L5A16"]

QUANT_README = {
    "W8A16": ("weight 8bit / activation 16bit(output·ffn) — qbcompiler 기본값 + QKᵀ 16bit override.\n"
              "**기본 배포본**이며 기존 최상위 경로(`<scheme>/pe_full.mxq`)와 동일한 설정이다."),
    "W4A16": ("weight 4bit(value만 8bit) / activation은 W8A16과 동일 + QKᵀ 16bit override.\n"
              "크기·처리량 이득이 크고 정확도는 떨어진다. 용도에 따라 선택."),
    "W4A8_L5A16": ("weight 4bit + activation 8bit. 단 outlier가 큰 상위 5개 텐서만 16bit로 유지\n"
                   "(우리 모델 프로파일링으로 선정: L3/L10/L12 mlp.c_proj, L10/L12 mlp.gelu). 속도 우선."),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--quant", required=True, choices=QUANTS)
    ap.add_argument("--src-dir", default=None, help="<quant>_<mode>.mxq 가 있는 디렉토리")
    ap.add_argument("--from-existing", action="store_true",
                    help="기존 최상위 <scheme>/pe_full.mxq 를 받아 <quant>/ 아래로 복제")
    ap.add_argument("--modes", nargs="*", default=MODES)
    ap.add_argument("--calib-md", default=None, help="CALIBRATION.md 경로(없으면 기존 것 재사용)")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    a = ap.parse_args()

    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(token=a.token)

    for mode in a.modes:
        if a.from_existing:
            src = hf_hub_download(repo_id=a.repo, filename=f"{mode}/pe_full.mxq", token=a.token)
        else:
            src = os.path.join(a.src_dir, f"{a.quant}_{mode}.mxq")
            if not os.path.exists(src):
                raise SystemExit(f"없음: {src}")
        dst = f"{a.quant}/{mode}/pe_full.mxq"
        print(f"업로드 {os.path.basename(src)} ({os.path.getsize(src)/1e6:.1f}MB) -> {dst}")
        api.upload_file(path_or_fileobj=src, path_in_repo=dst, repo_id=a.repo, repo_type="model")

        # CALIBRATION.md: 지정본 우선, 없으면 기존 모드 폴더 것을 그대로 옮긴다
        md = a.calib_md
        if md is None:
            try:
                md = hf_hub_download(repo_id=a.repo, filename=f"{mode}/CALIBRATION.md", token=a.token)
            except Exception:
                md = None
        if md:
            api.upload_file(path_or_fileobj=md, path_in_repo=f"{a.quant}/{mode}/CALIBRATION.md",
                            repo_id=a.repo, repo_type="model")

    readme = (f"# PE-Core-L14-336 — 양자화 스킴 `{a.quant}`\n\n{QUANT_README[a.quant]}\n\n"
              f"## 구성\n코어모드 4종: {', '.join(MODES)} — 각 폴더에 `pe_full.mxq`.\n\n"
              f"## 사용\n```python\nfrom pe_npu.inference import MXQInferenceFull\n"
              f"m = MXQInferenceFull.from_hf(scheme='global4', quant='{a.quant}')\n```\n\n"
              f"성능·정확도 비교표: 레포 `reports/performance/NPU_pe_quant_schemes.md`\n")
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write(readme); tmp = f.name
    api.upload_file(path_or_fileobj=tmp, path_in_repo=f"{a.quant}/README.md",
                    repo_id=a.repo, repo_type="model")
    os.unlink(tmp)
    print(f"[OK] {a.quant} 업로드 완료 → {a.repo}/{a.quant}/")


if __name__ == "__main__":
    main()
