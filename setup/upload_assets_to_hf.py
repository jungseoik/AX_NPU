"""
미리 컴파일된 PE NPU 자산을 HuggingFace Hub에 업로드.

기본: full NPU MXQ를 **코어모드 폴더별로** 올린다 (image->embedding 전부 NPU, QKᵀ 16bit).
  PIA-SPACE-LAB/MXQ_NPU/
    single/   pe_full.mxq + CALIBRATION.md
    multi/    pe_full.mxq + CALIBRATION.md
    global4/  pe_full.mxq + CALIBRATION.md
    global8/  pe_full.mxq + CALIBRATION.md
필요할 때 모드를 골라 당겨 쓴다:  MXQInferenceFull.from_hf(scheme="global8")

레거시(hybrid)용 pe_feat.mxq + pe_pool_head.pt 는 `--legacy`로 함께 올린다.

사전:
  - huggingface-cli login  (또는 HF_TOKEN)
  - 모드별 MXQ가 준비돼 있을 것:
      python -m pe_npu.compile --mode compile --save <dir>/pe_full_<mode>.mxq \
        --calib-data-path <calib_hwc> --device cpu --qk16 --scheme <mode>

사용:
  python setup/upload_assets_to_hf.py --src-dir scratchpad_repro/full_proof --calib-md scratchpad_repro/full_proof/CALIBRATION.md
  python setup/upload_assets_to_hf.py --modes single global8     # 일부만
  python setup/upload_assets_to_hf.py --legacy                   # feat+pool도
"""
import argparse
import os

DEFAULT_REPO = "PIA-SPACE-LAB/MXQ_NPU"
ALL_MODES = ["single", "multi", "global4", "global8"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO, help="HF repo_id")
    ap.add_argument("--src-dir", default="scratchpad_repro/full_proof",
                    help="pe_full_<mode>.mxq 들이 있는 디렉토리")
    ap.add_argument("--calib-md", default="scratchpad_repro/full_proof/CALIBRATION.md",
                    help="각 모드 폴더에 동봉할 calib 설명")
    ap.add_argument("--modes", nargs="+", default=ALL_MODES, choices=ALL_MODES)
    ap.add_argument("--legacy", action="store_true", help="pe_feat.mxq + pe_pool_head.pt도 업로드")
    ap.add_argument("--feat", default="pe_npu/out/pe_feat.mxq")
    ap.add_argument("--pool-head", default="pe_npu/out/pe_pool_head.pt")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = ap.parse_args()

    from huggingface_hub import HfApi
    api = HfApi(token=args.token)

    # 모드별 full MXQ + CALIBRATION.md
    for mode in args.modes:
        mxq = os.path.join(args.src_dir, f"pe_full_{mode}.mxq")
        if not os.path.exists(mxq):
            raise SystemExit(f"파일 없음: {mxq}\n  생성: python -m pe_npu.compile --mode compile "
                             f"--save {mxq} --calib-data-path <calib_hwc> --device cpu --qk16 --scheme {mode}")
        size_mb = os.path.getsize(mxq) / 1e6
        print(f"업로드 {mxq} ({size_mb:.0f}MB) -> {args.repo}/{mode}/pe_full.mxq")
        api.upload_file(path_or_fileobj=mxq, path_in_repo=f"{mode}/pe_full.mxq",
                        repo_id=args.repo, repo_type="model")
        if os.path.exists(args.calib_md):
            api.upload_file(path_or_fileobj=args.calib_md, path_in_repo=f"{mode}/CALIBRATION.md",
                            repo_id=args.repo, repo_type="model")
            print(f"       + {mode}/CALIBRATION.md")

    if args.legacy:
        for fp, name in [(args.feat, "pe_feat.mxq"), (args.pool_head, "pe_pool_head.pt")]:
            if not os.path.exists(fp):
                print(f"[skip legacy] 파일 없음: {fp}"); continue
            print(f"업로드 {fp} -> {args.repo}/{name}")
            api.upload_file(path_or_fileobj=fp, path_in_repo=name, repo_id=args.repo, repo_type="model")

    print(f"[OK] {args.repo} 업로드 완료. 사용: MXQInferenceFull.from_hf(scheme='single')")


if __name__ == "__main__":
    main()
