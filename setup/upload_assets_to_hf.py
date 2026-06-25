"""
미리 컴파일된 PE NPU 자산을 HuggingFace Hub에 업로드.

올리는 것:
  pe_full.mxq      # full NPU (trunk+attn_pool, QK^T 16bit, image->embedding). 권장, ~327MB
  pe_feat.mxq      # NPU trunk만 (aries2, INT8, weight 포함, ~300MB) — hybrid 레거시
  pe_pool_head.pt  # hybrid용 CPU pool head 가중치 (attn_pool + proj, ~53MB)

이걸 올려두면 다른 사람은 qbcompiler/원본 PE 가중치 없이 바로 추론할 수 있다.
  - 권장: `MXQInferenceFull.from_hf()`   (pe_full.mxq, CPU head 불필요)
  - 레거시: `MXQInferenceHybrid.from_hf()` (pe_feat.mxq + pe_pool_head.pt)

사전:
  - pip install huggingface_hub
  - huggingface-cli login   (또는 환경변수 HF_TOKEN)
  - 올릴 파일이 준비돼 있을 것:
      pe_full.mxq : python -m pe_npu.compile --mode compile --save pe_npu/out/pe_full.mxq \
                      --calib-data-path <calib_hwc> --device cpu --qk16
      pe_feat.mxq : python -m pe_npu.compile --mode compile --feat-only ...   (레거시)
      pe_pool_head.pt : python -m pe_npu.export_pool_head                     (레거시)

사용:
  python setup/upload_assets_to_hf.py --repo PIA-SPACE-LAB/MXQ_NPU            # full만(기본)
  python setup/upload_assets_to_hf.py --legacy                               # feat+pool도 함께
"""
import argparse
import os

DEFAULT_REPO = "PIA-SPACE-LAB/MXQ_NPU"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO, help="HF repo_id")
    ap.add_argument("--full", default="pe_npu/out/pe_full.mxq")
    ap.add_argument("--mxq", default="pe_npu/out/pe_feat.mxq", help="trunk MXQ (레거시 hybrid용)")
    ap.add_argument("--pool-head", default="pe_npu/out/pe_pool_head.pt", help="pool head (레거시 hybrid용)")
    ap.add_argument("--legacy", action="store_true", help="pe_feat.mxq + pe_pool_head.pt도 함께 업로드")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = ap.parse_args()

    from huggingface_hub import HfApi
    api = HfApi(token=args.token)

    files = [(args.full, "pe_full.mxq")]
    if args.legacy:
        files += [(args.mxq, "pe_feat.mxq"), (args.pool_head, "pe_pool_head.pt")]

    for fp, name in files:
        if not os.path.exists(fp):
            raise SystemExit(
                f"파일 없음: {fp}\n"
                f"  - full MXQ: python -m pe_npu.compile --mode compile --save {fp} "
                f"--calib-data-path <calib_hwc> --device cpu --qk16\n"
                f"  - trunk MXQ(레거시): python -m pe_npu.compile --mode compile --feat-only ...\n"
                f"  - pool head(레거시): python -m pe_npu.export_pool_head"
            )
    for fp, name in files:
        size_mb = os.path.getsize(fp) / 1e6
        print(f"업로드 {fp} ({size_mb:.0f}MB) -> {args.repo}/{name}")
        api.upload_file(path_or_fileobj=fp, path_in_repo=name, repo_id=args.repo, repo_type="model")
    print(f"[OK] {args.repo} 업로드 완료. 사용: MXQInferenceFull.from_hf('{args.repo}')")


if __name__ == "__main__":
    main()
