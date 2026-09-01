"""HF private 레포에서 Mobilint ARIES SDK 번들을 받아 로컬에 배치.

번들 버전은 setup/sdk_versions.json 이 단일 기준이다(현재 1.0 / 1.1, 기본 1.0).
버전마다 HF 폴더와 로컬 경로가 다르다.

  1.0 → HF sdk/v1.0  (없으면 sdk/aries2_v1.2.0 로 폴백) → 로컬 download/
  1.1 → HF sdk/v1.1                                     → 로컬 download/sdk/v1.1/

사용:
  python setup/fetch_sdk_from_hf.py                  # 기본 버전
  python setup/fetch_sdk_from_hf.py --sdk 1.1        # 1.1v 번들
  python setup/fetch_sdk_from_hf.py --folder sdk/실험폴더   # 원시 폴더 직접 지정(예외용)
"""
import argparse, os, shutil, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sdk_resolve import resolve  # noqa: E402

DEFAULT_REPO = "PIA-SPACE-LAB/MXQ_NPU"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--sdk", default=None, help="번들 버전 라벨 (1.0 | 1.1)")
    ap.add_argument("--folder", default=None, help="HF 폴더 직접 지정 (예: sdk/aries2_v1.2.0)")
    ap.add_argument("--out", default=None, help="기본: 해당 버전의 로컬 경로")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = ap.parse_args()

    v = resolve(args.sdk)
    out = args.out or v["local_dir_abs"]
    os.makedirs(out, exist_ok=True)

    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(token=args.token)
    repo_files = api.list_repo_files(args.repo)

    candidates = [args.folder] if args.folder else [v["hf_folder"], v.get("hf_folder_legacy")]
    prefix = None
    for cand in [c for c in candidates if c]:
        pre = cand.rstrip("/") + "/"
        if any(f.startswith(pre) for f in repo_files):
            prefix = pre
            break
    if not prefix:
        raise SystemExit(f"HF {args.repo}에 {candidates} 폴더 없음. "
                         f"먼저 'python setup/upload_sdk_to_hf.py --sdk {v['sdk']}' 로 업로드했는지 확인.")

    print(f"[fetch] 번들 {v['label']} | HF {args.repo}/{prefix} → {out}")
    for f in [f for f in repo_files if f.startswith(prefix) and not f.endswith("/")]:
        name = os.path.basename(f)
        if name == "README.md":
            continue
        cached = hf_hub_download(repo_id=args.repo, filename=f, token=args.token)
        dst = os.path.join(out, name)
        if os.path.abspath(cached) != os.path.abspath(dst):
            shutil.copy(cached, dst)
        print(f"  받음: {name} ({os.path.getsize(dst)/1e6:.1f}MB)")
    print(f"[OK] SDK({v['label']}) → {out}")
    print(f"     다음: sudo bash .claude/skills/npu-setup/setup_npu_cli.sh --sdk {v['sdk']}")


if __name__ == "__main__":
    main()
