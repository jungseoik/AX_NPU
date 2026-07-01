"""HF private 레포에서 Mobilint ARIES SDK를 받아 로컬 download/ 에 배치.

신규 서버: 레포 clone → HF 로그인(`huggingface-cli login` 또는 HF_TOKEN) → 이 스크립트 →
download/ 에 드라이버/런타임/컴파일러가 채워짐 → npu-setup skill로 설치.

사용: python setup/fetch_sdk_from_hf.py [--version aries2_v1.2.0] [--repo PIA-SPACE-LAB/MXQ_NPU]
"""
import argparse, os, shutil

DEFAULT_REPO = "PIA-SPACE-LAB/MXQ_NPU"
DEFAULT_VER = "aries2_v1.2.0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--version", default=DEFAULT_VER)
    ap.add_argument("--out", default=None, help="기본: <repo_root>/download")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = ap.parse_args()

    from huggingface_hub import HfApi, hf_hub_download
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = args.out or os.path.join(repo_root, "download")
    os.makedirs(out, exist_ok=True)

    api = HfApi(token=args.token)
    prefix = f"sdk/{args.version}/"
    files = [f for f in api.list_repo_files(args.repo) if f.startswith(prefix) and not f.endswith("/")]
    if not files:
        raise SystemExit(f"HF {args.repo}에 {prefix} 없음. 먼저 upload_sdk_to_hf.py로 업로드했는지 확인.")

    for f in files:
        name = os.path.basename(f)
        if name == "README.md":  # 폴더 README는 스킵
            continue
        cached = hf_hub_download(repo_id=args.repo, filename=f, token=args.token)
        dst = os.path.join(out, name)
        if os.path.abspath(cached) != os.path.abspath(dst):
            shutil.copy(cached, dst)
        print(f"  받음: {name} -> {dst}")
    print(f"[OK] SDK → {out}. 다음: sudo bash .claude/skills/npu-setup/setup_npu_cli.sh")


if __name__ == "__main__":
    main()
