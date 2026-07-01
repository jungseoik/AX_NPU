"""Mobilint ARIES SDK(드라이버/런타임/컴파일러)를 HF private 레포에 업로드 (버전 폴더).

목적: mxq와 함께 SDK를 한 곳에서 관리 → 신규 서버는 HF 로그인만으로 mxq+SDK 다 받아 설치.
⚠️ Mobilint SDK는 벤더 비공개 바이너리 → **레포는 반드시 private 유지.** (public 전환 금지)

업로드: download/ 의 SDK 파일 → `sdk/<version>/` + README.md
  sdk/aries2_v1.2.0/
    qbcompiler-1.1.2+aries2-py3-none-any.whl
    qbruntime_aries2-v4_v1.2.0_amd64.tar.gz
    qbruntime_aries2-v4_v1.2.0_arm64.tar.gz
    mobilint-aries2-driver_v1.13.tar.gz
    README.md

사용: python setup/upload_sdk_to_hf.py [--version aries2_v1.2.0] [--repo PIA-SPACE-LAB/MXQ_NPU]
"""
import argparse, os, glob

DEFAULT_REPO = "PIA-SPACE-LAB/MXQ_NPU"
DEFAULT_VER = "aries2_v1.2.0"

SDK_README = """# Mobilint ARIES SDK — {ver}

이 레포(private)의 SDK 번들. mxq와 함께 신규 서버 세팅에 쓴다.
**⚠️ Mobilint 벤더 비공개 바이너리 — 이 레포는 절대 public 전환 금지.**

## 구성 / 버전
| 파일 | 구성요소 | 버전 |
|------|---------|------|
| qbcompiler-1.1.2+aries2-py3-none-any.whl | 컴파일러(qbcompiler) | 1.1.2 |
| qbruntime_aries2-v4_v1.2.0_amd64.tar.gz | 런타임+CLI (x86_64) | 1.2.0 |
| qbruntime_aries2-v4_v1.2.0_arm64.tar.gz | 런타임+CLI (arm64) | 1.2.0 |
| mobilint-aries2-driver_v1.13.tar.gz | 커널 드라이버 | 1.13 |

- 대상 HW: ARIES MLA100 PCIe (Aries2), 펌웨어 **1.2.5** (드라이버/유틸로 업데이트, `mobilint-cli`).
- 호환표 원문: 레포 `docs/compatibility.md`.

## 사용 (신규 서버)
```bash
# 레포 clone 후 HF 로그인만 하면:
python setup/fetch_sdk_from_hf.py            # 이 폴더를 download/ 로 다운로드
sudo bash .claude/skills/npu-setup/setup_npu_cli.sh   # 드라이버+런타임+CLI 설치 → status
```
출처: https://dl.mobilint.com (원본, 계정 필요). 여기 사본은 조직 내부 세팅 편의용.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--version", default=DEFAULT_VER, help="sdk/<version>/ 폴더명")
    ap.add_argument("--download-dir", default="download")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = ap.parse_args()

    from huggingface_hub import HfApi
    api = HfApi(token=args.token)
    info = api.model_info(args.repo)
    if not info.private:
        raise SystemExit(f"[중단] {args.repo}가 PUBLIC입니다. SDK 업로드는 private 레포에만 허용.")

    files = sorted(glob.glob(os.path.join(args.download_dir, "*.whl")) +
                   glob.glob(os.path.join(args.download_dir, "*.tar.gz")))
    if not files:
        raise SystemExit(f"업로드할 SDK 파일 없음: {args.download_dir}/*.whl|*.tar.gz")

    prefix = f"sdk/{args.version}"
    for fp in files:
        name = os.path.basename(fp)
        print(f"업로드 {fp} ({os.path.getsize(fp)/1e6:.0f}MB) -> {args.repo}/{prefix}/{name}")
        api.upload_file(path_or_fileobj=fp, path_in_repo=f"{prefix}/{name}",
                        repo_id=args.repo, repo_type="model")
    # README
    readme = SDK_README.format(ver=args.version)
    api.upload_file(path_or_fileobj=readme.encode(), path_in_repo=f"{prefix}/README.md",
                    repo_id=args.repo, repo_type="model")
    print(f"       + {prefix}/README.md")
    print(f"[OK] SDK 업로드 완료 → {args.repo}/{prefix}  (private)")


if __name__ == "__main__":
    main()
