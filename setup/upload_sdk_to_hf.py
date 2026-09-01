"""Mobilint ARIES SDK 번들(드라이버/런타임/컴파일러)을 HF private 레포에 버전 폴더로 업로드.

번들 버전 정의: setup/sdk_versions.json (현재 1.0 / 1.1)
⚠️ Mobilint SDK는 벤더 비공개 바이너리 → **레포는 반드시 private 유지.**

  sdk/v1.0/   드라이버 1.13 / qbruntime 1.2.0 / qbcompiler 1.1.2
  sdk/v1.1/   드라이버 1.14 / qbruntime 1.4.0 / qbcompiler 1.2.0
  sdk/versions.json  버전 정의(라벨 → 폴더/파일/컴파일 요건) 사본

사용:
  python setup/upload_sdk_to_hf.py --sdk 1.1              # 1.1v 번들 업로드
  python setup/upload_sdk_to_hf.py --sdk 1.0              # 1.0v 번들 업로드
  python setup/upload_sdk_to_hf.py --sdk 1.1 --manifest-only
"""
import argparse, glob, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sdk_resolve import MANIFEST, resolve  # noqa: E402

DEFAULT_REPO = "PIA-SPACE-LAB/MXQ_NPU"

README_TMPL = """# Mobilint ARIES SDK — {label}

이 레포(private)의 SDK 번들. mxq와 함께 신규 서버 세팅에 쓴다.
**⚠️ Mobilint 벤더 비공개 바이너리 — 이 레포는 절대 public 전환 금지.**

{summary}

## 구성
| 파일 | 구성요소 | 버전 |
|------|---------|------|
{rows}

## 컴파일 요건
{compile_note}

## 사용 (신규 서버)
```bash
# 레포 clone + HF 로그인 후:
python setup/fetch_sdk_from_hf.py --sdk {sdk}                        # 이 폴더를 로컬로 다운로드
sudo bash .claude/skills/npu-setup/setup_npu_cli.sh --sdk {sdk}      # 드라이버+런타임+CLI 설치 → status
```
버전 목록: `python setup/sdk_resolve.py --list`
출처: https://dl.mobilint.com (원본, 계정 필요). 여기 사본은 조직 내부 세팅 편의용.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--sdk", default=None, help="번들 버전 라벨 (1.0 | 1.1)")
    ap.add_argument("--folder", default=None, help="HF 폴더 직접 지정(예외용)")
    ap.add_argument("--manifest-only", action="store_true", help="versions.json 만 업로드")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = ap.parse_args()

    v = resolve(args.sdk)
    prefix = (args.folder or v["hf_folder"]).rstrip("/")

    from huggingface_hub import HfApi
    api = HfApi(token=args.token)
    info = api.model_info(args.repo)
    if not info.private:
        raise SystemExit(f"[중단] {args.repo}가 PUBLIC입니다. SDK 업로드는 private 레포에만 허용.")

    # 버전 정의 파일은 항상 최신으로 올려둔다(다른 서버가 라벨만으로 해석 가능하도록)
    api.upload_file(path_or_fileobj=MANIFEST, path_in_repo="sdk/versions.json",
                    repo_id=args.repo, repo_type="model")
    print("업로드 sdk/versions.json")
    if args.manifest_only:
        return

    files = sorted(glob.glob(os.path.join(v["local_dir_abs"], "*.whl")) +
                   glob.glob(os.path.join(v["local_dir_abs"], "*.tar.gz")))
    files = [f for f in files if os.path.isfile(f)]     # 심볼릭 링크도 실체 있으면 통과
    if not files:
        raise SystemExit(f"업로드할 SDK 파일 없음: {v['local_dir']}/*.whl|*.tar.gz")

    for fp in files:
        name = os.path.basename(fp)
        print(f"업로드 {name} ({os.path.getsize(fp)/1e6:.0f}MB) -> {args.repo}/{prefix}/{name}")
        api.upload_file(path_or_fileobj=fp, path_in_repo=f"{prefix}/{name}",
                        repo_id=args.repo, repo_type="model")

    rows = "\n".join(
        f"| {os.path.basename(c['path']) if c['path'] else c['glob']} | {name} | {c['version']} |"
        for name, c in v["components"].items())
    cp = v["compile"]
    compile_note = (f"- docker 필요: `{cp['docker_image']}`\n" if cp["docker_required"]
                    else "- 호스트에서 바로 컴파일 가능(컴파일러 whl에 mmc 내장)\n")
    compile_note += f"- python {cp['python']} / numpy {cp['numpy']}\n- {cp['note']}"
    readme = README_TMPL.format(label=v["label"], sdk=v["sdk"], summary=v["summary"],
                                rows=rows, compile_note=compile_note)
    api.upload_file(path_or_fileobj=readme.encode(), path_in_repo=f"{prefix}/README.md",
                    repo_id=args.repo, repo_type="model")
    print(f"       + {prefix}/README.md")
    print(f"[OK] SDK({v['label']}) 업로드 완료 → {args.repo}/{prefix}  (private)")


if __name__ == "__main__":
    main()
