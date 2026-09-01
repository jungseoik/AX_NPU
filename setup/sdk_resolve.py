"""SDK 번들 버전(1.0 / 1.1) → 실제 경로·파일명 해석기.

setup/sdk_versions.json 을 단일 기준으로 읽어, 셸/파이썬 어디서든 같은 값을 쓰게 한다.

사용:
  python setup/sdk_resolve.py --sdk 1.1 --shell     # eval 가능한 셸 변수 출력
  python setup/sdk_resolve.py --sdk 1.1 --json      # 전체 정의 JSON
  python setup/sdk_resolve.py --list                # 버전 목록·요약
  python setup/sdk_resolve.py --sdk 1.1 --get compile.docker_image
"""
from __future__ import annotations
import argparse, glob, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "setup", "sdk_versions.json")


def load() -> dict:
    with open(MANIFEST, encoding="utf-8") as f:
        return json.load(f)


def resolve(sdk: str | None = None) -> dict:
    m = load()
    sdk = sdk or os.environ.get("SDK_VERSION") or m["default"]
    sdk = str(sdk).lower().lstrip("v").rstrip("v")          # "v1.1", "1.1v" 모두 허용
    if sdk not in m["versions"]:
        raise SystemExit(f"[sdk_resolve] 알 수 없는 버전 {sdk!r} — 사용 가능: {list(m['versions'])}")
    v = dict(m["versions"][sdk])
    v["sdk"] = sdk
    v["local_dir_abs"] = os.path.join(ROOT, v["local_dir"])
    # 컴포넌트별 실제 파일 경로(있으면) 해석
    for name, comp in v["components"].items():
        hits = sorted(glob.glob(os.path.join(v["local_dir_abs"], comp["glob"])))
        comp["path"] = hits[0] if hits else None
        comp["present"] = bool(hits)
    return v


def _pick_image(v: dict) -> str:
    """GPU가 보이면 cuda 이미지, 아니면 cpu 이미지. (컴파일 코드가 CPU로 자동 폴백하므로 cpu로 충분)"""
    cp = v["compile"]
    has_gpu = os.path.exists("/dev/nvidiactl") or os.path.exists("/proc/driver/nvidia/version")
    return (cp.get("docker_image_cuda") if has_gpu else cp.get("docker_image_cpu")) or ""


def dig(d: dict, path: str):
    cur = d
    for key in path.split("."):
        cur = cur[key]
    return cur


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sdk", default=None, help="1.0 | 1.1 (기본: manifest default 또는 $SDK_VERSION)")
    ap.add_argument("--shell", action="store_true", help="eval 가능한 셸 변수로 출력")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--get", default=None, help="점 표기 경로로 값 하나만 출력 (예: compile.docker_image)")
    a = ap.parse_args()

    if a.list:
        m = load()
        print(f"{'버전':6s} {'라벨':7s} {'로컬 경로':22s} {'HF 폴더':18s} 요약")
        for k, v in m["versions"].items():
            mark = " (기본)" if k == m["default"] else ""
            print(f"{k+mark:6s} {v['label']:7s} {v['local_dir']:22s} {v['hf_folder']:18s} {v['summary']}")
        return 0

    v = resolve(a.sdk)
    if a.get:
        val = dig(v, a.get)
        print("" if val is None else val)
        return 0
    if a.json:
        print(json.dumps(v, ensure_ascii=False, indent=1))
        return 0
    if a.shell:
        c = v["components"]
        out = {
            "SDK_VERSION": v["sdk"],
            "SDK_LABEL": v["label"],
            "SDK_DIR": v["local_dir_abs"],
            "SDK_HF_FOLDER": v["hf_folder"],
            "DRIVER_GLOB": c["driver"]["glob"],
            "RUNTIME_GLOB": c["runtime_amd64"]["glob"],
            "COMPILER_GLOB": c["compiler"]["glob"],
            "DRIVER_PATH": c["driver"]["path"] or "",
            "RUNTIME_PATH": c["runtime_amd64"]["path"] or "",
            "COMPILER_PATH": c["compiler"]["path"] or "",
            "DOCKER_REQUIRED": "1" if v["compile"]["docker_required"] else "0",
            "DOCKER_IMAGE_CPU": v["compile"].get("docker_image_cpu") or "",
            "DOCKER_IMAGE_CUDA": v["compile"].get("docker_image_cuda") or "",
            "DOCKER_IMAGE": _pick_image(v),
        }
        for k, val in out.items():
            print(f"{k}='{val}'")
        return 0

    # 기본 출력: 사람이 읽는 요약
    print(f"SDK {v['label']} — {v['summary']}")
    print(f"  로컬 경로 : {v['local_dir']}")
    print(f"  HF 폴더   : {v['hf_folder']}")
    for name, comp in v["components"].items():
        state = "있음" if comp["present"] else "없음"
        base = os.path.basename(comp["path"]) if comp["path"] else comp["glob"]
        print(f"  {name:14s} {comp['version']:8s} [{state}] {base}")
    cp = v["compile"]
    print(f"  컴파일     : docker 이미지 — 이 호스트 권장 {_pick_image(v)}")
    print(f"               cpu={cp.get('docker_image_cpu')} / cuda={cp.get('docker_image_cuda')}")
    print(f"               python {cp['python']} / numpy {cp['numpy']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
