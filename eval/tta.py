"""TTA_인증용 평가 데이터셋 — 압축/업로드/다운로드 관리.

이 레포는 NPU 있는 여러 서버로 옮겨다니며 쓰므로, 2GB대 영상 데이터를 git에 넣지 않고
HF Hub(private dataset)에 zip 1개로 올려두고 토큰만 있으면 재현되게 한다.

  HF: PIA-SPACE/AX_NPU_TTA (repo_type="dataset", private)
        README.md
        TTA_인증용.zip     # 아래 폴더 구조를 그대로 담은 zip (무압축 STORED)

  로컬: eval/datasets/TTA_인증용/{falldown,fire,intrusion,smoke}/
            *.mp4          # 원본 클립 50개
            *.json         # 이벤트 구간 라벨 (mp4와 1:1)
            clips/*.mp4    # 이벤트 구간만 잘라낸 클립

사용:
  python -m eval.tta download          # HF -> eval/datasets/TTA_인증용/ (있으면 skip)
  python -m eval.tta stats             # 로컬 데이터 개수/용량/이벤트 통계
  python -m eval.tta pack              # 로컬 폴더 -> TTA_인증용.zip
  python -m eval.tta upload            # zip + 데이터셋 카드 HF 업로드 (관리자용)

토큰: 환경변수 HF_TOKEN (또는 `hf auth login` 캐시). private 레포라 필수.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from pathlib import Path

HF_REPO = "PIA-SPACE/AX_NPU_TTA"
REPO_TYPE = "dataset"
NAME = "TTA_인증용"
ZIP_NAME = f"{NAME}.zip"

CATEGORIES = ["falldown", "fire", "intrusion", "smoke"]

# eval/datasets/ — 실데이터 위치 (.gitignore 처리됨)
DATA_ROOT = Path(__file__).resolve().parent / "datasets"
DATA_DIR = DATA_ROOT / NAME
ZIP_PATH = DATA_ROOT / ZIP_NAME

# pack으로 만든 zip의 sha256 (download 시 무결성 검증). pack이 갱신해 준다.
ZIP_SHA256 = "7bfa08a287bbb04a86dd9cf16507cdcbfb316298fc3bd1ce2d75156df6b5bb99"


# ---------------------------------------------------------------- helpers
def _sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _token(token: str | None = None) -> str | None:
    return token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


# ---------------------------------------------------------------- download
def ensure_tta(data_dir: Path | str = DATA_DIR, repo_id: str = HF_REPO,
               token: str | None = None, force: bool = False,
               verify: bool = True) -> Path:
    """로컬에 데이터가 있으면 그대로, 없으면 HF에서 zip을 받아 풀고 경로를 반환.

    토큰만 있으면 어느 서버에서든 동일한 데이터가 재현된다.
    """
    data_dir = Path(data_dir)
    if data_dir.exists() and any(data_dir.iterdir()) and not force:
        return data_dir

    from huggingface_hub import hf_hub_download

    zip_path = Path(hf_hub_download(
        repo_id=repo_id, repo_type=REPO_TYPE, filename=ZIP_NAME, token=_token(token),
    ))
    if verify and ZIP_SHA256 != "__FILL_ME__":
        got = _sha256(zip_path)
        if got != ZIP_SHA256:
            raise RuntimeError(f"zip sha256 불일치: {got} != {ZIP_SHA256}")

    data_dir.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_dir.parent)   # zip 내부 최상위가 TTA_인증용/
    return data_dir


# ---------------------------------------------------------------- pack
def pack(data_dir: Path | str = DATA_DIR, zip_path: Path | str = ZIP_PATH) -> Path:
    """데이터 폴더를 zip으로 묶는다. mp4는 이미 압축돼 있어 STORED(무압축)로 담는다."""
    data_dir, zip_path = Path(data_dir), Path(zip_path)
    files = sorted(p for p in data_dir.rglob("*") if p.is_file())
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
        for p in files:
            zf.write(p, arcname=str(Path(data_dir.name) / p.relative_to(data_dir)))
    return zip_path


# ---------------------------------------------------------------- upload
def upload(zip_path: Path | str = ZIP_PATH, repo_id: str = HF_REPO,
           token: str | None = None, private: bool = True,
           card: Path | str | None = None) -> str:
    """zip(+데이터셋 카드)을 HF private dataset 레포에 올린다. 관리자용."""
    from huggingface_hub import HfApi

    api = HfApi(token=_token(token))
    api.create_repo(repo_id, repo_type=REPO_TYPE, private=private, exist_ok=True)
    api.upload_file(path_or_fileobj=str(zip_path), path_in_repo=ZIP_NAME,
                    repo_id=repo_id, repo_type=REPO_TYPE)
    if card and Path(card).exists():
        api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md",
                        repo_id=repo_id, repo_type=REPO_TYPE)
    return f"https://huggingface.co/datasets/{repo_id}"


# ---------------------------------------------------------------- stats
def stats(data_dir: Path | str = DATA_DIR) -> dict:
    """카테고리별 파일 개수·용량·이벤트 통계를 dict로 반환."""
    data_dir = Path(data_dir)
    out = {"root": str(data_dir), "categories": {}}
    tot_ev: dict[str, int] = {}
    for c in CATEGORIES:
        d = data_dir / c
        mp4 = sorted(d.glob("*.mp4"))
        js = sorted(d.glob("*.json"))
        clips = sorted((d / "clips").glob("*.mp4"))
        ev: dict[str, int] = {}
        for j in js:
            for e in json.loads(j.read_text()).get("events", []):
                ev[e["category"]] = ev.get(e["category"], 0) + 1
                tot_ev[e["category"]] = tot_ev.get(e["category"], 0) + 1
        out["categories"][c] = {
            "video": len(mp4), "label": len(js), "clips": len(clips),
            "video_mb": round(sum(p.stat().st_size for p in mp4) / 2**20),
            "clips_mb": round(sum(p.stat().st_size for p in clips) / 2**20),
            "events": ev,
        }
    out["events_total"] = tot_ev
    return out


def _print_stats(s: dict) -> None:
    print(f"{'category':10} {'video':>6} {'label':>6} {'clips':>6} {'videoMB':>8} {'clipsMB':>8}  events")
    for c, v in s["categories"].items():
        print(f"{c:10} {v['video']:6} {v['label']:6} {v['clips']:6} "
              f"{v['video_mb']:8} {v['clips_mb']:8}  {v['events']}")
    print("\n전체 이벤트:", s["events_total"], "총", sum(s["events_total"].values()))


# ---------------------------------------------------------------- CLI
def main() -> None:
    ap = argparse.ArgumentParser(description="TTA_인증용 평가 데이터셋 관리")
    ap.add_argument("cmd", choices=["download", "stats", "pack", "upload"])
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--zip", default=str(ZIP_PATH))
    ap.add_argument("--repo", default=HF_REPO)
    ap.add_argument("--force", action="store_true", help="download: 이미 있어도 다시 받음")
    ap.add_argument("--public", action="store_true", help="upload: private 대신 public으로 생성")
    a = ap.parse_args()

    if a.cmd == "download":
        p = ensure_tta(a.data_dir, a.repo, force=a.force)
        print("OK:", p)
        _print_stats(stats(p))
    elif a.cmd == "stats":
        _print_stats(stats(a.data_dir))
    elif a.cmd == "pack":
        z = pack(a.data_dir, a.zip)
        print(f"OK: {z}  {z.stat().st_size / 2**30:.2f} GB")
        print("sha256:", _sha256(z))
    elif a.cmd == "upload":
        card = Path(__file__).resolve().parent / "TTA_DATASET_CARD.md"
        print("OK:", upload(a.zip, a.repo, private=not a.public, card=card))


if __name__ == "__main__":
    main()
