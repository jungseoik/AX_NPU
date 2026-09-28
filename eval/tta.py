"""TTA 평가 데이터셋 — 압축/업로드/다운로드 관리 (여러 데이터셋 공통).

이 레포는 NPU 있는 여러 서버로 옮겨다니며 쓰므로, GB 단위 영상 데이터를 git에 넣지 않고
HF Hub(private dataset)에 데이터셋별 zip 1개로 올려두고 토큰만 있으면 재현되게 한다.

  HF: PIA-SPACE/AX_NPU_TTA (repo_type="dataset", private)
        README.md
        <데이터셋명>/<데이터셋명>.zip      ← 데이터셋마다 폴더 1개
        TTA_인증용.zip                     ← (레거시) 최상위 경로. 폴더 경로 없으면 자동 폴백

  로컬: eval/datasets/<데이터셋명>/...

등록된 데이터셋 (`DATASETS`):
  · TTA_인증용              이상행동 4종(쓰러짐/화재/침입/연기) 15초 영상 200 + 초 단위 json 라벨
  · TTA_3종_엘쓰_에쓰_엘폭  엘리베이터 쓰러짐/에스컬레이터 쓰러짐/엘리베이터 폭력
                            10초 클립 450 + **프레임 단위 csv 라벨**, 양성:음성 = 40:60

사용:
  python -m eval.tta download                              # 기본 = TTA_인증용
  python -m eval.tta download -d TTA_3종_엘쓰_에쓰_엘폭
  python -m eval.tta list                                  # 등록된 데이터셋·로컬 상태
  python -m eval.tta stats   -d <이름>
  python -m eval.tta pack    -d <이름>                     # 로컬 폴더 -> zip (+sha256 출력)
  python -m eval.tta upload  -d <이름>                     # zip + 카드 HF 업로드 (관리자용)

토큰: 환경변수 HF_TOKEN (또는 `hf auth login` 캐시). private 레포라 필수.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

HF_REPO = "PIA-SPACE/AX_NPU_TTA"
REPO_TYPE = "dataset"

# eval/datasets/ — 실데이터 위치 (.gitignore 처리됨)
DATA_ROOT = Path(__file__).resolve().parent / "datasets"
HERE = Path(__file__).resolve().parent


@dataclass
class DatasetSpec:
    name: str
    categories: list
    label_ext: str                   # "json"(초 단위 구간) | "csv"(프레임 0/1)
    desc: str
    sha256: str = ""                 # pack 이 갱신. 빈 값이면 다운로드 시 검증 생략
    card: str = ""                   # 데이터셋 카드 파일명 (eval/ 기준)
    legacy_root_zip: bool = False    # 최상위 <name>.zip 으로도 존재(폴백 허용)

    @property
    def zip_name(self) -> str:
        return f"{self.name}.zip"

    @property
    def repo_path(self) -> str:
        """HF 내부 경로 — 데이터셋마다 폴더 하나."""
        return f"{self.name}/{self.zip_name}"

    @property
    def data_dir(self) -> Path:
        return DATA_ROOT / self.name

    @property
    def zip_path(self) -> Path:
        return DATA_ROOT / self.zip_name

    @property
    def card_path(self) -> Path:
        return HERE / (self.card or f"{self.name}_CARD.md")


DATASETS: dict[str, DatasetSpec] = {
    "TTA_인증용": DatasetSpec(
        name="TTA_인증용",
        categories=["falldown", "fire", "intrusion", "smoke"],
        label_ext="json",
        desc="이상행동 4종(쓰러짐/화재/침입/연기) 15초 CCTV 영상 200건 + 이벤트 구간 라벨(초) + clips",
        sha256="7bfa08a287bbb04a86dd9cf16507cdcbfb316298fc3bd1ce2d75156df6b5bb99",
        card="TTA_DATASET_CARD.md",
        legacy_root_zip=True,
    ),
    "TTA_3종_엘쓰_에쓰_엘폭": DatasetSpec(
        name="TTA_3종_엘쓰_에쓰_엘폭",
        categories=["elv_falldown", "esc_falldown", "elv_violence"],
        label_ext="csv",
        desc="엘리베이터 쓰러짐/에스컬레이터 쓰러짐/엘리베이터 폭력 10초 클립 450건 "
             "(항목당 150 = 양성 60 + 정상 54 + 하드네거티브 36) + 프레임 단위 라벨",
        sha256="c3c8f79878704b3015e9e57a656e68b787bd3e31c8314f459ad688f336fda782",
        card="TTA_3CAT_DATASET_CARD.md",
    ),
}

DEFAULT = "TTA_인증용"


# ---------------------------------------------------------------- helpers
def _sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _token(token: str | None = None) -> str | None:
    return token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def get(name: str = DEFAULT) -> DatasetSpec:
    if name not in DATASETS:
        raise KeyError(f"등록되지 않은 데이터셋 '{name}'. 사용 가능: {list(DATASETS)}")
    return DATASETS[name]


# ---------------------------------------------------------------- download
def ensure(name: str = DEFAULT, data_dir: Path | str | None = None,
           repo_id: str = HF_REPO, token: str | None = None,
           force: bool = False, verify: bool = True) -> Path:
    """로컬에 있으면 그대로, 없으면 HF에서 zip을 받아 풀고 경로를 반환.

    HF 경로는 `<이름>/<이름>.zip` 을 먼저 보고, 없으면 레거시 최상위 `<이름>.zip` 으로 폴백한다.
    """
    ds = get(name)
    data_dir = Path(data_dir) if data_dir else ds.data_dir
    if data_dir.exists() and any(data_dir.iterdir()) and not force:
        return data_dir

    from huggingface_hub import hf_hub_download

    tries = [ds.repo_path] + ([ds.zip_name] if ds.legacy_root_zip else [])
    last = None
    for fn in tries:
        try:
            zip_path = Path(hf_hub_download(repo_id=repo_id, repo_type=REPO_TYPE,
                                            filename=fn, token=_token(token)))
            break
        except Exception as e:                       # 경로 없으면 다음 후보
            last = e
    else:
        raise FileNotFoundError(f"HF에서 {tries} 를 찾지 못했습니다") from last

    if verify and ds.sha256:
        got = _sha256(zip_path)
        if got != ds.sha256:
            raise RuntimeError(f"zip sha256 불일치: {got} != {ds.sha256}")

    data_dir.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_dir.parent)               # zip 내부 최상위가 <이름>/
    return data_dir


def ensure_tta(data_dir: Path | str | None = None, repo_id: str = HF_REPO,
               token: str | None = None, force: bool = False, verify: bool = True) -> Path:
    """[하위호환] TTA_인증용 전용 진입점."""
    return ensure(DEFAULT, data_dir, repo_id, token, force, verify)


# ---------------------------------------------------------------- pack
def pack(name: str = DEFAULT, data_dir: Path | str | None = None,
         zip_path: Path | str | None = None) -> Path:
    """데이터 폴더를 zip으로 묶는다. mp4는 이미 압축돼 있어 STORED(무압축)로 담는다."""
    ds = get(name)
    data_dir = Path(data_dir) if data_dir else ds.data_dir
    zip_path = Path(zip_path) if zip_path else ds.zip_path
    files = sorted(p for p in data_dir.rglob("*") if p.is_file())
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
        for p in files:
            zf.write(p, arcname=str(Path(data_dir.name) / p.relative_to(data_dir)))
    return zip_path


# ---------------------------------------------------------------- upload
def upload(name: str = DEFAULT, zip_path: Path | str | None = None,
           repo_id: str = HF_REPO, token: str | None = None,
           private: bool = True, card: Path | str | None = None) -> str:
    """zip(+데이터셋 카드)을 HF private dataset 레포에 올린다. 관리자용.

    zip 은 `<이름>/<이름>.zip`, 카드는 `<이름>/README.md` 로 올라간다
    (레포 최상위 README.md 는 전체 인덱스라 덮어쓰지 않는다).
    """
    from huggingface_hub import HfApi

    ds = get(name)
    zip_path = Path(zip_path) if zip_path else ds.zip_path
    card = Path(card) if card else ds.card_path

    api = HfApi(token=_token(token))
    api.create_repo(repo_id, repo_type=REPO_TYPE, private=private, exist_ok=True)
    api.upload_file(path_or_fileobj=str(zip_path), path_in_repo=ds.repo_path,
                    repo_id=repo_id, repo_type=REPO_TYPE)
    if card.exists():
        api.upload_file(path_or_fileobj=str(card), path_in_repo=f"{ds.name}/README.md",
                        repo_id=repo_id, repo_type=REPO_TYPE)
    return f"https://huggingface.co/datasets/{repo_id}/tree/main/{ds.name}"


# ---------------------------------------------------------------- stats
def stats(name: str = DEFAULT, data_dir: Path | str | None = None) -> dict:
    """카테고리별 개수·용량·양성/음성 통계. 라벨 형식(json/csv)에 맞춰 센다."""
    ds = get(name)
    data_dir = Path(data_dir) if data_dir else ds.data_dir
    out = {"name": ds.name, "root": str(data_dir), "label": ds.label_ext, "categories": {}}
    tot_ev: dict[str, int] = {}

    for c in ds.categories:
        d = data_dir / c
        mp4 = sorted(d.glob("*.mp4"))
        lab = sorted(d.glob(f"*.{ds.label_ext}"))
        rec = {"video": len(mp4), "label": len(lab),
               "video_mb": round(sum(p.stat().st_size for p in mp4) / 2**20)}
        if ds.label_ext == "json":
            ev: dict[str, int] = {}
            for j in lab:
                for e in json.loads(j.read_text()).get("events", []):
                    ev[e["category"]] = ev.get(e["category"], 0) + 1
                    tot_ev[e["category"]] = tot_ev.get(e["category"], 0) + 1
            rec["events"] = ev
            rec["clips"] = len(sorted((d / "clips").glob("*.mp4")))
        else:                                        # csv = 프레임 0/1
            pos = 0
            for cp in lab:
                rows = list(csv.DictReader(open(cp, encoding="utf-8-sig")))
                if not rows:
                    continue
                col = [k for k in rows[0] if k != "clip_frame"][0]
                if any(int(r[col]) for r in rows):
                    pos += 1
            rec["pos"], rec["neg"] = pos, len(lab) - pos
            tot_ev[c] = tot_ev.get(c, 0) + pos
        out["categories"][c] = rec
    out["events_total"] = tot_ev
    return out


def _print_stats(s: dict) -> None:
    csv_mode = s["label"] == "csv"
    if csv_mode:
        print(f"{'category':16} {'video':>6} {'label':>6} {'pos':>5} {'neg':>5} {'videoMB':>8}")
        for c, v in s["categories"].items():
            print(f"{c:16} {v['video']:6} {v['label']:6} {v['pos']:5} {v['neg']:5} {v['video_mb']:8}")
        tot = sum(v["video"] for v in s["categories"].values())
        pos = sum(v["pos"] for v in s["categories"].values())
        print(f"\n총 {tot}건 · 양성 {pos} / 음성 {tot - pos}"
              f" ({100 * pos / tot:.0f}% 양성)" if tot else "\n(비어 있음)")
    else:
        print(f"{'category':10} {'video':>6} {'label':>6} {'clips':>6} {'videoMB':>8}  events")
        for c, v in s["categories"].items():
            print(f"{c:10} {v['video']:6} {v['label']:6} {v['clips']:6} "
                  f"{v['video_mb']:8}  {v['events']}")
        print("\n전체 이벤트:", s["events_total"], "총", sum(s["events_total"].values()))


def _print_list() -> None:
    print(f"{'데이터셋':26} {'라벨':5} {'로컬':>6}  설명")
    for n, d in DATASETS.items():
        local = "있음" if d.data_dir.exists() and any(d.data_dir.iterdir()) else "없음"
        print(f"{n:26} {d.label_ext:5} {local:>6}  {d.desc}")
    print(f"\nHF: https://huggingface.co/datasets/{HF_REPO}  (경로: <이름>/<이름>.zip)")


# ---------------------------------------------------------------- CLI
def main() -> None:
    ap = argparse.ArgumentParser(description="TTA 평가 데이터셋 관리")
    ap.add_argument("cmd", choices=["download", "stats", "pack", "upload", "list"])
    ap.add_argument("-d", "--dataset", default=DEFAULT, help=f"기본 {DEFAULT}")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--zip", default=None)
    ap.add_argument("--repo", default=HF_REPO)
    ap.add_argument("--force", action="store_true", help="download: 이미 있어도 다시 받음")
    ap.add_argument("--public", action="store_true", help="upload: private 대신 public")
    a = ap.parse_args()

    if a.cmd == "list":
        _print_list()
        return
    if a.cmd == "download":
        p = ensure(a.dataset, a.data_dir, a.repo, force=a.force)
        print("OK:", p)
        _print_stats(stats(a.dataset, p))
    elif a.cmd == "stats":
        _print_stats(stats(a.dataset, a.data_dir))
    elif a.cmd == "pack":
        z = pack(a.dataset, a.data_dir, a.zip)
        print(f"OK: {z}  {z.stat().st_size / 2**30:.2f} GB")
        print("sha256:", _sha256(z))
        print(f"  → eval/tta.py 의 DATASETS['{a.dataset}'].sha256 에 반영할 것")
    elif a.cmd == "upload":
        print("OK:", upload(a.dataset, a.zip, a.repo, private=not a.public))


if __name__ == "__main__":
    main()
