"""새 카테고리의 이벤트 문구 → 프롬프트 CSV 확장 (+ 스펙 JSON 자동 생성).

기존 풀은 `장면(25) × 인원상황(15) × 이벤트문구` 의 완전 조합이다. 새 카테고리를 붙일 때
사람이 쓸 것은 **이벤트 문구 6~10개뿐**이고, 장면·인원상황 축과의 조합은 이 도구가 만든다.
(그 두 축은 프롬프트 선택 단계에서 어차피 자동으로 골라진다.)

    # 1) 템플릿 복사해서 문구를 채운다
    cp wave_npu/specs/new_categories_TEMPLATE.json my_cats.json
    # 2) 확장
    python -m wave_npu.extend_prompts --in my_cats.json
    # 3) 텍스트 임베딩 재추출 → 파이프라인
    python -m wave_npu.text --csv <out_csv> --out wave_npu/cache/text_feats_v2.npz
    python -m wave_npu.pipeline --spec <out_spec> --text wave_npu/cache/text_feats_v2.npz
"""
from __future__ import annotations

import argparse
import csv
import json
import os

from .config import DEFAULT_CSV, Spec
from .prompts import build


def extend(cfg: dict):
    base = cfg.get("base_csv", DEFAULT_CSV)
    space = build(base)
    rows = list(csv.DictReader(open(base, encoding="utf-8")))
    used = {int(c) for c in space.phrases}
    print(f"[extend] base={base} rows={len(rows)} scenes={len(space.scenes)} "
          f"occs={len(space.occs)} 기존 class={sorted(used)}")

    new = []
    for name, spec in cfg["categories"].items():
        if spec.get("source", "pe") != "pe":
            print(f"  · {name:14s} source={spec['source']} → 프롬프트 불필요(검출기 기반), 건너뜀")
            continue
        cls = int(spec["cls"])
        if cls in used:
            raise SystemExit(f"class {cls} 는 이미 쓰인다({name}). 새 번호를 주세요.")
        phrases = [p.strip().rstrip(".") for p in spec["phrases"] if p.strip()]
        if not phrases:
            raise SystemExit(f"{name}: phrases 가 비었습니다.")
        for sc in space.scenes:
            for oc in space.occs:
                for ph in phrases:
                    new.append({"ID": 0, "class": cls, "prompt": f"{sc}. {oc}. {ph}. "})
        print(f"  + {name:14s} cls={cls} 문구 {len(phrases)}개 → "
              f"{len(space.scenes) * len(space.occs) * len(phrases)} 프롬프트")

    for ph in cfg.get("normal_extra", []):
        ph = ph.strip().rstrip(".")
        for sc in space.scenes:
            for oc in space.occs:
                new.append({"ID": 0, "class": cfg.get("normal_class", 0),
                            "prompt": f"{sc}. {oc}. {ph}. "})
    if cfg.get("normal_extra"):
        print(f"  + normal 하드네거티브 {len(cfg['normal_extra'])}개 → "
              f"{len(space.scenes) * len(space.occs) * len(cfg['normal_extra'])} 프롬프트")

    out_csv = cfg["out_csv"]
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ID", "class", "prompt"])
        w.writeheader()
        w.writerows(rows + new)
    print(f"[extend] wrote {out_csv}  ({len(rows)} + {len(new)} = {len(rows) + len(new)} 프롬프트)")

    # 스펙 자동 생성
    if cfg.get("out_spec"):
        base_spec = Spec.load(cfg["base_spec"]) if cfg.get("base_spec") else Spec()
        cats = dict(base_spec.categories)
        for name, s in cfg["categories"].items():
            entry = {"source": s.get("source", "pe")}
            if entry["source"] == "pe":
                entry["cls"] = int(s["cls"])
            for k in ("eval_folders", "negatives_exclude", "signal"):
                if k in s:
                    entry[k] = s[k]
            cats[name] = entry
        sp = Spec(name=cfg.get("name", "extended"),
                  root=cfg.get("root", base_spec.root),
                  folders=tuple(cfg.get("folders", base_spec.folders)),
                  prompt_csv=out_csv, normal_class=base_spec.normal_class, categories=cats)
        sp.save(cfg["out_spec"])
        print(f"[extend] wrote {cfg['out_spec']}  cats={sp.names}")
        for c in sp.names:
            print(f"    {c:14s} source={sp.categories[c].get('source', 'pe'):7s} "
                  f"protocol={sp.protocol(c)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="카테고리 문구 JSON")
    a = ap.parse_args()
    extend(json.load(open(a.inp, encoding="utf-8")))


if __name__ == "__main__":
    main()
