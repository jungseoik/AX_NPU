"""데이터셋·카테고리 스펙 — 카테고리를 코드가 아니라 **설정**으로 다룬다.

새 카테고리(8종 추가 등)를 붙일 때 고쳐야 할 곳이 한 군데가 되도록, 폴더 구성·프롬프트
클래스 번호·신호 원천(PE 유사도 / YOLO 사람검출)을 전부 이 스펙에 모았다.

```json
{
  "name": "TTA_인증용",
  "root": "eval/datasets/TTA_인증용",
  "folders": ["falldown", "fire", "intrusion", "smoke"],
  "prompt_csv": "third_party/PIA_Wave/data/....csv",
  "normal_class": 0,
  "categories": {
    "falldown":  {"cls": 1, "source": "pe"},
    "intrusion": {"source": "person", "negatives_exclude": ["falldown"]}
  }
}
```

- `cls`      : 프롬프트 CSV 의 class 번호 (source="pe" 일 때 필수)
- `source`   : "pe"(PE-Core 임베딩 × 프롬프트 유사도) | "person"(YOLO11 사람 검출)
- `negatives_exclude` : 이 카테고리의 음성 집합에서 제외할 폴더. 라벨 누락으로 원리적으로
  구분 불가능한 폴더가 있을 때 쓴다.
- `eval_folders` : 평가를 이 폴더들로 **한정**한다(negatives_exclude 보다 강한 제약).
  intrusion 이 그 예다 — falldown 영상은 96%에 사람이 있는데 intrusion 라벨이 0이라
  교차 음성으로는 원리적으로 구분되지 않는다(reports 참고). 프로토콜이 달라지므로
  **카테고리별 평가 범위를 반드시 결과표에 함께 적는다.**
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CSV = "third_party/PIA_Wave/data/text_features_tuningfree_v2_soil최종.csv"


@dataclass
class Spec:
    name: str = "TTA_인증용"
    root: str = "eval/datasets/TTA_인증용"
    folders: tuple = ("falldown", "fire", "intrusion", "smoke")
    prompt_csv: str = DEFAULT_CSV
    normal_class: int = 0
    categories: dict = field(default_factory=lambda: {
        "falldown": {"cls": 1, "source": "pe"},
        "fire": {"cls": 2, "source": "pe"},
        "smoke": {"cls": 3, "source": "pe"},
    })

    # --- 파생 ---
    @property
    def names(self) -> tuple:
        return tuple(self.categories.keys())

    def by_source(self, source: str) -> tuple:
        return tuple(k for k, v in self.categories.items() if v.get("source", "pe") == source)

    @property
    def cat2cls(self) -> dict:
        return {k: v["cls"] for k, v in self.categories.items() if "cls" in v}

    def negatives_exclude(self, cat: str) -> tuple:
        return tuple(self.categories[cat].get("negatives_exclude", ()))

    def eval_folders(self, cat: str) -> tuple:
        """평가를 한정할 폴더. 비어 있으면 전체(=교차 음성)."""
        return tuple(self.categories[cat].get("eval_folders", ()))

    def protocol(self, cat: str) -> str:
        ef, ne = self.eval_folders(cat), self.negatives_exclude(cat)
        if ef:
            return f"{'+'.join(ef)} 영상 내"
        if ne:
            return f"전체−{'/'.join(ne)}"
        return "전체 교차"

    @classmethod
    def load(cls, path: str = None) -> "Spec":
        if not path:
            return cls()
        d = json.load(open(path, encoding="utf-8"))
        return cls(name=d.get("name", "spec"), root=d["root"],
                   folders=tuple(d["folders"]), prompt_csv=d.get("prompt_csv", DEFAULT_CSV),
                   normal_class=int(d.get("normal_class", 0)), categories=d["categories"])

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        json.dump({"name": self.name, "root": self.root, "folders": list(self.folders),
                   "prompt_csv": self.prompt_csv, "normal_class": self.normal_class,
                   "categories": self.categories}, open(path, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)


TTA = Spec()
# intrusion 은 "사람 등장 = 이벤트"라 falldown 영상(사람 96% 존재, intrusion 라벨 0)과
# 교차 음성으로는 원리적으로 구분되지 않는다 → 자기 폴더 안에서만 평가한다(결정 사항).
TTA_WITH_INTRUSION = Spec(
    categories={**TTA.categories,
                "intrusion": {"source": "person", "eval_folders": ["intrusion"]}})
