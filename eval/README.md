# eval — 평가용 데이터셋 관리

NPU 모델(PE-Core / YOLO11 / Qwen3-VL) 평가에 쓰는 데이터셋을 여기서 관리한다.
**실데이터는 git에 넣지 않는다.** HF Hub(private)에 zip으로 올려두고, 토큰만 있으면
어느 서버에서든 `python -m eval.tta download` 한 줄로 동일하게 재현한다.

| 데이터셋 | 용도 | 건수 | 라벨 | 관리 모듈 |
|---|---|---|---|---|
| `TTA_인증용` | 이상행동 4종(쓰러짐/화재/침입/연기) 이벤트 구간 검출 평가 | 영상 200 | `*.json` 구간(초) | `eval/tta.py` |
| `TTA_3종_엘쓰_에쓰_엘폭` | 엘리베이터 쓰러짐 / 에스컬레이터 쓰러짐 / 엘리베이터 폭력 (TTA T9·T10·T6) | 클립 450 | `*.csv` 프레임 0/1 | `eval/tta.py` |

HF: `PIA-SPACE/AX_NPU_TTA` (dataset, private) — **데이터셋마다 폴더 하나**
(`<이름>/<이름>.zip` + `<이름>/README.md`).

---

## 빠른 시작 (다운로드)

```bash
export HF_TOKEN=hf_xxx                 # private 레포라 필수 (또는 `hf auth login`)
python -m eval.tta list                # 등록된 데이터셋·로컬 보유 여부
python -m eval.tta download            # 기본 = TTA_인증용 -> eval/datasets/TTA_인증용/
python -m eval.tta download -d TTA_3종_엘쓰_에쓰_엘폭
python -m eval.tta stats -d <이름>     # 개수/용량/양성·음성 통계
```

- 이미 `eval/datasets/TTA_인증용/`이 있으면 건너뛴다(`--force`로 재다운로드).
- 받은 zip은 sha256으로 무결성 검증 후 풀린다.

```python
from eval.tta import ensure
root = ensure()                                        # TTA_인증용
root = ensure("TTA_3종_엘쓰_에쓰_엘폭")                # 없으면 받아서 풀고, 폴더 경로 반환
```

### 어디에 쓰나

이상행동 검출(침입/쓰러짐/화재/연기) 파이프라인의 **정확도·이벤트 구간 검출 성능 평가** 입력.
벤치/분석 결과 문서는 `reports/` 에 남긴다 → [`reports/README.md`](../reports/README.md)

---

## TTA_인증용

### 출처

- 원천: NAS 192TB `10.128.30.36:/volume1/AI_data` 의 `TTA/TTA_인증용_재인코딩/`
- 같은 위치에 재인코딩 전 원본(`TTA/TTA_인증용/`)도 있으나, **평가에는 재인코딩본을 쓴다**
  (원본은 일부 파일의 컨테이너/코덱이 균일하지 않음). 레포에서는 `TTA_인증용`으로 이름을 통일.
- NAS는 사내망 전용이라 외부/신규 서버에서는 위 HF 경로로 받는 것이 정식 경로다.

### 폴더 구조

```
eval/datasets/TTA_인증용/
├── falldown/
│   ├── <장소>__<YYYYMMDD_HHMMSS>__t171.mp4      # 평가 대상 영상 50개
│   ├── <장소>__<YYYYMMDD_HHMMSS>__t171.json     # 이벤트 구간 라벨 (mp4와 1:1)
│   └── clips/
│       └── <영상명>_ev01_<category>_<시작>-<끝>.mp4   # 이벤트 구간만 자른 클립
├── fire/          (동일 구조)
├── intrusion/     (동일 구조)
└── smoke/         (동일 구조)
```

### 개수·용량

| 카테고리 | 영상(mp4) | 라벨(json) | clips | 영상 용량 | clips 용량 | 이벤트 |
|---|---|---|---|---|---|---|
| falldown | 50 | 50 | 50 | 380 MB | 107 MB | falldown 50 |
| fire | 50 | 50 | 100 | 347 MB | 436 MB | fire 50, smoke 50 |
| intrusion | 50 | 50 | 52 | 273 MB | 118 MB | intrusion 52 |
| smoke | 50 | 50 | 50 | 336 MB | 229 MB | smoke 50 |
| **합계** | **200** | **200** | **252** | **1.31 GB** | **890 MB** | **252** |

- 전체 2.2 GB / 파일 652개.
- **이벤트 총 252건**: falldown 50, fire 50, smoke 100, intrusion 52.
- `fire` 폴더 영상에는 fire와 smoke 이벤트가 함께 달려 있어 clips가 영상 수의 2배(100)다.
  `intrusion`은 한 영상에 침입 이벤트가 2건인 케이스가 있어 52다.
- 이벤트 길이: 평균 10.7초, 중앙값 12.1초, 범위 0.7~15.0초.

### 라벨 포맷 (json)

```json
{
  "clip": "중앙로_1발매기실__20260531_030000__t171.mp4",
  "category": "fire",
  "origin": "base:2026-07-08/5152ff28-2be",
  "events": [
    { "category": "fire",  "timestamp": [1.500, 15.042], "duration": 13.542 },
    { "category": "smoke", "timestamp": [2.292, 15.042], "duration": 12.750 }
  ]
}
```

- `category`(최상위) = 폴더 기준 대표 카테고리, `events[].category` = 실제 이벤트 종류(혼재 가능).
- `timestamp` = `[시작초, 끝초]`, 영상 기준 초 단위(소수 3자리).
- `clips/` 파일명의 `_<시작>-<끝>` 구간이 이 `timestamp`와 대응한다.

---

## TTA_3종_엘쓰_에쓰_엘폭

TTA 인증 시험의 **T9 엘리베이터 내 쓰러짐 / T10 에스컬레이터 내 쓰러짐 / T6 폭력** 항목용.

### 출처

- 원천: NAS 200TB `10.128.30.51:/volume1/home/sj.park/tta_dataset` 의 10초 클립 벤치마크.
- `helmet`(안전모 미착용)은 TTA 12종에 없는 항목이라 **제외**했다.
- 정답은 원본의 `gt_label.zip`(정규화 정답지)에서 가져왔다. 원본 폴더 안 `per_frame/*.csv` 는
  subtype 별로 컬럼명이 달라(예: 폭력 하드네거티브가 `sittingdown`) 그대로 쓰면 라벨이 어긋난다.
- 반입 스크립트: `python -m wave_npu.import_clips` (평탄화 + MANIFEST 생성 + md5 검증).

### 폴더 구조

```
eval/datasets/TTA_3종_엘쓰_에쓰_엘폭/
├── elv_falldown/
│   ├── elv_falldown_1.mp4     # 클립
│   ├── elv_falldown_1.csv     # 프레임 단위 정답 (mp4와 1:1, 같은 stem)
│   └── ...
├── esc_falldown/
├── elv_violence/
└── MANIFEST.csv               # 답안지
```

### 개수·구성 — 항목당 150건, 양성:음성 = 40:60

| 폴더 | TTA | 총 | 양성 | 정상 | 하드네거티브 | 용량 |
|---|---|---|---|---|---|---|
| elv_falldown | T9 | 150 | 60 | 54 | 36 (앉기 20 + 인접 16) | 438 MB |
| esc_falldown | T10 | 150 | 60 | 54 | 36 (앉기 36) | 689 MB |
| elv_violence | T6 | 150 | 60 | 54 | 36 (앉기 20 + 인접 16) | 473 MB |
| **합계** | | **450** | **180** | **162** | **108** | **1.6 GB** |

- **음성(정상·하드네거티브)이 있어 오탐(FP)이 측정된다.** `TTA_인증용`과 가장 큰 차이.
  "무조건 이벤트"로 찍으면 F1 0.57 → 지표가 자명하게 달성되지 않는다.
- 하드네거티브 = 쓰러짐과 혼동되는 "앉기", 이벤트 **직전·직후**의 정상 구간.

### 라벨 포맷 (csv)

```csv
clip_frame,falldown
0,0
1,0
120,1
```

`clip_frame` = 클립 내 프레임 번호(0부터), 둘째 컬럼 = 이벤트 0/1. 행 수 = 영상 프레임 수(450건 전수 일치).

**왜 초가 아니라 프레임인가**: 이 데이터는 fps 가 7~50 으로 섞여 있고 28.8/29.1/29.7 같은
값도 있어, 초로 변환하면 구간 경계에서 왕복 오차가 난다. 프레임 라벨이 원본 그대로다.
`wave_npu` 는 두 형식을 모두 읽는다(스펙의 `label_format: "json" | "csv"`).

### 영상 스펙 (원본 유지)

| 폴더 | 해상도 | fps | 길이 |
|---|---|---|---|
| elv_falldown | 1920×1080(103) / 352×240(46) / 704×480(1) | 25~50 | 6~10초 |
| esc_falldown | 1920×1080(91) / 1280×720(41) / 960×480(11) / 기타 7 | **7~30** | 5~20초 |
| elv_violence | 1920×1080(102) / 352×240(41) / 704×480(7) | 25~50 | 6~10초 |

`esc_falldown` 에 **352×240 등 480p 미만**이 섞여 있다 — 성적서에 해상도 구성을 병기할 것.

---

## 관리자용 (업로드/갱신)

데이터가 바뀌면 zip을 다시 만들어 올리고, `eval/tta.py` 의 `DATASETS[<이름>].sha256` 을 갱신한다.

```bash
export HF_TOKEN=hf_xxx
python -m eval.tta pack   -d <이름>   # eval/datasets/<이름>/ -> <이름>.zip (STORED, 무압축)
                                      #   출력된 sha256을 DATASETS[<이름>].sha256 에 반영
python -m eval.tta upload -d <이름>   # <이름>/<이름>.zip + <이름>/README.md 업로드
```

**새 데이터셋 추가**: `eval/tta.py` 의 `DATASETS` 에 `DatasetSpec` 한 줄 + 카드 md 하나.
`label_ext` 로 라벨 형식(`json`=초 단위 구간 / `csv`=프레임 0/1)을 지정한다.

- mp4는 이미 압축돼 있어 zip은 **무압축(STORED)** 로 묶는다(압축률 이득 없음, 시간만 소모).
- HF 는 동일 blob 을 중복 저장하지 않는다 — 같은 zip 을 다른 경로에 올리면 업로드가 즉시 끝난다.

## 주의

- `eval/datasets/` 는 `.gitignore` 대상이다. 데이터/zip을 커밋하지 말 것.
- HF 레포는 **private**. 토큰 없이는 받을 수 없고, 토큰을 커밋하지 말 것(`.env` 사용).
- HF 최상위의 `TTA_인증용.zip` 은 폴더 구조 도입 이전의 **레거시 경로**다. 같은 파일이
  `TTA_인증용/TTA_인증용.zip` 에도 있고 `ensure()` 가 폴더 경로 → 레거시 순으로 폴백한다.
  구버전 체크아웃이 모두 정리되면 레거시 쪽을 지워도 된다.
