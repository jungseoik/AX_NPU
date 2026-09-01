# eval — 평가용 데이터셋 관리

NPU 모델(PE-Core / YOLO11 / Qwen3-VL) 평가에 쓰는 데이터셋을 여기서 관리한다.
**실데이터는 git에 넣지 않는다.** HF Hub(private)에 zip으로 올려두고, 토큰만 있으면
어느 서버에서든 `python -m eval.tta download` 한 줄로 동일하게 재현한다.

| 데이터셋 | 용도 | HF | 관리 모듈 |
|---|---|---|---|
| `TTA_인증용` | 이상행동 4종(쓰러짐/화재/침입/연기) 이벤트 구간 검출 평가 | `PIA-SPACE/AX_NPU_TTA` (dataset, private) | `eval/tta.py` |

---

## 빠른 시작 (다운로드)

```bash
export HF_TOKEN=hf_xxx                 # private 레포라 필수 (또는 `hf auth login`)
python -m eval.tta download            # -> eval/datasets/TTA_인증용/
python -m eval.tta stats               # 개수/용량/이벤트 통계 확인
```

- 이미 `eval/datasets/TTA_인증용/`이 있으면 건너뛴다(`--force`로 재다운로드).
- 받은 zip은 sha256으로 무결성 검증 후 풀린다.

```python
from eval.tta import ensure_tta
root = ensure_tta()                    # 없으면 받아서 풀고, 폴더 경로 반환
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

## 관리자용 (업로드/갱신)

데이터가 바뀌면 zip을 다시 만들어 올리고, `eval/tta.py`의 `ZIP_SHA256`을 갱신한다.

```bash
export HF_TOKEN=hf_xxx
python -m eval.tta pack        # eval/datasets/TTA_인증용/ -> TTA_인증용.zip (STORED, 무압축)
                               #   출력된 sha256을 eval/tta.py ZIP_SHA256에 반영
python -m eval.tta upload      # zip + 데이터셋 카드(TTA_DATASET_CARD.md) 업로드
```

- mp4는 이미 압축돼 있어 zip은 **무압축(STORED)** 로 묶는다(압축률 이득 없음, 시간만 소모).
- 새 데이터셋을 추가할 때는 `eval/tta.py`를 본떠 모듈 하나 + 이 표에 한 줄 추가.

## 주의

- `eval/datasets/` 는 `.gitignore` 대상이다. 데이터/zip을 커밋하지 말 것.
- HF 레포는 **private**. 토큰 없이는 받을 수 없고, 토큰을 커밋하지 말 것(`.env` 사용).
