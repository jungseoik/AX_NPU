---
license: other
language: [ko]
task_categories: [video-classification]
tags: [anomaly-detection, cctv, evaluation, npu, tta]
pretty_name: AX_NPU TTA 인증용 평가 데이터셋
---

# AX_NPU TTA 인증용 평가 데이터셋

이상행동 4종(쓰러짐/화재/침입/연기) **이벤트 구간 검출** 평가용 CCTV 영상 데이터셋.
Mobilint ARIES NPU 모델 평가(`AX_NPU` 레포)에 사용한다. **사내 전용(private).**

## 구성

`TTA_인증용.zip` 하나로 배포한다(무압축 STORED, 약 2.2 GB / 파일 652개).

```
TTA_인증용/
├── falldown/  *.mp4(50)  *.json(50)  clips/*.mp4(50)
├── fire/      *.mp4(50)  *.json(50)  clips/*.mp4(100)
├── intrusion/ *.mp4(50)  *.json(50)  clips/*.mp4(52)
└── smoke/     *.mp4(50)  *.json(50)  clips/*.mp4(50)
```

| 카테고리 | 영상 | 라벨 | clips | 영상 용량 | clips 용량 | 이벤트 |
|---|---|---|---|---|---|---|
| falldown | 50 | 50 | 50 | 380 MB | 107 MB | falldown 50 |
| fire | 50 | 50 | 100 | 347 MB | 436 MB | fire 50, smoke 50 |
| intrusion | 50 | 50 | 52 | 273 MB | 118 MB | intrusion 52 |
| smoke | 50 | 50 | 50 | 336 MB | 229 MB | smoke 50 |
| **합계** | **200** | **200** | **252** | **1.31 GB** | **890 MB** | **252** |

- `*.mp4` = 평가 대상 영상, `*.json` = 이벤트 구간 라벨(mp4와 1:1), `clips/` = 이벤트 구간만 자른 클립.
- 이벤트 총 252건(falldown 50 / fire 50 / smoke 100 / intrusion 52), 길이 평균 10.7초(0.7~15.0초).
- `fire` 영상에는 fire·smoke 이벤트가 함께 달려 clips가 영상 수의 2배다.

## 라벨 포맷

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

`timestamp` = `[시작초, 끝초]` (영상 기준 초, 소수 3자리).

## 사용법

```bash
export HF_TOKEN=hf_xxx
python -m eval.tta download     # AX_NPU 레포에서 -> eval/datasets/TTA_인증용/
```

또는 직접:

```python
from huggingface_hub import hf_hub_download
z = hf_hub_download("PIA-SPACE/AX_NPU_TTA", "TTA_인증용.zip", repo_type="dataset")
```

자세한 내용은 `AX_NPU` 레포 `eval/README.md` 참고.

## 출처·라이선스

사내 NAS(192TB) `TTA/TTA_인증용_재인코딩/` 를 재인코딩본 기준으로 정리한 것.
사내 평가 목적 한정, 외부 배포 금지.
