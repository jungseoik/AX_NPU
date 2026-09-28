---
license: other
language: [ko]
task_categories: [video-classification]
tags: [anomaly-detection, cctv, evaluation, npu, tta, elevator, escalator, violence]
pretty_name: AX_NPU TTA 3종 평가 데이터셋 (엘쓰·에쓰·엘폭)
---

# TTA_3종_엘쓰_에쓰_엘폭

**엘리베이터 쓰러짐 / 에스컬레이터 쓰러짐 / 엘리베이터 폭력** 3종의 **10초 클립 평가 데이터셋**.
TTA 인증 시험(TTA-26-03361)의 T9 / T10 / T6 항목 평가에 쓴다. **사내 전용(private).**

`TTA_3종_엘쓰_에쓰_엘폭.zip` 하나로 배포한다(무압축 STORED, 약 1.6 GB / 900 파일).

## 구성 — 항목당 150건, 양성:음성 = 40:60

| 폴더 | TTA 항목 | 총 | 양성 | 정상 | 하드네거티브 |
|---|---|---|---|---|---|
| `elv_falldown` | T9 엘리베이터 내 쓰러짐 | 150 | 60 | 54 | 36 (앉기 20 + 이벤트 인접 16) |
| `esc_falldown` | T10 에스컬레이터 내 쓰러짐 | 150 | 60 | 54 | 36 (앉기 36) |
| `elv_violence` | T6 폭력 (엘리베이터 내) | 150 | 60 | 54 | 36 (앉기 20 + 이벤트 인접 16) |
| **합계** | | **450** | **180** | **162** | **108** |

- 3종 모두 **양성 60 : 음성 90 (40:60)** 으로 통일 → 항목 간 점수 비교가 가능하다.
- 음성 안에 **하드네거티브**(오탐 유도)가 들어 있다: 쓰러짐과 혼동되는 "앉기", 폭력/낙상
  **직전·직후의 정상 구간**. 단순 정탐률이 아니라 변별력을 잰다.
- "무조건 이벤트"로 찍으면 F1 은 0.57 에 그친다(자명 달성 불가).

## 폴더 구조

```
TTA_3종_엘쓰_에쓰_엘폭/
├── elv_falldown/
│   ├── elv_falldown_1.mp4     # 클립
│   ├── elv_falldown_1.csv     # 프레임 단위 정답 (mp4와 1:1, 같은 stem)
│   └── ...  (elv_falldown_N=양성 / elv_normal_N=정상 / elv_hardneg_N=하드네거티브)
├── esc_falldown/   (es_falldown_N / es_normal_N / es_hardneg_N)
├── elv_violence/   (elv_violence_N / elv_normal_N / elv_hardneg_N)
└── MANIFEST.csv    # 답안지
```

파일명 접두사가 곧 subtype 이다(`*_falldown_` `*_violence_` = 양성, `*_normal_` = 정상,
`*_hardneg_` = 하드네거티브). 정확한 구분은 `MANIFEST.csv` 를 쓴다.

## 라벨 포맷 — 프레임 단위 CSV

```csv
clip_frame,falldown
0,0
1,0
...
120,1
```

- `clip_frame` = 클립 내 프레임 번호(0부터), 두 번째 컬럼 = 해당 이벤트 0/1.
- 컬럼명은 폴더마다 하나로 통일돼 있다(`falldown` / `violence`).
- 행 수 = 영상 프레임 수 (450건 전수 일치 확인).

**왜 초 단위 timestamp 가 아니라 프레임인가**: 이 데이터는 fps 가 7~50 으로 섞여 있고
28.8 / 29.1 / 29.7 같은 값도 있다. 초로 변환하면 구간 경계에서 왕복 오차가 난다.
프레임 라벨이 원본 그대로다.

## MANIFEST.csv

`item, name, subtype, place, is_pos, event_col, fps, width, height,
frames_video, frames_label, event_frames, src`

- 정답 = `is_pos`(클립 단위) 또는 짝 CSV(프레임 단위).
- `subtype` = `falldown|violence` (양성) / `normal` / `hardneg_sitting` / `hardneg_adjacent`.
- `place` = 촬영 장소 도메인 (`gaepo1/gaepo4/samsung/yeoksam2`, `ABB/DTRO/Hyundai_v2`).
- `src` = 원본 NAS 경로(provenance).

## 영상 스펙 (원본 유지, 보정하지 않음)

| 폴더 | 해상도 | fps | 길이 |
|---|---|---|---|
| elv_falldown | 1920×1080(103) / 352×240(46) / 704×480(1) | 25~50 | 6~10초 |
| esc_falldown | 1920×1080(91) / 1280×720(41) / 960×480(11) / 기타 7 | **7~30** | 5~20초 |
| elv_violence | 1920×1080(102) / 352×240(41) / 704×480(7) | 25~50 | 6~10초 |

`esc_falldown` 에 **352×240 등 480p 미만**이 섞여 있다. 시험 성적서에는 해상도 구성을 함께 적는다.

## 사용법

```bash
export HF_TOKEN=hf_xxx
python -m eval.tta download -d TTA_3종_엘쓰_에쓰_엘폭   # → eval/datasets/TTA_3종_엘쓰_에쓰_엘폭/
python -m eval.tta stats    -d TTA_3종_엘쓰_에쓰_엘폭
```

## 출처·주의

- 원천: 사내 NAS 200TB `10.128.30.51:/volume1/home/sj.park/tta_dataset` 의 클립 벤치마크.
  `helmet`(안전모 미착용)은 TTA 12종에 없는 항목이라 **제외**했다.
- 정답은 원본의 `gt_label.zip`(정규화된 정답지)에서 가져왔다. 원본 폴더 안의
  `per_frame/*.csv` 는 subtype 별로 컬럼명이 달라(예: 폭력 하드네거티브가 `sittingdown`)
  그대로 쓰면 라벨이 어긋난다.
- 영상은 원본과 md5 동일(무작위 표본 검증).
