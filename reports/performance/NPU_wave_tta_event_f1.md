# PIA_Wave zero-shot 이벤트 탐지 NPU 이식 + TTA 프레임 F1 (falldown/fire/smoke)

`third_party/PIA_Wave`(GPU/TensorRT)의 zero-shot 이벤트 탐지를 ARIES NPU(PE-Core MXQ, full NPU)로
옮기고, `eval/datasets/TTA_인증용` 200영상에서 **프레임 레벨 per-category F1**을 측정·최적화했다.
프롬프트 최적화는 `third_party/APO-AI-GUI`의 부분집합 선택 아이디어를 이 목적함수에 맞춰 이식했다.
코드는 전부 `wave_npu/` (third_party는 gitignore라 원본 무수정).

측정 서버: NPU 8장 중 **4장(aries1/3/4/5)만 유휴**, 나머지는 타 프로세스 점유. GPU 없음(CPU 96코어).

---

## 1. 결론

**목표(3종 모두 F1 90% 이상) 달성.** native 24fps 전 프레임 72,200개, 전체 200영상 교차 음성.

| 카테고리 | **F1** | P | R | 규칙 | 평활창 | 양성비율 | "전부 양성" 자명 F1 |
|---|---|---|---|---|---|---|---|
| falldown | **0.9805** | 0.9926 | 0.9686 | `mean_margin` | 25f (1.0s) | 9.8 % | 0.178 |
| fire | **0.9841** | 0.9918 | 0.9765 | `zmean_margin` | 13f (0.5s) | 22.4 % | 0.366 |
| smoke | **0.9153** | 0.9269 | 0.9040 | `topmean_margin` | 25f (1.0s) | 42.9 % | 0.601 |
| **macro** | **0.9600** | | | | | | |

- 원본 PIA_Wave 설정(규칙 `iou_std` + 프롬프트 16,125개 전부)의 macro **0.8919 → 0.9600**.
- 임계값·프롬프트를 train 영상에서만 정하고 **한 번도 안 본 test 영상**에서 재도 macro
  **0.9525~0.9540** (§5) — 이 숫자는 in-sample 상한에만 기대고 있는 게 아니다.
- 프롬프트는 16,125개 중 **13개**만 쓴다. 텍스트 임베딩 비용도 그만큼 줄어든다.
- 남은 병목은 **smoke**(0.915). 오차의 대부분은 연기가 옅은 영상 전체 미탐과, fire 영상에서
  연기 라벨 구간 밖 오탐 — 라벨 경계 불일치가 섞여 있어 순수 모델 오차만은 아니다(§8).

## 2. 평가 프로토콜 (이 문서의 숫자를 읽는 법)

- **프레임 레벨 per-category F1.** 카테고리마다 독립적인 이진 판정(해당 프레임에 그 이벤트가 있나).
- **음성 = 전체 200영상 교차.** 어떤 영상의 라벨에 카테고리 c 이벤트가 없으면 그 영상의 모든 프레임은
  c에 대해 음성으로 센다(intrusion 50영상 포함).
- **임계값은 200영상 in-sample 최적** = 상한 성능. 과적합 폭은 영상 단위 8:2 분할로 따로 확인.
- 영상 파일명이 폴더 간 중복(176 unique / 200 files)이라 키는 `<폴더>/<파일명>`.

### 왜 이 프로토콜인가 — 음성을 어디서 잡느냐가 지표를 좌우한다

| 카테고리 | 전체 200영상 양성비율 | "전부 양성" 자명 F1 | 해당 카테고리 라벨된 영상만 |
|---|---|---|---|
| falldown | 9.4 % | 0.172 | 0.546 |
| fire | 22.0 % | 0.361 | **0.937** |
| smoke | 42.2 % | 0.594 | **0.915** |

이벤트가 15초 영상의 85%를 덮기 때문에, 해당 카테고리 영상만 모아 재면 fire·smoke는
**아무 모델 없이 "항상 이벤트"만 찍어도 90%를 넘는다.** 그래서 교차 음성(전체 200영상)을 쓴다.

## 3. 구조 — 임베딩을 한 번만 뽑는다

무거운 단계는 `비디오 → 프레임 → PE-Core 임베딩(NPU)` 하나뿐이다. 이걸 캐시하면 프롬프트·결정규칙·
임계값·평활 실험이 전부 캐시 위 numpy 연산이라 초 단위로 반복된다. 이 분리가 이번 작업에서
수백 번의 실험을 가능하게 한 유일한 이유다.

| 단계 | 자원 | 규모 | 시간 |
|---|---|---|---|
| 비디오 → 임베딩 | **NPU 4장** (W8A16, single, 64 img/s) | 200영상 × 361프레임 = 72,200 | **21.7분**, 1회 |
| 프롬프트 → 텍스트 임베딩 | CPU (PE text tower) | 16,125 | 6분, 1회 |
| 이후 모든 실험 | CPU numpy | — | 0.15 s/평가 |

전 프레임(24fps)을 뽑아두므로 2/3/6fps 실험은 재디코딩 없이 서브샘플이다.
탐색 루프용 고속 경로(`FastMeanScorer`)는 클래스 평균·분산을 마스크 벡터와의 행렬-벡터 곱으로
구해 gather를 없앴다 — 기존 대비 **비트 동일, 5배** (0.81 s → 0.15 s/평가).

## 4. 무엇이 점수를 올렸나

기여를 큰 순서로. 모든 수치는 전체 200영상 in-sample macro F1.

| # | 조치 | macro F1 | 비고 |
|---|---|---|---|
| 0 | 원본 PIA_Wave 규칙(`iou_std`) 그대로, 프롬프트 16,125개 전부 | 0.8919 | smoke 0.768 이 병목 |
| 1 | **결정 규칙 교체** (`iou_std` → `topmean_margin`/`mean_margin`) | 0.9425 | +0.051 |
| 2 | **프롬프트 선택** (16,125 → 13개, APO 이식 좌표상승법) | 0.9526 | +0.010 |
| 3 | **카테고리별 규칙·평활창 독립 선택** + native 24fps 전 프레임 | **0.9600** | +0.007 |

### (1) 결정 규칙 — 원본 `iou_std` 가 가장 나빴다

원본은 프레임마다 normal 프롬프트 유사도 분포와 이벤트 프롬프트 유사도 분포의
`[mean±std]` 구간 IoU를 재고 "IoU가 낮으면 이벤트"로 본다. 분포의 **폭(std)** 에 크게 의존하는데,
프롬프트를 몇 개만 남기면 std가 무너져 지표가 불안정하다. 단순 평균 차이(`mean_margin`)나
프레임 내 z-정규화 평균 차이(`zmean_margin`)가 전 구간에서 더 낫다.

규칙별 성능이 카테고리마다 갈린다 — 임계값이 이미 카테고리별이므로 규칙도 카테고리별로 고르는 것이
정당하고, 실제로 그렇게 해야 이득이 난다.

전체 16,125 프롬프트 기준, 규칙별 카테고리 최고 F1 (평활창은 각자 최적):

| 규칙 | falldown | fire | smoke | 최고 macro |
|---|---|---|---|---|
| `iou_std` (원본) | 0.977 | 0.938 | **0.768** | 0.8919 |
| `mean_margin` | **0.982** | 0.943 | 0.897 | 0.9389 |
| `zmean_margin` | 0.932 | 0.969 | 0.901 | 0.9261 |
| `topmean_margin` | 0.977 | 0.956 | **0.899** | **0.9425** |
| `max_margin` | 0.941 | 0.955 | 0.831 | 0.9076 |
| `topk_frac` (APO 판정 규칙) | 0.936 | **0.972** | 0.689 | 0.8625 |
| `softmax` | 0.932 | 0.965 | 0.663 | 0.8487 |

원본 규칙이 무너지는 곳은 정확히 **smoke**(0.768)다.

`topk_frac`(상위 k개 프롬프트 다수결 = APO-AI-GUI의 판정 규칙)은 fire엔 좋지만 smoke엔 붕괴한다.
smoke 프롬프트가 normal의 "안개/연무/먼지" 프롬프트와 순위 경쟁에서 계속 밀리기 때문이다.

### (2) 프롬프트 선택 — 16,125개 중 13개

프롬프트 풀은 완전한 **factorial 구조**였다: `장면(25) × 인원상황(15) × 이벤트문구(클래스별 22/8/6/6)`.
그래서 APO의 무구조 16k 부분집합 탐색 대신 **세 축 위의 좌표상승법**(축마다 backward 제거 + forward
추가를 둘 다 시도해 좋은 쪽 채택)으로 탐색했다 — 탐색 공간이 25+15+42로 줄어 빠르고 과적합도 덜하다.

선택 결과(13개):

```
장면      : It is a gas station
인원상황  : People are scattered throughout the area
normal    : Clouds are drifting by
falldown  : lying on the floor / lying down on the floor / fallen face-down /
            upper body lying / lower body lying / legs of a person lying / upper body of a person lying  (7개)
fire      : Flames are burning / A fire is blazing / The fire is spreading  (3개)
smoke     : Dense smoke is filling the area  (1개)
```

장면이 "지하철역"이 아니라 "주유소"로 뽑힌 것을 의미로 해석하지 말 것. 모든 프롬프트에 같은 접두
문장이 붙고 판정은 카테고리 간 **상대** 유사도 차이만 쓰므로, 이 축은 "장면을 맞히는" 역할이 아니라
카테고리 분리를 가장 크게 벌려주는 문맥을 고르는 역할에 가깝다(기전은 검증하지 않았다).
`min_keep=4`로 5개 장면(library/lobby/public park/school classroom/office)을 강제해도 성능이
같다는 점이 이 해석을 뒷받침한다.

### (3) 효과가 없었거나 해로웠던 것

- **비디오별 기준선 보정**(프레임 점수에서 그 영상의 중앙값을 뺌): macro 0.942 → **0.612**. 치명적.
  smoke 이벤트가 영상의 84%를 덮어서 영상 내부 분위수를 빼면 신호 자체가 사라진다. 카메라별 오프셋
  제거라는 통상의 처방이 이 데이터에선 정확히 반대로 작동한다.
- **카테고리별 독립 프롬프트 선택**(카테고리마다 마스크를 따로 탐색): macro 0.9600 — 단일 마스크 +
  카테고리별 규칙(0.9600)과 **동일**. 복잡도만 늘어 채택하지 않았다.

## 5. 과적합 검증 — 프롬프트 선택까지 train 에서만

확정 프로토콜은 200영상 in-sample 튜닝이지만, 13개로 붕괴한 선택이 브리틀한지 확인하기 위해
**영상 단위 8:2 층화분할**로 프롬프트 선택·임계값 결정을 train 에서만 하고 test 에서 측정했다.

| 설정 | 프롬프트 | train macro | **test macro** | test falldown / fire / smoke |
|---|---|---|---|---|
| min_keep=1 | 14개 | 0.9551 | **0.9525** | 0.974 / 0.969 / 0.915 |
| min_keep=4 | 420개 | 0.9513 | **0.9540** | 0.984 / 0.965 / 0.913 |

train→test 하락이 0.003 이내이고, 프롬프트를 420개로 늘려 붙잡아도 차이가 없다.
**13개로 줄인 것은 과적합이 아니다.** 미지 영상에서도 세 카테고리 모두 0.91 이상을 유지한다.

## 6. NPU 이식 — 무엇을 어떻게 바꿨나

원본 PIA_Wave는 `encoders.py`에 pluggable 인코더 레지스트리가 있어서, **vision tower만 갈아끼우면**
나머지 파이프라인은 그대로다. NPU 전처리(`pe_npu.preprocess`)와 PIA_Wave 전처리가 이미 동일
(`uint8 → resize 336 bilinear+antialias → /255 → normalize 0.5`)이라 변환 계층도 필요 없었다.

- `wave_npu/encoder.py` — `MXQInferenceFull` 을 감싼 `ZeroShotEncoder` 구현을 레지스트리에 등록.
  원본 `--model_type pe-core-trt` → `--model_type npu`.
- `wave_npu/pia_wave.py` — 원본 스크립트를 그 인코더와 함께 실행하는 런처.
  `third_party/`는 gitignore 대상(외부 레포 클론)이라 **원본 파일은 한 줄도 고치지 않았다.**
- text tower는 추론 경로가 아니다(프롬프트당 1회). CPU torch 그대로 둔다 — 이 서버엔 GPU가 없다.

NPU 처리량(유휴 4장, W8A16 / single, 배치 64~256): **64 img/s** (카드당 16 img/s, CLAUDE.md 기록과 일치).
배치 크기를 64→256으로 키워도 변화 없어 NPU 바운드다. 200영상 전 프레임 72,200장 = 21.7분.

## 7. 재현

```bash
set -a; . ./.env; set +a          # HF_TOKEN (MXQ private 레포)
PY=~/miniconda3/envs/pe_npu_host/bin/python

mobilint-cli status                                   # 유휴 카드 확인 후 --device-ids 지정
$PY -m wave_npu.embed  --device-ids 1,3,4,5 --batch 128    # 21.7분
$PY -m wave_npu.text                                        # 6분
$PY -m wave_npu.run baseline --fps 2 --wins 1,3,5,7,9,13 --cals none,median
$PY -m wave_npu.run optimize --fps 2 --rule mean_margin --win 5 --rounds 3 --out out/wave/mask_mean.npz
$PY -m wave_npu.run combo    --fps 0 --mask out/wave/mask_mean.npz --wins 1,13,25,37,49,61,73
$PY -m wave_npu.run optimize --fps 2 --rule mean_margin --win 5 --rounds 3 --holdout   # 일반화 확인
```

산출물: `out/wave/` (임베딩 캐시 190 MB, 텍스트 임베딩, 마스크, 결과 json).

## 8. 남은 오차 — smoke 0.915 를 뜯어보면

`diagnose` 기준 오차가 있는 영상은 200개 중 106개, 그중 상위는 두 부류로 갈린다.

- **미탐(FN) — smoke 폴더 영상 전체를 놓침**: 상위 6건 중 5건이 `0FP / 135~275FN` 형태,
  즉 그 영상에서 연기를 아예 못 본다. 연기가 옅거나 화면 일부에만 있는 경우다.
  전역 crop(프레임 전체를 336×336으로 리사이즈)이라 작은 연기가 희석된다 — 타일링/ROI가 다음 후보다.
- **오탐(FP) — fire 폴더 영상**: `265FP / 0FN` 같은 형태. fire 영상에는 smoke 라벨이 함께 달려
  있는데, 라벨 구간(예: 2.29초~)과 실제 연기 발생 시점이 어긋나면 그 앞 구간이 전부 FP로 잡힌다.
  **라벨 경계 불일치**가 섞여 있어, 이 부분은 모델을 고쳐 줄일 수 있는 오차가 아니다.

다음에 시도할 만한 것(비용 순):
1. **타일링**(3×3 등) — 작은 연기/쓰러짐의 희석을 막는다. 임베딩 재추출이 필요(타일 수 배).
   PIA_Wave 에 `VERSION_1_SAHI` 와 KD 쪽 `03_PIA_Wave_LoRA_Tiling` 선례가 있다.
2. **프롬프트 생성**(APO의 Perception-LM) — 이번엔 기존 16k 풀에서 **선택만** 했다.
   지하철역 CCTV 도메인 문구를 새로 생성하면 여지가 더 있다. GPU 서버 필요(이 서버엔 GPU 없음).
3. **양자화 스킴** — 현재 W8A16. `reports/performance/NPU_pe_quant_tuning_matrix_120.md` 의
   **W8A16+튜닝(cos 0.9946)** 은 크기·속도가 같은 무상 교체 후보다. mxq만 갈아끼우면 된다.

## 9. 주의

- 이 문서의 수치는 **프레임 레벨**이다. 이벤트 단위(구간 검출) 지표가 필요하면 별도로 정의해야 한다.
- 임계값은 카테고리마다 하나씩, 200영상 전체에서 고른 값이다. 실배포 임계값이 아니라 **평가셋 최적치**다.
- 음성 라벨은 **부재로 추정**한다 — intrusion 영상에 연기가 실제로 없는지는 확인하지 않았다.
  있다면 그만큼 smoke precision 이 과소평가된다.
- 측정 시점에 NPU 8장 중 4장은 타 프로세스가 점유 중이었다. 처리량 수치는 4장 기준이다.
