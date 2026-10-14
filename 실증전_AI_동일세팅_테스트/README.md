# 실증 전 AI 동일세팅 테스트

운영(`macs-inference`)과 **동일한 세팅**으로 2단계 검출 파이프라인을 NPU에서 돌려,
TTA 인증용 데이터셋의 **쓰러짐·화재·연기** 3종을 **프레임 레벨 F1** 으로 평가한다.

## 격리 원칙

- 코드·산출물은 **전부 이 폴더 안**에만 둔다.
- 데이터셋(`eval/datasets/TTA_인증용`)과 `wave_npu/`, `third_party/` 는 **읽기만** 한다.
- `wave_npu` 의 평가 경로(`tta_cert`, `score.py`)는 건드리지 않는다. 규칙 자체가 다르다
  (wave_npu 는 `mean_margin` 등 자체 규칙, 운영은 `topk`).

## 운영과 무엇을 맞췄나

값의 출처는 전부 **살아있는 매니페스트**다. 임의로 고른 값이 없다.

```
kubectl get deploy -n macs-inference es-inference-node0-gpu1 -o json
kubectl get cm     -n macs-inference inference-server-config -o json   # ★ envFrom
```

★ **`env` 만 보면 안 된다.** Deployment 는 `envFrom` 으로 ConfigMap 2개를 붙이고 있고,
`PE_QUEUE_SIZE=3` / `PE_ALARM_DURATION_THRESHOLD=2` 는 **거기에만** 있다.
`env` 에 없다고 "미설정 → config.py 기본값(5창중3)" 으로 단정하면 틀린다(실제로 틀렸다).

| 구분 | 값 | 근거 |
|---|---|---|
| 1차 모델 | PE `W4A16` / `sws_optq` / `single` | `PE_NPU_QUANT`·`PE_NPU_TUNING`·`PE_NPU_SCHEME` |
| 1차 자산 | `text_features.json` 13,449개 | `MODEL_PERCEPTION_ENCODER_TXT_FEATURE_PATH` hostPath 원본 |
| 1차 판정 | `topk`, `TOP_CANDIDATE=13` | `PE_NPU_METHOD` 미설정 → `config.py` 기본값 |
| 알람 FSM | 최근 **3창 중 2회** 이상 → 발화 | ConfigMap `inference-server-config` (env 아님) |
| 입력률 | **1 fps** | `node-gpu-model-mappings.json` PENpuService `fps:1` |
| 2차 모델 | ft bs32 (`model-ft2`) | 실증 로그 `AX_NPU_VL3_2B_mxq:ft/model` 과 sha256 일치 |
| 2차 입력 | 256×192, 1프레임 | 서버 `NPU_PE_2STAGE_VLM_SIZE` 기본값(= 운영값) |
| 2차 문안 | `2stage_prompts` / **`ax_npu`** variant | 서버 `prompts.py` `DEFAULT_VARIANT` |

### MXQ 경로가 왜 HF 캐시인가

매니페스트는 `pe_w8a16_global4.mxq` 를 `/npu-assets/pe_full.mxq` 로 마운트하지만
**`MODEL_PE_NPU_MXQ_PATH` 가 빈 값이라 그 마운트는 참조되지 않는다.**
`service.py` 가 `QUANT/TUNING/SCHEME` 으로 경로를 조립하고, 로컬에 없으면 HF 에서 받는다.
실제로 로드되는 건 HF 캐시의 `W4A16/sws_optq/single/pe_full.mxq` 다. 그걸 그대로 지정했다.

### 2차도 운영 서버를 그대로 쓴다

ENTRYPOINT 를 덮어써서 `Classifier` 를 직접 잡으면 **그 순간 "운영 동일" 이 깨진다** —
`VLMPool` 의 카드 선택(inflight 기준)·배치 묶기(최대 32)·세마포어(96)를 전부 건너뛴다.
그래서 컨테이너를 **있는 그대로** 띄우고 `POST /v1/chat/completions` 로 묻는다.
부수 효과로 빨라진다(162건 18.2초 → 8.4초). 배치가 실제로 묶이기 때문이다.

문안도 서버 기준이다. `ax_tta/vqa_prompts.yaml` 은 내용이 같더라도 **출처가 아니다**.
서버 `prompts.py` 주석: *"실제로 운영 문안이 레포 상수와 어긋난 채로 돌고 있었다."*
카테고리는 폴더명이 아니라 YAML 의 `aliases`(`<cat>_ret`)로 찾는다 — 운영 판정도 그렇다.

### 증거 이미지는 반드시 원본 해상도로

운영은 2차에 **원본 프레임**을 넘긴다. 1차 PE 입력용 336×336(종횡비 파괴)을 그대로
2차에 주면 2차가 보는 그림이 달라져 판정이 뒤집힌다. 실제로 한 번 그렇게 만들었고,
같은 프레임인데 파이프라인은 `no`, 2차 단독은 `yes` 가 나왔다.
`stage1.py` 는 PE용 336 과 증거용 원본 JPEG 를 **따로** 들고 간다.
검증: 증거 JPEG 와 `stage2_only.py` 가 보내는 JPEG 의 md5 가 같아야 한다.

### 판정 로직을 베끼지 않은 이유

전처리·추론·판정·알람 FSM 을 **운영 모듈 그대로 import** 한다.

```python
from pia_prod.AI.modules.pe_npu.event import PENpuEventManager      # topk + 5창3 FSM
from pia_prod.AI.modules.pe_npu.multi_npu import MultiNPUInferenceFull
from pia_prod.AI.modules.pe_npu.parallel_preprocess import ParallelPreprocessor
```

베껴 쓰면 미묘하게 달라지고, **달라졌다는 걸 증명할 방법이 없다.**

## 평가 설계에서 조심한 것

**교차 음성.** `TTA_인증용` 에는 정상(음성) 영상이 **0건**이다. 카테고리 폴더 안에서만
재면 "무조건 이벤트"가 만점을 받아 수치가 무의미해진다. 그래서 각 카테고리를
**150건 전체**에 대해 평가한다 — 해당 폴더 50건이 양성 후보, 나머지 100건이 음성이다.

**1 fps 고정.** 알람 FSM 이 "5창 중 3" 이라, 24fps 로 넣으면 같은 FSM 이 0.2초 창이 되어
운영과 전혀 다른 동작이 된다. 운영 입력률인 1 fps 로 맞춰야 "5초 중 3초" 의미가 산다.

**BGR→RGB.** 운영은 `PE_NPU_BGR2RGB=false`, 즉 "입력은 이미 RGB" 전제다. `cv2` 는 BGR 을
주므로 읽는 쪽에서 변환한다. 안 하면 R↔B 가 뒤집혀 **에러 없이 정확도만 조용히 무너진다.**

**프레임 예측의 정의 = 알람이 떠 있는 프레임**(상태 1 시작 / 2 지속).
2차가 기각한 에피소드는 전체가 음성이 된다(운영에서 알람이 생성되지 않으므로).

**화재 영상은 화재만 본다.** fire 폴더 50건 **전부**가 smoke 라벨을 함께 달고 있다
(불이 나면 연기가 난다). 연기 평가에서 이들을 음성으로 쓰면 실제로 연기가 보이는 화면이라
연기 검출이 전량 오탐이 된다 — 그래서 음성으로 쓰지 않고 **분모에서 뺀다**(`EVAL_EXCLUDE`).
GT 는 폴더명이 아니라 `events[].category` 로 거른다.

**2차 판정은 실행마다 0~4건 흔들린다.** 같은 입력·같은 문안으로 두 번 돌려 162건 중
1건이 뒤집혔다. 설정을 비교할 때는 3회 이상 돌려 범위가 겹치는지 볼 것
(`reports/evaluation/NPU_tta4ax_pe_vs_vlm.md` 에 같은 현상이 기록돼 있다).

## 실행

```bash
# ① 1차 — 호스트 conda (qbruntime 1.2.0)
~/miniconda3/envs/pe_npu_host/bin/python stage1.py

# ② 2차 — 운영 bs32 검증서버를 띄우고 HTTP 로 질의 (서버 기동 ~26s + 162건 8.4s)
SUDOPW=<sudo암호> ./run_stage2.sh

# ③ 2차 단독 (1차·FSM 없이 매 초 전 프레임 VQA) — 서버가 떠 있어야 한다
~/miniconda3/envs/gov_vlm/bin/python stage2_only.py --workers 48

# ④ 프레임 F1 (1차 단독 / 1차+2차 / 2차 단독)
~/miniconda3/envs/gov_vlm/bin/python evaluate_stage2_only.py

# ⑤ 검토용 그리드 페이지
~/miniconda3/envs/gov_vlm/bin/python gen_review.py
(cd out/review && python -m http.server 8901 --bind 127.0.0.1)
```

## 파일

| 파일 | 역할 |
|---|---|
| `settings.py` | 운영 세팅 동결. 바꾸면 "동일" 전제가 깨진다 |
| `stage1.py` | PE NPU 추론 + topk 판정 + 알람 FSM → `out/stage1.json`, `evidence/` |
| `stage2_client.py` | 운영 검증서버에 HTTP 질의 → `out/stage2.json` |
| `run_stage2.sh` | bs32 검증서버 기동 → health 대기 → 위 클라이언트 실행 → 정리 |
| `stage2_only.py` | 2차 단독 — 매 초 전 프레임 VQA → `out/stage2_only.json` |
| `evaluate.py` | 프레임 F1 (1차 단독 / 1차+2차) → `out/f1.json` |
| `evaluate_stage2_only.py` | 세 방식 비교 → `out/f1_compare.json` |
| `gen_review.py` | 판정 검토 그리드(GT/1차/최종/2차단독 4줄) → `out/review/` |

## 결과 (2026-10-14)

프레임 레벨 F1. 150건 2,250프레임(연기는 100건 1,500프레임).

| 카테고리 | 1차 단독 | 1차+2차 | 2차 단독 |
|---|---:|---:|---:|
| falldown | 65.24 | 61.98 | **87.28** |
| fire | 86.62 | 70.20 | 84.27 |
| smoke | **57.43** | 21.34 | 44.72 |
| **매크로** | 69.76 | **51.17** | **72.09** |

**운영 구성(1차+2차)이 세 종목 모두에서 가장 낮다.** 두 단계를 직렬로 걸면 각 단계의
미탐이 누적된다(재현율 1차 49% → 합산 32%).

오탐 기준으로 보면 2차의 가치가 카테고리마다 갈린다.

| | 1차 정밀도 | 1차+2차 정밀도 | 2차가 막은 오탐 | 2차가 깎은 정탐 |
|---|---:|---:|---:|---:|
| falldown | 100.00% | 100.00% | 0 | 알람 2건 |
| fire | 93.37% | **100.00%** | **FP 40프레임·영상 6건 전부** | 알람 26건 |
| smoke | 100.00% | 100.00% | 0 | **알람 72건** |

화재만 2차가 제값을 한다. 쓰러짐·연기는 1차 오탐이 이미 0이라 2차가 줄일 것이 없고
정탐만 깎는다.

★ **단, 이 데이터에는 평범한 일상 장면이 0건이다.** TTA_인증용은 전부 이벤트 영상이고
음성 역할은 "다른 카테고리 이벤트 영상"이 대신한다. DTRO 24시간 실증에서는 falldown
1차가 102건을 올려 94건이 오탐이었다 — 거기선 2차가 실제로 일을 했다.
라우팅을 바꾸려면 정상 장면이 포함된 데이터로 재확인이 필요하다.

### TTA 인증 수치(98.05)와 비교하면 안 되는 이유

| | TTA 인증 | 여기 |
|---|---|---|
| 판정 규칙 | `mean_margin` 등 | 운영 `topk` |
| 프롬프트 | 13개 선별 | 13,449개 배포본 |
| 임계값·평활창 | **이 데이터로 탐색**(in-sample) | 운영 고정값 |
| 알람 FSM | 없음 | 3창중2 |
| 프레임레이트 | 361프레임(24fps) | 15프레임(1fps) |

TTA 수치는 **튜닝 상한**이고 여기 수치는 **배포본 현장 성능**이다.
