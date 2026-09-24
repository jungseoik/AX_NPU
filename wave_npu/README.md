# wave_npu — TTA 데이터셋 이벤트 탐지 평가 (ARIES NPU)

> **이건 평가(evaluation) 워크스트림이다.** 목표는 TTA 인증용 데이터셋에서 카테고리별
> 프레임 F1 을 맞추는 것이고, 운영 추론 경로(`pe_npu` / `yolo_npu`)와는 목적이 다르다.
> 그래서 **코드·스펙·결정 산출물·문서를 전부 이 패키지 안에서 관리한다** — 레포 전역
> `reports/` 에는 넣지 않는다.
>
> | | 위치 | git |
> |---|---|---|
> | 코드 | `wave_npu/*.py` | ✅ |
> | 문서·보고서 | [`wave_npu/docs/`](docs/) | ✅ |
> | 카테고리 스펙 | [`wave_npu/specs/`](specs/) | ✅ |
> | 결정 산출물 (마스크·배포설정·결과) | [`wave_npu/artifacts/`](artifacts/) | ✅ |
> | 재생성 가능한 캐시 (수백 MB) | `wave_npu/cache/` | ❌ gitignore |


`third_party/PIA_Wave`(GPU/TensorRT)의 zero-shot 영상 이벤트 탐지를 **PE-Core vision tower만
NPU(MXQ, full NPU)로 갈아끼워** 돌리고, TTA 인증용 데이터셋에서 **프레임 레벨 per-category F1**을
측정·최적화한다. 프롬프트 최적화는 `third_party/APO-AI-GUI`의 아이디어(부분집합 선택)를 이 목적함수에
맞춰 옮겨온 것이다.

> `third_party/`는 gitignore 대상(외부 레포 클론)이라 **원본 파일은 건드리지 않는다.**
> NPU 인코더는 레지스트리에 등록만 하고(`wave_npu/encoder.py`), 실험 코드는 전부 이 패키지에 있다.

## 설계의 핵심 — 임베딩을 한 번만 뽑는다

무거운 건 `비디오 → 프레임 → PE-Core 임베딩(NPU)` 하나뿐이다. 이걸 한 번 뽑아 캐시하면
프롬프트·결정규칙·임계값·평활 실험은 전부 캐시 위의 numpy 연산이라 **초 단위로** 반복된다.

```
비디오 200개 × 361프레임 ──NPU 4장──> emb_tta/<cat>/<video>.npz  (72,200 × 1024)   ~22분, 1회
프롬프트 16,125개 ────────CPU────────> text_feats.npz             (16,125 × 1024)   ~6분, 1회
                                    └─> 이후 모든 실험은 CPU numpy
```

## 모듈

| 모듈 | 역할 |
|------|------|
| `embed` | 비디오 → 프레임 → NPU 임베딩 캐시. 전 프레임(24fps)을 뽑아두면 2/3/6fps는 재디코딩 없이 서브샘플 |
| `text` | 프롬프트 CSV → PE-Core text tower 임베딩 (CPU. text는 추론 경로가 아님) |
| `gt` | TTA 라벨(json, 초) → 프레임 레벨 이진 라벨 |
| `data` | 임베딩 + GT 조립, 영상 단위 서브셋 |
| `prompts` | 프롬프트 풀의 factorial 구조 파싱 (장면 25 × 인원상황 15 × 이벤트문구 6~22) |
| `score` | 결정 규칙들(iou_std / mean_margin / topmean_margin / …) + 시간 평활 + 비디오별 보정 |
| `evaluate` | 프레임 레벨 P/R/F1, 임계값 스윕 |
| `optimize` | 프롬프트 선택 최적화 (세 축 좌표상승법) |
| `run` | 실험 드라이버 (baseline / optimize / final) |
| `encoder`, `pia_wave` | PIA_Wave 원본 스크립트를 NPU 인코더로 실행하는 어댑터·런처 |

## 평가 프로토콜 (확정)

- **프레임 레벨 per-category F1** (falldown / fire / smoke).
- **음성 = 전체 200영상 교차.** 어떤 영상의 라벨에 카테고리 c 이벤트가 없으면 그 영상의 모든
  프레임은 c에 대해 음성(intrusion 50영상 포함).
  → 카테고리별 양성비율 falldown 9.4% / fire 22.0% / smoke 42.2%.
  "무조건 양성"만 찍었을 때 F1은 0.17 / 0.36 / 0.59 이므로 지표가 자명하게 달성되지 않는다.
  (해당 카테고리가 라벨된 영상만 모아 재면 fire 0.94·smoke 0.92가 그냥 나와 무의미하다.)
- **임계값은 200영상 in-sample 최적**(= 상한 성능). 과적합 폭 확인용으로 영상 단위 8:2 분할
  held-out 수치도 함께 낸다.
- 영상 파일명이 폴더 간 중복되므로(176 unique / 200 files) 키는 `<폴더>/<파일명>`.

## 사용

```bash
set -a; . ./.env; set +a          # HF_TOKEN (MXQ private 레포)
PY=~/miniconda3/envs/pe_npu_host/bin/python

# 1) 임베딩 캐시 (NPU). 노는 카드를 확인해서 지정할 것 — mobilint-cli status
$PY -m wave_npu.embed --out wave_npu/cache/emb_tta --device-ids 1,3,4,5 --batch 128
# 2) 텍스트 임베딩 (CPU)
$PY -m wave_npu.text  --out wave_npu/cache/text_feats.npz
# 3) 결정규칙 × 평활 × 보정 스윕
$PY -m wave_npu.run baseline --fps 2 --wins 1,3,5,7,9,13 --cals none,median
# 4) 프롬프트 선택 최적화
$PY -m wave_npu.run optimize --rule <best> --win <best> --out wave_npu/artifacts/mask.npz
# 5) 전 프레임 최종 평가 (+ held-out 참고치)
$PY -m wave_npu.run final --fps 0 --mask wave_npu/artifacts/mask.npz --out wave_npu/artifacts/final.json
```

원본 PIA_Wave 스크립트를 NPU로 돌리려면:

```bash
$PY -m wave_npu.pia_wave 02_Inference --model_type npu --npu-devices 1,3,4,5 -v <dir> -o results -c fire ...
```

---

## 새 카테고리 추가 절차 (8종 확장용)

카테고리는 **코드가 아니라 설정**이다(`wave_npu/config.py` + `wave_npu/specs/*.json`).
새 카테고리를 붙일 때 사람이 할 일은 **이벤트 문구 6~10개를 쓰는 것**뿐이다.

### 신호 원천을 먼저 고른다 — 이게 핵심 판단이다

| 유형 | `source` | 예 | 교차 음성이 성립하나 |
|---|---|---|---|
| **장면 상태** (화면이 그 상태인가) | `pe` | fire, smoke, falldown, violence | ✅ 다른 카테고리 영상은 그 상태가 아니다 |
| **객체 등장** (무언가 나타났나) | `person` | intrusion, loitering | ❌ **안 된다** |

객체 등장형이 교차 음성으로 안 되는 이유: intrusion 실측에서 falldown 영상의 **96.2%에 사람이
있는데 intrusion 라벨은 0**이다. 라벨 누락이라 어떤 모델로도 구분할 수 없고, F1 상한이 0.41에
막힌다. 그래서 `eval_folders` 로 자기 폴더 안에서만 평가한다(F1 0.929). **프로토콜이 달라지므로
결과표에 반드시 함께 적는다.**

### 절차

```bash
PY=~/miniconda3/envs/pe_npu_host/bin/python

# 1) 데이터 배치 — 기존과 같은 형식: <카테고리>/*.mp4 + 동명 *.json (timestamp 는 초 단위)
# 2) 문구 작성
cp wave_npu/specs/new_categories_TEMPLATE.json my_cats.json && vi my_cats.json
# 3) 프롬프트 CSV 확장 + 스펙 자동 생성 (장면 25 × 인원상황 15 조합은 도구가 만든다)
$PY -m wave_npu.extend_prompts --in my_cats.json
# 4) 캐시 추출 (새 영상만 — 기존 파일은 건너뜀)
$PY -m wave_npu.embed  --root <데이터루트> --out wave_npu/cache/emb_v2 --device-ids <유휴카드>
$PY -m wave_npu.text   --csv wave_npu/specs/prompts_11cat.csv --out wave_npu/cache/text_v2.npz
$PY -m wave_npu.person --root <데이터루트> --out wave_npu/cache/person_v2 --fps 0 --device-ids <유휴카드>   # person 소스가 있을 때만
# 5) 파이프라인 한 줄 (프롬프트 선택 → 카테고리별 규칙·창 → F1)
$PY -m wave_npu.pipeline --spec wave_npu/specs/tta_11cat.json \
     --emb wave_npu/cache/emb_v2 --text wave_npu/cache/text_v2.npz --person wave_npu/cache/person_v2
```

### 문구 쓰는 요령 (실측에서 효과가 확인된 것)

- 영어 평서문, 8~18단어. **부정문(no / not / without)은 쓰지 말 것** — CLIP 계열은 부정을 제대로
  인코딩하지 못해 반대 의미로 매칭된다.
- 같은 사건을 **다른 각도로** 6~10개. 예: falldown 이 잘 되는 이유는 전신/상반신/하반신/다리처럼
  부분 관찰을 나눠 쓴 문구가 있기 때문이다.
- **`normal_extra`(오탐 유발 정상 장면)가 성능에 크게 기여한다.** 기존 풀이 화재 오탐용으로
  "석양/붉은 조명/렌즈 플레어", 연기용으로 "안개/먼지/렌즈 얼룩", 쓰러짐용으로
  "쪼그려앉기/무릎꿇기/앉기"를 갖고 있는 것이 그 예다. 새 카테고리마다 "이건 이벤트가 아닌데
  비슷해 보이는 장면"을 같이 써 줄 것.
- 장면·인원상황 축은 **쓰지 말 것** — 도구가 조합하고, 어떤 조합이 좋은지는 선택 단계가 정한다.

### 규모 감각 (NPU 4장 기준)

| 단계 | 비용 |
|---|---|
| 임베딩 추출 | 영상 200개(15초, 24fps 전 프레임) = 22분. 영상 수에 비례 |
| 텍스트 임베딩 | 프롬프트 16k = 6분(CPU). 카테고리당 +2,250개(문구 6개 기준) ≈ +1분 |
| 사람 검출 | 영상 200개 전 프레임 = 약 15분 |
| 프롬프트 선택 | 2fps 서브샘플에서 좌표상승 3라운드 = 5~10분(CPU). 카테고리 수에 선형 |
