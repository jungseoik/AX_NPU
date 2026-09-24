# wave_npu — PIA_Wave(zero-shot 이벤트 탐지)를 ARIES NPU로

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
$PY -m wave_npu.embed --out out/wave/emb_tta --device-ids 1,3,4,5 --batch 128
# 2) 텍스트 임베딩 (CPU)
$PY -m wave_npu.text  --out out/wave/text_feats.npz
# 3) 결정규칙 × 평활 × 보정 스윕
$PY -m wave_npu.run baseline --fps 2 --wins 1,3,5,7,9,13 --cals none,median
# 4) 프롬프트 선택 최적화
$PY -m wave_npu.run optimize --rule <best> --win <best> --out out/wave/mask.npz
# 5) 전 프레임 최종 평가 (+ held-out 참고치)
$PY -m wave_npu.run final --fps 0 --mask out/wave/mask.npz --out out/wave/final.json
```

원본 PIA_Wave 스크립트를 NPU로 돌리려면:

```bash
$PY -m wave_npu.pia_wave 02_Inference --model_type npu --npu-devices 1,3,4,5 -v <dir> -o results -c fire ...
```
