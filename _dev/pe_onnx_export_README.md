# PE-Core-L14-336 Vision Encoder → MXQ 변환

Product-AI-mono의 `perception_encoder` 모듈이 사용하는 모델
**PE-Core-L14-336** (Meta Perception Encoder, CLIP 계열)의 vision encoder를
Mobilint ARIES NPU용 **MXQ**로 변환하는 절차와 코드.

가중치는 HuggingFace `facebook/PE-Core-L14-336`에서 받고, vision tower(`model.visual`)만
떼어내 ONNX로 export한 뒤 `qbcompiler`로 MXQ 컴파일한다.

## 결과 요약 (검증 완료, 2026-06-15)

| 단계 | 결과 |
|------|------|
| ONNX export (vision encoder) | 성공. 입력 `(B,3,336,336)` → 출력 `(B,1024)`, 약 1.2GB (fp32) |
| onnxsim 단순화 | 성공. **If 노드 1 → 0** (3965 → 1725 노드) |
| MXQ 컴파일 (qbcompiler, GPU) | **성공.** 829 layers, Aries2, MXQ v7, 약 314MB |

핵심 발견: PE의 RoPE/위치임베딩 동적 조건문이 ONNX `If` 노드로 남는데, qbcompiler가
`If`의 서브그래프 속성(`then_branch`)을 지원하지 않아 컴파일이 막힌다.
**onnxsim 단순화로 `If` 노드를 제거하는 단계가 필수.** (export 스크립트에 통합되어 있음)

> 이는 "NPU에서 실행 가능한 형태로 컴파일됨"을 검증한 것이다. 실제 추론(정확도/속도)은
> NPU 카드 장착 후 가능하며, 현재 calibration은 동작 검증용 랜덤 데이터를 사용했다(정확도 무의미).

## 사전 준비

- 컴파일 컨테이너 `mblt_compiler` 가 떠 있어야 한다. 이 워크플로우는 Product-AI-mono의
  perception_models 코드를 참조하므로, **레포 상위 디렉토리를 통째로 마운트**한다.

```bash
# 상위 디렉토리(AX_NPU)를 /workspace로 마운트 → AX_NPU와 Product-AI-mono 모두 접근 가능
docker rm -f mblt_compiler 2>/dev/null
docker run -dit --gpus all --ipc=host --name mblt_compiler \
  -v /home/gpuadmin/Repo/seoik/AX_NPU:/workspace -w /workspace \
  mobilint/qbcompiler:1.1-cuda12.8.1-ubuntu22.04 /bin/bash

docker exec mblt_compiler pip install /workspace/AX_NPU/download/qbcompiler-1.1.2+aries2-py3-none-any.whl
docker exec mblt_compiler pip install onnxsim   # If 노드 제거에 필요
```

컨테이너에는 torch 2.7.1+cu128 / torchvision / einops / timm / huggingface_hub 가 이미 설치되어 있다.

## 재현 절차

### 1. ONNX export (+ 자동 단순화)

```bash
docker exec -w /workspace/AX_NPU/pe_onnx_export mblt_compiler \
  python export_pe_onnx.py --out ./out/PE-Core-L14-336_vision.onnx --device cpu
```

- HuggingFace에서 `PE-Core-L14-336.pt` 가중치를 다운로드한다 (최초 1회, 인터넷 필요).
- vision encoder를 ONNX로 export하고, 이어서 onnxsim으로 단순화(If 노드 제거)한다.
- `--no-simplify` 로 단순화를 끌 수 있으나, 그러면 다음 컴파일 단계가 실패한다.
- perception_models 경로는 기본적으로 이 스크립트 기준 `../../Product-AI-mono/packages` 를 사용한다.
  다르면 환경변수 `PE_PACKAGES_PATH` 로 지정한다.

### 2. MXQ 컴파일

```bash
docker exec -w /workspace/AX_NPU/pe_onnx_export mblt_compiler \
  python compile_pe.py \
    --onnx ./out/PE-Core-L14-336_vision.onnx \
    --save ./out/PE-Core-L14-336_vision.mxq \
    --scheme single
```

- `--calib-data-path` 없이 실행하면 `use_random_calib=True`(동작 검증, 정확도 무의미).
- `--scheme` 은 `single`(기본) / `multi` / `global` / `global4` / `global8` / `all` 중 선택.
  멀티코어 활용까지 보려면 `all`.

### 2-1. (정확도용) 실제 calibration 재컴파일

랜덤 calibration MXQ는 NPU 동작 검증용이라 정확도가 낮을 수 있다. 실제 정확도가 필요하면
도메인 이미지로 calibration 데이터를 만들어 재컴파일한다. (원리: 양자화 INT8 범위를 실제
activation 분포로 산출 — `.claude/setup-notes.md` 양자화 섹션 참고)

**calibration 데이터 준비** (`prepare_calib.py` — PE 전처리 336+normalize 0.5 적용):

```bash
# COCO val2017 (비gated, 권장). 먼저 이미지 다운로드:
#   wget http://images.cocodataset.org/zips/val2017.zip && unzip val2017.zip
docker exec -w /workspace/AX_NPU/pe_onnx_export mblt_compiler \
  python prepare_calib.py --dataset coco --src ./coco/val2017 --num 1000 --out ./calib_coco

# ImageNet-1k (HF gated: 'hf auth login' + 데이터셋 약관 동의 필요)
docker exec -w /workspace/AX_NPU/pe_onnx_export mblt_compiler \
  python prepare_calib.py --dataset imagenet --num 1000 --out ./calib_imagenet
```

**재컴파일** (`--calib-data-path` 지정 → per-channel + percentile 클리핑 적용):

```bash
docker exec -w /workspace/AX_NPU/pe_onnx_export mblt_compiler \
  python compile_pe.py \
    --onnx ./out/PE-Core-L14-336_vision_sim.onnx \
    --save ./out/PE-Core-L14-336_vision_coco.mxq \
    --calib-data-path ./calib_coco --scheme all
```

데이터셋 추천: PE는 범용 비전 인코더라 COCO(다양한 실사 장면, 감시와 유사)로 충분.
ImageNet(단일객체 중심)과 둘 다 만들어 카드에서 정확도(코사인 유사도)를 비교하면 선택 근거가 된다.
운영 CCTV 프레임이 확보되면 그것이 최선.

### 3. 결과 확인

```bash
ls -la out/PE-Core-L14-336_vision.mxq          # 약 314MB
# MXQ 헤더: magic ".MXQ", format 0x00000700 = MXQv7
```

## 파일 구성

| 파일 | 설명 |
|------|------|
| `export_pe_onnx.py` | PE vision encoder를 HF에서 받아 ONNX로 export + onnxsim 단순화 |
| `compile_pe.py` | ONNX → MXQ 컴파일 (qbcompiler) |
| `out/` | 생성물 (onnx, mxq) — 용량이 커서 git 커밋 대상에서 제외 권장 |

## 주의사항 / 한계

- **dynamic batch**: export 시 batch 축을 dynamic으로 두었으나, export 단계에서 RoPE 관련
  `TracerWarning`이 발생한다. onnxsim은 입력 shape를 `[1,3,336,336]`로 고정해 단순화하므로,
  현재 MXQ는 batch=1 기준이다. 배치 추론이 필요하면 `inference_scheme` 또는 고정 batch export를 검토할 것.
- **정확도**: 랜덤 calibration이라 정확도는 의미 없다. 실제 추론 정확도가 필요하면
  도메인 이미지로 calibration을 다시 수행해야 한다 (`compile_pe.py`에서 `use_random_calib`를
  끄고 `calib_data_path` 지정).
- **ONNX 크기**: vision encoder fp32 ONNX가 약 1.2GB로 크다. 디스크 여유를 확인할 것.

## 다음 단계 (NPU 카드 장착 후)

`docs/` 와 `.claude/setup-notes.md` 의 추론 환경 구축 절차(드라이버 → 펌웨어 → 런타임)를 따른 뒤,
이 `PE-Core-L14-336_vision.mxq` 를 런타임(`qbruntime`)으로 NPU에 올려 추론을 검증한다.
