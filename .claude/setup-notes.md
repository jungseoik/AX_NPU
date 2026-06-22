# AX_NPU 셋업 상세 노트

`CLAUDE.md`의 상세 보충 문서. 필요할 때만 참조한다.

## 하드웨어: ARIES MLA100 PCIe Card

(`docs/aries-mla100-pcie-card.md`)

- Mobilint ARIES NPU 기반 PCIe AI 가속 카드, NPU 아키텍처 코드명 **Aries2**
- 25W 전력, 최대 80 TOPS
- 호스트 요구사항: PCIe x8 슬롯 1개 이상
- 지원 OS: x86-64 / aarch64, Ubuntu 20.04 / 22.04 / 24.04 (22.04.5 권장), Windows 10/11
- 외부 호스트가 필요한 가속기 (SoC 아님)

## 호스트 환경 현황

- OS: Ubuntu 24.04.1 LTS (x86_64) — 지원 범위 내
- Docker 29.5.3, NVIDIA Container Toolkit 1.19.1
- GPU: RTX PRO 6000 Blackwell, 드라이버 580.82.07
- 호스트 Python: miniconda 3.13 (추론 시 conda libstdc++ 충돌 주의 → venv 권장)

## 컴파일 환경

> **전제: NPU가 기본, GPU는 옵션.** 컴파일은 NPU가 아니라 호스트 CPU/GPU에서 도는데, **CPU만으로 가능**하고
> GPU가 있으면 더 빠를 뿐(`--device cpu`(기본) / `--device gpu`(옵션)). 추론은 NPU(`/dev/aries*`)에서만.

### 방법 A — conda (GPU 없는 NPU 서버 기본) ✅
GPU 없는 서버에서 컴파일하는 기본 경로. qbcompiler의 `mmc`(C++ 백엔드)가 **torch 2.7.1 + CUDA torch의
`libtorch_cuda.so`** 에 ABI로 묶여 있어, **torch 버전을 2.7.1로 정확히 맞춰야** 한다(GPU 없어도 .so만 있으면 됨).
```bash
conda create -y -n pe_compile python=3.10
conda activate pe_compile
pip install "torch==2.7.1" "torchvision==0.22.1"          # CUDA 빌드(.so 필요), GPU 없어도 OK
pip install tensorflow-cpu "transformers>=4.54" "onnxruntime>=1.19.2" onnx msgpack tqdm \
            "numpy<2" opencv-python-headless PyYAML pycocotools typeguard pydantic pydantic_settings einops timm huggingface_hub
pip install --no-deps download/qbcompiler-1.1.2+aries2-py3-none-any.whl
# 컴파일 (NPU 서버 기본 = --device cpu)
python -m pe_npu.compile --mode compile --save pe_npu/out/pe_feat.mxq --feat-only --device cpu
```
> ⚠️ torch가 2.7.1이 아니면 `mmc: undefined symbol` / `libtorch_cuda.so 없음` 에러. (이게 컴파일 셋업 시 겪는 함정.)

### 방법 B — Docker (GPU 있을 때 옵션)
GPU 머신이면 공식 이미지가 torch 2.7.1+cu128 내장이라 위 버전맞춤이 불필요(편함). **GPU 없으면 `--gpus all` 빼고**도 시도 가능하나 방법 A 권장.
```bash
docker pull mobilint/qbcompiler:1.1-cuda12.8.1-ubuntu22.04
docker run -dit --ipc=host --name mblt_compiler \
  $( [ -e /dev/nvidia0 ] && echo --gpus all )        # GPU 있을 때만 --gpus all (옵션)
  -v "$PWD/..":/workspace -w /workspace \
  mobilint/qbcompiler:1.1-cuda12.8.1-ubuntu22.04 /bin/bash
docker exec mblt_compiler pip install /workspace/AX_NPU/download/qbcompiler-1.1.2+aries2-py3-none-any.whl
```
- 컨테이너 사전설치: torch 2.7.1+cu128, torchvision 0.22.1, onnx, einops, timm, huggingface_hub, numpy 등.

### 컴파일 재현 명령

```bash
# ResNet50 (기본 동작 검증)
docker exec -w /workspace/AX_NPU/_dev/compile_test mblt_compiler bash -lc 'python model_compile.py \
  --onnx-path ./resnet50.onnx --calib-data-path ./calib_images --save-path ./resnet50.mxq'

# 커스텀 모델 PE-Core-L14-336 → 상세는 tutorial_pe_npu/README.md (python -m pe_npu.compile)
```

### 컴파일 검증 결과 (구 `_dev/compile_test/` — gitignore된 로컬 스크래치, 현재 제거됨. ResNet50 동작검증 기록만 남김)

- `resnet50.onnx` (102MB float32) → `resnet50.mxq` (26MB INT8)
- MXQ 포맷 0x70000 = MXQv7, Hardware Version = Aries2
- `inference_scheme="all"` → Single / Multi / Global / GlobalCluster 4모드 빌드 성공
- calibration은 동작 검증용 랜덤 이미지 사용 (정확도 무의미)

## 추론 환경 (NPU 카드 장착 후 진행)

필요 파일은 `download/`에 이미 받아둠.

1. **드라이버**: `download/mobilint-aries2-driver_v1.13.tar.gz` 또는 `apt install mobilint-aries-driver`
   - Secure Boot 활성화 시 MOK 등록 필요 (SSH 불가, 모니터/키보드 직접 연결)
   - 재부팅 후 `ls -al /dev/aries*` → `/dev/aries0` 확인
   - PCI 인식 확인: `sudo lspci -vd 209f:`
2. **펌웨어**: `aries_flash_firmware update` (인터넷 필요)
3. **런타임**: `download/qbruntime_aries2-v4_v1.2.0_amd64.tar.gz` (x86_64용) 또는 `apt install mobilint-qb-runtime` / `pip install mobilint-qb-runtime`
   - arm64 파일은 이 호스트(x86_64)엔 불필요
4. **유틸리티**: `apt install mobilint-cli` (상태 확인, MXQ 버전, 벤치마크)
5. **추론**: `.mxq`를 NPU에 올려 실행 (`docs/tutorial_resnet50.md`, `docs/programming_guide.md`)

## 버전 호환성

- 받아둔 조합: 드라이버 1.13 / 런타임 1.2.0 / MXQv7 → 상호 호환
- 호환표상 드라이버 1.11+, 런타임 1.0.0+가 MXQv7 지원
- 참조: `docs/compatibility.md`

## 양자화 / Calibration 상세

- ARIES NPU는 정수(INT8/INT4) 연산 전용 하드웨어. float 모델을 NPU에 올리려면 양자화 컴파일 필수. bfloat16/float16 네이티브 추론 경로 없음.
- `weight_dtype`(float32/float16), `CompileConfig.dtype`(float)은 NPU 실행 정밀도가 아니라 **컴파일/calibration 과정의 호스트 연산 정밀도**. 양자화를 끄는 옵션이 아님.
- float 그대로 돌리려면 NPU가 아니라 GPU/CPU에서 원본 모델 실행.
- 정확도 손실 줄이기: 8bit 유지(4bit는 손실 큼), per-channel + percentile calibration, 필요 시 SpinQuant/SearchWeightScale.
- calibration data = 양자화 정수 눈금 범위를 정하려고 모델에 흘려보내는 대표 입력 샘플.
  - 정확도 중요 → 실제 도메인 데이터(예: ImageNet)
  - 컴파일 동작 검증만 → `mxq_compile(use_random_calib=True)` 한 줄로 충분
- 양자화 config 상세: `../mblt-sdk-tutorial/compilation/_guides/01_about_quantization_config.KR.md`

## 참고 자료 위치

- SDK 공식 문서: `docs/`
- 컴파일 튜토리얼/모델별 예제: `../mblt-sdk-tutorial/compilation/` (image_classification, llm, vlm, stt, object_detection 등)
- 사전 컴파일 모델: `../mblt-model-zoo/`
- 다운로드 센터: https://dl.mobilint.com (계정 필요)
- 받아둔 SDK 파일: `download/` / PE 컴파일·추론 패키지: `pe_npu/` / 실험 보관: `_dev/compile_test/`
