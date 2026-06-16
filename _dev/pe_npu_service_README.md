# PE Vision Encoder NPU 대체 — 테스트/통합 준비

Product-AI-mono `perception_encoder` 모듈의 **비전 인코더 추론 파트(현재 TensorRT)를
Mobilint NPU로 대체**하는 신규 서비스 모듈을 위한 준비 코드와 테스트 절차.

현재 `perception_encoder/service.py`는 다음과 같이 추론한다:

```python
self.model = TRTInference(PERCEPTION_ENCODER_TRT_PATH)   # trt_load.py
visual_vectors = self.model(image_cuda)                  # (B,3,336,336) -> (B,1024)
```

여기서 `TRTInference`를 같은 인터페이스의 **`MXQInference`(NPU)** 로 교체하는 것이 목표다.

## 구성 파일

| 파일 | 설명 | 필요 환경 |
|------|------|----------|
| `mxq_inference.py` | `TRTInference` 호환 NPU 추론 래퍼(`MXQInference`). drop-in 교체용 | qbruntime + NPU |
| `compare_backends.py` | 원본 PyTorch vs ONNX vs NPU 출력 비교 (코사인 유사도) | 백엔드별 선택 |
| `check_npu.sh` | NPU 장착 후 드라이버~런타임 단계별 점검 | NPU 호스트 |
| `README.md` | 본 문서 (절차/체크리스트/통합 가이드) | - |

> MXQ/ONNX 산출물은 `../pe_onnx_export/out/` 에 있다. 생성 방법은 `../pe_onnx_export/README.md` 참고.

## 사전 검증 (NPU 없이 지금 가능)

컴파일 컨테이너(`mblt_compiler`)에서 **원본 PyTorch vs ONNX** 정합성을 먼저 확인한다.
(ONNX가 원본과 일치하면, 남은 변수는 NPU 양자화 오차뿐이다.)

```bash
docker exec mblt_compiler pip install onnxruntime   # 최초 1회
docker exec -w /workspace/AX_NPU/pe_npu_service mblt_compiler \
  python compare_backends.py --onnx ../pe_onnx_export/out/PE-Core-L14-336_vision_sim.onnx
```

## 사전 준비 (호스트 상태 점검 결과 2026-06-15)

| 항목 | 상태 | 비고 |
|------|------|------|
| Secure Boot | **disabled** | MOK 등록 불필요 → 원격에서도 드라이버 설치 가능 |
| linux-headers / build-essential | 설치됨 | 드라이버 빌드 사전요건 충족 |
| Mobilint 드라이버/런타임/cli | **미설치** | 아래 절차로 설치 |
| 받아둔 파일 | 있음 | `../download/` (드라이버 v1.13, 런타임 v1.2.0) |
| 컨테이너 런타임(qbruntime) | **설치 완료** | `mblt_compiler`에 cp310 wheel 설치·import 검증됨 |

**추론 실행 환경은 컨테이너**다. 호스트 conda는 Python 3.13인데 런타임 wheel은 cp38~cp312만
제공되어 호스트 직접 설치가 불가능하다. 컨테이너(Python 3.10)에는 이미 qbruntime을 설치해 두었다.

### 호스트 드라이버 설치 (카드 장착 전 미리 가능)

```bash
sudo bash prepare_host_for_npu.sh   # APT 저장소 + mobilint-aries-driver + mobilint-cli
```

드라이버는 dkms로 빌드되며, 카드를 꽂고 재부팅하면 자동 로드되어 `/dev/aries0`이 생성된다.

## NPU 장착 후 테스트 절차 (체크리스트)

- [ ] **1. (전원 OFF) 카드 장착 → 부팅 → 재부팅 1회**
- [ ] **2. 호스트 환경 점검** — `bash check_npu.sh`
  - 드라이버 모듈 / `/dev/aries0` / PCI(209f) 통과 확인
  - 실패 시 `.claude/setup-notes.md`의 추론 환경 절차로 해결
- [ ] **3. 펌웨어 최신화** — `aries_flash_firmware update` (필요 시 재부팅)
- [ ] **4. 추론 컨테이너 (재)생성** — 컴파일 컨테이너에 NPU 디바이스를 연결
  ```bash
  docker rm -f mblt_compiler 2>/dev/null
  docker run -dit --gpus all --ipc=host --name mblt_compiler \
    --device /dev/aries0:/dev/aries0 \
    -v /home/gpuadmin/Repo/seoik/AX_NPU:/workspace -w /workspace \
    mobilint/qbcompiler:1.1-cuda12.8.1-ubuntu22.04 /bin/bash
  # 컴파일러 + 런타임 재설치
  docker exec mblt_compiler pip install /workspace/AX_NPU/download/qbcompiler-1.1.2+aries2-py3-none-any.whl
  docker exec mblt_compiler bash -lc 'cd /tmp && tar xzf /workspace/AX_NPU/download/qbruntime_aries2-v4_v1.2.0_amd64.tar.gz && pip install /tmp/qbruntime_aries2-v4_v1.2.0_amd64/qbruntime/qbruntime/python/*cp310*.whl'
  ```
- [ ] **5. MXQ 로드 + 단독 추론**
  ```bash
  docker exec -w /workspace/AX_NPU/pe_npu_service mblt_compiler \
    python mxq_inference.py --mxq ../pe_onnx_export/out/PE-Core-L14-336_vision.mxq --core-mode single
  # 배치(멀티코어):
  docker exec -w /workspace/AX_NPU/pe_npu_service mblt_compiler \
    python mxq_inference.py --mxq ../pe_onnx_export/out/PE-Core-L14-336_vision_all.mxq --core-mode multi --batch 4
  ```
- [ ] **6. 원본 대비 정확도 비교**
  ```bash
  docker exec -w /workspace/AX_NPU/pe_npu_service mblt_compiler \
    python compare_backends.py \
      --onnx ../pe_onnx_export/out/PE-Core-L14-336_vision_sim.onnx \
      --mxq  ../pe_onnx_export/out/PE-Core-L14-336_vision.mxq
  ```
  - 판정: `pth vs npu` 코사인 유사도 ≥ 0.99 면 양호. 낮으면 calibration 데이터를
    실제 도메인 이미지로 바꿔 재컴파일(`../pe_onnx_export/compile_pe.py`의 `use_random_calib` 해제).
  - 참고: `pth vs onnx`는 이미 cos=1.000000 검증됨 → NPU 차이는 곧 양자화 오차.
- [ ] **7. 코어 모드별 성능/정확도 스윕** — `_all.mxq` + `--core-mode` (single/multi/global4/global8)
- [ ] **8. 서비스 통합** — `service.py`의 `TRTInference`를 `MXQInference`로 교체 (아래)

## 신규 서비스 모듈 통합 방향

`perception_encoder`를 복제해 NPU 버전 모듈(예: `perception_encoder_npu`)을 만들 때,
바꾸는 핵심은 **모델 로딩/추론 한 곳**이다. 인터페이스가 동일하므로 나머지(전처리,
이벤트, ROI, 텍스트 벡터)는 그대로 재사용한다.

```python
# 기존 (service.py)
from .trt_load import TRTInference
self.model = TRTInference(PERCEPTION_ENCODER_TRT_PATH)

# NPU 버전
from .mxq_inference import MXQInference
self.model = MXQInference(PERCEPTION_ENCODER_MXQ_PATH, core_mode="single")
```

호출부 `visual_vectors = self.model(image_cuda)` 는 그대로 둔다 (입출력 시그니처 동일).

### 통합 시 확인할 점

- **입력 레이아웃/dtype**: NPU MXQ의 기대 입력(NCHW vs NHWC, float32)이 컴파일 설정에
  따라 다를 수 있다. `MXQInference`가 모델 input shape를 조회해 자동 정렬하지만,
  실제 NPU에서 한 번 검증 필요. 현재 PE는 normalize를 모델 밖(`preprocess_image`)에서
  수행하므로 MXQ 입력은 정규화된 float32다.
- **출력 정규화**: `service._init_default_values`는 `zero_mask_vec`를 norm으로 나눈다.
  NPU 출력도 동일하게 후처리하면 된다 (벡터 차원 1024 동일).
- **배치 처리**: `multi` 모드(4-batch) 또는 `single`+멀티스레딩. 스트림 수와 지연 요구에
  맞춰 선택. `service._detect`는 stream 배치를 한 번에 인코딩하므로 `multi`가 자연스럽다.

## 한계 / 미해결

- **batch 차원**: 현재 MXQ는 단일 샘플(batch=1) 기준이며, 배치는 NPU 코어 모드(multi)로
  처리한다. `MXQInference.infer`는 입력 배치를 순회해 코어에 분산하는 단순 구현이다.
  최고 처리량이 필요하면 멀티스레딩/async 분산을 별도 설계해야 한다(`docs/advanced_usage.md`).
- **text encoder 미포함**: 이번 변환은 vision encoder만 다룬다. `perception_encoder`는
  텍스트 피처를 `text_features.json`으로 미리 보관해 쓰므로 vision만으로 충분할 수 있으나,
  텍스트도 NPU화하려면 별도 컴파일이 필요하다.
- **정확도 기준선**: NPU 비교 전, 원본 PyTorch vs ONNX 정합성을 먼저 통과시켜야 NPU
  양자화 오차를 분리해 평가할 수 있다.
