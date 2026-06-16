# AX_NPU 프로젝트

Mobilint **ARIES MLA100 PCIe Card** (NPU 아키텍처 Aries2) 에서 딥러닝 모델을 컴파일·추론하는 프로젝트.
호스트: Ubuntu 24.04 / x86_64 / GPU RTX PRO 6000.

## 현재 상태

- **컴파일 환경: 구축·검증 완료.** NPU 카드 없이 Docker+GPU만으로 동작.
  - ResNet50 ONNX → MXQ(v7) 컴파일 성공 (`compile_test/`)
  - 커스텀 모델 PE-Core-L14-336 vision encoder → MXQ(v7) 컴파일 성공 (`pe_onnx_export/`). onnxsim으로 ONNX If 노드 제거가 필수였음.
    - `..._vision.mxq` (single, 314MB), `..._vision_all.mxq` (single/multi/global4/global8 전부, 338MB)
- **NPU 서비스 준비: 코드 완료, 실행은 카드 후.** 목표 = `perception_encoder`의 비전인코더(TensorRT)를 NPU로 대체하는 신규 서비스 모듈. (`pe_npu_service/`)
  - `MXQInference` = `TRTInference` 드롭인 대체(qbruntime), 출력 비교 스크립트, NPU 단계별 점검 스크립트 준비됨
- **추론 환경: 미구축.** NPU 카드가 아직 미장착. 카드 장착 후 드라이버→펌웨어→런타임 설치 필요.
- **컨테이너 마운트 주의**: `mblt_compiler`는 현재 레포 상위(`/home/gpuadmin/Repo/seoik/AX_NPU`)를 `/workspace`로 마운트한다(Product-AI-mono 참조용). 따라서 컨테이너 내 경로는 `/workspace/AX_NPU/...`.

## 헷갈리지 말 것 (핵심 규칙)

- **추론은 NPU 카드가 있어야 가능.** 지금은 컴파일까지만 된다. 커스텀 모델도 "컴파일(=NPU 호환성 검증)"은 지금 가능하나 "실제 추론"은 카드 장착 후.
- **NPU에 올리려면 양자화(INT8/INT4) 필수.** ARIES는 정수 연산 전용 하드웨어라 float/bf16 네이티브 추론 경로가 없다. `weight_dtype`/`dtype=float`는 양자화를 끄는 옵션이 아니라 컴파일 과정의 호스트 연산 정밀도일 뿐.
- 컴파일은 상주 Docker 컨테이너 **`mblt_compiler`** 에서 수행 (qbcompiler 1.1.2 설치됨).
- 받아둔 SDK 파일은 `download/`, 컴파일 작업물은 `compile_test/`.

## 상세 참조

전체 현황 대시보드·디렉토리·카드 후 흐름 → **`README.md`** (루트)
NPU 카드 장착 후 통합 테스트 → **`pe_npu_service/run_npu_tests.sh`**
설치 단계·재현 명령·버전 호환·양자화 상세는 → **`.claude/setup-notes.md`**
커스텀 모델(PE) 변환 재현 → **`pe_onnx_export/README.md`**
NPU 서비스 테스트/통합(체크리스트 포함) → **`pe_npu_service/README.md`**
SDK 공식 문서 → `docs/` · 컴파일 예제 → `../mblt-sdk-tutorial/compilation/`

## Skill (참조 규칙)

`mblt-model-zoo` / `mblt-sdk-tutorial` 레포 작업 시 해당 작업 규칙은 `.claude/skills/` 에 복사해 둠.
- `.claude/skills/mblt-model-zoo.md` — Model Zoo 레포 편집/검증 규칙
- `.claude/skills/mblt-sdk-tutorial.md` — SDK 튜토리얼 레포 편집/검증 규칙
