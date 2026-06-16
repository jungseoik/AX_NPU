# AX_NPU — Mobilint NPU 통합 작업 공간

Mobilint **ARIES MLA100 PCIe Card** (NPU 아키텍처 Aries2)에서 딥러닝 모델을 컴파일하고
추론하기 위한 작업 공간. 최종 목표는 Product-AI-mono의 `perception_encoder` 비전 인코더
추론(현재 TensorRT)을 **NPU로 대체하는 신규 서비스 모듈**이다.

호스트: Ubuntu 24.04 / x86_64 / GPU RTX PRO 6000 / SSH 헤드리스 (GUI 불필요, 전부 CLI).

---

## 전체 준비 상태 (2026-06-15)

| 단계 | 상태 | 위치 |
|------|------|------|
| 컴파일 환경 (Docker + GPU + qbcompiler 1.1.2) | ✅ 완료 | 컨테이너 `mblt_compiler` |
| ResNet50 컴파일 검증 | ✅ 완료 | `compile_test/` |
| PE 비전인코더 ONNX export | ✅ 완료 | `pe_onnx_export/` |
| PE → MXQ 컴파일 (single + all 스킴) | ✅ 완료 | `pe_onnx_export/out/` |
| 원본 PyTorch vs ONNX 정합성 | ✅ cos=1.000000 | `pe_npu_service/compare_backends.py` |
| NPU 추론 코드 (`MXQInference` 등) | ✅ 작성·import 검증 | `pe_npu_service/` |
| 컨테이너 런타임 (qbruntime 1.2.0) | ✅ 설치 | 컨테이너 |
| 호스트 드라이버 (aries.ko) 빌드·설치·자동로드 | ✅ 완료 | `/lib/modules/.../updates/aries.ko` |
| **NPU 카드 물리 장착** | ⏳ 대기 | - |
| NPU 실측 추론 / 정확도 검증 | ⏳ 카드 후 | `pe_npu_service/run_npu_tests.sh` |

→ **카드만 꽂으면 끝.** 카드 전 호스트/코드 준비는 모두 완료.

---

## 디렉토리 안내

| 경로 | 내용 |
|------|------|
| `download/` | 받아둔 SDK 파일 (드라이버 v1.13, 런타임 v1.2.0, 컴파일러 whl) |
| `compile_test/` | ResNet50 컴파일 동작 검증 (기본 예제) |
| `pe_onnx_export/` | PE 비전인코더 다운로드→ONNX export→MXQ 컴파일 |
| `pe_npu_service/` | NPU 추론 래퍼 + 테스트 + 호스트/NPU 점검 스크립트 |
| `docs/` | **Mobilint 공식 SDK 문서 (벤더 원본)** |
| `.claude/` | 프로젝트 컨텍스트, 셋업 상세, skill |

## 문서 인덱스 (상세는 여기 참조)

| 문서 | 내용 |
|------|------|
| `.claude/setup-notes.md` | 설치 단계·재현 명령·버전 호환·양자화 상세 |
| `pe_onnx_export/README.md` | PE 모델 ONNX 변환 + MXQ 컴파일 재현 절차 |
| `pe_npu_service/README.md` | NPU 테스트 체크리스트 + 서비스 통합 가이드 |
| `docs/` | 드라이버/런타임/컴파일러 설치, 멀티코어 등 공식 문서 |

---

## NPU 카드 장착 후 진행 흐름

```
1. (전원 OFF) 카드를 PCIe x8 장착 → 부팅
2. sudo modprobe aries                      # 드라이버 로드 (또는 재부팅 시 자동)
3. bash pe_npu_service/check_npu.sh         # 드라이버/디바이스/PCI/런타임 점검
4. (선택) aries_flash_firmware status       # 펌웨어 확인, 구버전이면 update
5. bash pe_npu_service/run_npu_tests.sh     # ★ 통합 테스트 한 번에 실행
```

5번 한 줄이 아래를 **순차 자동 실행**한다:
- 추론 컨테이너 재생성 (`--device /dev/aries0`, 컴파일러+런타임 설치)
- MXQ 로드 + 단독 추론 (single 코어)
- 배치 추론 (multi 코어, batch=4)
- 원본 PyTorch 대비 정확도 비교 (코사인 유사도)

> 자동화 범위: **물리 장착·재부팅·`modprobe`(sudo)는 사람**이, 그 뒤 컨테이너 구성·추론·비교는 **스크립트가 자동**.
