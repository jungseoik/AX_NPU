# docs/ — Mobilint ARIES SDK 공식 문서 (인덱스)

Mobilint에서 제공하는 ARIES NPU SDK 공식 문서 묶음(일부 발췌).
문서끼리 상대링크로 상호참조하므로 **원문/파일명은 그대로 두고**, 여기 인덱스로 탐색한다.
(전체 흐름은 [getting_started.md](getting_started.md)가 허브)

## 🚀 시작 · 개요
| 문서 | 내용 |
|------|------|
| [getting_started.md](getting_started.md) | 시작 가이드 — 설치~추론 전체 흐름 허브 |
| [aries-mla100-pcie-card.md](aries-mla100-pcie-card.md) | 하드웨어: ARIES MLA100 PCIe Card 스펙 |
| [compatibility.md](compatibility.md) | 드라이버/펌웨어/런타임/MXQ 버전 호환표 |
| [release_note.md](release_note.md) | SDK 릴리즈 노트 |

## 🔧 설치 · 셋업
| 문서 | 내용 |
|------|------|
| [installing_driver.md](installing_driver.md) | NPU 드라이버 설치 (apt / 소스) |
| [update_firmware.md](update_firmware.md) | 펌웨어 업데이트 |
| [installing_runtime_library.md](installing_runtime_library.md) | 런타임 라이브러리(libqbruntime) 설치 |
| [installing_compiler.md](installing_compiler.md) | 컴파일러(qbcompiler) 설치 |
| [installing_utility.md](installing_utility.md) | 유틸리티(mobilint-cli) 설치 |
| [kubernetes_device_plugin.md](kubernetes_device_plugin.md) | 쿠버네티스 NPU device plugin |

## 💻 프로그래밍 · 사용
| 문서 | 내용 |
|------|------|
| [programming_guide.md](programming_guide.md) | NPU 프로그래밍 가이드 (C++/Python) |
| [advanced_usage.md](advanced_usage.md) | 고급 사용 (멀티스레딩/async 파이프라인 등) |
| [multicore.md](multicore.md) | 멀티코어 활용 (Single/Multi/Global 모드) |
| [utility_usage.md](utility_usage.md) | 유틸리티(mobilint-cli) 사용법 |
| [tutorial_resnet50.md](tutorial_resnet50.md) | ResNet50 모델 실행 예제 (basic) |

---
> 이 문서들은 Mobilint 공식 SDK 문서다. 프로젝트 자체 분석/벤치 문서는 [`../reports/`](../reports/README.md) 참조.
> 다운로드 센터: https://dl.mobilint.com (계정 필요)
