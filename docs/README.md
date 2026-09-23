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

---

## 버전별 공식 문서 (2026-09-23 수집)

`docs.mobilint.com` 의 **"Download source file"** 이 가리키는 `_sources/<page>.md` **원본 마크다운**을
그대로 받은 것이다(HTML 변환 아님). 재수집: `python reports/scripts/fetch_mobilint_docs.py`.

| 폴더 | 대상 | 페이지 | 비고 |
| --- | --- | ---: | --- |
| [`compiler_v1.3/`](compiler_v1.3/) | **qbcompiler 1.3** | 16 | ★ **신규** — `dynamicRope` 지원 문서 포함 |
| [`runtime_v1.4/`](runtime_v1.4/) | **qbruntime 1.4** | 14 | ★ 신규 |
| 이 폴더 최상위 `*.md` | qbruntime **1.2** 계열 | 15 | 기존 수집본(현 운영 버전) |

### ★ compiler 1.3 에서 확인한 것

- `CHANGELOG.md` — **"Added support for custom masks and dynamic RoPE"** (1.3.0)
- `transformer.md` §Dynamic RoPE and Dynamic Mask — `LlmConfig.Attributes.Runtime` 의
  **`dynamicRope`** / **`dynamicMask`**.
  `dynamicRope=True` 면 RoPE 위치 데이터가 컴파일 고정이 아니라 **런타임 동적**이 된다.
  → 벤더가 말한 "신규 SDK 에서 dynamic RoPE 지원"이 문서로 확인된다.
  단 *"runtime input requirements 가 늘고 최적화 여지가 줄어드니 필요할 때만 켜라"* 는 단서가 붙어 있다.
- `vision.md` — 비전 모델 전용 페이지(1.2 에는 없던 항목)

→ 배경: [`../reports/inquiries/06_qwen3vl_prefill_optimization/REPLY_2.md`](../reports/inquiries/06_qwen3vl_prefill_optimization/REPLY_2.md)
