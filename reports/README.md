# reports/ — 분석·벤치마크 문서 인덱스

PE-Core-L14-336 NPU 추론 프로젝트의 실측/분석 문서를 주제별로 정리.

## 📊 performance/ — 성능 (지연·처리량·병렬화·컴파일)
| 문서 | 내용 |
|------|------|
| [NPU_batch_latency.md](performance/NPU_batch_latency.md) | 단일 NPU 배치 지연/처리량, 코어 모드, bit4 양자화 한계 (실측) |
| [NPU_multicard_62ch_benchmark.md](performance/NPU_multicard_62ch_benchmark.md) | 멀티카드(7×ARIES=56코어) 1→62채널 분산 추론 지연 (실측) |
| [NPU_coremode_benchmark.md](performance/NPU_coremode_benchmark.md) | 코어모드 4종(Single/Multi/Global4/Global8) × 다채널 지연·메모리 (현재 서버 실측) |
| [NPU_pipeline_stage_latency.md](performance/NPU_pipeline_stage_latency.md) | _detect 파이프라인 단계별(전처리/NPU/pool/event) + e2e 지연, 채널 스윕(1~16) |
| [NPU_preprocess_parallel.md](performance/NPU_preprocess_parallel.md) | 고채널 병목인 CPU 전처리 병렬화 (스레드/멀티프로세스) |
| [NPU_poolhead_batch_nogain.md](performance/NPU_poolhead_batch_nogain.md) | CPU pool head 배치/스레드 최적화가 무효한 이유 (실측) |
| [compile_benchmark.md](performance/compile_benchmark.md) | 컴파일 시간 GPU vs CPU |

## 🔢 quantization/ — 양자화
| 문서 | 내용 |
|------|------|
| [quantization_reference.md](quantization/quantization_reference.md) | INT8/INT4 양자화 배경·근거 |
| [QUANT_TUNING_guide.md](quantization/QUANT_TUNING_guide.md) | 양자화 튜닝 가이드 |

## 🧩 design/ — 설계·해결
| 문서 | 내용 |
|------|------|
| [SOLUTION_single_io_compile.md](design/SOLUTION_single_io_compile.md) | 단일 입출력 컴파일 + hybrid(NPU trunk + CPU pool) 정확도 0.997 해결 |

## ✅ testing/ — 테스트
| 문서 | 내용 |
|------|------|
| [test_results.md](testing/test_results.md) | 컴파일/추론 테스트 결과 종합 |

## 📨 vendor/ — Mobilint 커뮤니케이션
| 문서 | 내용 |
|------|------|
| [mobilint_support_inquiry.md](vendor/mobilint_support_inquiry.md) | Mobilint 기술지원 문의 정리 |

## 🛠 scripts/ · assets/
- `scripts/` — 벤치마크 재현 스크립트 (`bench_*.py`). 실행: `conda activate pe_npu_host` 후 해당 스크립트.
- `assets/` — 벤치 산출물 (csv 원자료, png 차트).
