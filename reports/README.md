# reports/ — 분석·벤치마크 문서 인덱스

PE-Core-L14-336 NPU 추론 프로젝트의 실측/분석 문서를 주제별로 정리.

> ★ **현재 상태: full NPU (image→embedding 전부 NPU, cos 0.99)**. attn_pool INT8 붕괴는
> QKᵀ matmul 16bit로 해결됨 → [vendor/mobilint_resolution_attn_pool.md](vendor/mobilint_resolution_attn_pool.md).
> 그 이전 hybrid(NPU trunk + CPU pool) 분석 문서들은 히스토리로 보존.

## 📊 performance/ — 성능 (지연·처리량·병렬화·컴파일)
| 문서 | 내용 |
|------|------|
| [NPU_full_vs_hybrid.md](performance/NPU_full_vs_hybrid.md) | ★ full NPU vs hybrid — CPU attn_pool 병목 제거 실측 (QKᵀ 16bit) |
| [NPU_full_pipeline_e2e.md](performance/NPU_full_pipeline_e2e.md) | ★ [애프터] full NPU 코어모드 4종 × 채널 스윕 단계별 e2e (Pool 단계 제거) |
| [NPU_throughput_modes_correct.md](performance/NPU_throughput_modes_correct.md) | ★ 다채널 처리량·모드선택 (올바른 1모델+멀티스레드 sync 패턴, 출력검증) |
| [NPU_batch_latency.md](performance/NPU_batch_latency.md) | 단일 NPU 배치 지연/처리량, 코어 모드, bit4 양자화 한계 (실측) |
| [NPU_multicard_62ch_benchmark.md](performance/NPU_multicard_62ch_benchmark.md) | 멀티카드(7×ARIES=56코어) 1→62채널 분산 추론 지연 (실측) |
| [NPU_multicard_62ch_full.md](performance/NPU_multicard_62ch_full.md) | ★ [full NPU] 멀티카드 1→62ch 배치 지연 (비포와 동일 구조, attn_pool도 NPU) |
| [NPU_coremode_benchmark.md](performance/NPU_coremode_benchmark.md) | 코어모드 4종(Single/Multi/Global4/Global8) × 다채널 지연·메모리 (현재 서버 실측) |
| [NPU_1card_coremode_16ch.md](performance/NPU_1card_coremode_16ch.md) | ★ NPU 1장 코어모드 4종 × 1~16채널 순수추론 증가폭 (슬롯 거동·모드선택) |
| [NPU_pipeline_stage_latency.md](performance/NPU_pipeline_stage_latency.md) | _detect 파이프라인 단계별(전처리/NPU/pool/event) + e2e 지연, 채널 스윕(1~16) |
| [NPU_coremode_pipeline_e2e.md](performance/NPU_coremode_pipeline_e2e.md) | **종합**: 코어모드 3종 × 단계별 × 채널(최대 56=7×8) — NPU 병목 어디서 커지는지 |
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
| [SOLUTION_single_io_compile.md](design/SOLUTION_single_io_compile.md) | [비포] 단일 입출력 컴파일 + hybrid(0.997). 현재는 full NPU(QKᵀ16bit, cos 0.99) → vendor/mobilint_resolution_attn_pool.md |

## ✅ testing/ — 테스트
| 문서 | 내용 |
|------|------|
| [test_results.md](testing/test_results.md) | 컴파일/추론 테스트 결과 종합 |

## 📨 vendor/ — Mobilint 커뮤니케이션
| 문서 | 내용 |
|------|------|
| [mobilint_resolution_attn_pool.md](vendor/mobilint_resolution_attn_pool.md) | ★ **[해결]** attn_pool INT8 붕괴 원인(QKᵀ outlier)·해결(score matmul 16bit) → full NPU cos 0.99 |
| [mobilint_inquiry_attn_pool.md](vendor/mobilint_inquiry_attn_pool.md) | attention pooling head INT8 붕괴 문의 (해결됨, 당시 기록 보존) |
| [inquiries/qwen3vl_batch_serving/](vendor/inquiries/qwen3vl_batch_serving/README.md) | Qwen3-VL-2B 배치·모드별 서빙(vLLM) NPU 1장 동시요청 테스트 요청 + vllm-mblt 버그 2건 (미발송) |
| [mobilint_support_inquiry.md](vendor/mobilint_support_inquiry.md) | Mobilint 기술지원 문의 정리 |

## 🛠 scripts/ · assets/
- `scripts/` — 벤치마크 재현 스크립트. 실행: `conda activate pe_npu_host` 후 해당 스크립트.
  - `*_full.py` / `bench_modes_threaded.py` / `bench_throughput_correct.py` / `profile_full_modes.py` = **현재(full NPU, AX_NPU 자기완결)**.
  - `bench_multinpu.py` / `bench_scaling.py` / `bench_preprocess.py` / `profile_stages.py` = **[레거시] hybrid 시절 재현용, Product-AI-mono import 필요**(단독 실행 불가).
- `assets/` — 벤치 산출물 (csv 원자료, png 차트).
