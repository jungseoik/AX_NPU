# reports/ — 분석·벤치마크 문서 인덱스

PE-Core-L14-336 NPU 추론 프로젝트의 실측/분석 문서를 주제별로 정리.

> ★ **현재 상태: full NPU (image→embedding 전부 NPU, cos 0.99)**. attn_pool INT8 붕괴는
> QKᵀ matmul 16bit로 해결됨 → [vendor/mobilint_resolution_attn_pool.md](vendor/mobilint_resolution_attn_pool.md).
> 그 이전 hybrid(NPU trunk + CPU pool) 분석 문서들은 히스토리로 보존.

## 📊 performance/ — 성능 (지연·처리량·병렬화·컴파일)

> **네이밍 규칙** — 파일명은 `NPU_<주제>_<단계>` 꼴. before/after 스토리는 접미사(`_hybrid`=before / `_full`=after,
> `_before`/`_opt`/`_after`)나 번호(`_1_`/`_2_`/`_3_`)로 단계를 표시. H1 제목의 `[..]` 태그도 동일하게 맞춰져 있다.

### PE-Core: hybrid(before) → full(after) 마이그레이션  ★ 현재 = full NPU
attn_pool의 QKᵀ를 16bit로 올려 CPU pool을 없애고 전부 NPU로 옮긴 스토리. `_hybrid`=이전, `_full`=현재.
| 문서 | 내용 |
|------|------|
| [NPU_pe_hybrid_vs_full.md](performance/NPU_pe_hybrid_vs_full.md) | ★ **[비교]** full NPU vs hybrid — CPU attn_pool 병목 제거 실측 (QKᵀ 16bit) |
| [NPU_pe_pipeline_e2e_full.md](performance/NPU_pe_pipeline_e2e_full.md) | ★ [full·after] 코어모드 4종 × 채널 스윕 단계별 e2e (Pool 단계 제거) |
| [NPU_pe_multicard_62ch_full.md](performance/NPU_pe_multicard_62ch_full.md) | ★ [full·after] 멀티카드 1→62ch 배치 지연 (hybrid와 동일 구조, attn_pool도 NPU) |
| [NPU_pe_throughput_modes_full.md](performance/NPU_pe_throughput_modes_full.md) | ★ [full·after] 다채널 처리량·모드선택 (올바른 1모델+멀티스레드 sync 패턴, 출력검증) |
| [NPU_pe_1card_coremode_full.md](performance/NPU_pe_1card_coremode_full.md) | ★ [full·after] NPU 1장 코어모드 4종 × 1~16채널 순수추론 증가폭 (슬롯 거동) |
| [NPU_pe_pipeline_e2e_hybrid.md](performance/NPU_pe_pipeline_e2e_hybrid.md) | [hybrid·before] 코어모드 3종 × 단계별 × 채널(최대 56) — NPU 병목 어디서 커지는지 |
| [NPU_pe_multicard_62ch_hybrid.md](performance/NPU_pe_multicard_62ch_hybrid.md) | [hybrid·before] 멀티카드(7×ARIES=56코어) 1→62채널 분산 추론 지연 (full의 짝) |
| [NPU_pe_stage_latency_hybrid.md](performance/NPU_pe_stage_latency_hybrid.md) | [hybrid·before] _detect 단계별(전처리/NPU/pool/event) + e2e 지연, 채널 스윕(1~16) |
| [NPU_pe_poolhead_nogain_hybrid.md](performance/NPU_pe_poolhead_nogain_hybrid.md) | [hybrid·before] CPU pool head 배치/스레드 최적화가 무효한 이유 (짝 없음, full에서 소멸) |

### 전처리 최적화 (① 시도 → ② 실험 → ③ 채택)
고채널 병목인 CPU 전처리를 줄이는 과정. 번호가 읽는 순서, ③이 최종 채택안.
| 문서 | 내용 |
|------|------|
| [NPU_preprocess_1_parallel.md](performance/NPU_preprocess_1_parallel.md) | [전처리 ①] 병렬화(스레드/멀티프로세스) — 1차 시도 |
| [NPU_preprocess_2_uint8_offload.md](performance/NPU_preprocess_2_uint8_offload.md) | [전처리 ②] NPU 오프로드(uint8) 실험: normalize는 폴딩되나 resize 불가라 이득 없음 |
| [NPU_preprocess_3_cv2_decision.md](performance/NPU_preprocess_3_cv2_decision.md) | [전처리 ③ 채택] 의사결정(e2e): 비용 원천=resize, torchvision→cv2(56ch −25%·CPU↓, 정확도 0.99→0.97 opt-in) |

### npu_intrusion 침입 서비스 e2e (before → 최적화 → after)
YOLO11 침입감지 서비스 모듈 e2e를 측정→진단→재측정한 스토리.
| 문서 | 내용 |
|------|------|
| [NPU_npu_intrusion_e2e_before.md](performance/NPU_npu_intrusion_e2e_before.md) | ★ [e2e·before] 실제 _detect 5단계(ROIcrop→전처리→추론→NMS→교집합+알람) 채널 스윕×카드 1→7 — NPU 아닌 CPU가 병목, ROI crop 마스킹이 최대비용 |
| [NPU_npu_intrusion_e2e_opt.md](performance/NPU_npu_intrusion_e2e_opt.md) | ★ [e2e·최적화] 문제(마스킹/오버서브/argmax)→해결(rect슬라이스+스레드 / 풀분리 / person-only fast-path). e2e 424→121ms(3.5×), 알람 16/16 동일 |
| [NPU_npu_intrusion_e2e_after.md](performance/NPU_npu_intrusion_e2e_after.md) | ★ [e2e·after] 최적화 후 동일 재측정 — 단계별 before/after, 카드 스케일링 회복(56ch 384→113ms), 처리량 151→531 img/s |

### 모델별 (YOLO11 · Qwen3-VL)
| 문서 | 내용 |
|------|------|
| [NPU_yolo11_coremode_batch.md](performance/NPU_yolo11_coremode_batch.md) | ★ [YOLO11] 4사이즈(n/s/m/l)×4모드 컴파일(패치0)·코어모드×배치 1→64·카드수(1~7)×배치 스케일링·mAP(INT8 vs fp32) |
| [NPU_qwen3vl_multicard_batch.md](performance/NPU_qwen3vl_multicard_batch.md) | [Qwen3-VL] 멀티카드×배치(동시요청) VQA 1토큰 지연 (dev_no 카드지정, VLMPool) |

### 공통·기타 (스토리 무관)
| 문서 | 내용 |
|------|------|
| [NPU_batch_latency.md](performance/NPU_batch_latency.md) | 단일 NPU 배치 지연/처리량, 코어 모드, bit4 양자화 한계 (실측) |
| [NPU_coremode_benchmark.md](performance/NPU_coremode_benchmark.md) | 코어모드 4종(Single/Multi/Global4/Global8) × 다채널 지연·메모리 (현재 서버 실측) |
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
| [sdk_v11_compat.md](testing/sdk_v11_compat.md) | ★ SDK 1.1v 호환성 — 기존 HF mxq는 런타임 1.4.0에서 재컴파일 불필요(cos 0.993/15.4 img/s), 컴파일은 cpu 이미지로 충분 |
| [NPU_pe_quant_schemes.md](performance/NPU_pe_quant_schemes.md) | ★ 양자화 스킴 비교 W8A16/W4A16/W4A8 × 코어모드 4종 × 1~20채널 지연 + cos. W4A16 동작 확인(크기 −42%, 처리량 +13%), W4A8은 붕괴 |

## 📮 inquiries/ — Mobilint 문의 스레드 (번호 = 문의 순서, 클수록 최신)
전체 인덱스: [inquiries/README.md](inquiries/README.md)

| # | 스레드 | 내용 | 상태 |
|---|------|------|------|
| 01 | [01_pe_single_io_compile/](inquiries/01_pe_single_io_compile/INQUIRY.md) | PE 비전 인코더 25-서브그래프 분할 문제 | 미발송(자체 해결) |
| 02 | [02_attn_pool_int8/](inquiries/02_attn_pool_int8/INQUIRY.md) | attention pooling head INT8 붕괴 문의 (당시 기록 보존) | 해결 → vendor/ |
| 03 | [03_qwen3vl_batch_serving/](inquiries/03_qwen3vl_batch_serving/README.md) | ★ Qwen3-VL-2B 배치·코어모드 서빙(vLLM) NPU 1장 동시요청 + vllm-mblt 버그 2건 | 미발송 |
| 04 | [04_vit_quantization_speed/](inquiries/04_vit_quantization_speed/REPLY.md) | ★**최신** ViT 양자화 속도(W8A16/W4A16/W4A8) + uint8 입력 + single·async 권고, 컴파일 예제 3종 | 회신 수신(검증 미착수) |

## 📨 vendor/ — Mobilint 응답·해결 기록 (받은 것)
| 문서 | 내용 |
|------|------|
| [mobilint_resolution_attn_pool.md](vendor/mobilint_resolution_attn_pool.md) | ★ **[해결]** attn_pool INT8 붕괴 원인(QKᵀ outlier)·해결(score matmul 16bit) → full NPU cos 0.99 |
| [mobilint_reply_email.md](vendor/mobilint_reply_email.md) | Mobilint 답장 이메일 원문 (attn_pool 건 = 문의 02) |

> 04번 회신(ViT 양자화)은 스레드 폴더 안에 있다 → [inquiries/04_vit_quantization_speed/REPLY.md](inquiries/04_vit_quantization_speed/REPLY.md)

## 🧪 평가용 데이터셋 (측정 입력)

벤치/정확도 문서가 쓰는 **평가 데이터는 이 레포에 커밋하지 않는다.** HF private에 zip으로 두고
토큰만 있으면 재현한다 — 상세: [`eval/README.md`](../eval/README.md)

| 데이터셋 | 내용 | 받기 |
|---|---|---|
| `TTA_인증용` | 이상행동 4종(falldown/fire/intrusion/smoke) 영상 200 + 이벤트 라벨 200 + clips 252 (2.2GB) | `export HF_TOKEN=... && python -m eval.tta download` |

## 🛠 scripts/ · assets/
- `scripts/` — 벤치마크 재현 스크립트. 실행: `conda activate pe_npu_host` 후 해당 스크립트.
  - `*_full.py` / `bench_modes_threaded.py` / `bench_throughput_correct.py` / `profile_full_modes.py` = **현재(full NPU, AX_NPU 자기완결)**.
  - `bench_multinpu.py` / `bench_scaling.py` / `bench_preprocess.py` / `profile_stages.py` = **[레거시] hybrid 시절 재현용, Product-AI-mono import 필요**(단독 실행 불가).
- `assets/` — 벤치 산출물 (csv 원자료, png 차트).
