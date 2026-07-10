# PE-Core-L14-336 NPU 테스트 결과 리포트

> **[UPDATE 2026-06] 이 문서(2026-06-15 작성)의 §4·§6 "막힘/측정불가/기술지원 문의 필요" 서술은 이후 해결되어 superseded.**
> 단일 입출력 컴파일 = **5개 모델 패치로 자체 해결**(§6.5 [해결됨]), 정확도 = **hybrid(NPU trunk + CPU attn_pool)로 원본 대비 cos 0.9987**.
> 멀티카드/전처리/실서비스까지 완료. **그리고 그 hybrid(CPU attn_pool)도 이후 superseded** — attn_pool의
> QKᵀ matmul을 16bit로 올려 **full NPU**(image→embedding 전부 NPU, cos 0.99)가 현재 기본이다.
> 최신: [`../vendor/mobilint_resolution_attn_pool.md`](../vendor/mobilint_resolution_attn_pool.md),
> [`../performance/NPU_pe_pipeline_e2e_full.md`](../performance/NPU_pe_pipeline_e2e_full.md). 아래 §4·§6은 당시 기록(historical).

- 대상 모델: **PE-Core-L14-336** vision encoder (Meta Perception Encoder, CLIP 계열 ViT-L/14)
- 목표: Product-AI-mono `perception_encoder`의 비전 인코더(현 TensorRT)를 NPU로 대체
- 하드웨어: **ARIES MLA100 PCIe Card** (Aries2, 8코어/2클러스터, 16GB, 펌웨어 1.2.5)
- 호스트: Ubuntu 24.04 / x86_64 / 드라이버 1.13.0 / 런타임 qbruntime 1.2.0
- 작성일: 2026-06-15

---

## 1. 컴파일 결과 (GPU, qbcompiler 1.1.2)

| 항목 | 결과 |
|------|------|
| ONNX export (vision encoder) | ✅ (B,3,336,336) → (B,1024), fp32 ~1.2GB |
| onnxsim 단순화 | ✅ If 노드 1→0 (3965→1725 노드) — ViT RoPE 조건문 제거, **컴파일 필수 단계** |
| MXQ 컴파일 (single) | ✅ **94초**, 314MB |
| MXQ 컴파일 (all: single+multi+global4+global8) | ✅ **193초**, 323MB |
| MXQ 포맷 | MXQv7 (0x70000), Hardware Aries2 |
| 양자화 | INT8 (channel-wise, symmetric). 입력은 Float32/16/Int8 허용 |

> calibration은 동작검증용 random. 실제 데이터 calibration은 ViT 25-서브그래프 분할 이슈로
> 별도 경로 필요 (별첨 참고).

---

## 2. NPU 인식/상태 (mobilint-cli status)

```
NPU 0  Aries(aries0)   Firmware 1.2.5   Temp 35°C   Pwr 2.04W/6.49W
       Clock 50MHz/150MHz   Memory 0MB/16384MB   NPU-Util 0.00% (idle)
Driver: Aries 1.13.0
```
- PCI 인식(209f), `/dev/aries0`, 드라이버/런타임 전부 정상.

---

## 3. 추론 동작 + 멀티코어 성능 (mblt-benchmark, infer-type float)

이미지 1장 = (336,336,3). **Throughput**(초당 처리량)과 **Latency**(1장 처리 지연)는 다르다.

| 코어 모드 | 동작 방식 | Throughput (FPS) | Latency (ms/frame) |
|-----------|----------|-----------------:|-------------------:|
| **global8** (8코어 1추론 분담) | 1장을 8코어가 나눠 처리 | **17.68** | **≈ 56.6** ← 최저 |
| single (8코어 독립) | 8장을 8코어가 병렬 | 11.98 | 83.5* |
| global4 (4코어 1추론 분담) | 1장을 4코어가 나눠 처리 | 10.79 | 92.7 |
| multi (클러스터 4배치) | 클러스터당 4장 묶음 | 5.99 | 167* |
| single (1코어) | 1장을 1코어가 처리 | 4.62 | **216** (순수 단일 latency) |

- `*` single/multi의 latency는 throughput 역수(1000/FPS)로, **여러 장을 병렬 처리하는 평균치**다.
  단일 프레임의 실제 지연은 1코어 기준 **216ms**이며, 병렬은 그 지연을 겹쳐서 처리량을 높이는 것.
- global 모드는 1장을 여러 코어가 **분담**하므로 throughput=프레임 처리율이고, latency 자체가 낮다.

**VMS 실시간 카메라 스트림 관점:**
- 실시간성(프레임 지연 최소)이 중요 → **global8: 56.6 ms/frame ≈ 초당 17~18 프레임 처리.**
  일반 CCTV(15~30fps)에서 프레임을 실시간에 가깝게 따라갈 수 있는 수준.
- 카메라가 많아 처리량이 중요 → single 다코어로 여러 스트림을 병렬 처리.
- 참고: 1코어→8코어가 4.62→11.98(약 2.6배)에 그치는 건 ViT-L이 무거워 메모리 대역폭이
  병목이기 때문. 그래서 코어 분담(global8)이 독립 병렬(single)보다 유리하다.

> ✅ **NPU에서 PE 비전인코더 연산이 실제로 동작**함을 확인 (INT8 MXQ).
> ⚠️ 단 이 수치는 benchmark가 25-입력을 자동 생성해 측정한 것 (§4의 추론 인터페이스 한계 참조).
> 단일 입출력 MXQ가 완성되면 직접 `infer`로 per-frame latency를 더 정확히 재측정 가능.

---

## 4. 원본(PyTorch) 대비 정확도 (compare_backends.py)

| 비교 | 코사인 유사도 | 비고 |
|------|--------------:|------|
| 원본 PyTorch vs ONNX | **1.000000** (MAE 1e-6) | export 무손실 ✅ |
| 원본 PyTorch vs NPU(MXQ) | **측정 불가** | 아래 한계 참조 |

**측정 불가 사유 (중요):** 현재 MXQ는 NPU에 로드·launch까지 정상이지만,
`get_model_input_shape`가 **입력 25개 / 출력 25개**를 노출한다:
- 입력: `(1,577,1024)` × 24 (블록 사이 activation) + `(336,336,3)` × 1 (image)
- 출력: `(1,1,1024)`(최종 임베딩) + `(3,577,1024)` × 24

즉 컴파일러가 ViT를 24블록+1 = **25개 서브그래프로 분할해 입출력을 전부 노출**했다.
image 한 장만 넣는 단순 `infer`는 중간 24개 입력을 채울 수 없어 `Input shape is invalid`.
(`mblt-benchmark`는 25개 입력을 자동 생성해 성능만 측정하므로 동작함 → §3 수치는 유효.)

**이것은 calibration이 막혔던 원인(§별첨)과 동일**하다: ONNX 직접 컴파일 시 ViT가 25조각으로
쪼개진다. 실사용 가능한 단일 image→embedding MXQ를 얻으려면 **vlm vision처럼
`ModelParser`로 단일 입출력 모델을 구성해 컴파일**해야 한다.

---

## 5. 멀티코어/멀티스레딩 활용 방식 (정리)

- **코어 모드** (컴파일 시 결정, `inference_scheme`):
  - single: 8개 독립 코어. 멀티스레딩으로 쉴 새 없이 요청해야 throughput 극대화.
  - multi: 클러스터(4코어) 단위 4배치 처리.
  - global4/global8: 1개 추론을 4/8코어가 협동 → latency 감소 (무거운 모델 유리).
- **멀티스레딩 / 비동기** (`advanced_usage.md`):
  - 블로킹 `infer()`는 단일 스레드에서 코어를 다 못 채움.
  - 해결: 직접 멀티스레딩 구현 또는 `infer_async()` (`set_async_pipeline_enabled(True)`).
  - `infer_async` 제약: RNN/LSTM/LLM 미지원, CPU offload 미지원, 단일 배치만.
- **자원 분배**: `ModelConfig`로 코어 수/특정 코어 지정 가능 (여러 모델 동시 운용 시 유용).
- **트레이싱**: `start_tracing_events()` → Perfetto UI로 NPU 사용량 시각화.

---

## 6. 결론 / 다음 단계

**검증됨 ✅**
- 커스텀 PE 비전인코더 **컴파일** (single 94초 / all 193초, MXQv7 INT8)
- NPU **인식·로드·연산 동작** (mobilint-cli status, benchmark)
- **멀티코어** throughput 비교 (global8 17.68 > single 11.98 > global4 10.79 > multi 5.99 회/초)
- 원본 PyTorch ↔ ONNX **무손실** (cos=1.0)

**막힌 부분 ❌ (동일 근본원인: ONNX 직접 컴파일 시 ViT 25-서브그래프 분할)**
- 실제 image→embedding 추론 (MXQ가 25개 입출력 노출)
- 실데이터 calibration (25개 입력 매핑 불가)
- 원본 대비 NPU 정확도 측정 (추론 인터페이스 문제)

### 단일 입출력 MXQ — [해결됨] ✅

> 처음엔 ONNX/mblt/torch/RoPE상수화가 모두 25분할 또는 trace 실패였으나, **torch FX 경로 +
> 모델 패치(SDK 무수정)로 단일 입출력 컴파일 + NPU 추론에 성공**했다. 상세: `reports/design/SOLUTION_single_io_compile.md`.

| 경로 | 결과 |
|------|------|
| ONNX 직접 (`backend=onnx`) | ❌ 25 서브그래프 분할 |
| MBLT 경유 | ❌ 25분할 |
| Torch 직접 (패치 없이) | ❌ RoPE fx-trace 실패 |
| RoPE 상수화 + ONNX | ❌ 25분할 (RoPE는 주원인 아님) |
| **Torch + 5개 모델 패치** | ✅ **입력1/출력1, NPU 추론 OK, pth대비 cos 0.936(random calib)** |

해결 핵심: RoPE 상수화 + einops→네이티브 + posemb 상수화 + qkv 슬라이싱 + attn-pooling 패치를
모델 객체에 적용 후 `mxq_compile(backend="torch")`. op는 원래 다 지원됐고(101개), 문제는 그래프
트레이스 형태였다. → `pe_torch_compile.py`, `out/pe_torch_single.mxq`.

> **확정된 근본 원인:** RoPE를 cos/sin 상수로 박고 `If` 노드를 0으로 만들어도 **24개 transformer
> block이 각각 독립 서브그래프로 분할**된다(입력 25 = 중간 24×`(1,577,1024)` + image `(336,336,3)`,
> 출력 25 = 임베딩 + block별 24개). 즉 **분할 원인은 RoPE가 아니라 transformer block(attention)
> 구조 자체**다. ONNX에 Softmax 25개(24 block + 1 attn-pooling)가 있고 컴파일러가 각 block을
> 경계로 분할한다.
>
> 검증: 같은 image에 24개 중간 입력만 zeros↔random으로 바꾸면 첫 출력이 cos=0.958로 변한다 →
> 24개 입력이 결과에 영향(무시 불가) → image만으로 정상 추론 불가 = 진짜 분할.
>
> 표준 absolute-PE ViT(Model Zoo의 DeiT/FlexiViT/SigLIP)는 단일 입출력으로 컴파일된다.
> Qwen2-VL ViT도 RoPE를 쓰지만 **qbcompiler SDK 내부의 전용 패치**(`VisionModelForQwen2VL`)로
> 단일 입출력이 된다. **→ [해결됨] PE용 SDK 패치 없이도, 모델 객체에 5개 패치를 적용하는 방식으로
> 단일 입출력 컴파일에 성공했다(위 표). 기술지원 문의는 불필요해져 미발송.**
| ModelParser (vlm 방식) | ❌ 미적용 | vlm 코드가 **Qwen2-VL 전용**(`VisionModelForQwen2VL`), PE 구조에 직접 적용 불가 |

`split_blocks`/`split_parts`는 "더 쪼개는" 옵션이라 합치는 데 쓸 수 없음.

**다음 단계 — [전부 완료됨] ✅**
- ~~핵심 과제: 단일 입출력 컴파일 경로 확보~~ → **5개 모델 패치로 해결**(SDK 무수정, 기술지원 불필요).
- ~~실데이터 calibration~~ → COCO val2017로 INT8 calib 동작.
- ~~원본 대비 정확도 측정~~ → **hybrid(NPU trunk + CPU attn_pool) cos 0.9987** 측정 완료.
- ~~service.py TRTInference→MXQ 교체~~ → Product-AI-mono `pe_npu` 모듈로 구현(멀티NPU 자동분산 + 전처리 병렬화 포함).
- 멀티카드/전처리 후속 분석: `../performance/NPU_pe_multicard_62ch_hybrid.md`, `../performance/NPU_preprocess_1_parallel.md`.
- 참고: 양자화 원리/근거는 `reports/quantization/quantization_reference.md`.
