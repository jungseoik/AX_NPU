# PE-Core-L14-336 NPU 처리량/지연시간 분석

배치로 들어오는 추론 요청을 ARIES NPU에서 처리할 때의 지연시간(latency)과 처리량(throughput)을
실측한 결과. "배치 N장이 동시에 들어오면 다 끝나는 데 얼마 걸리나"에 대한 근거 자료다.

모든 수치는 NPU trunk(24 transformer block, INT8 feat MXQ) 기준이며, async 파이프라인의 최선값이다.
(당시 hybrid: attn_pool head는 CPU float, 장당 약 2ms. 현재 full NPU에선 attn_pool도 NPU로 흡수 — `NPU_pe_pipeline_e2e_full.md`.)

---

## 1. NPU 하드웨어 스펙 (ARIES MLA100 PCIe)

`mobilint-cli status` 기준.

| 항목 | 값 |
|------|-----|
| 제품 | Mobilint ARIES (aries0), Aries2 아키텍처 |
| 코어 구성 | **8 Local Core = 2 Cluster x 4 Core** (+ 클러스터별 Global Core) |
| 메모리 | 16384 MB (16 GB), 8코어 공유 |
| 드라이버 / 펌웨어 | Aries 1.13.0 / 1.2.5 |
| 연산 | INT8 정수 전용 (float/bf16 네이티브 경로 없음) |
| 코어 모드 | Single / Multi / Global4 / Global8 모두 지원 |

핵심: **물리적 동시 병렬도 = 8** (코어 8개). 같은 순간 NPU에서 동시에 도는 추론은 최대 8건.

---

## 2. 코어 모드 (컴파일 시 고정)

코어 모드는 "코어를 몇 개 쓰냐"가 아니라 **"코어들을 어떻게 협력시키냐"**의 구분이다.
컴파일 단계에서 정해지며(`pe_npu/compile.py --scheme`), 컴파일된 MXQ는 그 모드로만 실행된다.

| 모드 | 동작 | 용도 |
|------|------|------|
| **Single** | 8코어가 **각자 독립**으로 1장씩 처리 (8코어 = 8장 동시) | async/멀티스레딩으로 throughput 최대 |
| **Multi** | 4코어(1 Cluster)가 **협력**해 처리 | (이론상) 배치 처리 |
| **Global4/8** | 4/8코어가 **1장을 분할**해 빨리 | 단건 latency 감소 |

"Single"이 코어 1개라는 뜻이 아니다 — 각 코어가 단독으로 모델 전체를 수행한다는 뜻이고,
여러 코어를 동시에 쓸 수 있다.

---

## 3. 배치 크기별 총 지연시간 (Single + async, 채택안)

`set_async_pipeline_enabled(True)` 후 B장을 `infer_async`로 한꺼번에 제출하고 `future.get()`으로 회수.

| 배치 B (동시 요청) | 총 지연 (B장 전부 완료) | 장당 실효 | throughput |
|:---:|:---:|:---:|:---:|
| 1 | 284 ms | 284 ms | 3.5 img/s |
| 2 | 307 ms | 154 ms | 6.5 img/s |
| 4 | 366 ms | 91 ms | 10.9 img/s |
| **8** | **527 ms** | **66 ms** | **15.2 img/s** |
| 16 | 1030 ms | 64 ms | 15.5 img/s |
| 32 | 2028 ms | 63 ms | 15.8 img/s |

### 읽는 법
- **1장 = 284 ms**, **8장 배치 = 527 ms.** 8장을 처리하고도 1장의 1.85배 시간밖에 안 걸린다.
- 8장을 순차로 따로 했다면 284 x 8 = 2272 ms → 배치로 묶으면 **약 4.3배 단축**.
- 장당 실효 지연이 284 → 66 ms로 떨어진다.
- **B > 8이면** per-image 63~64 ms로 고정, 총 지연은 거의 선형 증가(8장씩 웨이브).
- 실용 근사: **B <= 8 → 약 280~530 ms, B > 8 → 대략 64 ms x B.**

### 처리량 천장: ~15.8 img/s (장당 63 ms)
1코어 284 ms이므로 8코어 완벽 병렬이면 28 img/s(장당 35 ms)가 이론치지만, 실측은 15.8(장당 63ms).
실효 병렬도 약 4.5/8. 원인:
- Global Core 중재 + 8코어의 16GB 메모리 대역폭 경합 (하드웨어 한계)
- Python GIL + async 스케줄링 오버헤드 (소프트웨어, C++로 일부 회복 가능)

---

## 4. 시도했으나 효과 없던 최적화 (실측)

| 시도 | 결과 |
|------|------|
| `set_activation_slots` 4/8/16/24 스윕 | 15.8 img/s 불변 (4는 오히려 악화). 파이프라인 깊이로는 개선 없음 |
| `infer_hwc([x]*B)` 리스트 입력 | `Input shape is invalid`. Single MXQ는 한 호출에 여러 장 못 받음(입력 shape이 단건) |

---

## 5. Multi 모드 실측 — 됐지만 더 느림

배치 워크로드에 맞을까 하여 `--scheme multi`로 재컴파일해 측정.

- **컴파일 성공**: "HW: Aries2, Build Mode: Multi", `mxqtool` Core Mode: Multi. 입력 shape은 여전히 단건 `[336,336,3]`.
- **그러나 Single+async보다 약 3배 느림:**

| 방식 | throughput | 장당 |
|------|:---:|:---:|
| Single + async | **15.8 img/s** | **63 ms** |
| Multi + async | 5.2 img/s | 193 ms |
| Multi 단건 동기 | 2.8 img/s | 357 ms (> Single 단건 284 ms) |

### 왜 Multi가 더 느린가
Multi/Global은 여러 코어가 **한 데이터를 나눠** 처리하는 협력 모드라, 코어 간 중간결과 통신과
Global Core 중재(target cores 10개) 비용이 든다. ViT trunk는 이미 코어 하나에 잘 들어가므로,
이 협력 비용이 이득을 초과해 손해다. Single은 코어들이 서로 간섭 없이 각자 1장씩 처리해 throughput에 최적.

---

## 6. 양자화를 더 낮춰 속도를 올릴 수 있나 — bit_4 실측 (결론: 불가)

"INT8보다 더 낮추면(4bit) 빨라지냐"를 실측. **결론: 안 된다. mixed-precision 옵션은 존재하지만 이 경로에서 무시(no-op)된다.**

### 배경
- `docs/programming_guide.md`의 "입력 데이터 타입 UINT8/INT8/float32"는 **입력 텐서 형식**이고, weight 양자화 비트와는 다른 축.
- qbcompiler API에는 `BitConfig.Transformer.mixed_precision`(필드: `apply, bit_2, bit_4, bit_8, prune, importance_threshold_low/high`)이 있어 weight를 2/4/8비트 비율로 섞을 수 있게 **보인다**.
- 단 `docs`/`release_note` 어디에도 mixed-precision/4bit 언급이 **전혀 없다** → 비공식·실험적 기능 의심.

### 실측 (동일 calib 16장, feat trunk, `pe_npu/compile.py --bit4`)
| 설정 | 크기 | 단건 ms | img/s(B32) | pth 대비 cos |
|------|:---:|:---:|:---:|:---:|
| INT8 (baseline) | 314 MB | 284.3 | 15.7 | 0.9238 |
| bit4=0.3 | 314 MB | 284.6 | 15.8 | 0.9238 |
| bit4=0.5 | 314 MB | 284.2 | 15.9 | 0.9238 |
| bit4=0.7 | 314 MB | 284.7 | 15.8 | 0.9238 |

### 판정: mixed_precision은 no-op
- 크기·latency·throughput·**cos(소수 4자리)까지 4종 완전 동일.** 우연이면 미세하게라도 달라야 하므로, `bit_config(mixed_precision)`가 컴파일 경로(PE torch backend + aries2)에서 **무시됨**이 확정.
- 컴파일은 "successful"로 끝나지만(에러 없음) 실제 양자화 비트에 영향 0. docs 미기재 = 실험적/미지원 정황과 일치.
- (주의) 위 cos 0.9238은 calib 16장이라 낮다. 정식 배포본은 COCO 200장 calib: **full NPU `pe_full.mxq`(--qk16) = cos 0.99**, 레거시 trunk `pe_feat.mxq`(hybrid+CPU pool) = cos 0.997. 여기선 4종 상대비교가 목적이라 calib을 통일했을 뿐, 절대 정확도 비교 아님.

### 결론
- **양자화로 속도를 더 짜내는 길은 막혀 있다. INT8이 ARIES의 실질 최저 정밀도.** (bit_4는 API엔 있으나 무효)
- 속도를 줄이려면 양자화가 아니라 **연산량 축소**(입력 해상도 336→224, 더 작은 PE 모델) 또는 **C++ 런타임**(GIL 제거) 뿐.

---

## 7. 결론 / 권고

- **배치 throughput 목적이면 Single + async가 최선이다 (실측 확정).** Multi는 3배 느렸다.
- 한 배치(B <= 8)는 약 280~530 ms에 처리된다. 배치를 그대로 `infer_async`로 제출하면 된다.
- **Global4/8**은 단건 1장을 284 ms보다 빨리 끝내는 latency용이라, 배치 처리량 목적엔 의미 없다
  (협력 오버헤드로 throughput은 오히려 낮음).
- throughput을 더 짜내는 현실적 카드는 **C++ 런타임**(Python GIL 제거)뿐. 구현 비용이 크다.

### 못 넘는 천장
- 물리 병렬 8코어 (하드웨어)
- 1코어 284 ms = 모델 크기 / INT8 연산량이 결정. 줄이려면 모델 경량화 또는 입력 해상도 축소 필요.

---

## 재현

```bash
# 코어 모드별 컴파일 (기본 single)
python -m pe_npu.compile --mode compile --save ./pe_npu/out/pe_feat.mxq --feat-only --scheme single
#   --scheme multi|global4|global8 로 다른 모드 컴파일 (단, 위 결과상 throughput엔 single이 최선)

# 배치 latency / 멀티코어 측정
tutorial/pe_npu/multicore_benchmark.ipynb   # 동기 vs async vs 멀티스레딩 + 그래프
```

측정 환경: conda `pe_npu_host`(Python 3.11) + ARIES `/dev/aries0`, async 파이프라인, 각 6회 최선값.
