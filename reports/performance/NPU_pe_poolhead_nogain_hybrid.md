# [hybrid · before] pool head 배치화는 이득 없음 (음성 결과, 실측)

> **[UPDATE 2026-06]** 이 문서는 **hybrid 시절 CPU pool head**에 대한 것. 이후 attn_pool의 QKᵀ를
> 16bit로 올려 **full NPU**(attn_pool도 NPU, cos 0.99)가 되면서 **CPU pool head 자체가 사라졌다** →
> 이 배치화 고민은 더 이상 해당 없음. → [`../vendor/mobilint_resolution_attn_pool.md`](../vendor/mobilint_resolution_attn_pool.md)

(아래는 hybrid 당시 기록) CPU로 도는 **pool head(attn_pool + proj)**를 채널마다(per-item) 도는 방식
대신 **`(B,577,1024)`로 한 번에 배치 처리**하면 빨라질 것 같지만 — **재보니 오히려 ~20% 느리다.**
"pool 배치하면 빠르지 않나?"는 재실험 없이 이 문서로 갈음한다.

> 결론 먼저: **`multi_npu.py`/`inference.py`의 per-item loop를 그대로 둘 것.** 배치화 금지.

## 측정 (NPU 불필요, 순수 CPU)

- 코드: `reports/scripts/bench_poolhead.py` (vendored `AttentionPooling(1024, heads=8)` 그대로 사용)
- 환경: torch 2.12 CPU / 64코어. `loop`=현재 per-item, `batch`=pool head 1회.
- `kv` = pool head 내부의 k/v projection(577토큰 × 1024×1024)만 따로 잰 값.
- 단위 ms (min of 40 reps). loop/batch 결과는 `allclose`로 동치 확인.

| threads | B | loop | batch | speedup | kv(=loop의 대부분) |
|---|---|---|---|---|---|
| **32** | 8  | 16.5 | 19.2 | 0.86x | 4.0 |
| **32** | 32 | 44.1 | 55.6 | 0.79x | 31.0 |
| **32** | 62 | **81.9** | **101.4** | **0.81x** | 73.0 |
| 8 | 62 | 139.2 | 167.2 | 0.83x | 112.5 |
| 1 | 62 | 919.0 | 964.4 | 0.95x | 762.0 |

## 왜 이득이 없나

1. **compute-bound (오버헤드 X).** B=62/32스레드에서 pool head 82ms 중 **73ms가 k/v projection** —
   순수 행렬곱 FLOPs다. loop든 batch든 연산량이 동일하다. 파이썬 루프 오버헤드는
   1-스레드에서 loop≈batch(0.95~1.01x)로 확인되듯 수백 ms 중 1ms도 안 돼 무의미.
2. **per-item matmul이 이미 코어를 다 쓴다.** `(1,577,1024)@(1024,1024)`도 충분히 커서 32코어를
   포화시킨다. 배치로 키우면 코어 이득 없이 **큰 텐서 메모리 트래픽/캐시 악화**로 BLAS가 손해 → 0.8x.
3. **현재 loop는 NPU↔CPU를 overlap한다.** `for f in futures: f.get(); pool(...)` 구조라 카드 i를
   CPU pool하는 동안 카드 i+1..은 NPU에서 계속 추론한다. 배치는 "전 카드 `get()` 대기 → 그다음 pool"
   이라 이 겹침을 깨뜨린다 (느려지는 둘째 이유).

## pool head를 진짜 줄이려면 (CPU 한정)

루프 재배치로는 불가. FLOPs 자체를 건드려야 한다 (모두 정확도 0.997 트레이드오프 검증 필요):
- pool head 비용의 ~88%가 **577토큰 전부에 대한 k/v projection**. 토큰 수 축소 또는 attn_pool 별도 양자화.
- pool head를 NPU/GPU로? attn_pool은 NPU INT8에서 깨져(cos 0.46) CPU로 뺀 부분이라 원점
  (`../design/SOLUTION_single_io_compile.md`).

## 연관

- `reports/performance/NPU_pe_multicard_62ch_hybrid.md` — 멀티카드 분산 (이 pool head가 도는 맥락)
- `reports/performance/NPU_preprocess_1_parallel.md` — 고채널 CPU 병목(전처리)
