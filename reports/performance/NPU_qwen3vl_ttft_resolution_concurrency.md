# [실측] Qwen3-VL — TTFT × 해상도(720p/1080p) × 카드 내 동시성

배치 컴파일 문의(2026-09-05 벤더 회신)를 계기로, **지금 자산으로 무엇이 되고 안 되는지**를
실측한 기록. 배경·코어모드 구조는
[`../vendor/mobilint_update_vlm_batch_coremode.md`](../vendor/mobilint_update_vlm_batch_coremode.md).

| 항목 | 값 |
| --- | --- |
| 측정일 | 2026-09-07 |
| 모델 | `mobilint/Qwen3-VL-2B-Instruct` (HF 커밋 `7202ab5b`, 2026-06-29 — **최신**) |
| 라이브러리 | `mblt-model-zoo` **2.4.2** (신규 env `vlm_zoo242`), transformers 4.57.1 |
| NPU | 드라이버 1.13 / 런타임 1.2.0, `/dev/aries7` 단일 카드 |
| 코어모드 | vision/text 둘 다 `global8`(config 기본값) |
| 지표 | **TTFT** = `max_new_tokens=1` 생성 시간, median(워밍업 1회 제외) |
| 원자료 | [`../assets/qwen3vl_concurrency_1card.json`](../assets/qwen3vl_concurrency_1card.json) |

## 1. 해상도는 TTFT에 영향이 없다

| 해상도 | 전처리(CPU) | TTFT | TTFT 최저 |
| --- | ---: | ---: | ---: |
| 720p | 24.4 ms | 166.0 ms | **162.2** |
| 1080p | 33.1 ms | 176.4 ms | **161.3** |

**최저값이 사실상 동일하다**(162.2 vs 161.3). median 차이 10ms는 잡음이다.

이유는 vision mxq 가 **static** 이기 때문이다. 입력 형상이 `[1024, 64, 6]` 로 고정 —
시각 토큰 수가 1024개로 박혀 있어 어떤 해상도든 프로세서가 같은 크기로 만들어 넣는다.
**NPU 연산량이 물리적으로 동일하다.**

차이는 **CPU 전처리에만** 생긴다(24.4 → 33.1ms, +8.7ms). 1080p 픽셀이 2.25배라 리사이즈가 그만큼 더 든다.

> **실무 함의**: 해상도를 올려도 **정확도는 안 오르고**(어차피 1024토큰으로 뭉개진다)
> **전처리 비용만 는다.** 업스트림에서 미리 줄여 보내는 편이 낫다.

## 2. 카드 1장 동시성 — 2~4스레드에서 1.4~1.65배

16장 처리, `global8`, B=1 반복.

| 해상도 | 스레드 | 총 16장 | 장당 | img/s | 1스레드 대비 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 720p | 1 | 3403.3 ms | 212.7 ms | 4.70 | 1.00× |
| 720p | **2** | 2415.1 ms | 150.9 ms | **6.62** | **1.41×** |
| 720p | 4 | 2439.7 ms | 152.5 ms | 6.56 | 1.40× |
| 720p | 8 | 2455.4 ms | 153.5 ms | 6.52 | 1.39× |
| 1080p | 1 | 4043.8 ms | 252.7 ms | 3.96 | 1.00× |
| 1080p | 2 | 2594.2 ms | 162.1 ms | 6.17 | 1.56× |
| 1080p | **4** | 2451.2 ms | 153.2 ms | **6.53** | **1.65×** |
| 1080p | 8 | 2446.1 ms | 152.9 ms | 6.54 | 1.65× |

**NPU 는 in-flight 1 인데도 스레드가 이득이다.** 순차로 돌리면 `전처리(CPU) → 추론(NPU) → 전처리 → …`
로 두 자원이 번갈아 노는데, 스레드를 쓰면 한 건이 NPU 에 있는 동안 다른 건의 전처리가 CPU 에서 돈다.

**해상도 페널티가 사라진다**는 게 그 증거다. 1스레드에서는 1080p 가 720p 보다 16% 느리지만
(3.96 vs 4.70), 4스레드에서는 **6.53 vs 6.56 으로 동일**해진다. 전처리가 NPU 대기 뒤에 완전히 숨는다.

**포화점**: 720p 는 2스레드, 1080p 는 4스레드. 그 이상은 평평하다(대기만 늘어난다).
전처리가 무거울수록 포화점이 뒤로 밀린다.

> 현재 `VLMPool` 은 **카드 간** 분산만 한다. **카드 안** 스레드는 안 쓴다.
> 카드당 2~4 스레드를 추가하면 위 배수만큼 더 얻는다. → `tutorial/pe_npu/vlm_npu.py` 개선 대상.

## 3. 배치는 불가능하다 — mxq 자산 제약

벽이 셋인데 성격이 다르다.

| 벽 | 증상 | 원인 | 해결 |
| --- | --- | --- | --- |
| 프로세서가 이미지 2장 이상 거부 | `NotImplementedError: Only one image input is supported` | 라이브러리 1.3.1 | ✅ **2.4.2 로 해결** |
| **text 배치 = 3입력 MXQ 필요** | `ValueError: Batched Qwen3-VL text inference requires a 3-input [inputs, rope, deepstack] MXQ (the current Batch16 build). The legacy 2-input build ...` | **mxq 자산** | ❌ 벤더 릴리스 대기 |
| text `core_mode="multi"` 불가 | `QbRuntimeError: Model_MXQAndModelConfigNotMatch` | **mxq 자산** | ❌ text mxq 에 Multi 번들 없음 |
| 한 프롬프트에 이미지 여러 장 | `NotImplementedError: ... requires a dynamic-vision Qwen3-VL release` | **mxq 자산**(static vision) | ❌ dynamic vision mxq 필요 |

**우리 자산 실측**

| mxq | 입력 수 | 코어모드 번들 |
| --- | ---: | --- |
| `_vision.mxq` | 1 (`images_0`) — static | Single 1 / Multi 5 / Global4 5 / Global8 10 |
| `_text.mxq` | **2** (`inputs_embeds/reshape`, `deepstack_visual_embeds_0`) — legacy | Single 1 / Global4 5 / Global8 10 (**Multi 없음**) |

HF `mobilint/Qwen3-VL-2B-Instruct` 최신 커밋은 `7202ab5b`(2026-06-29)로 **우리가 쓰는 것과 동일**하다.
Batch16 빌드는 아직 배포되지 않았다.

vision 에는 Multi 번들이 있지만 **혼자서는 못 쓴다** — 여러 장을 vision 에 넣으려면 배치 요청이
성립해야 하는데 text 에서 먼저 막힌다. vision/text 를 둘 다 `multi` 로 주면 text 가
`Model_MXQAndModelConfigNotMatch` 로 죽는다.

## 4. 결론

- **해상도**: TTFT 무관. 업스트림에서 줄여 보낼 것.
- **지금 할 수 있는 최적화**: 카드당 2~4 스레드 → **1.4~1.65배**. 코드 변경만으로 가능.
- **배치**: 벤더의 Batch16 text mxq 릴리스 전까지 불가. 라이브러리는 이미 준비돼 있다(2.4.2).
- **벤더 문의 후보**: Qwen3-VL-2B 의 3입력 Batch16 빌드 배포 일정, dynamic-vision 빌드 여부.

## 부록. 재현

```bash
conda create -y -n vlm_zoo242 --clone gov_vlm && pip install "mblt-model-zoo==2.4.2"
python /tmp/w4a16/vlm_ttft.py  --device 7 --batches 1,2,4     # 해상도별 TTFT
python /tmp/w4a16/vlm_conc.py  --device 7 --threads 1,2,4,8 --n 16   # 동시성 스윕
```
