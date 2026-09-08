# [06] Qwen3-VL-2B prefill 최적화 — 신규 릴리즈 SDK 요청 및 실증기간 장비 문의

| 항목 | 내용 |
| --- | --- |
| 수신 | Mobilint 기술문의 (참조: 서현석 매니저님) |
| 관련 | AX 미니트랙 실증 / Qwen3-VL-2B 최적화 |
| 목적 | ① prefill 최적화된 신규 릴리즈 SDK·튜토리얼 요청 ② 실증기간 상위 NPU 교체 테스트 가능 여부 |
| 상태 | 발송 대기 (초안) |

> **내부 메모**: 첨부 수치는 2026-09-07 자체 실측이다.
> 원본 리포트 [`../../performance/NPU_qwen3vl_ttft_resolution_concurrency.md`](../../performance/NPU_qwen3vl_ttft_resolution_concurrency.md),
> 원자료 [`../../assets/qwen3vl_concurrency_1card.json`](../../assets/qwen3vl_concurrency_1card.json).
> 배치가 막힌 원인(text mxq legacy 2입력) 규명은 [`../../vendor/mobilint_update_vlm_batch_coremode.md`](../../vendor/mobilint_update_vlm_batch_coremode.md).

---

안녕하세요~ PIA-SPACE 정서익입니다.

다름이 아니라, 현재 **Qwen3-VL-2B 를 최적화하는 작업**이 필요해서 연락드립니다.
AX 미니트랙 서현석 매니저님과 황지욱 엔지니어님께 **신규 릴리즈 버전이 있다**는 이야기를 들었습니다.

저희가 필요로 하는 목표는 **Qwen3-VL-2B 를 16채널에서 1000ms 안쪽으로 처리**하는 것입니다.

- 현재 Qwen3-VL-2B 에 가용한 NPU 는 **2장** 입니다.
- **16채널 환경** = 720p 이미지 16장이 동시요청 또는 배치로 들어올 때 처리되는 시간입니다.
- 저희는 **TTFT(prefill 이후 첫 토큰까지의 시간)** 가 중요합니다.
  과업 특성상 **분류를 위해 토큰 1개만** 보기 때문입니다.

## 1. 현재 저희 측정 결과

먼저 저희가 어디까지 확인했는지 공유드립니다.

### 측정 환경

| 항목 | 값 |
| --- | --- |
| 모델 | `mobilint/Qwen3-VL-2B-Instruct` (HF 커밋 `7202ab5b`, 2026-06-29 — 현재 최신) |
| NPU | ARIES **1장**(`/dev/aries7`), 드라이버 **1.13** / 펌웨어 1.2.5 |
| 런타임 | qbruntime **1.2.0** (SDK 번들 1.0v) |
| 라이브러리 | `mblt-model-zoo` **2.4.2**, `transformers` 4.57.1 |
| **코어모드** | vision / text 모두 **`global8`** (config.json 기본값, `target_clusters=[0,1]`) |
| 측정 방식 | `max_new_tokens=1` 생성 시간 = TTFT, median (워밍업 1회 제외) |

### 단건 TTFT — 해상도와 무관합니다

| 해상도 | 전처리(CPU) | TTFT | TTFT 최저 |
| --- | ---: | ---: | ---: |
| 720p | 24.4 ms | 166.0 ms | **162.2 ms** |
| 1080p | 33.1 ms | 176.4 ms | **161.3 ms** |

vision MXQ 입력이 `[1024, 64, 6]` 고정이라 해상도와 무관하게 시각 토큰 1024개로 정규화되고,
NPU 연산량이 동일한 것으로 이해했습니다. 차이는 CPU 전처리(+8.7ms)에서만 발생했습니다.

### 16장 처리 — 카드 1장 기준

`global8`, 배치 없이 B=1 을 스레드로 동시 제출한 결과입니다.

| 해상도 | 스레드 | 16장 총 시간 | 장당 | 처리량 |
| --- | ---: | ---: | ---: | ---: |
| 720p | 1 (순차) | 3403 ms | 212.7 ms | 4.70 img/s |
| 720p | **2** | **2415 ms** | **150.9 ms** | **6.62 img/s** |
| 720p | 4 | 2440 ms | 152.5 ms | 6.56 img/s |
| 720p | 8 | 2455 ms | 153.5 ms | 6.52 img/s |
| 1080p | 1 (순차) | 4044 ms | 252.7 ms | 3.96 img/s |
| 1080p | **4** | **2451 ms** | **153.2 ms** | **6.53 img/s** |

2~4 스레드에서 포화되어 그 이상은 개선되지 않았습니다.

### 목표까지 남은 거리

카드 1장에서 장당 150.9ms 이므로, **카드 2장에 8장씩 분산하면 약 1207ms** 로 추정됩니다.
목표 1000ms 대비 **약 21% 초과**입니다.

| | 16장 처리 시간 |
| --- | ---: |
| NPU 1장 (실측) | 2415 ms |
| NPU 2장 (추정) | **약 1207 ms** |
| **목표** | **1000 ms 이내** |

### 배치 추론 시도 결과

배치로 처리하면 개선될 것으로 보고 시도했으나, 아래 메시지로 진행하지 못했습니다.

```
ValueError: Batched Qwen3-VL text inference requires a 3-input
            [inputs, rope, deepstack] MXQ (the current Batch16 build).
            The legacy 2-input build ...
```

확인해보니 저희가 받은 text MXQ 는 입력이 2개(`inputs_embeds/reshape`, `deepstack_visual_embeds_0`)인
legacy 빌드였습니다. 또한 text MXQ 에는 Multi 번들이 없어(`Single`/`Global4`/`Global8` 만 존재)
`core_mode="multi"` 지정 시 `Model_MXQAndModelConfigNotMatch` 가 발생했습니다.
`mblt-model-zoo` 2.4.2 라이브러리 쪽은 배치 경로가 준비되어 있는 것으로 보입니다.

## 2. 문의드리는 사항

### ① 신규 릴리즈 SDK 및 튜토리얼 문서

**신규 릴리즈에서 prefill 단계가 더 최적화되었다**고 들었습니다.
혹시 **해당 SDK 와 사용방법 튜토리얼 문서를 제공받을 수 있을지** 문의드립니다.

함께 확인하고 싶은 점은 아래와 같습니다.

- 신규 릴리즈에 **3-input Batch16 text MXQ** 가 포함되는지, 포함된다면 배포 일정
- 배치 사용 시 저희 목표(16채널 1000ms)에 도달 가능한 수준인지
- vision/text 코어모드 권장 조합 (현재 저희는 양쪽 `global8` 기본값 사용 중)

### ② 실증 기간 중 상위 NPU 교체 테스트 가능 여부

AX 미니트랙 실증 기간은 **1~2개월** 정도로 예상하고 있습니다.

- **10월 15일 설치**, 이후 **약 30일 운영** (그 이전 테스트 기간 포함)

이 실증 기간 동안 **좀 더 성능이 좋은 NPU 를 잠시 교체하여 테스트가 가능할지** 문의드립니다.
결국 안정적인 서비스 검증 및 배포 실증을 위해서는 해당 작업이 필요할 것으로 보입니다.

---

측정에 사용한 스크립트나 추가 데이터가 필요하시면 언제든 말씀해 주시면 전달드리겠습니다.

바쁘신 중에 늘 상세히 답변 주셔서 감사합니다.

감사합니다.
PIA-SPACE 정서익 드림
