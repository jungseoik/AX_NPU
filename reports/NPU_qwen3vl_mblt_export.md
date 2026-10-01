# Qwen3-VL 체크포인트 → MBLT export 규명 (2026-10-01)

신규 학습본을 벤더와 같은 레시피로 컴파일하려면 `.mblt` 를 직접 만들어야 한다.
벤더 재현 패키지는 **이미 export 된 `.mblt` 에서 시작**하고, 그 호출법은 문서에 없다.
qbcompiler 내부를 읽어 규명하고 **스톡으로 재현 검증**한 기록.

- 구현: `qwen3vl_recompile/` (export / 캘리브 / 컴파일 / NPU A/B)
- 환경: GPU 서버 6000-pro (RTX PRO 6000 Blackwell SE ×2) / `mblt_vlm_repro:compile`
- 관련: `reports/RUNBOOK_qwen3vl_bs32_compile.md`, `reports/performance/NPU_qwen3vl_bs32_server.md`

---

## 1. 결론

**스톡 체크포인트에서 벤더와 동일한 `.mblt` 를 재현했다.**

| | 우리 | 벤더 | 차이 | 소요 |
|---|---:|---:|---:|---:|
| `vision.mblt` | 1543.7 MB | 1543.7 MB | **+0.00%** | 12.5 s |
| `decoder.mblt` | 6579.9 MB | 6579.8 MB | **+0.00%** | 36.3 s |

`mblt_compile()` 은 **그래프 변환만** 하므로 빠르다(49초). 2시간 걸리는 건 그다음
`mxq_compile` 의 양자화 최적화다.

---

## 2. 호출법 — 6가지 함정

`qbcompiler.mblt_compile()` 은 공개 API 지만 Qwen3-VL VLM 용 호출법이 문서에 없다.
컴파일러 안의 전용 패처(`model_dict_new/parser/patcher/models/hf_models/qwen3vl.py`,
781줄)를 읽어 알아냈다. **한국어 주석이 있다** — Mobilint 가 PIA 용으로 만든 경로다:

```python
def set_rope(self, position_ids):
    """Prefill 시 호출하여 cos/sin 테이블 생성"""
```

| # | 증상 | 원인 | 해법 |
|---|---|---|---|
| 1 | `ValueError: For torch parsing, feed dict is required` | 추적용 예시 입력 없음 | 프로세서로 실제 입력 생성 |
| 2 | `AttributeError: 'NoneType' has no '__name__'` | `mblt_compile` 은 `hf_config` 로더를 **타지 않는다**(그건 `mxq_compile` 의 `_parsing_task` 전용) | 모델 객체 직접 전달 |
| 3 | `FX tracing failed while tracing Qwen3VLForConditionalGeneration` | 전체 VLM 은 추적 불가(`use_cache`/`past_key_values` 가 fx proxy 와 충돌) | 패처의 `VisionModelForQwen3VL` / `Projection` 을 **각각** export |
| 4 | `UnboundLocalError: local variable 'ov'` | RoPE 의 in-place `freqs_t[..., idx] = ...` | `CachedQwen3VLTextRotaryEmbedding` + `set_rope()` 로 호스트 선계산 |
| 5 | `NotImplementedError: HFProxyableDynamicCache` | KV 캐시가 그래프 **출력**에 섞임 | `use_cache=False` + logits 만 반환하는 래퍼 |
| 6 | `tensor a (64) must match tensor b (83)` | HF 원본은 deepstack 을 압축형(N_visual)으로 masked-scatter | **전체형(T)** 으로 주고 `_deepstack_process` 를 덧셈으로 교체 |

**전부 "지원 안 함" 이 아니라 호출 인자 문제였고, 매번 패처 안에 정답이 있었다.**

### 왜 두 조각으로 쪼개야 하나

벤더가 `.mblt` 를 2개 준 이유가 여기 있다. `manifest.json` 의 입출력 이름이
패처 클래스의 forward 시그니처와 **정확히 대응**한다:

| 패처 클래스 | `manifest.json` |
|---|---|
| `VisionModelForQwen3VL.forward` → `(hidden_states, deepstack_feature_lists)` | vision_outputs 4개 (merger 1 + deepstack merger 3) |
| `Projection.forward(inputs_embeds, deepstack_visual_embeds, …)` | decoder_inputs |

`PatchedPatchMerger` 가 fc 를 **1×1 conv 로 바꿔** NCHW 를 유지하는 것까지 출력 이름
(`…linear_fc1/reshape/gelu/conv2d`)에 드러난다.

### 크기 의존 상수를 그래프 입력으로 빼낸다

vision 과 decoder 가 같은 패턴을 쓴다. 패처 주석:

> "These are fed to `forward` as inputs so the graph stays size-agnostic
> (they become **graph inputs, not constants**)."

| | 호스트 선계산 | 그래프 입력이 됨 |
|---|---|---|
| vision | `compute_side_inputs(grid)` | `pos_embeds`, `cos`, `sin` |
| decoder | `CachedRoPE.set_rope(pos)` | `cos` |

벤더 캘리브 입력 이름이 이것과 1:1로 맞는다:
```
vision  : cos [1,1,-1,128] / pos_embeds/reshape [1,1,-1,1024] / float_1_channel_last [1,1,-1,1536]
decoder : inputs_embeds/reshape [1,-1,2048] / deepstack_visual_embeds_0 [1,3,-1,2048] / cos/reshape_0 [1,-1,256]
```
`float_1_channel_last` 는 `fold_pixel_values()` 의 채널라스트 변환 결과다.

---

## 3. batch32 에 `split_legacy_deepstack.py` 는 **불필요**

벤더 패키지의 그 스크립트는 **레거시 포맷 전용 어댑터**다. 신규 export 에 돌리면 거부한다:

```
split_legacy_deepstack.py: error:
  This adapter is for the checked legacy format; re-export newer MBLT normally
```

| | 벤더 mblt | 우리 mblt |
|---|---|---|
| 끝 4바이트 | `\x00\x00\x00\x00` | `MBLT` |
| 33바이트 푸터 | 유효 (`375506`) | 신규 포맷 |
| deepstack 입력 | 1개(뭉침) → split 필요 | 처음부터 분리 |

벤더 `decoder.mblt` 는 옛 export 라 deepstack 이 `deepstack_visual_embeds_0` 하나로
뭉쳐 있었고, batch32 용으로 3개(`slice_2/3/4`)로 쪼개야 했다.
**신규 export 는 그 단계가 없다.** 앞서 리스크로 꼽았던 항목이 해소됐다.

---

## 4. 재사용 가능 / 재생성 필요

| 입력 | 가중치 종속 | 처리 |
|---|:---:|---|
| `.mblt` | ✅ | 재생성 (`export_mblt.py`) |
| 캘리브 활성값 | ✅ | 재생성 (`make_calib.py`) |
| `rotation.pth` | ❌ | **재사용** |
| `configs/*.json` | ❌ | 재사용 |

`rotation.pth` 는 2048×2048 float64 로, **직교성 오차 5.96e-08 / 절댓값 고유값 1개**
= 하다마드 행렬이다. 데이터로 학습한 게 아니라 고정 직교행렬이므로 가중치가 바뀌어도
그대로 쓴다(hidden_size 2048 유지 조건).

캘리브는 **활성값**이라 재생성이 필수다 — `inputs_embeds` 는 embed_tokens 와 비전 출력에
직접 의존한다. 비전 출력은 **PyTorch(fp32)로 계산**해 "mxq 를 만들려고 mxq 가 필요한"
순환을 피했다.

---

## 5. 미검증

- **우리 mblt + 우리 캘리브 → mxq → NPU** 조합은 이 문서 작성 시점에 실행 중이다.
  ③ `mxq_compile` 자체는 **벤더 입력으로 입증**됐다(2시간 7분, NPU 지연 1% 미만·라벨 동일).
- 캘리브 비전 출력을 fp32 로 계산한다. 벤더가 NPU 출력을 썼는지는 모른다.
- 파인튜닝으로 가중치 분포가 바뀌면 **양자화 정확도는 재검증이 필요하다**
  (PE-Core 에서 attn_pool 이 INT8 에서 cos 0.46 으로 무너진 전례).
