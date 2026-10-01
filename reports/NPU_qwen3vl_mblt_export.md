# Qwen3-VL 신규 학습본 재컴파일 — export~NPU 전 구간 규명 (2026-10-01)

신규 학습본을 벤더와 같은 레시피로 컴파일하려면 `.mblt` 를 직접 만들어야 한다.
벤더 재현 패키지는 **이미 export 된 `.mblt` 에서 시작**하고, 그 호출법은 문서에 없다.
qbcompiler 내부를 읽어 규명하고, **스톡 통제군과 실제 학습본으로 NPU 까지 검증**한 기록.

- 구현: `qwen3vl_recompile/` (export / 캘리브 / 컴파일 / NPU A/B)
- 환경: GPU 서버 6000-pro (RTX PRO 6000 Blackwell SE ×2) / `mblt_vlm_repro:compile`
- 관련: `reports/RUNBOOK_qwen3vl_bs32_compile.md`, `reports/performance/NPU_qwen3vl_bs32_server.md`

---

## 1. 결론

**생 체크포인트 → export → 캘리브 → mxq → NPU 전 구간이 동작하고, 벤더 산출물과 출력이 동등하다.**

NPU 실측(batch32, 카드 1장, COCO 이미지, yes/no 프롬프트 교차):

| 대상 | 라벨 일치 | 단건 | 8건 | 20건 |
|---|---|---:|---:|---:|
| 스톡 통제군(우리 파이프라인) | **8/8 · 8/8 · 20/20** | −0.4% | −1.2% | +1.0% |
| **학습본**(`PIA-SPACE/AX_NPU_VL3_2B`) | **8/8 · 8/8 · 20/20** | −0.1% | −1.5% | −1.1% |

로짓 크기도 PyTorch 원본과 같은 자리다 — NPU `no 44~45 / yes 34~41` vs 원본 `46~50 / 31~39`.

산출물 크기(벤더 대비):

| | 우리 | 벤더 | 차이 |
|---|---:|---:|---:|
| `vision.mblt` | 1543.7 MB | 1543.7 MB | +0.00% |
| `decoder.mblt` | 6579.9 MB | 6579.8 MB | +0.00% |
| `vision.mxq` | 434,963,448 B | 434,961,465 B | +0.0005% |
| `decoder-b32.mxq` | 925,962,130 B | 926,369,797 B | −0.044% |

`mblt_compile()` 은 **그래프 변환만** 하므로 빠르다(49초). 오래 걸리는 건 그다음
`mxq_compile` 의 양자화 최적화다(vision 27분 / decoder-b32 8분).

★ **스톡 통제군이 이 작업의 핵심 장치였다.** "학습본이 깨진다" 는 현상만으로는
파이프라인 결함인지 체크포인트 특성인지 못 가른다. 같은 코드에 **스톡 가중치**를
태워 벤더와 대조하면 그게 갈린다 — 실제로 결함 5개 중 4개를 이 통제군이 잡아냈다.

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
| 5 | `NotImplementedError: HFProxyableDynamicCache` | HF 가 내부에서 만든 생 `DynamicCache` 가 인자로 흐름 | `past_key_values=DynamicCache()` 를 **feed_dict 로** 전달(추적기가 그 자리에 전용 프록시를 만든다) + logits 만 반환 |
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

## 5. mxq → NPU 에서 드러난 결함 5가지

export 가 "돌아가는 것" 과 "맞는 것" 은 다르다. mblt 크기가 벤더와 +0.00% 로 같아도
아래 다섯은 전부 잘못돼 있었다. **넷은 조용히 틀린다** — 컴파일도 되고 지연도 같다.

| # | 결함 | 증상 | 어떻게 드러났나 |
|---|---|---|---|
| 1 | 캘리브 항목 수 | `quantizeFB failed. The size of info … are not same` | 즉시 |
| 2 | 캘리브 이름을 벤더 json 에서 베낌 | **20분 뒤** `could not find input info in memory: …` | quantizeFB 는 개수만 본다 |
| 3 | `use_cache=False` | 2번과 같은 자리에서 사망 | **연산자 수 대조**(벤더 799 vs 769)로 발견 |
| 4 | `logits_to_keep` 미지정 | `Unexpected final logit rows: (84, 151936)` + mxq 17% 비대 | 벤더 출력 `[1,1,151936]` 과 대조 |
| 5 | rope 캘리브가 단순 concat | **조용히 열화**(스톡에서도 `no` → `Yes`) | 스톡 통제군이 벤더와 안 맞아서 |

### 3번 — KV 캐시를 끄면 안 된다

`use_cache=False` 로 내보내면 그래프에 HF 캐시 업데이트 연산이 안 생기고, 컴파일러가
그걸 접어 만드는 `StatefulKVCacheWrapper`(계층당 2개 = **56개**)가 통째로 빠진다.
prefill 산출물은 KV 캐시를 채워야 하므로 메모리 플래닝에서 죽는다.

캐시를 켜면 이번엔 `NotImplementedError: HFProxyableDynamicCache` 가 나는데,
**캐시 객체를 `feed_dict` 로 넘기면** 추적기가 그 자리에 전용 프록시를 만들어 해결된다
(`hf_cache_patcher.py`). 즉 끄는 게 아니라 **출력에서만 빼는 것**이 정답이다.

### 5번 — rope 캘리브는 rotateTensor 레이아웃으로

런타임(`mblt_model_zoo` 의 `_build_vision_rotate_tensor()` / `MobilintQwen3VLRotaryEmbedding`)이
NPU 에 넣는 rope 는 단순 `cat(cos, sin)` 이 아니라 **인터리브 표**다:

```
rt[0:dim:2]      =  cos[:half]        rt[dim:2dim:2]   = sin[half:]
rt[1:dim:2]      = -sin[:half]        rt[dim+1:2dim:2] = cos[half:]
```

concat 으로 캘리브를 뜨면 **채널별 양자화 범위가 다른 채널에 배정**된다. 그래프는
멀쩡히 돌고 지연도 같아서 출력만 조용히 열화한다. 고치자 스톡 통제군이 벤더와 라벨
완전 일치로 바뀌었고, 비전 출력 코사인도 ds 0.75→0.91 / merger **0.55→0.89** 로 올랐다.

---

## 6. ★ embed_tokens 는 회전해서 넣어야 한다

가장 오래 숨어 있던 함정. 벤더가 배포한 `model.safetensors` 의 `embed_tokens` 는
**생 가중치가 아니라 `embed_tokens @ R`** 다(R = `rotation.pth` 하다마드).
decoder 가 `SpinR1` 로 컴파일돼 **회전된 기저**에서 동작하기 때문이다.

```
스톡 원본 @ R   vs  벤더 포장본      최대오차 0.000e+00   ← 바이트 단위 동일
스톡 원본       vs  벤더 포장본      코사인 ≈ 0          ← 노름·std 는 같은데 직교
```

생 임베딩을 넣으면 **그래프는 멀쩡히 돌고 지연도 같은데** 로짓 스케일만 ~5배 눌린다:

| | yes | no | 고른 토큰 |
|---|---:|---:|---|
| 벤더 | 33~38 | 47~52 | `no` |
| 학습본(생 임베딩) | 6~8 | 8~10 | `\n` |
| 학습본(회전 적용) | 34~41 | 44~45 | `no` |
| PyTorch 학습본 원본 | 31~39 | 46~50 | `no` |

**no > yes 순서는 유지된 채 전체가 눌린다** → 고빈도 토큰에 밀린다. 양자화 열화로
착각하기 딱 좋아서, 실제로 W8 로 올려도 OPTQ 를 켜도 낫지 않았다(둘 다 낭비).
`model.visual.pos_embed.weight` 는 회전 대상이 **아니다**.

이 함정이 오래 숨은 이유: 스톡 통제군은 **벤더의(이미 회전된) safetensors** 를 그대로
썼기 때문에 통과했다. 우리가 직접 포장한 디렉토리만 깨졌다.
→ 조립은 `pack_model_dir.py` 로 한다.

**진단법**: `dump_logits.py` 로 로짓 *크기* 를 본다. 기저가 어긋나면 순서는 보존된 채
스케일이 무너지고, 양자화 열화는 순서가 흐트러진다 — 구분이 된다.

---

## 7. mxq 만 갈아끼우는 drop-in 은 안 된다

벤더 mblt 로 컴파일한 산출물은 I/O 이름·순서가 벤더와 같지만, **우리가 export 한 mblt 는 다르다.**

| | 벤더 | 우리 export |
|---|---|---|
| 비전 입력 순서 | `[cos, pos, folded]` | `[folded, pos, cos]` |
| 비전 출력 이름 | `model_deepstack_merger_list_<i>_…` | `add/reshape_49/…` |
| decoder rope 이름 | `cos/reshape_0` | `language_model_rotary_emb_cos_0` |

- **출력 순서는 제어 가능**하다(래퍼의 반환 순서 → `VisionIO`).
- **입력 순서는 제어 불가**다 — 그래프의 **데이터플로우 순서**라 feed_dict 순서로도
  forward 시그니처로도 안 바뀐다(둘 다 시도해 확인). 그런데 런타임은 이 리스트를
  **위치로** 넘긴다 → `ValueError: Input shape is invalid.`
- **이름이 다르면 벤더 설정의 이름 기반 지정이 조용히 무시된다.** 비전
  `layerOverrides.activation16Bits` 가 그렇다(출력 4개를 16bit 로 올리는 지정 —
  PE-Core 에서 attn_pool 만 16bit 로 올려 복구한 것과 같은 자리).
  → `compile_mxq.py` 가 **우리 그래프 출력 이름으로 교체**한다.

로더에 끼울 어댑터 3종을 `artifact_order.py` 에 모았다(벤더 산출물에는 항등으로 동작).

---

## 8. 검증 도구와 폐기한 방법

| 도구 | 용도 |
|---|---|
| `inspect_mblt.py` | mblt 입력 규격 조회. 캘리브가 어긋날 때 증상별 원인 수록 |
| `compare_vision_mxq.py` | **출력 순서 확정의 표준** — 검증된 mxq 와 직접 비교 |
| `dump_logits.py` | 토큰·로짓 덤프. 기저 어긋남 vs 양자화 열화 구분 |
| `verify_npu.py` | 기준 모델과 라벨·지연 A/B |

★ **PyTorch 를 기준으로 쓰려던 방법은 폐기했다.**
`model.get_image_features()` 와 비전 mxq 출력은 코사인이 0 근처로 나오는데,
**벤더 자신의 mxq + 스톡 체크포인트로 대조해도 0 이었다**. 즉 산출물 문제가 아니라
기준 계산이 틀린 것이었다(전처리 경로가 다르다). 한때 "비전이 깨졌다" 고 오판할 뻔했다 —
기준으로 쓸 것은 **이미 검증된 mxq** 다.

---

## 9. 남은 것

- 평가는 yes/no 라벨 + 지연이다. **학습본의 실제 과제(운영 프롬프트)로 정확도 평가는 별도**다.
- 캘리브는 COCO 115장 + 짧은 프롬프트다. 운영 프롬프트 분포로 뜨면 더 나을 수 있다(미실측).
- 캘리브의 비전 출력은 fp32 로 계산한다. 벤더가 NPU 출력을 썼는지는 모른다.
