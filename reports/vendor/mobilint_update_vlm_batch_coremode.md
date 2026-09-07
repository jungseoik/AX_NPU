# [벤더 회신 확인] Qwen3-VL — 배치 컴파일 / 코어모드 / 동시요청

2026-09-05 Mobilint 임범수 님 회신(VLM Batch Compile, Qwen3-VL 튜토리얼, vLLM override 버그)에 대해
**실제로 확인한 결과**. 서브모듈 `mblt-sdk-tutorial` 최신화(`659d052`) + 우리가 쓰는 mxq 직접 검사.

## 요약

| 벤더 회신 | 확인 결과 |
| --- | --- |
| VLM Batch Compile — 구현 완료, **차기 릴리즈 반영 예정** | ✅ 사실. **qbcompiler 1.2.0 에는 없다** |
| Qwen3-VL Non-batch 컴파일 튜토리얼 업데이트 | ✅ 반영됨 — `compilation/vlm/` |
| vLLM `--model-loader-extra-config` 버그 → vllm-mblt 0.11.2 수정 | ⚠️ 우리 스택은 vLLM 을 안 쓴다(해당 없음) |
| vision/text 가 각기 다른 core mode·target cores 를 가진다 | ✅ 사실이고 **우리 문서가 틀렸다** — 아래 §2 |

## 1. 배치 컴파일 — 1.2.0 에 없다

qbcompiler 1.2.0 컨테이너에서 직접 조회:

```
mxq_compile() batch 관련 파라미터: 없음
LlmConfig 필드: ['apply', 'npu_parallel_degree', 'attributes']   # batch 없음
Config 목록: BitConfig, CalibrationConfig, CompileConfig, EquivalentTransformationConfig,
            LlmConfig, ModConfig, OptqConfig, PreprocessingConfig, ResourceManagementConfig,
            SaveSampleConfig, SearchWeightScaleConfig, Uint8InputConfig
```

우리 mxq 의 `Max Batch Data Size: 0`, `config.json` 의 `max_batch_size: 1` 과도 일치한다.
**배치 추론은 차기 컴파일러를 기다려야 한다.**

## 2. ★ 코어모드 — 우리 문서가 틀렸다

**틀린 서술(정정 전)**: "Qwen3-VL 은 컴파일 모드가 global8 하나뿐이라 카드별 인스턴스로만 분산 가능"

**실제**: mxq 는 `inference_scheme="all"` 로 컴파일돼 **코어모드 4종을 모두 담고 있다.**
`global8` 은 `config.json` 의 기본값일 뿐이며 **런타임에 바꿀 수 있다.**

`mobilint/Qwen3-VL-2B-Instruct` mxq 실측(`mxqtool show` 번들 집계):

| mxq | Single | Multi | Global4 | Global8 |
| --- | ---: | ---: | ---: | ---: |
| `_vision.mxq` | 1 | 5 | 5 | 10 |
| `_text.mxq` | 1 | — | 5 | 10 |

벤더 튜토리얼도 같은 말을 한다 — `compilation/vlm/mxq_compile_language.py`:

> `# REGULUS only supports the single scheme; ARIES supports all schemes in one model.`

그리고 `config.json` 은 단지 기본값을 줄 뿐이다:

```json
"vision_config": { "core_mode": "global8", "target_clusters": [0, 1] },
"text_config":   { "core_mode": "global8", "target_clusters": [0, 1] },
"max_batch_size": 1
```

우리가 쓰는 `mblt-model-zoo` 도 4종을 모두 받는다(`utils/npu_backend.py`):

```python
core_mode: Literal["single", "multi", "global4", "global8"] = "single"
target_cores: Optional[List[Union[str, "CoreId"]]] = None
target_clusters: Optional[List[Union[int, "Cluster"]]] = None
```

즉 **재컴파일 없이 코어모드를 바꿀 수 있다.**

## 3. 동시요청 — 카드 1장을 쪼개 쓸 수 있다

벤더가 준 vLLM 예시가 핵심을 보여준다.

```
'{"dev_no": 0, "vision_core_mode": "global4", "vision_target_clusters": [0],
  "text_core_mode": "single", "text_target_cores": ["1:0"]}'
```

**vision 은 클러스터0 의 global4, text 는 코어 1:0 의 single** — 한 카드 안에서 vision/text 가
서로 다른 코어를 쓴다. 지금 우리 설정(양쪽 다 global8, `target_clusters=[0,1]`)은 **한 인스턴스가
8코어를 독점**하므로 카드당 1스트림이 한계다.

코어를 쪼개면 **카드당 여러 인스턴스**를 띄울 수 있다. 현재 우리 방식(카드별 1인스턴스,
`VLMPool(device_ids="auto")`)보다 동시성이 올라갈 여지가 있다.

> **미검증**: 위는 벤더 예시와 API 시그니처에서 도출한 것이고 **우리가 실측하지 않았다.**
> 코어를 쪼개면 인스턴스당 지연은 늘어난다(global8→single). 동시성 이득이 지연 손해를 넘는지는
> 측정해야 한다. 기존 실측(64동시 1장 12s → 7장 2.2s)은 카드별 1인스턴스 기준이다.
> → [`../performance/NPU_qwen3vl_multicard_batch.md`](../performance/NPU_qwen3vl_multicard_batch.md)

## 4. 다음에 할 것

1. **라이브러리 업그레이드 검토**: `mblt-model-zoo` 1.3.1 → 2.4.2 (§5)
2. **코어모드 스윕**: vision/text × {single, multi, global4, global8} 조합별 단건 지연·동시처리량 실측
3. **카드 분할 동시성**: 카드당 2~4 인스턴스(코어 분할) vs 카드당 1인스턴스(global8) 비교
4. 배치 컴파일(하드웨어)은 차기 qbcompiler 릴리즈 확인 후 재검토 — 단 **소프트웨어 배치는
   이미 v2.4.0 에 있다**(multi-slot)

## 부록. 확인에 쓴 명령

```bash
git submodule update --init --remote --recursive          # 659d052
mobilint-cli mxqtool show <Qwen3-VL-2B-Instruct_vision.mxq> | grep "Core Mode"
docker exec mblt_c12 python -c "from qbcompiler import mxq_compile; import inspect; \
  print([p for p in inspect.signature(mxq_compile).parameters if 'batch' in p.lower()])"
```

## 5. ★ 우리 실행 형태 vs 최신 — 최적화 여지가 크다

### 5-1. 우리가 지금 돌리는 방식

| 항목 | 현재 |
| --- | --- |
| 런타임 라이브러리 | **`mblt-model-zoo==1.3.1`** (핀) |
| 진입 | `AutoModelForImageTextToText.from_pretrained(trust_remote_code=True)` + `AutoProcessor` |
| 코어모드 | vision/text 둘 다 **global8**, `target_clusters=[0,1]` (config.json 기본값 그대로) |
| 배치 | `max_batch_size=1` — 인스턴스당 in-flight 1건 |
| 동시성 | `VLMPool`: **카드당 1인스턴스**, ThreadPoolExecutor 로 카드 간 분산 |
| vision mxq | **static** (입력 1개 `images_0`) → 비디오 불가, 이미지 전용 |

### 5-2. 최신(v2.4.2)과의 격차 — 74커밋

| 버전 | 들어온 것 | 우리에게 의미 |
| --- | --- | --- |
| v2.1.0 | core-mode fan-out to subconfigs | vision/text 코어모드를 따로 주기 쉬워짐 |
| v2.2.1 | `npu_prefill_chunk_size` 런타임 전파 | prefill 튜닝 노브 |
| v2.3.0 | **Qwen3-VL 배치 추론 지원** + MRoPE/dynamic vision + transformers 4.x/5.x | 다중 이미지·배치 경로 |
| v2.4.0 | **multi-slot NPU backend (sw-batch)** | 한 가속기에 **N개 Model 인스턴스**를 띄우고 ThreadPoolExecutor 로 동시 `.infer` |

v2.4.0 커밋 설명이 우리가 PE-Core 에서 쓰는 패턴과 정확히 같다:

> *"N Models per Accelerator with per-Model batching … runs concurrent `.infer` via ThreadPoolExecutor"*

즉 **PE-Core 에서 이미 검증한 "카드당 N모델 + 멀티스레드 동기 infer"를 VLM 에도 쓸 수 있게 된 것**이다.

### 5-3. 확인된 최적화 여지 4가지

1. **라이브러리 업그레이드 (1.3.1 → 2.4.2)** — 위 4개 기능이 전부 여기 딸려온다.
   호환성: v2.4.2 요구 `transformers>=4.54.0,<=5.12.1`, 우리 설치본 **4.57.1 → 범위 안**.
2. **vision `core_mode="multi"` → 하드웨어 배치.** `modeling_qwen3_vl.py` 에 명시적 분기가 있다:
   ```python
   if not is_dynamic and core_mode == "multi" and len(npu_inputs) > 1:
       encoder_outputs = mxq_model.infer(np.stack(npu_inputs, axis=0))   # 한 번에 여러 장
   ```
   우리 vision mxq 에 **Multi 번들이 5개** 있으므로 지금 자산 그대로 쓸 수 있다.
   (text mxq 에는 Multi 가 없다 — vision 전용 경로다)
3. **카드 분할** — vision=global4/cluster0, text=single/core1:0 로 나누면 카드당 다중 인스턴스.
   현재는 global8 로 8코어를 독점해 카드당 1스트림이 한계다.
4. **`dev_no` 리스트 지원** — v2.4.2 는 `dev_no: Union[int, list[int]]` 라 한 인스턴스가
   여러 카드를 걸칠 수 있다. 지금 `VLMPool` 이 파이썬 레벨에서 하는 분산을 라이브러리가 대신할 수 있다.

### 5-4. 주의

- **전부 미실측이다.** 코드·시그니처·mxq 번들에서 확인한 "가능성"이고, 실제 이득은 재봐야 한다.
  코어를 쪼개면 인스턴스당 지연은 늘어난다(global8→single/global4).
- **1.3.1 → 2.4.2 는 메이저 업그레이드**라 `VLMPool`/`load_vlm` 이 그대로 도는지 확인이 필요하다.
  특히 config 필드 마이그레이션(`_migrate_target_cores` 등)이 들어가 있다.
- 우리 vision mxq 는 static 이라 **비디오는 안 된다.** dynamic vision mxq 를 쓰는 신규 릴리스가
  있는지는 별도 확인 필요.

### 5-5. 권장 순서

1. 별도 conda env 에 `mblt-model-zoo==2.4.2` 설치 → 기존 `demo_vlm_qwen3` 흐름이 그대로 도는지 확인
2. vision `core_mode="multi"` 다중 이미지 배치 실측 (현재 자산 그대로 가능)
3. 카드 분할(vision global4 + text single) 동시성 실측 — 카드당 1인스턴스 대비
4. 이득 확인되면 `vlm_npu.py` / `VLMPool` 갱신 + 핀 버전 상향
