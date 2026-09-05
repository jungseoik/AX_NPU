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

1. **코어모드 스윕**: vision/text × {single, multi, global4, global8} 조합별 단건 지연·동시처리량 실측
2. **카드 분할 동시성**: 카드당 2~4 인스턴스(코어 분할) vs 카드당 1인스턴스(global8) 비교
3. 배치 컴파일은 차기 qbcompiler 릴리즈 확인 후 재검토

## 부록. 확인에 쓴 명령

```bash
git submodule update --init --remote --recursive          # 659d052
mobilint-cli mxqtool show <Qwen3-VL-2B-Instruct_vision.mxq> | grep "Core Mode"
docker exec mblt_c12 python -c "from qbcompiler import mxq_compile; import inspect; \
  print([p for p in inspect.signature(mxq_compile).parameters if 'batch' in p.lower()])"
```
