# 메일 발송본 — Qwen3-VL-2B 배치·코어모드 서빙 문의

---

**제목**: [PIA-SPACE] Qwen3-VL-2B vLLM 서빙 — 배치/코어모드 MXQ 또는 컴파일 방법 문의

**수신**: Mobilint 기술지원팀

---

안녕하세요, PIA-SPACE 정시준입니다.

먼저 지난번 attention pooling 성능 저하 건, 상세히 분석해 주셔서 정말 감사했습니다. 알려주신 대로 **QK matmul을 layer pattern으로 탐지해 해당 output을 16bit로 지정**하는 방식을 저희 full model에 적용했더니, 원본 대비 cos similarity가 **0.99까지 잘 복구**되었습니다. 덕분에 image→embedding 전 과정을 NPU로 돌릴 수 있게 되어 큰 도움이 되었습니다.

이번에는 Qwen3-VL-2B를 NPU로 서빙하는 과정에서 배치·코어모드 관련 문의가 있어 다시 메일 드립니다.

## 1. 배경 — 저희가 궁극적으로 확인하려는 것

저희는 **vLLM + `vllm-mblt` 플러그인**으로 `mobilint/Qwen3-VL-2B-Instruct`를 OpenAI 호환 HTTP API(`/v1/chat/completions`)로 서빙해 실제 서비스에 투입하려고 합니다.

궁극적인 목적은 **"이 서버가 저희 목표 지연시간(target latency) 안에서 동시요청을 몇 개까지 감당할 수 있는지"** 를 확인하는 것입니다. 즉 요청당 응답이 **목표 지연시간(예: 요청당 ○○초 이내)** 을 유지하면서 동시에 몇 개의 요청까지 처리 가능한지 — 그 한계점(카드당 수용 가능한 동시요청 수)을 실측해 배포 규모(카드 수·채널 수)를 산정하려고 합니다.

이를 위해 **NPU 1장에서 코어모드(single/multi/global4/global8) × 배치사이즈**를 바꿔가며 동시요청 수를 늘려보고, 목표 지연시간을 넘지 않는 지점을 찾으려 합니다. 이 측정 자체는 저희가 직접 진행할 예정이며, 그러려면 **배치/모드별로 컴파일된 MXQ** 또는 **직접 컴파일할 수 있는 방법**이 필요해서 문의드립니다.

측정하려는 형태는 대략 아래 표와 같습니다.

| core_mode | max_batch | 동시요청 1 | 2 | 4 | 8 |
|-----------|----------:|:---:|:---:|:---:|:---:|
| single    | 1 / 4 / 8 | … | … | … | … |
| multi     | 1 / 4 / 8 | … | … | … | … |
| global4   | 1 / 4 / 8 | … | … | … | … |
| global8   | 1 / 4 / 8 | … | … | … | … |

(셀 = 총지연 / 처리량, 목표 지연시간 유지 여부, NPU 메모리)

## 2. 현재 상황 및 저희가 문서에서 확인한 내용

먼저 관련 문서를 저희 나름대로 확인했고, 아래와 같이 이해했습니다(확인한 문서 출처를 함께 적었습니다). 이 이해가 Qwen3-VL에도 맞는지 확인 부탁드리는 것이 이번 문의의 핵심입니다.

- 배포된 `mobilint/Qwen3-VL-2B-Instruct`의 config는 **`max_batch_size=1`, `core_mode=global8`(text/vision) 고정**입니다. 그래서 vLLM에서 동시요청이 직렬로 큐잉됩니다(부하테스트 시 NPU 메모리 2.5GB 고정, 총지연은 요청 수에 선형으로 증가) → 동시요청이 늘면 목표 지연시간을 넘깁니다.
- **`vllm-mblt` 레포**(github.com/mobilint/vllm-mblt) **README의 "Runtime Tuning"** 확인: `--model-loader-extra-config`로 `core_mode`/`max_batch_size`를 override할 수 있고, `max_batch_size`는 vLLM `max_num_seqs`로 반영되는 것으로 이해했습니다. 다만 README는 실제 배치 실행을 **"batch-compiled MXQ"**(예: `mobilint/Llama-3.2-1B-Instruct-Batch32`)를 전제로 설명하고 있어, 저희는 **진짜 batch>1은 배치로 컴파일된 MXQ가 있어야 가능**하다고 이해했습니다.
- **Mobilint 공식 문서**(docs.mobilint.com)**의 Multicore 페이지**(v1.2/en/multicore) 확인: ARIES는 4개 코어모드(Single/Multi/Global4/Global8)를 모두 지원하며, 코어모드는 컴파일 시 `inference_scheme`으로 결정(각 모드별 MXQ)되는 것으로 이해했습니다. 다만 문서 예제는 vision CNN(resnet50)이라 **VLM(비전+언어 디코더)에 그대로 적용되는지는 불명확**합니다.
- **Qwen3-VL 컴파일 자료는 어느 레포에서도 찾지 못했습니다**: **`mblt-sdk-tutorial` 레포**(github.com/mobilint/mblt-sdk-tutorial)**의 `compilation/vlm`**은 **Qwen2-VL 전용**(language 디코더+vision 인코더 분리 컴파일 레시피 존재)이고, **`mblt-model-zoo` 레포**(github.com/mobilint/mblt-model-zoo)와 **`vllm-mblt` 레포**는 추론 런타임만 제공합니다. 세 레포 모두 Qwen3-VL의 컴파일/배치 예제는 없었습니다.

## 3. 문의 사항

**Q1. (확인) Qwen3-VL의 동시요청을 늘리려면 batch-compiled MXQ가 필요한 것이 맞나요?**
위 이해대로라면, 배포된 batch=1 MXQ에 `--model-loader-extra-config '{"max_batch_size":N}'`을 줘도 스케줄러 `max_num_seqs`만 올라갈 뿐 NPU 단에서 실제 배치 추론은 안 되고, **동시요청을 실제로 늘리려면 batch로 컴파일된 Qwen3-VL MXQ**(`Llama-3.2-1B-Instruct-Batch32`처럼)가 필요한 것으로 이해했습니다. 이 이해가 맞는지 확인 부탁드립니다. (batch 없이 `core_mode`를 single로 두어 코어별 슬롯으로 동시성을 얻는 방식이 VLM에도 유효한지도 함께 알려주시면 좋겠습니다.)

**Q2. (확인) Qwen3-VL(VLM)도 4개 코어모드 전부로 컴파일 가능한가요?**
`docs/multicore.md`상 ARIES는 4모드를 모두 지원하고 코어모드는 컴파일 시 `inference_scheme`으로 결정된다고 이해했으나, 문서 예제가 vision CNN이라 **VLM(특히 language 디코더 부분)이 single/multi/global4/global8 4종 모두로 컴파일 가능한지**는 확신이 서지 않습니다(배포본은 global8만 제공). 실제로 제공/컴파일 가능한 코어모드 범위와, 모드별 특성(단건 지연 vs 동시요청 처리량)을 알려주시면 감사하겠습니다.

**Q3. 위를 바탕으로, 아래 둘 중 하나로 지원이 가능할까요?**

- **(A) 배치/모드별 MXQ 제공 (우선 희망)** — `Llama-3.2-1B-Instruct-Batch32`처럼 **batch>1로 컴파일된 `Qwen3-VL-2B` MXQ**(예: Batch 4/8/16), 가능하면 Q2에서 확인된 **core_mode 변형**까지 받을 수 있을까요? VLM에서 배치가 이미지 입력에 주는 제약과 권장 batch/정확도도 함께 안내 부탁드립니다.

- **(B) 자체 컴파일용 "레시피" 제공 ((A)가 번거로우시면)** — 확인해보니 저희 **qbcompiler 1.1.2에 Qwen3-VL 파서와 패칭 클래스(`CachedQwen3VLTextRotaryEmbedding`, `Qwen3VLForConditionalGenerationWrapper`, deepstack 처리 등)가 이미 구현**돼 있어, `mblt-sdk-tutorial`의 Qwen2-VL 절차를 템플릿 삼아 **저희가 직접 컴파일을 시도**해보려 합니다. 다만 다음이 문서화돼 있지 않아 확인이 필요합니다:
  - **① Qwen3-VL용 `CompileConfig` 레시피** — 어떤 레이어를 `activation16Bits`로 둘지, `equivalentTransformation`(QK/UD/SpinR1/SpinR2, vision의 HeadOutChRotation 등) 설정값. Qwen2-VL 튜토리얼의 값(`inputs_embeds/reshape`, `model_merger_fc2` 등)은 Qwen3-VL에 그대로 맞지 않을 것으로 보입니다.
  - **② calibration 데이터 사양** — language/vision 각각 어떤 데이터·형식·개수를 권장하시는지. (저희는 COCO를 준비해 두었습니다.)
  - **③ config 변환/패키징 규격** — `get_config.py`류에서 Qwen3-VL용으로 바꿔야 할 필드(model_type, mxq_path, core_mode/batch 지정 등)와 최종 배포본 config 형식.

**Q4. (관련) VLM에서 `--model-loader-extra-config` override가 실패하는데, 올바른 방법이 있나요?**
`vllm-mblt` README(Runtime Layout Overrides)에는 `dev_no`/`core_mode` 등을 `--model-loader-extra-config`로 override할 수 있다고 안내돼 있으나 **예제가 텍스트 모델**뿐입니다. 저희가 이를 VLM(`mobilint/Qwen3-VL-2B-Instruct`)에 `'{"dev_no":0, "core_mode":...}'`로 적용했더니, 해당 kwarg가 그대로 `from_pretrained`를 거쳐 `Qwen3VLForConditionalGeneration.__init__() got unexpected keyword 'dev_no'` (TypeError)로 엔진이 종료됐습니다. **VLM에서도 이 override가 지원되는지, 지원된다면 올바른 지정 방법**을 알려주시면 감사하겠습니다. (Q3에서 모드별 MXQ를 받으면 이 부분은 필요 없어질 수도 있습니다.)

## 4. (참고 공유) vllm-mblt 0.1.0 — `config.vocab_size` AttributeError 버그

서빙 검증 중 확인한 버그로, **순정 `vllm-mblt 0.1.0` + 공식 README 명령으로 저희가 직접 재현**했습니다.

- **증상**: `mblt_worker.py`의 `_make_cached_sampling_state`가 `top_k` 정규화 시
  `sampling_params.top_k if top_k > 0 else self.model.config.vocab_size` 로 `vocab_size`에 접근합니다.
  `mobilint/Qwen3-VL-2B-Instruct`의 config에는 top-level `vocab_size`가 없고 `text_config.vocab_size`(151936)에만 있어,
  **`top_k <= 0`(top_k 비활성, 예: `top_k=-1` 또는 `0`) 파라미터로 요청하면** 아래 오류로 EngineCore가 종료됩니다.
  ```
  File ".../vllm_mblt/mblt_worker.py", line 1445, in _make_cached_sampling_state
      else self.model.config.vocab_size
  AttributeError: 'MobilintQwen3VLConfig' object has no attribute 'vocab_size'
  ```
- **재현 방법**(수정 없는 순정 vllm-mblt 0.1.0, README "Serve a VLM Model" 경로 그대로):
  1. `vllm serve mobilint/Qwen3-VL-2B-Instruct --trust-remote-code`
  2. `top_k: -1`을 포함한 요청 전송(텍스트만으로도 재현) → EngineCore 종료.
  - 참고: **기본 top_k(모델 config 기본값 20) 요청은 텍스트·이미지 모두 정상**입니다. `top_k`를 끄는(≤0) 요청에서만 발생합니다.
- **수정**: `getattr(config, "vocab_size", None) or config.get_text_config().vocab_size` 폴백으로 해결됩니다.
  이미 `mblt-model-zoo`의 `utils/benchmark_utils.py`에 동일 폴백(`_resolve_config_vocab_size`)이 있어 `vllm-mblt`에도 적용하면 될 것으로 보입니다.
  저희는 이 폴백을 로컬 패치해 정상 구동 중이며, **패치 후 `top_k=-1` 요청 정상 동작까지 확인**했습니다.

## 5. 테스트 환경

- **OS**: Ubuntu 22.04.1 LTS (kernel 6.5.0-41-generic)
- **CPU**: Intel Xeon Gold 6526Y ×2 (총 64 threads), RAM 188GB
- **NPU**: Mobilint **ARIES ×7**(`/dev/aries0~6`, PCIe), **driver 1.13.0 / firmware 1.2.5**, 카드당 16GB
- **SW**: `vllm==0.11.2` / `vllm-mblt==0.1.0` / `mblt-model-zoo[transformers]==1.5.1` / `mobilint-qb-runtime==1.2.0` / Python 3.11
- **Model**: `mobilint/Qwen3-VL-2B-Instruct`

필요하시면 위 vocab_size 건의 **크래시 로그(traceback) 전문**과 적용한 **폴백 패치 diff**를 바로 전달드리겠습니다.

바쁘신 와중에 확인 부탁드립니다. 감사합니다.

PIA-SPACE 정시준 드림
si.jung@pia.space
