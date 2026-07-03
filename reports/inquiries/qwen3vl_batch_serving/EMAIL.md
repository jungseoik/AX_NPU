# 메일 발송본 — Qwen3-VL-2B 배치·코어모드 서빙 문의

---

**제목**: [PIA-SPACE] Qwen3-VL-2B vLLM 서빙 — 배치/코어모드 MXQ 또는 컴파일 방법 문의

**수신**: Mobilint 기술지원팀

---

안녕하세요, PIA-SPACE 정시준입니다.
항상 지원해 주셔서 감사합니다. Qwen3-VL-2B를 NPU로 서빙하는 과정에서 배치·코어모드 관련 문의가 있어 메일 드립니다.

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

## 2. 현재 막힌 지점

- 배포된 `mobilint/Qwen3-VL-2B-Instruct`의 config는 **`max_batch_size=1`, `core_mode=global8`(text/vision) 고정**입니다. 그래서 vLLM에서 동시요청이 직렬로 큐잉됩니다(부하테스트 시 NPU 메모리 2.5GB 고정, 총지연은 요청 수에 선형으로 증가). → 동시요청이 늘면 목표 지연시간을 넘기게 됩니다.
- 배치는 MXQ 컴파일 시점에 결정되는 값으로 이해하고 있는데(실제로 `max_batch_size`는 실행 중 override 경로 자체가 없어 보입니다), **공식 문서·튜토리얼에는 Qwen3-VL 컴파일이나 배치 관련 안내가 없습니다.** (`mblt-sdk-tutorial/compilation/vlm`은 Qwen2-VL 전용이고, `mblt-model-zoo`/`vllm-mblt`는 추론 런타임만 제공)

## 3. 문의 사항

**Q1. `max_batch_size`가 vLLM 관점에서 정확히 무엇을 의미하나요?**
표의 축을 잘못 잡으면 측정이 헛수고가 되어 개념부터 확인하고 싶습니다. MXQ의 `max_batch_size=N`이
(a) vLLM 연속배칭에서 **서로 다른 HTTP 요청 N개를 한 번의 추론으로 묶는 것**(= 동시요청 처리량 증가)인지,
(b) **한 요청 내부의 배치 차원**(예: 이미지 N장)을 병렬 처리하는 것이고 요청 간 동시성과는 무관한 것인지 궁금합니다.
결국 **"NPU 1장의 동시요청 수"를 목표 지연시간 안에서 늘리려면 `max_batch_size` 컴파일이 답인지, 아니면 `core_mode`를 single로 두는 것이 답인지** — 올바른 방법을 알려주시면 감사하겠습니다.

**Q2. Qwen3-VL-2B는 어떤 core_mode로 컴파일/제공이 가능한가요?**
저희는 single/multi/global4/global8 4종을 비교하고 싶은데, language 파트가 포함된 VLM이 이 4종 모두로 컴파일 가능한 모델인지 확실치 않습니다. **제공/컴파일 가능한 core_mode 목록**과 모드별 특성(단건 지연 vs 동시요청 처리량)을 알려주시면 좋겠습니다.

**Q3. 위를 바탕으로, 아래 둘 중 하나로 지원이 가능할까요?**

- **(A) 배치/모드별 MXQ 제공 (우선 희망)** — `Llama-3.2-1B-Instruct-Batch32`처럼 **batch>1로 컴파일된 `Qwen3-VL-2B` MXQ**(예: Batch 4/8/16), 가능하면 Q2에서 확인된 **core_mode 변형**까지 받을 수 있을까요? VLM에서 배치가 이미지 입력에 주는 제약과 권장 batch/정확도도 함께 안내 부탁드립니다.

- **(B) 자체 컴파일용 자료 제공 ((A)가 번거로우시면)** — 저희가 직접 배치·모드별로 컴파일할 수 있도록 **① calibration 데이터셋(또는 생성 스크립트)** 과 **② Qwen3-VL 컴파일 코드/레시피**(vision+language, RoPE/그래프 패치, batch_size 지정, 패키징 규격)를 받을 수 있을까요? (저희 qbcompiler 1.1.2에 `qwen3vl` 파서는 있으나, Qwen2-VL 튜토리얼 자산을 Qwen3-VL로 옮기는 방법이 문서화돼 있지 않습니다.)

**Q4. (관련) VLM에서 로드 시 core_mode/dev_no를 override하는 올바른 방법이 있나요?**
실행 중 코어모드를 바꿔보려고 vLLM의 `--model-loader-extra-config '{"dev_no":0, "core_mode":...}'` 로 값을 넘겼더니, 해당 kwarg가 그대로 `from_pretrained`를 거쳐 `Qwen3VLForConditionalGeneration.__init__() got unexpected keyword 'dev_no'` (TypeError)로 엔진이 종료됐습니다. 텍스트 모델에서는 정상 동작하는데 VLM에서만 실패하는 것으로 보여, **VLM에서 로드 시 레이아웃(core_mode/dev_no 등)을 지정하는 올바른 방법**을 알려주시면 감사하겠습니다. (Q3에서 모드별 MXQ를 받으면 이 부분은 필요 없어질 수도 있습니다.)

## 4. (참고 공유) vllm-mblt 0.1.0 — Qwen3-VL 서빙 중 확인한 수정 필요 지점

서빙 과정에서 발견한 부분이라 참고로 공유드립니다.

- **`config.vocab_size` AttributeError** — `mblt_worker.py`의 `_make_cached_sampling_state`(및 `_pack_prompt_token_ids`)가 `self.model.config.vocab_size`에 접근하는데, `mobilint/Qwen3-VL-2B-Instruct`의 config는 top-level `vocab_size`가 없고 `text_config.vocab_size`(151936)에 있어 이미지 요청 시 EngineCore가 종료됩니다.
  - 이 현상은 **`mobilint/vllm-mblt` 레포(github.com/mobilint/vllm-mblt) README의 "Serve a VLM Model" 안내 경로 그대로**(`vllm serve mobilint/Qwen3-VL-2B-Instruct --trust-remote-code`)에서, 저희 쪽 수정 없이 재현됩니다. 문제의 접근은 `vllm-mblt` 원본 코드(`vllm_mblt/mblt_worker.py`)에 있고, 저희 Docker 패치는 이를 우회하기 위해 폴백을 덧댄 것입니다.
  - 참고로 `mblt-model-zoo`의 `utils/benchmark_utils.py`에는 이미 `config.vocab_size` → `text_config.vocab_size` 폴백 로직이 있어, `vllm-mblt`에도 동일하게 적용하면 될 것으로 보입니다. 저희는 우선 `text_config.vocab_size` 폴백으로 로컬 패치해 정상 구동 중입니다.

## 5. 재현 정보 (요청 시)

Docker(compose) + vllm-mblt 서빙 구성, 동시요청 부하테스트 스크립트, 크래시 로그 전문, 위 로컬 패치 diff는 필요하시면 바로 전달드리겠습니다.

바쁘신 와중에 확인 부탁드립니다. 감사합니다.

PIA-SPACE 정시준 드림
si.jung@pia.space
