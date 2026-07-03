# 메일 발송본 — Qwen3-VL-2B 배치·코어모드 서빙 문의

---

**제목**: [PIA-SPACE] Qwen3-VL-2B vLLM 서빙 — 배치/코어모드 MXQ 또는 컴파일 방법 문의

**수신**: Mobilint 기술지원팀

---

안녕하세요, PIA-SPACE 정시준입니다.
항상 지원해 주셔서 감사합니다. Qwen3-VL-2B를 NPU로 서빙하는 과정에서 배치·코어모드 관련 문의가 있어 메일 드립니다.

## 1. 배경 — 저희가 하려는 것

저희는 **vLLM + `vllm-mblt` 플러그인**으로 `mobilint/Qwen3-VL-2B-Instruct`를 OpenAI 호환 HTTP API(`/v1/chat/completions`)로 서빙하려고 합니다.

목표는 **"NPU 1장당 Qwen3-VL 동시요청을 몇 개까지 받을 수 있는지"** 를 코어모드(single/multi/global4/global8)와 배치사이즈별로 실측해서, 카드당 채널 수 등 배포 설계를 잡는 것입니다. 이 측정 자체는 저희가 직접 진행할 예정이고, 그러려면 **배치/모드별로 컴파일된 MXQ** 또는 **직접 컴파일할 수 있는 방법**이 필요해서 문의드립니다.

측정하려는 형태는 대략 아래 표와 같습니다.

| core_mode | max_batch | 동시요청 1 | 2 | 4 | 8 |
|-----------|----------:|:---:|:---:|:---:|:---:|
| single    | 1 / 4 / 8 | … | … | … | … |
| multi     | 1 / 4 / 8 | … | … | … | … |
| global4   | 1 / 4 / 8 | … | … | … | … |
| global8   | 1 / 4 / 8 | … | … | … | … |

(셀 = 총지연 / 처리량, NPU 메모리)

## 2. 현재 막힌 지점

- 배포된 `mobilint/Qwen3-VL-2B-Instruct`의 config는 **`max_batch_size=1`, `core_mode=global8`(text/vision) 고정**입니다. 그래서 vLLM에서 동시요청이 직렬로 큐잉됩니다(부하테스트 시 NPU 메모리 2.5GB 고정, 총지연은 요청 수에 선형으로 증가).
- 실행 중에 모드/배치를 바꾸려고 `--model-loader-extra-config`로 `core_mode`/`max_batch_size`/`dev_no`를 넘기면 `Qwen3VLForConditionalGeneration.__init__() got unexpected keyword 'dev_no'` (TypeError)로 엔진이 크래시합니다.
- 배치는 MXQ 컴파일 시점에 결정되는 값으로 이해하고 있는데, **공식 문서·튜토리얼에는 Qwen3-VL 컴파일이나 배치 관련 안내가 없습니다.** (`mblt-sdk-tutorial/compilation/vlm`은 Qwen2-VL 전용이고, `mblt-model-zoo`/`vllm-mblt`는 추론 런타임만 제공)

## 3. 문의 사항

**Q1. `max_batch_size`가 vLLM 관점에서 정확히 무엇을 의미하나요?**
표의 축을 잘못 잡으면 측정이 헛수고가 되어 개념부터 확인하고 싶습니다. MXQ의 `max_batch_size=N`이
(a) vLLM 연속배칭에서 **서로 다른 HTTP 요청 N개를 한 번의 추론으로 묶는 것**(= 동시요청 처리량 증가)인지,
(b) **한 요청 내부의 배치 차원**(예: 이미지 N장)을 병렬 처리하는 것이고 요청 간 동시성과는 무관한 것인지 궁금합니다.
결국 **"NPU 1장의 동시요청 수"를 늘리려면 `max_batch_size` 컴파일이 답인지, 아니면 `core_mode`를 single로 두는 것이 답인지** — 올바른 방법을 알려주시면 감사하겠습니다.

**Q2. Qwen3-VL-2B는 어떤 core_mode로 컴파일/제공이 가능한가요?**
저희는 single/multi/global4/global8 4종을 비교하고 싶은데, language 파트가 포함된 VLM이 이 4종 모두로 컴파일 가능한 모델인지 확실치 않습니다. **제공/컴파일 가능한 core_mode 목록**과 모드별 특성(단건 지연 vs 동시요청 처리량)을 알려주시면 좋겠습니다.

**Q3. 위를 바탕으로, 아래 둘 중 하나로 지원이 가능할까요?**

- **(A) 배치/모드별 MXQ 제공 (우선 희망)** — `Llama-3.2-1B-Instruct-Batch32`처럼 **batch>1로 컴파일된 `Qwen3-VL-2B` MXQ**(예: Batch 4/8/16), 가능하면 Q2에서 확인된 **core_mode 변형**까지 받을 수 있을까요? VLM에서 배치가 이미지 입력에 주는 제약과 권장 batch/정확도도 함께 안내 부탁드립니다.

- **(B) 자체 컴파일용 자료 제공 ((A)가 번거로우시면)** — 저희가 직접 배치·모드별로 컴파일할 수 있도록 **① calibration 데이터셋(또는 생성 스크립트)** 과 **② Qwen3-VL 컴파일 코드/레시피**(vision+language, RoPE/그래프 패치, batch_size 지정, 패키징 규격)를 받을 수 있을까요? (저희 qbcompiler 1.1.2에 `qwen3vl` 파서는 있으나, Qwen2-VL 튜토리얼 자산을 Qwen3-VL로 옮기는 방법이 문서화돼 있지 않습니다.)

## 4. (겸사) vllm-mblt 0.1.0 버그 2건

서빙 중에 확인한 부분이라 함께 공유드립니다.

1. **`config.vocab_size` AttributeError** — `mblt_worker._make_cached_sampling_state`가 `self.model.config.vocab_size`에 접근하는데, `MobilintQwen3VLConfig`는 `vocab_size`가 `text_config` 아래에 있어 이미지 요청 시 EngineCore가 크래시합니다. (현재 `text_config.vocab_size` 폴백으로 임시 패치해 사용 중입니다.)
2. **`--model-loader-extra-config` TypeError** — 위 2번에서 언급한 건으로, VLM에서 런타임 레이아웃(core_mode/dev_no)을 override하는 올바른 방법을 알려주시면 감사하겠습니다.

---

Docker(compose) + vllm-mblt 서빙 구성, 동시요청 부하테스트 스크립트, 크래시 로그 전문은 필요하시면 바로 전달드리겠습니다.

바쁘신 와중에 확인 부탁드립니다. 감사합니다.

PIA-SPACE 정시준 드림
si.jung@pia.space
