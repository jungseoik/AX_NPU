---
name: qwen3-vl
description: ARIES NPU에서 Qwen3-VL(Vision-Language) 모델로 추론 코드를 짤 때 쓴다. 이미지+프롬프트->텍스트 응답. mobilint/Qwen3-VL-2B/4B/8B-Instruct를 표준 HF API(AutoModelForImageTextToText + mblt-model-zoo)로 그대로 가져와 쓴다(포팅 불필요). 단발/스트리밍/멀티턴 헬퍼 제공. "Qwen3-VL 추론", "NPU VLM", "이미지+프롬프트 응답", "멀티모달 LLM 코드" 같은 요청에 사용.
---

# Qwen3-VL NPU 추론

## 언제 쓰나

ARIES NPU에서 **이미지+프롬프트 -> 텍스트**(VLM) 추론 코드를 짤 때.
바로 쓸 수 있는 헬퍼: **`tutorial/pe_npu/vlm_npu.py`** (`load_vlm`(device_id=) / `ask` / `ask_stream` / `VLMChat` / `VLMPool`(멀티카드 동시요청 분산)).
따라하기 문서: `tutorial/pe_npu/README_VLM_qwen3.md`, 노트북 `tutorial/pe_npu/demo_vlm_qwen3.ipynb`.

> PE-Core(이미지->임베딩, `pe_npu` 패키지)와는 **별개**다. 여기 Qwen3-VL은 Mobilint가 이미
> 컴파일해 HF에 올린 멀티모달 LLM을 받아 쓰는 것. 핵심은 **"포팅이 없다"** — NPU 가속은
> 모델 레포의 `trust_remote_code`(mblt_model_zoo) 안에 있고, 우리는 표준 HF API만 호출한다.

## 핵심 사실 (실측 확인)

- **코어모드 = `global8`** — vision/text MXQ 모두 8코어 전부(2클러스터, target_clusters [0,1])를
  묶어 단일 추론. config.json에 박혀 있다. **단일 스트림 latency 최적화**, `max_batch_size=1`.
  (Single/Multi/Global4 변형도 같은 MXQ에 들어있지만 config가 global8을 기본 선택)
  - 확인법: `mobilint-cli mxqtool show <*.mxq>` (코어모드 목록), config.json의 `text_config.core_mode`/`vision_config.core_mode`.
- 지원 모델: `mobilint/Qwen3-VL-2B-Instruct` / `-4B-` / `-8B-Instruct`. (Qwen2-VL은 첫 입력에 이미지 1장 필수, Qwen3-VL은 텍스트만도 가능)
- 입력 텐서는 `.to("cpu")`. NPU 디스패치는 모델 내부가 처리.
- 대화 1건당 `MobilintCache` 1개. 새 대화면 새로 만들어 리셋(안 하면 문맥 오염).
- 기본 샘플링(`do_sample=True, temp 0.7`)이라 매 실행 답이 조금씩 다름. 재현하려면 `GenerationConfig(do_sample=False, ...)`.

## 설치 (버전 핀 중요)

```bash
pip install "mblt-model-zoo[transformers]==1.3.1" "transformers==4.57.1"
```
- `mblt-model-zoo==1.3.1` 핀: `>=`로 열면 1.5.x를 끌어와 더 최신 qbruntime을 요구할 수 있음(현재 검증 런타임 `mobilint-qb-runtime 1.2.0`).
- `transformers>=4.57` 필수: **Qwen3-VL은 4.57부터.** 4.55.x는 `No module named transformers.models.qwen3_vl`로 실패.
- 기존 `pe_npu_host`(PE-Core env)에 얹어도 됨 — `perception-models`의 pillow/tokenizers 핀 충돌 경고는 무시(PE-Core는 벤더 코드를 써서 무관, 둘 다 동작 실측).
- 검증 조합: mblt 1.3.1 / transformers 4.57.1 / tokenizers 0.22.2 / torch cpu / pillow 12.x / qbruntime 1.2.0.

## 바로 쓰는 코드

```python
# tutorial/pe_npu/ 에서 (또는 vlm_npu.py를 PYTHONPATH에)
from vlm_npu import load_vlm, ask, ask_stream, VLMChat

model, processor = load_vlm("mobilint/Qwen3-VL-2B-Instruct")

# 1) 단발: 이미지 + 프롬프트 -> 답변
print(ask(model, processor, "images/cat1.jpg", "Describe this image in detail."))

# 2) 스트리밍 (토큰 단위)
for tok in ask_stream(model, processor, "images/dog.jpg", "What is happening?"):
    print(tok, end="", flush=True)

# 3) 멀티턴 (첫 턴 이미지 -> 이후 텍스트턴이 참조)
chat = VLMChat(model, processor)
chat.ask("What animal is this?", image="images/cat1.jpg")
chat.ask("How many of them?")        # 텍스트만
```

## 헬퍼 없이 raw HF로 (원리)

```python
from transformers import AutoModelForImageTextToText, AutoProcessor, GenerationConfig
from mblt_model_zoo.hf_transformers.utils.cache_utils import MobilintCache

M = "mobilint/Qwen3-VL-2B-Instruct"
model     = AutoModelForImageTextToText.from_pretrained(M, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(M, trust_remote_code=True)
cache     = MobilintCache(model.get_cache_mxq_model())   # 대화 1건당 1개

messages = [{"role": "user", "content": [
    {"type": "image", "url": "images/cat1.jpg"},          # 로컬경로/URL=url, PIL=image
    {"type": "text",  "text": "Describe this image."},
]}]
inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                       padding=True, return_tensors="pt", return_dict=True).to("cpu")
out = model.generate(**inputs, past_key_values=cache,
                     generation_config=GenerationConfig(max_new_tokens=512, repetition_penalty=1.1,
                                                        chunk_size=128, use_cache=True))
print(processor.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## 트러블슈팅

- `No module named transformers.models.qwen3_vl` -> transformers가 4.57 미만. `pip install "transformers==4.57.1"`.
- `No module named mblt_model_zoo` -> `pip install "mblt-model-zoo[transformers]==1.3.1"`.
- NPU 로드 실패/장치 못 찾음 -> `mobilint-cli status`로 `/dev/aries0` 확인 (드라이버/런타임은 skill `npu-setup`).
- 다른 모델로 교체 전 `model.dispose()`로 NPU 자원 해제.

## 출처

Mobilint `mobilint-runtime-gui`(NPU 웹 데모, `mobilint-runtime-gui` 명령 -> `0.0.0.0:5000`)의 백엔드
`src/model_manager.py`(`category=="vlm"`) + `src/vlm_runner.py`에서 VLM 추론 코어만 떼어내 GUI 의존 없이 재구성.
