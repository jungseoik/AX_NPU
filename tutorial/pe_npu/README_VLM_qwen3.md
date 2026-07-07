# Qwen3-VL NPU 추론 튜토리얼 (Vision-Language)

ARIES NPU에서 **Qwen3-VL** 시리즈(이미지+텍스트 → 텍스트)를 추론한다.
**이미지 한 장 + 프롬프트를 주면 텍스트 답변을 받는** 구조.

- 실습 노트북: [`demo_vlm_qwen3.ipynb`](demo_vlm_qwen3.ipynb)
- 바로 쓰는 헬퍼: [`vlm_npu.py`](vlm_npu.py) (`load_vlm` / `ask` / `ask_stream` / `VLMChat`)
- 에이전트용 요약 skill: `.claude/skills/qwen3-vl/`
- 대상 모델: `mobilint/Qwen3-VL-2B / 4B / 8B-Instruct`

> **이건 PE-Core 비전인코더(이 레포 본체)와는 별개의 작업이다.**
> PE-Core는 이미지→임베딩(`pe_npu` 패키지, 직접 컴파일). 여기 Qwen3-VL은 Mobilint가
> 이미 컴파일해 HF(`mobilint/...`)에 올려둔 멀티모달 LLM을 **그대로 받아 쓰는** 것이다.

---

## 핵심: "포팅"이 필요 없다

NPU 가속 코드는 **모델 레포 안(`trust_remote_code`)에 전부 들어있다.** 우리 쪽 코드는
NPU를 전혀 몰라도 되고, 표준 HuggingFace API만 호출한다:

```python
from transformers import AutoModelForImageTextToText, AutoProcessor, GenerationConfig
from mblt_model_zoo.hf_transformers.utils.cache_utils import MobilintCache

model     = AutoModelForImageTextToText.from_pretrained("mobilint/Qwen3-VL-2B-Instruct", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("mobilint/Qwen3-VL-2B-Instruct", trust_remote_code=True)
cache     = MobilintCache(model.get_cache_mxq_model())   # NPU KV-cache (대화 1건당 1개)
```

`from_pretrained`이 HF에서 vision/text 두 MXQ + 토크나이저를 받아 NPU에 올린다 (최초 1회만 다운로드).

> 이 패턴은 Mobilint `mobilint-runtime-gui`(웹 데모 서버)의 `vlm_runner.py` / `model_manager.py`
> 동작을 그대로 떼어내 이 레포에서 **end-to-end 실행 검증**한 것이다.

---

## 코어모드 — 어떻게 "최적화"돼 있나 (single? multi? global4? global8?)

`mobilint/Qwen3-VL-*` MXQ는 vision/text **둘 다 `global8`** 로 실행된다
(config.json의 `text_config.core_mode` / `vision_config.core_mode` = `global8`, `target_clusters=[0,1]`).
2B·4B 모두 확인.

- **global8 = 한 ARIES 카드의 8코어 전부(2클러스터)를 묶어 단일 추론을 처리** → 단일 스트림 **latency 최적화**. `max_batch_size=1`.
- MXQ 자체는 Single/Multi/Global4/Global8을 **모두 담은 fat 파일**이지만, config가 global8을 기본 선택한다.
- 확인법: `mobilint-cli mxqtool show <model>_text.mxq` (담긴 코어모드 목록) / config.json의 `*.core_mode`.
- 코어모드 일반 설명·벤치: `docs/multicore.md`, `reports/performance/NPU_batch_latency.md`.

> 한 줄 답: **global8** (single/multi/global4 아님). 카드 하나를 통째로 써서 한 요청을 가장 빠르게 돌리는 모드.
> 대량 동시요청 throughput이 목적이면 single 8개 독립이 유리할 수 있으나, VLM 챗/단일스트림엔 global8이 맞다.

---

## 0. 설치

NPU 드라이버 + qbruntime(`mobilint-qb-runtime`)이 설치된 서버 기준 (`.claude/skills/npu-setup` 참고).
그 위에 파이썬 패키지만 깔면 된다.

### ⚠️ 버전 핀이 중요하다 (실측으로 확정)

| 패키지 | 핀 | 이유 |
|---|---|---|
| `mblt-model-zoo` | **`==1.3.1`** | `>=1.3.1`로 열면 1.5.x를 끌어오는데, 더 최신 qbruntime을 요구할 수 있음. 검증된 런타임은 `mobilint-qb-runtime 1.2.0`이라 1.3.1로 핀 |
| `transformers` | **`>=4.57`** (검증 `4.57.1`, mblt 상한 `4.57.6`) | **Qwen3-VL은 transformers 4.57부터 지원.** 4.55.x에서는 `No module named transformers.models.qwen3_vl`로 실패 |
| `torch` | CPU 휠 | NPU가 연산을 가져가므로 GPU torch 불필요 |

### (A) 새 conda env

```bash
conda create -n vlm_npu python=3.10 -y && conda activate vlm_npu
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install "mblt-model-zoo[transformers]==1.3.1" "transformers==4.57.1"
```

### (B) 이 레포의 기존 PE-Core env(`pe_npu_host`)에 얹기

PE-Core 추론에 쓰던 env에 그대로 추가해도 된다 (이 레포에서 실측 검증함):

```bash
conda activate pe_npu_host
pip install "mblt-model-zoo[transformers]==1.3.1" "transformers==4.57.1"
```

> 이때 pip가 **`perception-models가 pillow==11.0.0 / tokenizers==0.21.1을 요구하는데 충돌`** 경고를 띄운다.
> **무시해도 된다.** PE-Core는 벤더된 코드(`pe_npu/pe_vendor`)를 쓰지 perception-models 패키지를
> import하지 않는다. 업그레이드(transformers 4.55→4.57, pillow 11→12, tokenizers 0.21→0.22) 후
> **PE-Core 추론(`(1,1024)` 임베딩)과 VLM 둘 다 정상 동작**함을 확인했다.

검증된 조합: `mblt-model-zoo 1.3.1` · `transformers 4.57.1` · `tokenizers 0.22.2` · `torch 2.x+cpu` · `pillow 12.x` · `mobilint-qb-runtime 1.2.0`.

> 이미 `mobilint-runtime-gui`가 깔려 있으면 그 venv
> (`/usr/lib/mobilint/mobilint-runtime-gui/.venv`)를 그대로 써도 된다.

`mobilint-cli status`로 `/dev/aries0` 인식부터 확인할 것.

---

## 1. 가장 기본 — 이미지 + 프롬프트 → 답변

```python
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "url": "images/dog.jpg"},     # 로컬 경로 / URL / PIL.Image
        {"type": "text",  "text": "Describe this image in detail."},
    ],
}]

inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True,
    padding=True, return_tensors="pt", return_dict=True,
).to("cpu")                          # ← NPU 아니라 cpu. NPU 디스패치는 모델이 알아서 한다.

gen = GenerationConfig(max_new_tokens=512, repetition_penalty=1.1, chunk_size=128, use_cache=True)
out = model.generate(**inputs, past_key_values=cache, generation_config=gen)

# 입력 토큰 이후(=생성분)만 디코드
answer = processor.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(answer)
```

이미지 입력 형식:
- 로컬 경로 / URL: `{"type": "image", "url": "..."}`
- PIL.Image 객체: `{"type": "image", "image": pil_img}`

---

## 2. 스트리밍 (토큰 단위 출력)

`TextIteratorStreamer` + 별도 스레드로 생성되는 대로 흘려보낸다.

```python
from threading import Thread
from transformers import TextIteratorStreamer

streamer = TextIteratorStreamer(processor.tokenizer, skip_special_tokens=True, skip_prompt=True)
kwargs = dict(**inputs, streamer=streamer, past_key_values=cache, generation_config=gen)
Thread(target=model.generate, kwargs=kwargs, daemon=True).start()
for token in streamer:
    print(token, end="", flush=True)
```

---

## 3. 멀티턴 대화

같은 `cache`를 유지하고 `history`에 user/assistant를 누적한다.
첫 턴에 이미지를 주면, 이후 **텍스트만으로도** 그 이미지를 참조해 대화할 수 있다 (검증됨).

```python
history = []
def ask(prompt, image=None):
    content = ([{"type": "image", "url": image}] if image else []) + [{"type": "text", "text": prompt}]
    history.append({"role": "user", "content": content})
    inputs = processor.apply_chat_template(history, tokenize=True, add_generation_prompt=True,
                                           padding=True, return_tensors="pt", return_dict=True).to("cpu")
    out = model.generate(**inputs, past_key_values=cache, generation_config=gen)
    ans = processor.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    history.append({"role": "assistant", "content": [{"type": "text", "text": ans}]})
    return ans

ask("What animal is in this image?", image="images/cat1.jpg")   # → cat
ask("What color is it?")                                        # 텍스트만, 앞 이미지 참조
```

> **새 대화를 시작할 땐 `cache = MobilintCache(model.get_cache_mxq_model())`로 캐시를 새로 만든다.**
> 이전 대화 캐시를 그대로 쓰면 문맥이 섞인다.

---

## 4. 알아둘 것 / 트러블슈팅

| 항목 | 내용 |
|---|---|
| 모델 선택 | `Qwen3-VL-2B`(가벼움) / `-4B-` / `-8B-`(품질↑, 메모리·지연↑). model_id만 교체 |
| 캐시 리셋 | 새 대화마다 새 `MobilintCache`. 안 하면 문맥 오염 |
| device | 입력 텐서는 `.to("cpu")`. NPU로 직접 올리지 않음 (모델 내부가 디스패치) |
| 샘플링 | 기본 `do_sample=True, temperature=0.7, top_p=0.8` → 실행마다 답이 조금씩 달라짐. 재현하려면 `GenerationConfig(do_sample=False, ...)` |
| 모델 교체 | 다른 모델 로드 전 `model.dispose()`로 NPU 자원 해제 |
| 로드 실패 | `mobilint-cli status`로 `/dev/aries0` 인식 확인 (드라이버/런타임은 `.claude/skills/npu-setup`) |

## 사용 가능한 NPU VLM (mblt-model-zoo 실측)

- `mobilint/Qwen3-VL-2B-Instruct` ← 이 튜토리얼 기본
- `mobilint/Qwen3-VL-4B-Instruct`
- `mobilint/Qwen3-VL-8B-Instruct`
- (참고) `mobilint/Qwen2-VL-2B-Instruct` — Qwen2 계열은 **첫 입력에 이미지 1장 필수**. Qwen3-VL은 텍스트만도 가능.

---

## 참고 — 이 패턴의 출처

`mobilint-runtime-gui`(NPU 웹 데모, `mobilint-runtime-gui` 명령으로 `0.0.0.0:5000` 기동)의 백엔드:

- `/usr/lib/mobilint/mobilint-runtime-gui/src/model_manager.py` — 모델/프로세서/캐시 로딩 (`select()`의 `category == "vlm"`)
- `/usr/lib/mobilint/mobilint-runtime-gui/src/vlm_runner.py` — 메시지 구성 + `generate` 스트리밍

이 튜토리얼은 그중 VLM 추론 코어만 떼어내 GUI(Flask/SocketIO) 의존 없이 재구성한 것이다.
