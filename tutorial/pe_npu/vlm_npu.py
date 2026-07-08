"""
Qwen3-VL NPU 추론 헬퍼 (ARIES MLA100).

이미지+프롬프트 -> 텍스트. NPU 가속은 모델 레포의 trust_remote_code(mblt_model_zoo)가
처리하므로, 이 파일은 표준 HuggingFace API만 호출한다(= "포팅" 없음).

코어모드: mobilint/Qwen3-VL-* MXQ는 vision/text 모두 **global8**(8코어 전부, 2클러스터)로
config.json에 박혀 있다. 단일 스트림 latency 최적화. max_batch_size=1.

필요 패키지(검증):  mblt-model-zoo==1.3.1, transformers>=4.57(<=4.57.6), torch(cpu), qbruntime 1.2.0
    pip install "mblt-model-zoo[transformers]==1.3.1" "transformers==4.57.1"

사용:
    from vlm_npu import load_vlm, ask, ask_stream, VLMChat, VLMPool

    # 단일 카드 (device_id로 카드 지정 가능)
    model, processor = load_vlm("mobilint/Qwen3-VL-2B-Instruct")           # 또는 device_id=1
    print(ask(model, processor, "images/cat1.jpg", "Describe this image."))

    chat = VLMChat(model, processor)
    print(chat.ask("What animal is this?", image="images/cat1.jpg"))
    print(chat.ask("What color is it?"))            # 텍스트만 — 앞 이미지 참조

    # 멀티카드 — 동시요청을 장착 NPU 전부에 분산 (요청 병렬)
    pool = VLMPool("mobilint/Qwen3-VL-2B-Instruct", device_ids="auto")
    pool.ask("images/cat1.jpg", "Is there a cat? yes/no", max_new_tokens=1)
    pool.ask_batch([(img1, q1), (img2, q2), ...])   # 순서 보존, 전부 응답까지 블록
"""
from __future__ import annotations

import glob
import os
from concurrent.futures import ThreadPoolExecutor
from itertools import count
from threading import Lock, Thread
from typing import Iterator, Optional

from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoProcessor,
    GenerationConfig,
    TextIteratorStreamer,
)
from mblt_model_zoo.hf_transformers.utils.cache_utils import MobilintCache

# 지원 모델 (global8, 단일 ARIES 카드)
SUPPORTED = (
    "mobilint/Qwen3-VL-2B-Instruct",
    "mobilint/Qwen3-VL-4B-Instruct",
    "mobilint/Qwen3-VL-8B-Instruct",
)

# GUI 기본값과 동일
DEFAULT_GEN = dict(max_new_tokens=512, repetition_penalty=1.1, chunk_size=128, use_cache=True)


def detect_npu_devices() -> list[int]:
    """장착된 NPU device id(`/dev/ariesN`) 목록. 멀티카드 자동 감지용."""
    ids = []
    for p in glob.glob("/dev/aries*"):
        s = os.path.basename(p)[len("aries"):]
        if s.isdigit():
            ids.append(int(s))
    return sorted(ids)


def load_vlm(model_id: str = "mobilint/Qwen3-VL-2B-Instruct", device_id: int = None):
    """모델 + 프로세서 로드 (MXQ를 NPU에 launch). 최초 1회 HF 다운로드.

    device_id: 특정 카드(aries<device_id>)에 올리려면 지정. None이면 config 기본(보통 0).
               (dev_no는 from_pretrained 인자가 아니라 config 필드라 config로 넣는다.)
    """
    kw = {}
    if device_id is not None:
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        cfg.vision_config.dev_no = device_id
        cfg.text_config.dev_no = device_id
        kw["config"] = cfg
    model = AutoModelForImageTextToText.from_pretrained(model_id, trust_remote_code=True, **kw)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor


def _new_cache(model):
    """대화 1건당 1개. 새 대화를 시작할 때마다 새로 만들어 KV-cache를 리셋한다."""
    return MobilintCache(model.get_cache_mxq_model())


def _image_part(image):
    # 로컬 경로/URL은 url=, PIL.Image 등 객체는 image=
    return {"type": "image", "url": image} if isinstance(image, str) else {"type": "image", "image": image}


def _build_inputs(processor, messages):
    return processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        padding=True, return_tensors="pt", return_dict=True,
    ).to("cpu")  # NPU 아니라 cpu. NPU 디스패치는 모델 내부가 처리.


def ask(model, processor, image, prompt: str, **gen_kwargs) -> str:
    """이미지(경로/URL/PIL) + 프롬프트 -> 답변 문자열. 단발성(대화 이어가지 않음)."""
    messages = [{"role": "user", "content": [_image_part(image), {"type": "text", "text": prompt}]}]
    inputs = _build_inputs(processor, messages)
    gen = GenerationConfig(**{**DEFAULT_GEN, **gen_kwargs})
    out = model.generate(**inputs, past_key_values=_new_cache(model), generation_config=gen)
    return processor.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def ask_stream(model, processor, image, prompt: str, **gen_kwargs) -> Iterator[str]:
    """ask()의 스트리밍 버전. 토큰을 yield 한다."""
    messages = [{"role": "user", "content": [_image_part(image), {"type": "text", "text": prompt}]}]
    inputs = _build_inputs(processor, messages)
    streamer = TextIteratorStreamer(processor.tokenizer, skip_special_tokens=True, skip_prompt=True)
    gen = GenerationConfig(**{**DEFAULT_GEN, **gen_kwargs})
    kwargs = dict(**inputs, streamer=streamer, past_key_values=_new_cache(model), generation_config=gen)
    Thread(target=model.generate, kwargs=kwargs, daemon=True).start()
    yield from streamer


class VLMChat:
    """멀티턴 대화. 같은 KV-cache + history 누적. 첫 턴 이미지를 이후 텍스트턴이 참조한다."""

    def __init__(self, model, processor, **gen_kwargs):
        self.model = model
        self.processor = processor
        self.cache = _new_cache(model)
        self.history: list[dict] = []
        self.gen = GenerationConfig(**{**DEFAULT_GEN, **gen_kwargs})

    def reset(self):
        """대화를 새로 시작 (캐시/히스토리 리셋)."""
        self.cache = _new_cache(self.model)
        self.history.clear()

    def ask(self, prompt: str, image=None) -> str:
        content = ([_image_part(image)] if image is not None else []) + [{"type": "text", "text": prompt}]
        self.history.append({"role": "user", "content": content})
        inputs = _build_inputs(self.processor, self.history)
        out = self.model.generate(**inputs, past_key_values=self.cache, generation_config=self.gen)
        ans = self.processor.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        self.history.append({"role": "assistant", "content": [{"type": "text", "text": ans}]})
        return ans


class VLMPool:
    """멀티카드 Qwen3-VL — 카드별 인스턴스 + 요청 라운드로빈 분산 (PE/YOLO 멀티카드와 동일 방침).

    Qwen3-VL은 global8 단일모드(max_batch_size=1)라 한 인스턴스는 in-flight 1개만 처리.
    그래서 "배치"가 아니라 **동시요청을 카드에 나눠** 처리량을 올린다(카드당 1건씩 순차).

        pool = VLMPool("mobilint/Qwen3-VL-2B-Instruct", device_ids="auto")   # 장착 NPU 전부
        pool.ask("img.jpg", "Is there a bus? yes/no", max_new_tokens=1)       # 단건(카드 자동 배정)
        pool.ask_batch([(img1, q1), (img2, q2), ...])                         # 동시요청 분산, 순서 보존

    실측(2B, VQA 1토큰): 단건 ~180ms, 64동시 1장 12s → 7장 2.2s(5.5x).
    (reports/performance/NPU_qwen3vl_multicard_batch.md)
    """

    def __init__(self, model_id: str = "mobilint/Qwen3-VL-2B-Instruct",
                 device_ids="auto", **gen_kwargs):
        if device_ids == "auto":
            device_ids = detect_npu_devices()
            if not device_ids:
                raise RuntimeError("NPU(/dev/aries*)를 찾지 못했습니다.")
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.models, self.locks = [], []
        for d in device_ids:
            cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            cfg.vision_config.dev_no = d
            cfg.text_config.dev_no = d
            self.models.append(
                AutoModelForImageTextToText.from_pretrained(model_id, config=cfg, trust_remote_code=True))
            self.locks.append(Lock())   # 인스턴스당 in-flight 1 강제
        self.device_ids = list(device_ids)
        self.n = len(self.models)
        self.gen = GenerationConfig(**DEFAULT_GEN, **gen_kwargs) if gen_kwargs else GenerationConfig(**DEFAULT_GEN)
        self._rr = count()

    def __len__(self):
        return self.n

    def _ask_on(self, idx: int, image, prompt: str, gen) -> str:
        m = self.models[idx]
        with self.locks[idx]:                       # 같은 인스턴스 동시진입 방지(출력 깨짐 방지)
            messages = [{"role": "user", "content": [_image_part(image), {"type": "text", "text": prompt}]}]
            inputs = _build_inputs(self.processor, messages)
            out = m.generate(**inputs, past_key_values=_new_cache(m), generation_config=gen)
            return self.processor.tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def ask(self, image, prompt: str, **gen_kwargs) -> str:
        """단건 요청 → 카드 하나에 라운드로빈 배정."""
        gen = GenerationConfig(**{**DEFAULT_GEN, **gen_kwargs}) if gen_kwargs else self.gen
        return self._ask_on(next(self._rr) % self.n, image, prompt, gen)

    def ask_batch(self, requests, **gen_kwargs) -> list:
        """동시요청 [(image, prompt), ...] → 카드에 라운드로빈 분산, 순서 보존 답변 리스트.
        전부 응답받을 때까지 블록. (요청 j → 카드 j%n, 카드당 순차)"""
        gen = GenerationConfig(**{**DEFAULT_GEN, **gen_kwargs}) if gen_kwargs else self.gen
        results = [None] * len(requests)

        def job(j):
            img, pr = requests[j]
            results[j] = self._ask_on(j % self.n, img, pr, gen)

        with ThreadPoolExecutor(max_workers=self.n) as ex:
            list(ex.map(job, range(len(requests))))
        return results

    def dispose(self):
        for m in self.models:
            try:
                m.dispose()
            except Exception:
                pass


if __name__ == "__main__":
    import sys
    mid = sys.argv[1] if len(sys.argv) > 1 else "mobilint/Qwen3-VL-2B-Instruct"
    img = sys.argv[2] if len(sys.argv) > 2 else "images/cat1.jpg"
    prompt = sys.argv[3] if len(sys.argv) > 3 else "Describe this image in one sentence."
    m, p = load_vlm(mid)
    print(ask(m, p, img, prompt, max_new_tokens=64))
