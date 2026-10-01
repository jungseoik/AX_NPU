#!/usr/bin/env python3
"""우리가 재컴파일한 산출물과 벤더 산출물의 **I/O 순서·이름 차이**를 흡수한다.

벤더 mxq 와 우리 mxq 는 같은 텐서를 쓰지만 순서와 이름이 다르다:

    입력   벤더   ['cos', 'pos_embeds/reshape', 'float_1_channel_last']
           우리   ['float_1_channel_last', 'pos_embeds/reshape', 'cos']
    출력   벤더   ['model_deepstack_merger_list_{0,1,2}_...', 'model_merger_...']
           우리   ['model_merger_...', 'add/reshape_49/...', '_99/...', '_149/...']

입력 순서는 **데이터플로우 순서**라 export 쪽 시그니처나 feed_dict 순서로는 못 바꾼다
(둘 다 바꿔 봤지만 그대로였다). 그런데 `mblt_model_zoo` 의
`_prepare_dynamic_npu_inputs()` 는 `[cos, pos, folded]` 리스트를 **위치로** 넘긴다
→ 그대로 주면 `ValueError: Input shape is invalid.`

출력은 벤더 런타임이 이름 접두(`model_deepstack_merger_list_<i>_`)로 찾는데
우리 이름은 데이터플로우라 매칭이 실패한다
→ `ValueError: Cannot identify model_deepstack_merger_list_0_ in [...]`

그래서 **mxq 만 갈아끼우는 drop-in 은 성립하지 않는다.** 여기 두 함수를 로더에 끼우면 된다.
"""
from __future__ import annotations
import json
from pathlib import Path

# 비전 입력을 마지막 축 크기로 식별한다(이름 규칙보다 안전하다).
#   folded pixel = 1536 / pos_embeds = 1024 / cos|sin 병합 = 128
_BY_LAST_DIM = {1536: "float_1_channel_last", 1024: "pos_embeds/reshape", 128: "cos"}


def _names(mxq: str, key: str):
    import sys, os
    sys.path.insert(0, os.environ.get("NPU_PE_2STAGE_VENDOR_SRC", "/repro/src"))
    from pia_vlm.artifact import metadata
    return metadata(str(mxq), key)


def order_vision_inputs(mxq: str, prepared):
    """`_prepare_dynamic_npu_inputs()` 결과를 **그 mxq 가 선언한 입력 순서**로 재배열."""
    want = _names(mxq, "modelInputList")
    tagged = {}
    for t in prepared:
        last = int(t.shape[-1])
        nm = _BY_LAST_DIM.get(last)
        if nm is None:
            raise ValueError(f"비전 입력의 마지막 축 {last} 를 식별하지 못했다")
        tagged[nm] = t
    missing = [n for n in want if n not in tagged]
    if missing:
        raise ValueError(f"mxq 가 요구하는 입력 {missing} 를 만들지 못했다 (가진 것: {list(tagged)})")
    return [tagged[n] for n in want]


def resolve_vision_order(mxq: str, model_dir: str | None = None):
    """출력 순서를 정한다 — 벤더 이름 매칭 우선, 실패하면 `vision_output_order.json`.

    json 은 `verify_vision_order.py` 가 **PyTorch 기준과 코사인으로 대조해** 쓴 것이다.
    벤더도 자기 json 에 "validate numerically after recompilation" 이라고 적어 뒀다.
    """
    try:
        import sys, os
        sys.path.insert(0, os.environ.get("NPU_PE_2STAGE_VENDOR_SRC", "/repro/src"))
        from pia_vlm.artifact import vision_order
        return vision_order(str(mxq))
    except ValueError:
        pass
    if model_dir is None:
        model_dir = Path(mxq).parent
    f = Path(model_dir) / "vision_output_order.json"
    if not f.is_file():
        raise ValueError(
            f"{f} 가 없다. 이름 매칭이 실패한 산출물은 "
            f"`python3 verify_vision_order.py` 로 순서를 먼저 확정해야 한다")
    d = json.loads(f.read_text())
    order = d["output_order"]
    if sorted(order) != list(range(len(order))):
        raise ValueError(f"vision_output_order.json 의 순서가 잘못됐다: {order}")
    return order


_CURRENT = {"vp": None, "md": None}
_INSTALLED = False


def install(model_dir: str):
    """벤더 런타임에 위 두 가지를 끼워 넣는다.

    모델을 바꿔 가며 쓰므로(baseline → candidate) 현재 대상만 갈아끼운다.
    벤더 산출물에 대해서는 두 함수 모두 **원래 동작과 같다**
    (이름 매칭이 성공하고, 입력 순서도 이미 벤더 순서라 재배열이 항등이다).
    """
    global _INSTALLED
    import pia_vlm.runtime as R
    from pia_vlm.support import VisionInputs
    md = Path(model_dir)
    cfg = R.load_support(md)[0]
    _CURRENT["md"] = str(md)
    _CURRENT["vp"] = str(md / cfg.vision_config.mxq_path)

    if not _INSTALLED:
        R.vision_order = lambda p: resolve_vision_order(str(p), _CURRENT["md"])
        _patch_decoder_inputs(R)
        orig = VisionInputs._prepare_dynamic_npu_inputs
        VisionInputs._prepare_dynamic_npu_inputs = (
            lambda self, h, g: order_vision_inputs(_CURRENT["vp"], orig(self, h, g)))
        _INSTALLED = True
    return _CURRENT["vp"]


def _patch_decoder_inputs(R):
    """decoder rope 입력 이름 매칭을 완화한다.

    벤더 런타임은 `name.startswith("cos")` 로 rope 입력을 찾는데(벤더 이름이
    `cos/reshape_0`), 우리 export 는 모듈 경로가 붙어 `language_model_rotary_emb_cos_0`
    라 접두가 아니다 → `ValueError: Unknown decoder input: ...`.
    'cos' 를 **포함**하면 rope 로 본다. 벤더 이름도 그대로 걸리므로 동작은 같다.
    deepstack 쪽은 벤더 `deepstack_index()` 정규식이 우리 레이아웃
    (`deepstack_visual_embeds/reshape/slice[_01]`)을 이미 포함해 손댈 게 없다.
    """
    from pia_vlm.artifact import deepstack_index

    def _inputs(self, embeddings, deepstack, rope):
        def lookup(name):
            if name.startswith("inputs_embeds"):
                return embeddings
            if "deepstack" in name:
                if name == "deepstack_visual_embeds_0":
                    return deepstack
                split_names = [n for n in self.input_names if "deepstack" in n]
                if len(split_names) != 3:
                    raise ValueError(f"Unsupported DeepStack input layout: {split_names}")
                i = deepstack_index(name)
                return deepstack[i:i + 1]
            if "cos" in name:
                return rope
            raise ValueError(f"Unknown decoder input: {name}")
        return [lookup(name) for name in self.input_names]

    R.Classifier._inputs = _inputs
