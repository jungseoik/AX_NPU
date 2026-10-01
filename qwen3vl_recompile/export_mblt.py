#!/usr/bin/env python3
"""① 체크포인트 → vision.mblt + decoder.mblt  (Mobilint 중간표현 export)

벤더 재현 패키지는 **이미 export 된 .mblt 에서 시작**한다. 신규 학습본은 그게 없으므로
여기서 직접 만든다. 호출 방법은 qbcompiler 1.3.4+pia.capacity3 의 Qwen3-VL 패처
(`model_dict_new/parser/patcher/models/hf_models/qwen3vl.py`, 781줄)를 읽어 알아냈다.

★ 반드시 지켜야 하는 6가지 (전부 시행착오로 확인):
  1) 전체 `Qwen3VLForConditionalGeneration` 을 통째로 넣으면 **FX 추적이 실패**한다
     (transformers 의 use_cache/past_key_values 처리가 fx proxy 와 충돌).
     → 패처가 제공하는 `VisionModelForQwen3VL` / `Projection` 서브모듈을 각각 넣는다.
  2) `mblt_compile()` 은 `hf_config` 의 loader 를 **타지 않는다**(그건 mxq_compile 전용).
     → 모델 객체를 직접 만들어 넘긴다.
  3) RoPE 를 그래프 안에서 계산하면 in-place setitem 때문에 추적이 깨진다.
     → `CachedQwen3VLTextRotaryEmbedding` + `set_rope()` 로 **호스트 선계산**.
  4) KV 캐시는 **출력에서만** 뺀다. 캐시 객체가 그래프 출력에 섞이면 직렬화가 안 되지만,
     `use_cache=False` 로 꺼버리면 그래프에 HF 캐시 업데이트 연산이 아예 안 생기고,
     그러면 컴파일러가 그걸 접어 만드는 `StatefulKVCacheWrapper`(계층당 2개 = 56개)가
     통째로 빠진다(벤더 799 연산자 vs 캐시 끈 것 769). prefill 산출물은 KV 캐시를
     채워야 하므로 빠지면 메모리 플래닝에서 죽는다.
     → `use_cache=True` + `past_key_values=DynamicCache()` 를 **feed_dict 로** 넘기고
       `logits_to_keep=1` 로 **마지막 토큰 logits 만** 반환하는 래퍼.
  6) 그 래퍼는 `Projection` 을 **상속**해야 한다. nn.Module 로 감싸면 모듈 경로가
     한 단계 깊어져 상수 이름에 `p_` 접두가 붙는다.
  5) deepstack 은 **전체 시퀀스 길이(T)** 로 주고 `_deepstack_process` 를 덧셈으로 교체한다
     (HF 원본은 압축형(N_visual)을 masked-scatter 하는데 추적이 안 된다).

검증: 스톡 Qwen/Qwen3-VL-2B-Instruct(rev 89644892) 로 돌려 벤더 산출물과 대조 —
  vision.mblt  1543.7MB vs 1543.7MB  (+0.00%)
  decoder.mblt 6579.9MB vs 6579.8MB  (+0.00%)
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import torch, torch.nn as nn
from transformers.cache_utils import DynamicCache


def build(ckpt: str, image: str, prompt: str, size: int):
    import qbcompiler  # noqa: F401  (패처 등록을 위해 먼저 import)
    from qbcompiler.model_dict_new.parser.patcher.models.hf_models import qwen3vl as P
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from PIL import Image

    P.ensure_qwen3vl_classes_loaded(); P.get_patcher_cfg()
    proc = AutoProcessor.from_pretrained(ckpt)
    img = Image.open(image).convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    tpl = proc.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}],
        tokenize=False, add_generation_prompt=True)
    x = proc(text=[tpl], images=[img], return_tensors="pt")
    return P, proc, x, Qwen3VLForConditionalGeneration


def export(ckpt: str, out: Path, image: str, prompt: str, size: int) -> dict:
    import qbcompiler
    P, proc, x, Loader = build(ckpt, image, prompt, size)
    out.mkdir(parents=True, exist_ok=True)
    ids, grid = x["input_ids"], x["image_grid_thw"]
    T = ids.shape[1]
    report = {}

    # ---------- vision ----------
    m = Loader.from_pretrained(ckpt, dtype=torch.float32).eval()
    cfg = m.config
    class VisionIO(P.VisionModelForQwen3VL):
        """I/O **순서**를 벤더 산출물과 맞춘다 (상속이라 모듈 경로는 그대로).

        런타임(`mblt_model_zoo` 의 `_prepare_dynamic_npu_inputs`)은 입력을
        **위치로** `[cos, pos_embeds, folded]` 리스트로 넘긴다. 패처 원본
        `forward(images, pos_embeds, cos, sin)` 순서로 내보내면 mxq 입력이
        `[float_1_channel_last, pos_embeds/reshape, cos]` 가 되어 정반대라
        `ValueError: Input shape is invalid.` 로 거부된다.
        출력도 벤더는 `[ds0, ds1, ds2, merger]` 라 같은 순서로 돌려준다.
        """
        def forward(self, cos=None, sin=None, pos_embeds=None, images=None):
            hidden, ds = super().forward(images=images, pos_embeds=pos_embeds,
                                         cos=cos, sin=sin)
            return (*ds, hidden)

    vis = VisionIO(m.model).eval()
    pos_embeds, cos, sin = vis.compute_side_inputs(grid)   # 크기 의존 상수를 그래프 입력으로
    fd_v = {"cos": cos, "sin": sin, "pos_embeds": pos_embeds,
            "images": P.fold_pixel_values(x["pixel_values"].to(torch.float32))}
    # ★★ dynamic_axes 가 없으면 그래프의 시퀀스 축이 **고정**으로 굳어, 이후 캘리브레이션에서
    #    어떤 shape 을 줘도 `quantizeFB failed. Input shapes ... not same` 으로 거부된다.
    #    벤더 그래프는 시퀀스 축이 dynamic=True 다(md.to_dict() 로 확인). 형식은 {이름: [축인덱스]}.
    DYN_V = {"images": [3], "pos_embeds": [0], "cos": [2], "sin": [2]}
    t = time.perf_counter()
    qbcompiler.mblt_compile(model=vis, backend="torch", target_device="aries-rb",
                            device="gpu", feed_dict=fd_v, dynamic_axes=DYN_V,
                            mblt_save_path=str(out / "vision.mblt"))
    report["vision"] = {"sec": round(time.perf_counter() - t, 1),
                        "bytes": (out / "vision.mblt").stat().st_size,
                        "feed": {k: list(v.shape) for k, v in fd_v.items()},
                        "dynamic_axes": DYN_V}
    del m, vis

    # ---------- decoder ----------
    m = Loader.from_pretrained(ckpt, dtype=torch.float32).eval()
    H = cfg.text_config.hidden_size
    L = len(cfg.vision_config.deepstack_visual_indexes)
    lm = m.model.language_model
    pos = torch.arange(T).view(1, 1, T).repeat(3, 1, 1)
    cached = P.CachedQwen3VLTextRotaryEmbedding(lm.rotary_emb)
    cached.set_rope(pos)                       # ★ RoPE 호스트 선계산
    lm.rotary_emb = cached
    for nm in ("_deepstack_process", "deepstack_process"):   # ★ masked-scatter → 덧셈
        if hasattr(type(lm), nm):
            setattr(type(lm), nm, staticmethod(
                lambda hidden_states, visual_pos_masks, visual_embeds, *a, **k:
                    hidden_states + visual_embeds))

    class LogitsOnly(P.Projection):             # ★ KV 캐시는 **출력에서만** 뺀다
        """**Projection 을 상속**한다 — 감싸면 안 된다.

        nn.Module 로 감싸(`self.p = projection`) 내보내면 모듈 경로가 한 단계
        깊어져 연산자 이름이 벤더와 달라진다:
            벤더  language_model_layers_0_self_attn_k_proj
            감쌈  inputs_embeds/rmsnorm_0/reshape_1/conv2d   (+ 상수는 p_... 접두)
        상속하면 language_model/lm_head 가 루트에 그대로 남아 상수 이름의 `p_` 접두가 사라진다.

        ★ **`use_cache=False` 로 내보내면 안 된다.** 캐시를 끄면 그래프에
          HF 캐시 업데이트 연산이 안 생기고, 그러면 컴파일러가 그걸 접어 만드는
          `StatefulKVCacheWrapper`(계층당 2개 = 56개)가 통째로 빠진다
          (벤더 799 연산자 vs 캐시 끈 우리 769). prefill 산출물은 KV 캐시를
          채워야 하고 메모리 플래너가 그걸 할당하므로, 빠지면 한참 뒤
          `could not find input info in memory: ...` 로 죽는다.
          출력에서 캐시 객체만 빼고(`.logits`) **동작은 켠 채로** 둔다.
        """
        def forward(self, inputs_embeds=None, deepstack_visual_embeds=None,
                    visual_pos_masks=None, position_ids=None, attention_mask=None,
                    past_key_values=None):
            return super().forward(
                inputs_embeds=inputs_embeds,
                deepstack_visual_embeds=deepstack_visual_embeds,
                visual_pos_masks=visual_pos_masks, position_ids=position_ids,
                attention_mask=attention_mask, past_key_values=past_key_values,
                # ★ prefill 산출물은 **마지막 토큰 logits 하나만** 내야 한다.
                #   기본값 0 이면 Projection 이 slice(0, None) 로 전 위치를 내보내
                #   출력이 (1,1,T,151936) 가 되고, 런타임이
                #   `RuntimeError: Unexpected final logit rows: (84, 151936)` 로 거부한다.
                #   벤더 그래프 출력은 `lm_head` [1,1,151936] 다.
                logits_to_keep=1,
                use_cache=True).logits

    fd_d = {"inputs_embeds": m.model.get_input_embeddings()(ids).to(torch.float32),
            "deepstack_visual_embeds": torch.zeros(L, T, H, dtype=torch.float32),
            "visual_pos_masks": (ids == cfg.image_token_id),
            "position_ids": pos, "attention_mask": x["attention_mask"],
            # ★ 캐시는 **feed_dict 로 넘겨야** 한다. 그래야 추적기가 그 자리에
            #   HFProxyableDynamicCache 플레이스홀더를 만들어 `.update()` 를 프록시한다.
            #   그냥 use_cache=True 만 켜면 HF 가 내부에서 만든 생 DynamicCache 가
            #   인자로 흘러들어 `NotImplementedError: argument of type: ...DynamicCache`.
            "past_key_values": DynamicCache()}
    # decoder 의 **실제 그래프 입력은 2개뿐**이다. visual_pos_masks / position_ids /
    # attention_mask 는 추적 중 상수로 접혀 입력에 남지 않는다. 없는 이름을 주면
    #   ValueError: Unknown input `visual_pos_masks` in dynamic_axes
    # 로 즉시 거부되므로, 실제 입력만 대상으로 한다(시퀀스 축 = 1).
    DYN_D = {"inputs_embeds": [1], "deepstack_visual_embeds": [1]}
    t = time.perf_counter()
    dec = LogitsOnly(lm, m.lm_head).eval()      # 상속형이라 inner 를 다시 안 감싼다
    qbcompiler.mblt_compile(model=dec, backend="torch",
                            target_device="aries-rb", device="gpu", feed_dict=fd_d,
                            dynamic_axes=DYN_D,
                            mblt_save_path=str(out / "decoder.mblt"))
    report["decoder"] = {"sec": round(time.perf_counter() - t, 1),
                         "bytes": (out / "decoder.mblt").stat().st_size,
                         "feed": {k: list(v.shape) for k, v in fd_d.items() if hasattr(v, "shape")},
                         "dynamic_axes": DYN_D}
    report["meta"] = {"checkpoint": ckpt, "size": size, "seq_len": T,
                      "hidden": H, "deepstack_layers": L,
                      "qbcompiler": qbcompiler.__version__, "torch": torch.__version__}
    (out / "export-info.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="HF 체크포인트 경로 또는 repo id")
    p.add_argument("--out", type=Path, required=True, help="mblt 저장 디렉토리")
    p.add_argument("--image", required=True, help="추적용 예시 이미지 1장")
    p.add_argument("--prompt", default="Is this a dog? Answer yes or no.")
    p.add_argument("--size", type=int, default=224, help="처리 해상도(정사각, 32 배수)")
    a = p.parse_args()
    if a.size % 32:
        p.error("size 는 32 의 배수여야 한다 (patch 16 x merge 2)")
    r = export(a.checkpoint, a.out, a.image, a.prompt, a.size)
    for k in ("vision", "decoder"):
        print(f"  {k:8s} {r[k]['bytes']/2**20:9.1f} MB  {r[k]['sec']:6.1f}s")


if __name__ == "__main__":
    main()
