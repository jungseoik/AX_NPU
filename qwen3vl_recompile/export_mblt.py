#!/usr/bin/env python3
"""① 체크포인트 → vision.mblt + decoder.mblt  (Mobilint 중간표현 export)

벤더 재현 패키지는 **이미 export 된 .mblt 에서 시작**한다. 신규 학습본은 그게 없으므로
여기서 직접 만든다. 호출 방법은 qbcompiler 1.3.4+pia.capacity3 의 Qwen3-VL 패처
(`model_dict_new/parser/patcher/models/hf_models/qwen3vl.py`, 781줄)를 읽어 알아냈다.

★ 반드시 지켜야 하는 4가지 (전부 시행착오로 확인):
  1) 전체 `Qwen3VLForConditionalGeneration` 을 통째로 넣으면 **FX 추적이 실패**한다
     (transformers 의 use_cache/past_key_values 처리가 fx proxy 와 충돌).
     → 패처가 제공하는 `VisionModelForQwen3VL` / `Projection` 서브모듈을 각각 넣는다.
  2) `mblt_compile()` 은 `hf_config` 의 loader 를 **타지 않는다**(그건 mxq_compile 전용).
     → 모델 객체를 직접 만들어 넘긴다.
  3) RoPE 를 그래프 안에서 계산하면 in-place setitem 때문에 추적이 깨진다.
     → `CachedQwen3VLTextRotaryEmbedding` + `set_rope()` 로 **호스트 선계산**.
  4) KV 캐시가 그래프 출력에 섞이면 직렬화가 안 된다.
     → `use_cache=False` + logits 만 반환하는 래퍼.
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
    vis = P.VisionModelForQwen3VL(m.model).eval()
    pos_embeds, cos, sin = vis.compute_side_inputs(grid)   # 크기 의존 상수를 그래프 입력으로
    fd_v = {"images": P.fold_pixel_values(x["pixel_values"].to(torch.float32)),
            "pos_embeds": pos_embeds, "cos": cos, "sin": sin}
    t = time.perf_counter()
    qbcompiler.mblt_compile(model=vis, backend="torch", target_device="aries-rb",
                            device="gpu", feed_dict=fd_v,
                            mblt_save_path=str(out / "vision.mblt"))
    report["vision"] = {"sec": round(time.perf_counter() - t, 1),
                        "bytes": (out / "vision.mblt").stat().st_size,
                        "feed": {k: list(v.shape) for k, v in fd_v.items()}}
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
    inner = P.Projection(lm, m.lm_head).eval()

    class LogitsOnly(nn.Module):                # ★ KV 캐시를 출력에서 제거
        def __init__(self, p): super().__init__(); self.p = p
        def forward(self, inputs_embeds=None, deepstack_visual_embeds=None,
                    visual_pos_masks=None, position_ids=None, attention_mask=None):
            return self.p(inputs_embeds=inputs_embeds,
                          deepstack_visual_embeds=deepstack_visual_embeds,
                          visual_pos_masks=visual_pos_masks, position_ids=position_ids,
                          attention_mask=attention_mask, use_cache=False).logits

    fd_d = {"inputs_embeds": m.model.get_input_embeddings()(ids).to(torch.float32),
            "deepstack_visual_embeds": torch.zeros(L, T, H, dtype=torch.float32),
            "visual_pos_masks": (ids == cfg.image_token_id),
            "position_ids": pos, "attention_mask": x["attention_mask"]}
    t = time.perf_counter()
    qbcompiler.mblt_compile(model=LogitsOnly(inner).eval(), backend="torch",
                            target_device="aries-rb", device="gpu", feed_dict=fd_d,
                            mblt_save_path=str(out / "decoder.mblt"))
    report["decoder"] = {"sec": round(time.perf_counter() - t, 1),
                         "bytes": (out / "decoder.mblt").stat().st_size,
                         "feed": {k: list(v.shape) for k, v in fd_d.items()}}
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
