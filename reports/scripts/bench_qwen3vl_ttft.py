"""Qwen3-VL TTFT — 해상도(720p/1080p) × 배치 크기. 전처리(CPU)와 생성(NPU) 분리 측정."""
import argparse, json, os, statistics, sys, time
from pathlib import Path
sys.path.insert(0, "/home/gpuadmin/AX_NPU/tutorial/pe_npu")
import torch
from transformers import AutoConfig, AutoProcessor, AutoModelForImageTextToText
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--device", type=int, default=7)
ap.add_argument("--model", default="mobilint/Qwen3-VL-2B-Instruct")
ap.add_argument("--core-mode", default=None, help="미지정=config 기본(global8)")
ap.add_argument("--batches", default="1,2,4")
ap.add_argument("--repeat", type=int, default=5)
ap.add_argument("--out", default="/tmp/w4a16/vlm_ttft.json")
a = ap.parse_args()

cfg = AutoConfig.from_pretrained(a.model, trust_remote_code=True)
cfg.vision_config.dev_no = a.device
cfg.text_config.dev_no = a.device
if a.core_mode:
    cfg.vision_config.core_mode = a.core_mode
    cfg.text_config.core_mode = a.core_mode
vm = getattr(cfg.vision_config, "core_mode", "?"); tm = getattr(cfg.text_config, "core_mode", "?")
print(f"[cfg] device={a.device} vision={vm} text={tm}", flush=True)

t0 = time.monotonic()
model = AutoModelForImageTextToText.from_pretrained(a.model, trust_remote_code=True, config=cfg)
proc = AutoProcessor.from_pretrained(a.model, trust_remote_code=True)
print(f"[load] {time.monotonic()-t0:.1f}s", flush=True)

IMG = Path("/tmp/w4a16/vlm_imgs")
PROMPT = "Describe this image."
batches = [int(b) for b in a.batches.split(",")]
results = []

def run(res, bs):
    imgs = [Image.open(IMG / f"{res}_{i%4}.jpg").convert("RGB") for i in range(bs)]
    pre, ttft = [], []
    for r in range(a.repeat + 1):
        t = time.monotonic()
        msgs = [[{"role":"user","content":[{"type":"image","image":im},{"type":"text","text":PROMPT}]}] for im in imgs]
        inputs = proc.apply_chat_template(msgs if bs > 1 else msgs[0],
                                          add_generation_prompt=True, tokenize=True,
                                          return_dict=True, return_tensors="pt")
        p = (time.monotonic() - t) * 1000
        t = time.monotonic()
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=1, do_sample=False)
        g = (time.monotonic() - t) * 1000
        if r:                       # 첫 회는 워밍업
            pre.append(p); ttft.append(g)
    rec = dict(res=res, batch=bs, vision=vm, text=tm,
               preprocess_ms=round(statistics.median(pre), 1),
               ttft_ms=round(statistics.median(ttft), 1),
               ttft_min=round(min(ttft), 1), ttft_max=round(max(ttft), 1),
               total_ms=round(statistics.median(pre) + statistics.median(ttft), 1))
    rec["per_img_ms"] = round(rec["ttft_ms"] / bs, 1)
    results.append(rec)
    print(f"[done] {res:6s} B={bs:2d}  전처리 {rec['preprocess_ms']:7.1f}ms  "
          f"TTFT {rec['ttft_ms']:8.1f}ms (min {rec['ttft_min']:.1f})  장당 {rec['per_img_ms']:7.1f}ms", flush=True)

for res in ("720p", "1080p"):
    for bs in batches:
        try:
            run(res, bs)
        except Exception as e:
            print(f"[FAIL] {res} B={bs}: {type(e).__name__}: {str(e)[:160]}", flush=True)
            results.append(dict(res=res, batch=bs, error=f"{type(e).__name__}: {str(e)[:200]}"))

Path(a.out).write_text(json.dumps({"vision":vm,"text":tm,"results":results}, ensure_ascii=False, indent=1))
print(f"\n[save] {a.out}")
print("=== TTFT DONE ===")
