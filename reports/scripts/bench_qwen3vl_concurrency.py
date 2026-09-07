"""카드 1장 동시성 스윕 — 스레드 수별 총 처리시간/처리량 (global8, B=1 반복)."""
import argparse, json, statistics, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import torch
from transformers import AutoConfig, AutoProcessor, AutoModelForImageTextToText
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--device", type=int, default=7)
ap.add_argument("--threads", default="1,2,4,8")
ap.add_argument("--n", type=int, default=16, help="총 처리 이미지 수")
ap.add_argument("--repeat", type=int, default=3)
ap.add_argument("--out", default="/tmp/w4a16/vlm_conc.json")
a = ap.parse_args()

M = "mobilint/Qwen3-VL-2B-Instruct"
cfg = AutoConfig.from_pretrained(M, trust_remote_code=True)
cfg.vision_config.dev_no = a.device; cfg.text_config.dev_no = a.device
model = AutoModelForImageTextToText.from_pretrained(M, trust_remote_code=True, config=cfg)
proc = AutoProcessor.from_pretrained(M, trust_remote_code=True)
print(f"[cfg] device={a.device} vision={cfg.vision_config.core_mode} n={a.n}", flush=True)

IMG = Path("/tmp/w4a16/vlm_imgs")
def msg(im): return [{"role":"user","content":[{"type":"image","image":im},{"type":"text","text":"Describe this image."}]}]
def one(im):
    inp = proc.apply_chat_template(msg(im), add_generation_prompt=True, tokenize=True,
                                   return_dict=True, return_tensors="pt")
    with torch.no_grad(): model.generate(**inp, max_new_tokens=1, do_sample=False)

results = []
for res in ("720p", "1080p"):
    imgs = [Image.open(IMG/f"{res}_{i%4}.jpg").convert("RGB") for i in range(a.n)]
    for th in [int(x) for x in a.threads.split(",")]:
        ts = []
        for r in range(a.repeat + 1):
            t = time.monotonic()
            if th == 1:
                for im in imgs: one(im)
            else:
                with ThreadPoolExecutor(th) as ex: list(ex.map(one, imgs))
            if r: ts.append((time.monotonic() - t) * 1000)
        med = statistics.median(ts)
        rec = dict(res=res, threads=th, n=a.n, total_ms=round(med,1),
                   per_img_ms=round(med/a.n,1), img_s=round(a.n/(med/1000),2))
        results.append(rec)
        print(f"[done] {res:6s} threads={th:2d}  총 {rec['total_ms']:8.1f}ms  "
              f"장당 {rec['per_img_ms']:6.1f}ms  {rec['img_s']:5.2f} img/s", flush=True)

Path(a.out).write_text(json.dumps({"results":results}, ensure_ascii=False, indent=1))
print("\n=== CONC DONE ===")
