"""vllm-mblt 서버 동시요청 부하 테스트.
720p 랜덤 단색 이미지 + max_tokens=1 로 동시 N(1,2,4,6)요청 → 총지연 + NPU 메모리 peak + 실패지점.

실행: python loadtest_vllm.py [--url http://localhost:8000] [--model qwen3-vl] [--levels 1,2,4,6]
"""
import argparse, base64, io, time, threading, subprocess, statistics, re
from concurrent.futures import ThreadPoolExecutor
import numpy as np, httpx
from PIL import Image

COLORS = [(255,0,0),(255,255,0),(0,255,0),(0,0,255),(255,128,0),(128,0,255),(0,255,255),(255,0,255)]

def img_b64(color):  # 720p 단색 png → base64
    im = Image.fromarray(np.full((720,1280,3), color, np.uint8))
    b = io.BytesIO(); im.save(b, format="PNG"); return base64.b64encode(b.getvalue()).decode()

def payload(model, color):
    return {"model": model, "max_tokens": 1, "temperature": 0.0,
            "messages": [{"role":"user","content":[
                {"type":"image_url","image_url":{"url":f"data:image/png;base64,{img_b64(color)}"}},
                {"type":"text","text":"color?"}]}]}

def npu_mem_mb(dev=0):
    try:
        out = subprocess.run(["mobilint-cli","status"], capture_output=True, text=True, timeout=8).stdout
        for line in out.splitlines():
            if f"aries{dev})" in line:
                m = re.search(r"(\d+)\s*MB\s*/\s*\d+\s*MB", line)
                if m: return int(m.group(1))
    except Exception: pass
    return -1

class MemSampler(threading.Thread):
    def __init__(self, dev=0): super().__init__(daemon=True); self.dev=dev; self.stop=False; self.peak=0; self.base=npu_mem_mb(dev)
    def run(self):
        while not self.stop:
            v=npu_mem_mb(self.dev); self.peak=max(self.peak,v); time.sleep(0.5)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default="qwen3-vl")
    ap.add_argument("--levels", default="1,2,4,6")
    ap.add_argument("--repeat", type=int, default=3)
    args=ap.parse_args()
    levels=[int(x) for x in args.levels.split(",")]
    ep=f"{args.url}/v1/chat/completions"

    def one(color):
        t=time.perf_counter()
        try:
            r=httpx.post(ep, json=payload(args.model,color), timeout=180.0)
            return r.status_code, (time.perf_counter()-t)*1000, (r.json().get("usage") if r.status_code==200 else r.text[:80])
        except Exception as e:
            return -1, (time.perf_counter()-t)*1000, str(e)[:80]

    print("[warmup]"); one(COLORS[0])
    print(f"\n{'동시N':>5} | {'총지연(ms)':>10} | {'req평균(ms)':>11} | {'성공/전체':>9} | {'NPU peak(MB)':>12} | 비고")
    for n in levels:
        best=None
        for _ in range(args.repeat):
            sampler=MemSampler(); sampler.start()
            t0=time.perf_counter()
            with ThreadPoolExecutor(max_workers=n) as ex:
                res=list(ex.map(one, [COLORS[i%len(COLORS)] for i in range(n)]))
            total=(time.perf_counter()-t0)*1000
            sampler.stop=True; sampler.join(timeout=2)
            ok=sum(1 for s,_,_ in res if s==200)
            avg=statistics.mean([ms for _,ms,_ in res])
            fail=[info for s,_,info in res if s!=200]
            if best is None or total<best[0]: best=(total,avg,ok,sampler.peak,fail)
        total,avg,ok,peak,fail=best
        note="OK" if ok==n else f"실패:{fail[:1]}"
        print(f"{n:>5} | {total:>10.0f} | {avg:>11.0f} | {ok:>4}/{n:<4} | {peak:>12} | {note}")
    print(f"\n기준 NPU mem(유휴시 모델로드): ~{npu_mem_mb(0)}MB / 16384MB")

if __name__=="__main__":
    main()
