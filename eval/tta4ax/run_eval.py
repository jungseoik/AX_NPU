#!/usr/bin/env python3
"""추출한 프레임을 두 서버에 보내고 정답과 대조한다."""
import base64, json, time, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests

SERVERS = {"bs32": "http://localhost:8110", "ft": "http://localhost:8111"}
PROMPTS = {c: Path(f"/tmp/ax_eval/user_{c}.txt").read_text().strip()
           for c in ("fire", "smoke", "falldown")}
INDEX = json.loads(Path("/tmp/ax_eval/index.json").read_text())
WORKERS = int(sys.argv[1]) if len(sys.argv) > 1 else 16


def ask(base, rec):
    b64 = base64.b64encode(Path(rec["path"]).read_bytes()).decode()
    body = {"model": "npu", "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        {"type": "text", "text": PROMPTS[rec["category"]]}]}]}
    t = time.perf_counter()
    try:
        r = requests.post(f"{base}/v1/chat/completions", json=body, timeout=120)
        r.raise_for_status()
        ans = r.json()["choices"][0]["message"]["content"].strip().lower()
    except Exception as e:
        ans = f"<error:{type(e).__name__}>"
    return {**rec, "answer": ans, "ms": (time.perf_counter() - t) * 1000}


def main():
    out = {}
    for tag, base in SERVERS.items():
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            res = list(ex.map(lambda r: ask(base, r), INDEX))
        el = time.perf_counter() - t0
        out[tag] = res
        bad = sum(1 for r in res if r["answer"].startswith("<error"))
        print(f"  {tag}: {len(res)}건 {el:.1f}s ({el/len(res)*1000:.0f}ms/건) 오류 {bad}", flush=True)
    Path("/tmp/ax_eval/raw.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("=== EVAL_DONE ===")


if __name__ == "__main__":
    main()
