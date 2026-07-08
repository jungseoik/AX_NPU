"""Qwen3-VL VQA(1토큰) 카드수 × 배치(동시요청) 지연 매트릭스.
배치 B = 동시에 들어온 요청 수. 값 = B개 요청이 전부 응답받는 데 걸린 총 wall-clock(ms).
카드 N장에 라운드로빈 분산, 카드당 1요청씩 순차(인스턴스당 in-flight 1)."""
import time, threading
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoConfig, GenerationConfig
from mblt_model_zoo.hf_transformers.utils.cache_utils import MobilintCache

MODEL = "mobilint/Qwen3-VL-2B-Instruct"
import sys
IMG = sys.argv[1] if len(sys.argv) > 1 else "bus.jpg"   # 임의 이미지 경로
NCARDS = 7
CARDS = [1, 2, 4, 6, 7]
BATCHES = [1, 2, 4, 8, 16, 32, 64]
PROMPT = "Is there a bus in this image? Answer only yes or no."

proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
inputs = proc.apply_chat_template(
    [{"role": "user", "content": [{"type": "image", "url": IMG}, {"type": "text", "text": PROMPT}]}],
    tokenize=True, add_generation_prompt=True, padding=True, return_tensors="pt", return_dict=True).to("cpu")
GEN = GenerationConfig(max_new_tokens=1, use_cache=True)

insts = []
for d in range(NCARDS):
    cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    cfg.vision_config.dev_no = d; cfg.text_config.dev_no = d
    insts.append(AutoModelForImageTextToText.from_pretrained(MODEL, config=cfg, trust_remote_code=True))
    print(f"[load] card {d}", flush=True)


def one(m):
    m.generate(**inputs, past_key_values=MobilintCache(m.get_cache_mxq_model()), generation_config=GEN)


for m in insts:  # 전체 warmup
    one(m); one(m)


def run_batch(N, B):
    """B개 동시요청을 N카드에 라운드로빈. 카드 i는 자기 몫을 순차 처리. 전부 끝날 때까지 wall."""
    use = insts[:N]
    groups = [[j for j in range(B) if j % N == i] for i in range(N)]
    def worker(i):
        for _ in groups[i]:
            one(use[i])
    ths = [threading.Thread(target=worker, args=(i,)) for i in range(N) if groups[i]]
    s = time.perf_counter()
    for t in ths: t.start()
    for t in ths: t.join()
    return (time.perf_counter() - s) * 1000

# 매트릭스 측정 (각 셀 2회 중 최솟값)
res = {}
for N in CARDS:
    res[N] = {}
    for B in BATCHES:
        res[N][B] = min(run_batch(N, B) for _ in range(2))
    print(f"[done] cards={N}", flush=True)

hdr = "카드\\배치 | " + " | ".join(f"B={b:<4}" for b in BATCHES)
print("\n===== 총 지연 (ms) : B개 동시요청 전부 응답까지 =====", flush=True)
print(hdr); print("-" * len(hdr) * 2)
for N in CARDS:
    print(f"{N:>7}  | " + " | ".join(f"{res[N][b]:6.0f}" for b in BATCHES), flush=True)

print("\n===== 처리량 (req/s) =====", flush=True)
print(hdr)
for N in CARDS:
    print(f"{N:>7}  | " + " | ".join(f"{b/(res[N][b]/1000):6.1f}" for b in BATCHES), flush=True)
print("VLM_MATRIX_DONE", flush=True)
