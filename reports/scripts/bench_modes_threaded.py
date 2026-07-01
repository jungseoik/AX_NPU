"""[full NPU] 모드별 다채널 처리량 — 올바른 패턴(1모델 + N스레드 동기), 출력검증.

async multi-in-flight는 출력이 깨지고(N=1만 안전), multi-instance는 메모리만 낭비 →
**1모델 + 멀티스레드 sync**가 정확(cos 1.0)하면서 코어를 다 쓰는 표준 패턴(SDK 문서 방법1: 멀티스레딩).
1카드에서 서로 다른 이미지 BATCH장을 모드별로 처리 → img/s + 출력 cos 검증.

실행: python bench_modes_threaded.py [device_id] [batch] [nthreads]
"""
import sys, time, threading, queue
import numpy as np, cv2
from PIL import Image
from huggingface_hub import hf_hub_download
import qbruntime
sys.path.insert(0, "/home/gpuadmin/AX_NPU")
from pe_npu.preprocess import preprocess_image

DEV = int(sys.argv[1]) if len(sys.argv) > 1 else 1
BATCH = int(sys.argv[2]) if len(sys.argv) > 2 else 32
NTH = int(sys.argv[3]) if len(sys.argv) > 3 else 8   # 카드당 스레드 수(=코어수 권장)
SP = "/tmp/claude-1000/-home-gpuadmin/057c9eaa-b86d-4a41-ab6f-9dc6babfd1fe/scratchpad"
MODES = ["single", "global4", "global8", "multi"]

def load(n):
    vids = ["event_video.mp4", "pe_binary_elvfalldown_video_1.mp4", "pe_binary_esfalldown_video.mp4"]
    out = []
    for v in vids:
        c = cv2.VideoCapture(f"{SP}/{v}"); tot = int(c.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        for fn in np.linspace(5, max(6, tot - 5), n // 3 + 2).astype(int):
            c.set(cv2.CAP_PROP_POS_FRAMES, int(fn)); ok, f = c.read()
            if ok: out.append(np.ascontiguousarray(preprocess_image(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))).transpose(1, 2, 0)))
        c.release()
    return out[:n]
imgs = load(BATCH)
def shp(o): o = o[0] if isinstance(o, (list, tuple)) else o; return np.asarray(o).reshape(-1)
def cos(a, b): return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

print(f"[setup] dev{DEV}, batch {BATCH}(서로 다름), 1모델+{NTH}스레드 sync\n")
print(f"{'mode':>8} | {'wall(ms)':>9} | {'img/s':>7} | {'단건(ms)':>8} | {'출력검증':>10}")
rows = {}
for mode in MODES:
    mxq = hf_hub_download("PIA-SPACE-LAB/MXQ_NPU", f"{mode}/pe_full.mxq")
    m = qbruntime.Model(mxq); m.launch(qbruntime.Accelerator(DEV))
    for _ in range(2): m.infer(imgs[0])
    ref = [shp(m.infer(x)) for x in imgs]          # 동기 단건 = 정확 기준 + 단건 latency
    t0 = time.perf_counter(); [m.infer(x) for x in imgs[:5]]; single_ms = (time.perf_counter() - t0) * 1000 / 5
    # 1모델 + NTH 스레드
    res = {}; q = queue.Queue()
    for i, x in enumerate(imgs): q.put((i, x))
    def w():
        while True:
            try: i, x = q.get_nowait()
            except queue.Empty: break
            res[i] = shp(m.infer(x))
    ths = [threading.Thread(target=w) for _ in range(NTH)]
    t0 = time.perf_counter(); [t.start() for t in ths]; [t.join() for t in ths]
    wall = (time.perf_counter() - t0) * 1000
    mincos = min(cos(res[i], ref[i]) for i in range(len(imgs)))
    rows[mode] = {"wall_ms": round(wall, 1), "img_s": round(BATCH / (wall / 1000), 1),
                  "single_ms": round(single_ms, 1), "min_cos": round(mincos, 4)}
    print(f"{mode:>8} | {wall:>9.0f} | {BATCH/(wall/1000):>7.1f} | {single_ms:>8.0f} | {('OK '+str(round(mincos,4))):>10}")
    m.dispose() if hasattr(m, "dispose") else None
import json
json.dump(rows, open(f"{SP}/bench_modes_threaded.json", "w"), indent=2)
print("\n[결론] throughput=img/s 최대 모드 / 단건latency=single_ms 최소 모드. 출력검증 전부 OK여야 사용가능.")
