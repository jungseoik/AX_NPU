"""[full NPU] 모드별 '정확한' 다채널 처리량 — 1장 기준 (서로 다른 이미지, 출력 검증).

목표: 여러 다른 이미지를 한 카드(8코어)로 가장 빠르게 처리하는 코어모드 찾기.
async 한 모델 multi-in-flight는 출력이 깨지므로(검증됨), **모델 인스턴스 N개 + 스레드 + 동기 infer**로
8코어를 정확히 채운다. N = 8 / (이미지당 코어):
  single=8인스턴스(각1코어), global4=2(각4), global8=1(8코어), multi=1.

각 모드: 서로 다른 이미지 BATCH장을 처리하는 wall-clock → img/s. 출력은 동기 단일추론과 cos로 검증.
실행: python bench_throughput_correct.py [device_id] [batch]
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
SP = "/tmp/claude-1000/-home-gpuadmin/057c9eaa-b86d-4a41-ab6f-9dc6babfd1fe/scratchpad"
INSTANCES = {"single": 8, "global4": 2, "global8": 1, "multi": 1}

# 서로 다른 이미지 BATCH장
def load_imgs(n):
    vids = ["event_video.mp4", "pe_binary_elvfalldown_video_1.mp4", "pe_binary_esfalldown_video.mp4"]
    imgs = []
    for v in vids:
        c = cv2.VideoCapture(f"{SP}/{v}"); tot = int(c.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        for fn in np.linspace(5, max(6, tot - 5), n // len(vids) + 2).astype(int):
            c.set(cv2.CAP_PROP_POS_FRAMES, int(fn)); ok, f = c.read()
            if ok: imgs.append(np.ascontiguousarray(preprocess_image(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))).transpose(1, 2, 0)))
    return imgs[:n]
imgs = load_imgs(BATCH)
def shp(o): o = o[0] if isinstance(o, (list, tuple)) else o; return np.asarray(o).reshape(-1)
def cos(a, b): return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

print(f"[setup] device {DEV}, batch {BATCH} (서로 다른 이미지)\n")
print(f"{'mode':>8} | {'인스턴스':>6} | {'wall(ms)':>9} | {'img/s':>7} | {'출력검증':>8}")
for mode, ninst in INSTANCES.items():
    mxq = hf_hub_download("PIA-SPACE-LAB/MXQ_NPU", "pe_feat.mxq" if mode == "trunk" else f"{mode}/pe_full.mxq")
    # 동기 단일추론 기준(정확값)
    mref = qbruntime.Model(mxq); mref.launch(qbruntime.Accelerator(DEV))
    ref = [shp(mref.infer(x)) for x in imgs]; mref.dispose() if hasattr(mref, "dispose") else None
    # N개 인스턴스 (각각 동기), 스레드로 큐 소비
    models = [qbruntime.Model(mxq) for _ in range(ninst)]
    for m in models: m.launch(qbruntime.Accelerator(DEV))
    for m in models: m.infer(imgs[0])  # warmup
    results = {}
    def worker(m, q):
        while True:
            try: i, x = q.get_nowait()
            except queue.Empty: break
            results[i] = shp(m.infer(x))
    q = queue.Queue(); [q.put((i, x)) for i, x in enumerate(imgs)]
    t0 = time.perf_counter()
    ths = [threading.Thread(target=worker, args=(m, q)) for m in models]
    for t in ths: t.start()
    for t in ths: t.join()
    wall = (time.perf_counter() - t0) * 1000
    cs = [cos(results[i], ref[i]) for i in range(len(imgs))]
    ok = "OK" if min(cs) > 0.999 else f"깨짐{min(cs):.2f}"
    print(f"{mode:>8} | {ninst:>6} | {wall:>9.1f} | {BATCH/(wall/1000):>7.1f} | {ok:>8}")
    for m in models: m.dispose() if hasattr(m, "dispose") else None
print("\n[해석] img/s 높을수록 빠름. 출력검증 OK여야 실제 사용 가능. 8장 서버면 ×8 (대략).")
