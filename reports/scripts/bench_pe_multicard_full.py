"""통합 MXQInferenceFull: 62채널까지 스케일링(추론시간) + 출력 정확성 검증 + 기존 데모 호환."""
import sys, time, glob
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))
import numpy as np
import pe_npu

# 실제 이미지 62장 (COCO val2017), PE 전처리(336)
paths = sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "val2017/*.jpg"))[:62]
X = np.stack([pe_npu.preprocess_image(p) for p in paths], axis=0).astype(np.float32)
print(f"입력 {X.shape} (실제 val2017 62장)\n", flush=True)

# 기준: 단일카드 (global4)
single = pe_npu.MXQInferenceFull.load(scheme="global4", device_id=0)
print(f"[단일] cards={len(single)} mode={single.core_mode} slots={single.slots_per_card}", flush=True)
ref = single.infer(X)                                     # (62,1024)

# 멀티카드 auto (전 카드)
multi = pe_npu.MXQInferenceFull.load(scheme="global4", device_ids="auto")
print(f"[멀티] cards={len(multi)} mode={multi.core_mode} slots={multi.slots_per_card} W={multi.W}\n", flush=True)

# 출력 정확성: 멀티 62장 vs 단일 62장
om = multi.infer(X)
def cosrow(a,b):
    a=a/np.linalg.norm(a,axis=1,keepdims=True); b=b/np.linalg.norm(b,axis=1,keepdims=True)
    return (a*b).sum(1)
c = cosrow(om, ref)
print(f"[출력검증] 멀티 shape={om.shape}, cos(멀티 62 vs 단일 62) min={c.min():.6f} "
      f"mean={c.mean():.6f}  norm범위=[{np.linalg.norm(om,axis=1).min():.2f},{np.linalg.norm(om,axis=1).max():.2f}] "
      f"{'✅ 정확(cos≈1, garbage 아님)' if c.min()>0.9999 else '❌'}\n", flush=True)

# 채널 스윕 추론시간 (멀티 7카드, min of 3)
def best(m, x, rep=3):
    m.infer(x)                       # warmup
    ts = []
    for _ in range(rep):
        s = time.perf_counter(); m.infer(x); ts.append((time.perf_counter() - s) * 1000)
    return min(ts)
print(f"{'채널':>4} | {'멀티7카드 ms':>12} | {'단일1카드 ms':>12}", flush=True)
print("-"*38)
for B in [1, 8, 16, 32, 56, 62]:
    xb = X[:B]
    tm = best(multi, xb); ts = best(single, xb)
    print(f"{B:>4} | {tm:>12.0f} | {ts:>12.0f}", flush=True)
print("PE62_DONE", flush=True)
