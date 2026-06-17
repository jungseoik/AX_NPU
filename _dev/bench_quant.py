"""
bit_4 mixed-precision 양자화 스윕 벤치: 여러 feat MXQ에 대해
  - NPU trunk latency (단건 동기, async B=8 throughput)
  - pth 대비 hybrid 임베딩 cos (정확도)
를 한 번에 측정해 표로 출력. pth 기준은 한 번만 로드.

사용:
  python _dev/bench_quant.py <mxq1>:<label1> <mxq2>:<label2> ...
예:
  python _dev/bench_quant.py pe_npu/out/pe_feat.mxq:INT8 pe_npu/out/pe_feat_b4_50.mxq:bit4=0.5
"""
import sys, os, time, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import qbruntime
from pe_npu import preprocess_image, load_pe, MXQInferenceHybrid

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGES = sorted(glob.glob(os.path.join(HERE, "..", "tutorial_pe_npu", "images", "*.jpg")))


def cos(a, b):
    a, b = a.ravel(), b.ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def bench_latency(mxq, x_hwc, N=32):
    # 단건 동기 latency
    acc = qbruntime.Accelerator(); m = qbruntime.Model(mxq); m.launch(acc)
    m.infer(x_hwc)
    best = 1e9
    for _ in range(6):
        t = time.perf_counter(); m.infer(x_hwc); best = min(best, time.perf_counter() - t)
    single_ms = best * 1000
    m.dispose()
    # async B=N throughput
    mc = qbruntime.ModelConfig(); mc.set_async_pipeline_enabled(True)
    acc2 = qbruntime.Accelerator(); m2 = qbruntime.Model(mxq, mc); m2.launch(acc2)
    m2.infer_async(x_hwc).get()
    best = 1e9
    for _ in range(5):
        t = time.perf_counter()
        futs = [m2.infer_async(x_hwc) for _ in range(N)]
        [f.get() for f in futs]
        best = min(best, time.perf_counter() - t)
    ips = N / best
    m2.dispose()
    return single_ms, ips


def main():
    specs = [s.rsplit(":", 1) for s in sys.argv[1:]]
    if not specs:
        raise SystemExit("사용: python _dev/bench_quant.py <mxq>:<label> ...")
    if not IMAGES:
        raise SystemExit("측정 이미지 없음 (tutorial_pe_npu/images). download_images.py 먼저.")

    x = np.stack([preprocess_image(p) for p in IMAGES], axis=0)  # (N,3,336,336)
    x_hwc = np.transpose(x[0], (1, 2, 0)).copy()                 # latency용 단건 HWC

    print(f"측정 이미지 {len(IMAGES)}장, latency 단건 + async B=32\n")

    # pth 기준 임베딩 (한 번만)
    ref = load_pe("PE-Core-L14-336", mode="full", patch=False)
    with torch.no_grad():
        emb_pth = ref(torch.from_numpy(x)).numpy()

    rows = []
    for mxq, label in specs:
        if not os.path.exists(mxq):
            print(f"  [skip] {label}: 파일 없음 {mxq}"); continue
        single_ms, ips = bench_latency(mxq, x_hwc)
        hyb = MXQInferenceHybrid(mxq)
        emb = hyb.infer(x)
        mean_cos = float(np.mean([cos(emb_pth[i], emb[i]) for i in range(len(IMAGES))]))
        size_mb = os.path.getsize(mxq) / 1e6
        rows.append((label, size_mb, single_ms, ips, mean_cos))
        print(f"  측정완료: {label}")

    print("\n" + "=" * 72)
    print(f"{'설정':>14} | {'크기MB':>7} | {'단건ms':>7} | {'img/s(B32)':>10} | {'pth대비cos':>9}")
    print("-" * 72)
    for label, size_mb, single_ms, ips, mean_cos in rows:
        print(f"{label:>14} | {size_mb:>7.0f} | {single_ms:>7.1f} | {ips:>10.1f} | {mean_cos:>9.4f}")


if __name__ == "__main__":
    main()
