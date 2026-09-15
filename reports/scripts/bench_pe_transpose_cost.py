"""MultiNPUInferenceFull.infer 내부의 CHW->HWC 되돌리기 비용."""
import os, sys, time, statistics as st
import numpy as np, torch
sys.path.insert(0, os.environ.get("PIA_PACKAGES", "packages"))
from pia_prod.AI.modules.pe_npu.multi_npu import MultiNPUInferenceFull
md = st.median; CH = 62
def timed(fn, it=7):
    fn(); v=[]
    for _ in range(it):
        s=time.perf_counter(); fn(); v.append((time.perf_counter()-s)*1e3)
    return md(v)
def main():
    x = torch.randn(CH, 3, 336, 336)
    t_np = timed(lambda: x.detach().cpu().numpy().astype(np.float32, copy=False))
    arr = x.numpy()
    t_tr = timed(lambda: [np.ascontiguousarray(arr[i].transpose(1,2,0)) for i in range(CH)])
    m = MultiNPUInferenceFull(os.environ["MXQ"], device_ids=list(range(8)))
    t_all = timed(lambda: m(x), it=5)
    hwc = [np.ascontiguousarray(arr[i].transpose(1,2,0)) for i in range(CH)]
    t_pure = timed(lambda: list(m._pool.map(lambda i: m._infer_one(m.models[i % m.n], hwc[i]), range(CH))), it=5)
    print(f"infer(62ch,8카드) 전체 {t_all:7.1f} ms")
    print(f"  ├ torch->numpy        {t_np:6.1f} ms")
    print(f"  ├ CHW->HWC 되돌리기   {t_tr:6.1f} ms   ← 전처리에서 CHW로 만든 걸 다시 HWC로")
    print(f"  └ 순수 NPU 호출        {t_pure:6.1f} ms")
    m.dispose()
if __name__ == "__main__":
    main()
