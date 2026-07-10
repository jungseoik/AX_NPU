"""pool head(attn_pool+proj) CPU 레이턴시: per-item loop(현재) vs batch.

배경/결론은 reports/NPU_pe_poolhead_nogain_hybrid.md 참조. NPU 불필요(순수 CPU).
실행: python reports/bench_poolhead.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from pe_npu.pe_vendor.pe import AttentionPooling

EMBED, HEADS, TOK = 1024, 8, 577
torch.manual_seed(0)
pool = AttentionPooling(EMBED, HEADS).float().eval()
proj = torch.randn(EMBED, EMBED)


@torch.inference_mode()
def run_loop(feat):
    out = []
    for i in range(feat.shape[0]):
        p = pool(feat[i:i+1]).squeeze(1) @ proj
        out.append(p.reshape(-1).numpy())
    return np.stack(out, 0)


@torch.inference_mode()
def run_batch(feat):
    return (pool(feat).squeeze(1) @ proj).numpy()


@torch.inference_mode()
def kv_only(feat):                      # pool head 내부 dominant 비용 추정: k,v projection
    Wk = pool.attn.in_proj_weight[EMBED:2*EMBED]
    Wv = pool.attn.in_proj_weight[2*EMBED:]
    k = torch.nn.functional.linear(feat, Wk)
    v = torch.nn.functional.linear(feat, Wv)
    return k, v


def bench(fn, feat, warmup=5, rep=40):
    for _ in range(warmup):
        fn(feat)
    t = []
    for _ in range(rep):
        s = time.perf_counter(); fn(feat); t.append((time.perf_counter()-s)*1000)
    t.sort()
    return min(t), t[len(t)//2]          # min, median ms


for nth in (32, 8, 1):
    torch.set_num_threads(nth)
    print(f"\n##### torch threads = {nth} #####")
    print(f"{'B':>4} | {'loop(min/med)':>16} | {'batch(min/med)':>16} | {'spd(min)':>8} | {'kv min':>7}")
    print("-"*70)
    for B in [1, 8, 32, 62]:
        feat = torch.randn(B, TOK, EMBED)
        lo_min, lo_med = bench(run_loop, feat)
        ba_min, ba_med = bench(run_batch, feat)
        kv_min, _ = bench(kv_only, feat)
        print(f"{B:>4} | {lo_min:>7.2f}/{lo_med:<7.2f} | {ba_min:>7.2f}/{ba_med:<7.2f} | "
              f"{lo_min/ba_min:>6.2f}x | {kv_min:>6.2f}")
