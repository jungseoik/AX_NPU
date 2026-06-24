"""attention pooling head만 standalone ONNX로 export (랜덤 가중치, 구조만 공유용).
trunk/실가중치 비공개 유지. compiler가 보는 형태(MHA 분해)로 명시 op 노출.
입력 (1,577,1024) → 임베딩 (1,1024).
"""
import torch, torch.nn as nn, torch.nn.functional as F
import onnx
from collections import Counter

E, H, HD, N = 1024, 16, 64, 577   # embed, heads, head_dim, tokens

class AttnPoolHead(nn.Module):
    """CLIP attention pooling: probe 1개가 N 토큰에 cross-attn → MLP → proj. (decomposed)"""
    def __init__(self):
        super().__init__()
        self.probe = nn.Parameter(torch.randn(1, 1, E))
        self.q = nn.Linear(E, E); self.k = nn.Linear(E, E); self.v = nn.Linear(E, E)
        self.out_proj = nn.Linear(E, E)
        self.ln = nn.LayerNorm(E)
        self.mlp = nn.Sequential(nn.Linear(E, E*4), nn.GELU(), nn.Linear(E*4, E))
        self.proj = nn.Linear(E, E, bias=False)

    def forward(self, x):                       # x: (1, N, 1024) = trunk 출력 토큰
        B = x.shape[0]
        q = self.q(self.probe).reshape(1, 1, H, HD).permute(0, 2, 1, 3)   # (1,16,1,64)  query=1
        k = self.k(x).reshape(B, N, H, HD).permute(0, 2, 1, 3)            # (1,16,577,64)
        v = self.v(x).reshape(B, N, H, HD).permute(0, 2, 1, 3)
        o = F.scaled_dot_product_attention(q, k, v)                      # softmax(qkᵀ/√64)·v → (1,16,1,64)
        o = o.permute(0, 2, 1, 3).reshape(1, 1, E)
        o = self.out_proj(o)
        o = o + self.mlp(self.ln(o))                                     # residual MLP
        return (o @ self.proj.weight.T).reshape(1, E)                    # (1,1024) 임베딩

m = AttnPoolHead().eval()
dummy = torch.randn(1, N, E)
OUT = "/home/gpuadmin/AX_NPU/reports/vendor/assets/attn_pool_head.onnx"
import os; os.makedirs(os.path.dirname(OUT), exist_ok=True)
torch.onnx.export(m, dummy, OUT, input_names=["tokens_1x577x1024"], output_names=["embedding_1x1024"],
                  opset_version=17, dynamo=False)
print("ONNX 저장:", OUT)

g = onnx.load(OUT).graph
ops = Counter(n.op_type for n in g.node)
print(f"\n총 노드 {len(g.node)}개, op 분포:")
for op, c in ops.most_common():
    print(f"  {op:24s} {c}")
print("\n주요 연산 흐름 (순서):")
for n in g.node:
    print(f"  {n.op_type}")
