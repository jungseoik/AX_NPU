"""
PE-Core-L14-336 모델 로딩 + NPU 컴파일용 패치 (pe_npu 패키지 핵심).

perception_models(Product-AI-mono/packages) 코드를 참조해 CLIP vision encoder를 만들고,
qbcompiler(torch backend)가 추적/양자화할 수 있도록 5개 패치를 적용한다.

- 가중치 출처: HuggingFace `facebook/PE-Core-L14-336` (최초 1회 자동 다운로드)
- perception_models는 복사하지 않고 경로만 sys.path에 추가해 import한다.
  경로는 이 파일 기준 상대경로 ../../Product-AI-mono/packages (= AX_NPU/Product-AI-mono/packages).
  환경변수 PE_PACKAGES_PATH 로 덮어쓸 수 있다.

5개 패치(apply_pe_patches):
  1) RoPE freq 상수화 (고정 grid 24x24, update_grid 무력화)
  2) Rope2D.__call__ einops-free native 구현 (constant cos/sin buffer)
  3) SelfAttention.forward einops -> reshape/permute + SDPA
  4) AttentionPooling.forward MHA -> 명시적 q/k/v Linear + SDPA (INT8 정확도)
  5) _sample_abs_posemb 상수화 (ellipsis+newaxis 회피)
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
# pe_npu/ 기준 상대경로: AX_NPU/AX_NPU/pe_npu -> ../../Product-AI-mono/packages
DEFAULT_PKG = os.path.normpath(os.path.join(HERE, "..", "..", "Product-AI-mono", "packages"))

MODEL_NAME = "PE-Core-L14-336"
IMAGE_SIZE = 336
PATCH = 14


def _add_packages_to_path():
    """perception_models(Product-AI-mono/packages)를 sys.path에 추가하고 경로를 반환."""
    pkg = os.environ.get("PE_PACKAGES_PATH", DEFAULT_PKG)
    if not os.path.isdir(pkg):
        raise FileNotFoundError(
            f"perception_models 패키지 경로를 찾을 수 없습니다: {pkg}\n"
            f"PE_PACKAGES_PATH 환경변수로 Product-AI-mono/packages 경로를 지정하세요."
        )
    if pkg not in sys.path:
        sys.path.insert(0, pkg)
    return pkg


def _import_pe_modules():
    """perception_models의 pe / rope 모듈을 import (경로 추가 포함)."""
    _add_packages_to_path()
    from pia.ai.tasks.T2VRet.models.PE.perception_models.pe_core.vision_encoder import pe as pe_mod
    from pia.ai.tasks.T2VRet.models.PE.perception_models.pe_core.vision_encoder import rope as rope_mod
    return pe_mod, rope_mod


class VisionWrapper(torch.nn.Module):
    """CLIP에서 vision tower만 떼어내 (B,3,H,W) -> (B,D) 임베딩을 내는 래퍼."""

    def __init__(self, clip_model):
        super().__init__()
        self.visual = clip_model.visual

    def forward(self, image):
        return self.visual(image)


class FeatWrapper(torch.nn.Module):
    """진단/컴파일용: attn_pool 직전 forward_features (1,577,1024) 출력 (trunk만)."""

    def __init__(self, clip_model):
        super().__init__()
        self.v = clip_model.visual

    def forward(self, image):
        return self.v.forward_features(image, norm=True)


class PoolWrapper(torch.nn.Module):
    """진단용: pool 후 proj 전 출력 (pool 내부 vs proj 손실 구간 특정)."""

    def __init__(self, clip_model):
        super().__init__()
        self.v = clip_model.visual

    def forward(self, image):
        x = self.v.forward_features(image, norm=True)
        return self.v._pool(x)


def apply_pe_patches(model, pe_mod=None):
    """NPU(qbcompiler torch backend) 컴파일을 위해 PE vision encoder에 5개 패치를 적용.

    model: CLIP.from_config(...) 결과 (in-place 수정).
    pe_mod: perception_models의 pe 모듈 (없으면 import). SelfAttention/AttentionPooling
            클래스 메서드를 패치하므로 모듈 단위로 필요하다.
    반환: model (체이닝 편의).
    """
    if pe_mod is None:
        pe_mod, _ = _import_pe_modules()
    # Rope2D 클래스는 rope 모듈에서 import
    _add_packages_to_path()
    from pia.ai.tasks.T2VRet.models.PE.perception_models.pe_core.vision_encoder.rope import Rope2D

    visual = model.visual
    rope = visual.rope

    # ---- 패치 1: RoPE freq 상수화 (고정 grid 24x24) ----
    gh = gw = IMAGE_SIZE // PATCH
    rope.update_grid("cpu", gh, gw)
    freq = rope.freq.detach().clone()  # (1, seq, dim)
    rope.update_grid = lambda *a, **k: None  # 동적 grid 재계산 무력화
    rope.freq = freq

    # ---- 패치 2: Rope2D.__call__ einops-free native + constant cos/sin ----
    def rotate_half_native(x):
        x = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
        x1 = x[..., 0]
        x2 = x[..., 1]
        x = torch.stack((-x2, x1), dim=-1)
        return x.reshape(*x.shape[:-2], -1)

    freq_b = freq[:, None, :, :]  # (1,1,seq,dim)
    cos_c = freq_b.cos()
    sin_c = freq_b.sin()

    def static_call(self, q, k):
        q = q * cos_c + rotate_half_native(q) * sin_c
        k = k * cos_c + rotate_half_native(k) * sin_c
        return q, k

    Rope2D.__call__ = static_call

    # ---- 패치 3: SelfAttention.forward einops -> reshape/permute + SDPA ----
    def sa_forward(self, x, attn_mask=None):
        batch, seq, embed_dim = x.shape
        if self.split_qkv:
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
        else:
            proj = F.linear(x, self.in_proj_weight, self.in_proj_bias)
            q = proj[..., :embed_dim]
            k = proj[..., embed_dim:2 * embed_dim]
            v = proj[..., 2 * embed_dim:]
        h = self.num_heads
        q = q.reshape(batch, seq, h, -1).permute(0, 2, 1, 3)
        k = k.reshape(batch, seq, h, -1).permute(0, 2, 1, 3)
        v = v.reshape(batch, seq, h, -1).permute(0, 2, 1, 3)
        if self.rope:
            q, k = self.rope(q, k)
        attn = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=self.scale
        )
        attn = attn.permute(0, 2, 1, 3).reshape(batch, seq, embed_dim)
        return F.linear(attn, self.out_proj.weight, self.out_proj.bias)

    pe_mod.SelfAttention.forward = sa_forward

    # ---- 패치 4: AttentionPooling.forward MHA -> 명시적 q/k/v + SDPA ----
    def ap_forward(self, x):
        E = self.embed_dim
        H = self.num_heads
        mha = self.attn
        Wq, Wk, Wv = mha.in_proj_weight[:E], mha.in_proj_weight[E:2 * E], mha.in_proj_weight[2 * E:]
        bq, bk, bv = mha.in_proj_bias[:E], mha.in_proj_bias[E:2 * E], mha.in_proj_bias[2 * E:]
        B, N = x.shape[0], x.shape[1]
        q = F.linear(self.probe, Wq, bq)
        k = F.linear(x, Wk, bk)
        v = F.linear(x, Wv, bv)
        P = self.probe.shape[1]
        q = q.reshape(1, P, H, E // H).permute(0, 2, 1, 3)
        k = k.reshape(B, N, H, E // H).permute(0, 2, 1, 3)
        v = v.reshape(B, N, H, E // H).permute(0, 2, 1, 3)
        o = F.scaled_dot_product_attention(q, k, v)
        o = o.permute(0, 2, 1, 3).reshape(1, P, E)
        o = F.linear(o, mha.out_proj.weight, mha.out_proj.bias)
        o = o + self.mlp(self.layernorm(o))
        return o

    pe_mod.AttentionPooling.forward = ap_forward

    # ---- 패치 5: _sample_abs_posemb 상수화 ----
    abs_pe = visual._sample_abs_posemb(gh, gw).detach().clone()  # (1, seq, width)

    def sample_abs_posemb_const(self, grid_h, grid_w):
        return abs_pe

    type(visual)._sample_abs_posemb = sample_abs_posemb_const

    return model


def load_pe(model_name: str = MODEL_NAME, mode: str = "full", patch: bool = False):
    """PE-Core CLIP vision encoder를 로드해 래퍼로 반환.

    model_name: HF config 이름 (기본 PE-Core-L14-336).
    mode      : 'full'(VisionWrapper, 임베딩 1024) | 'feat'(FeatWrapper, trunk 1,577,1024)
                | 'pool'(PoolWrapper) | 'clip'(CLIP 원본 객체 그대로 반환).
    patch     : True면 apply_pe_patches 적용(컴파일용). 원본 PyTorch 정확도 기준이 필요하면 False.

    반환: torch.nn.Module (mode='clip'이면 CLIP 객체).
    """
    pe_mod, _ = _import_pe_modules()
    model = pe_mod.CLIP.from_config(model_name, device="cpu", load_default_weights=True).float().eval()
    if patch:
        apply_pe_patches(model, pe_mod=pe_mod)
    if mode == "clip":
        return model
    if mode == "full":
        return VisionWrapper(model).eval()
    if mode == "feat":
        return FeatWrapper(model).eval()
    if mode == "pool":
        return PoolWrapper(model).eval()
    raise ValueError(f"알 수 없는 mode: {mode} (full|feat|pool|clip)")
