"""
PE-Core-L14-336 vision encoder -> single-input/single-output MXQ via backend="torch".

Approach (mirrors SDK's VisionModelForQwen2VL):
  - Fix grid 24x24 (336/14), precompute RoPE freq -> store cos/sin path as constants.
  - Replace Rope2D.__call__ (a *plain* python class holding an nn.Module `self.rope`)
    with a buffer-based static method that does NOT call any nn.Module
    (this is the root cause of the "module is not installed as a submodule" NameError).
  - Register PE's apply_rotary_emb / rotate_half as FX autowrap leaves so the tracer
    does not descend into einops rearrange.
"""
import os
import sys
import argparse

import torch

HERE = "/workspace/AX_NPU/pe_onnx_export"
sys.path.insert(0, HERE)

IMAGE_SIZE = 336
PATCH = 14


def _parse_operators(wrapper, fd):
    """ModelParser로 sg0를 만들고 (layertype, name) 리스트 반환."""
    from qbcompiler.model_dict.parser.parser import ModelParser
    parser = ModelParser(
        model=wrapper, backend="torch", target_device="aries2",
        yolo_decode_include=True,
    )
    parser.cfg.allocate_to_devices = True
    parser.cfg.split_supported_concat = True
    parser.parse(feed_dict=fd, save_subgraph_type=1, debug=False)
    md, _ = parser.get_md_wd(body_only=False)
    sg0 = md.subgraphs[0]
    ops = []
    for op in sg0.operators:
        lt = op.layertype.name if hasattr(op.layertype, "name") else str(op.layertype)
        ops.append((lt, op.name))
    return ops


def _select_names(ops, spec):
    """spec('all'|'none'|'sub1,sub2'...)에 맞는 operator 이름 리스트 반환.
    substring 매칭은 operator 이름 또는 layertype 둘 다 대상으로 한다."""
    if spec is None or spec == "none" or spec == "":
        return []
    if spec == "all":
        return [name for _, name in ops]
    subs = [s.strip().lower() for s in spec.split(",") if s.strip()]
    out = []
    for lt, name in ops:
        hay = (name + " " + lt).lower()
        if any(s in hay for s in subs):
            out.append(name)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="parse", choices=["parse", "compile"])
    ap.add_argument("--save", default=os.path.join(HERE, "out", "pe_torch.mxq"))
    ap.add_argument("--calib-data-path", default=None,
                    help="실제 calibration npy 디렉토리/txt. 미지정 시 random calib")
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--calib-output", type=int, default=0, choices=[0, 1],
                    help="0=activation per-layer, 1=per-channel(정밀도↑)")
    ap.add_argument("--calib-method", type=int, default=1, choices=[0, 1, 2, 3],
                    help="양자화 방식. 1=W채널·A레이어 대칭, 3=zeropoint 비대칭")
    ap.add_argument("--use-et", action="store_true",
                    help="EquivalentTransformation(SmoothQuant류) 적용 — activation outlier 완화")
    ap.add_argument("--act16", default=None,
                    help="16bit activation override. 'all'=전체, 'none'=없음, "
                         "또는 쉼표구분 substring 필터(예: 'attn_pool,layernorm') — "
                         "operator 이름에 substring 매칭되는 레이어를 16bit activation으로 둠")
    ap.add_argument("--weight16", default=None,
                    help="16bit weight override. act16과 동일 문법(substring 매칭)")
    ap.add_argument("--act16-exclude", default=None,
                    help="act16 선택 결과에서 제외할 substring(쉼표구분). "
                         "예: --act16 all --act16-exclude 'inputconst_193,attn_pool,add_96'")
    ap.add_argument("--dump-names", default=None,
                    help="parse 모드에서 operator 이름/타입 목록을 이 파일에 덤프")
    ap.add_argument("--feat-only", action="store_true",
                    help="진단용: attn_pool 전 forward_features(1,577,1024) 출력")
    ap.add_argument("--pool-only", action="store_true",
                    help="진단용: pool 후 proj 전 출력 (pool 내부 vs proj 가름)")
    args = ap.parse_args()

    from export_pe_onnx import VisionWrapper, _add_packages_to_path
    _add_packages_to_path()
    from pia.ai.tasks.T2VRet.models.PE.perception_models.pe_core.vision_encoder import pe as pe_mod
    from pia.ai.tasks.T2VRet.models.PE.perception_models.pe_core.vision_encoder import rope as rope_mod
    from pia.ai.tasks.T2VRet.models.PE.perception_models.pe_core.vision_encoder.rope import (
        Rope2D, rotate_half, apply_rotary_emb,
    )

    print("[1] load PE-Core-L14-336")
    model = pe_mod.CLIP.from_config("PE-Core-L14-336", device="cpu", load_default_weights=True).float().eval()
    visual = model.visual
    rope = visual.rope

    print("[2] precompute RoPE freq for fixed grid 24x24")
    gh = gw = IMAGE_SIZE // PATCH
    rope.update_grid("cpu", gh, gw)
    freq = rope.freq.detach().clone()  # (1, seq, dim)
    print(f"    freq shape: {tuple(freq.shape)}")

    # neutralize dynamic grid recompute
    rope.update_grid = lambda *a, **k: None

    # store constant freq on the Rope2D instance (plain tensor, not a Module)
    rope.freq = freq

    # ---- Replace Rope2D.__call__ + apply_rotary_emb with einops-free native impl ----
    # PE's rotate_half/apply_rotary_emb use einops.rearrange which FX's MbltProxy
    # cannot pass through. Reimplement with native torch ops.
    # rotate_half: "... (d r) -> ... d r" (r=2), unbind, stack(-x2,x1), flatten back.
    def rotate_half_native(x):
        x = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
        x1 = x[..., 0]
        x2 = x[..., 1]
        x = torch.stack((-x2, x1), dim=-1)
        return x.reshape(*x.shape[:-2], -1)

    # precompute cos/sin of freq (freq already constant), broadcast shape (1,1,seq,dim)
    freq_b = freq[:, None, :, :]  # (1,1,seq,dim)
    cos_c = freq_b.cos()
    sin_c = freq_b.sin()

    def static_call(self, q, k):
        q = q * cos_c + rotate_half_native(q) * sin_c
        k = k * cos_c + rotate_half_native(k) * sin_c
        return q, k
    Rope2D.__call__ = static_call
    print("    patched Rope2D.__call__ (einops-free, constant cos/sin buffers)")

    # ---- Patch SelfAttention.forward to avoid einops.rearrange (FX/einops incompat) ----
    import torch.nn.functional as F

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
        # b s (h d) -> b h s d
        q = q.reshape(batch, seq, h, -1).permute(0, 2, 1, 3)
        k = k.reshape(batch, seq, h, -1).permute(0, 2, 1, 3)
        v = v.reshape(batch, seq, h, -1).permute(0, 2, 1, 3)
        if self.rope:
            q, k = self.rope(q, k)
        attn = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=self.scale
        )
        # b h s d -> b s (h d)
        attn = attn.permute(0, 2, 1, 3).reshape(batch, seq, embed_dim)
        return F.linear(attn, self.out_proj.weight, self.out_proj.bias)
    pe_mod.SelfAttention.forward = sa_forward
    print("    patched SelfAttention.forward (einops -> reshape/permute)")

    # ---- Patch AttentionPooling.forward: nn.MultiheadAttention -> 명시적 q/k/v Linear + SDPA ----
    # SDK는 nn.MultiheadAttention을 저수준 폴백(multi_head_attention_forward)으로 처리하는데
    # 이 분해가 INT8에 적대적이라 attn_pool에서 정확도가 0.95→0.46으로 박살난다.
    # Qwen2.5-VL/BLIP 비전 패치와 동일하게 fused in_proj를 q/k/v로 분리하고 SDPA를 명시적으로 쓴다.
    # (probe=query 1개가 image token에 cross-attention. batch 고정=1.)
    def ap_forward(self, x):
        E = self.embed_dim
        H = self.num_heads
        mha = self.attn
        Wq, Wk, Wv = mha.in_proj_weight[:E], mha.in_proj_weight[E:2 * E], mha.in_proj_weight[2 * E:]
        bq, bk, bv = mha.in_proj_bias[:E], mha.in_proj_bias[E:2 * E], mha.in_proj_bias[2 * E:]
        B, N = x.shape[0], x.shape[1]
        q = F.linear(self.probe, Wq, bq)              # (1, num_probe, E)
        k = F.linear(x, Wk, bk)                       # (B, N, E)
        v = F.linear(x, Wv, bv)
        P = self.probe.shape[1]
        q = q.reshape(1, P, H, E // H).permute(0, 2, 1, 3)   # (1,H,P,d)
        k = k.reshape(B, N, H, E // H).permute(0, 2, 1, 3)   # (B,H,N,d)
        v = v.reshape(B, N, H, E // H).permute(0, 2, 1, 3)
        o = F.scaled_dot_product_attention(q, k, v)          # (1,H,P,d)
        o = o.permute(0, 2, 1, 3).reshape(1, P, E)
        o = F.linear(o, mha.out_proj.weight, mha.out_proj.bias)
        o = o + self.mlp(self.layernorm(o))
        return o
    pe_mod.AttentionPooling.forward = ap_forward
    print("    patched AttentionPooling.forward (MHA -> explicit q/k/v Linear + SDPA)")

    # ---- Patch _sample_abs_posemb: avoid `[None, ...]` (ellipsis+newaxis unsupported) ----
    # grid is fixed 24x24 == posemb_grid_size, so the embedding is a constant.
    abs_pe = visual._sample_abs_posemb(gh, gw).detach().clone()  # (1, seq, width)
    def sample_abs_posemb_const(self, grid_h, grid_w):
        return abs_pe
    type(visual)._sample_abs_posemb = sample_abs_posemb_const
    print(f"    patched _sample_abs_posemb -> constant {tuple(abs_pe.shape)}")

    if args.feat_only:
        # 진단용: attn_pool 직전 토큰피쳐(1,577,1024)를 출력 (블록 vs pooling 손실 구간 특정)
        class FeatWrapper(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.v = m.visual
            def forward(self, image):
                return self.v.forward_features(image, norm=True)
        wrapper = FeatWrapper(model).eval()
        print("    [feat-only] attn_pool 전 forward_features 출력")
    elif args.pool_only:
        # 진단용: pool 후 proj 전 출력 (pool 내부 vs proj 손실 구간 특정)
        class PoolWrapper(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.v = m.visual
            def forward(self, image):
                x = self.v.forward_features(image, norm=True)
                return self.v._pool(x)
        wrapper = PoolWrapper(model).eval()
        print("    [pool-only] proj 전 pool 출력")
    else:
        wrapper = VisionWrapper(model).eval()
    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    print("[3] sanity forward")
    with torch.no_grad():
        out = wrapper(dummy)
    print(f"    output: {tuple(out.shape)}")

    from qbcompiler.model_dict.parser.backend.torch.util import wrap_tensor
    fd = {"image": wrap_tensor("image", dummy)}

    if args.mode == "parse":
        from qbcompiler.model_dict.parser.parser import ModelParser
        print("[4] ModelParser(backend=torch).parse()")
        parser = ModelParser(
            model=wrapper,
            backend="torch",
            target_device="aries2",
            yolo_decode_include=True,
        )
        parser.cfg.allocate_to_devices = True
        parser.cfg.split_supported_concat = True
        parser.parse(feed_dict=fd, save_subgraph_type=1, debug=False)
        md, wd = parser.get_md_wd(body_only=False)
        print(f"[OK] parse finished. subgraphs={len(md.subgraphs)}")
        sg0 = md.subgraphs[0]
        print(f"    sg0 inputs={sg0.inputs} outputs={sg0.outputs}")
        # operator 이름/타입 덤프 (16bit override 후보 이름 추출용)
        lines = []
        for op in sg0.operators:
            lt = op.layertype.name if hasattr(op.layertype, "name") else str(op.layertype)
            lines.append(f"{lt}\t{op.name}")
        print(f"    sg0 num_operators={len(sg0.operators)}")
        if args.dump_names:
            with open(args.dump_names, "w") as f:
                f.write("\n".join(lines) + "\n")
            print(f"[OK] dumped {len(lines)} operator names -> {args.dump_names}")
        else:
            # 타입별 개수 요약 + 샘플
            from collections import Counter
            cnt = Counter(l.split("\t")[0] for l in lines)
            print("    layertype counts:", dict(cnt))
    else:
        from qbcompiler import mxq_compile
        common = dict(
            model=wrapper, backend="torch", feed_dict=fd, save_path=args.save,
            target_device="aries2", yolo_decode_include=True,
            inference_scheme="single", device=args.device,
        )
        # ---- 16bit override (BitConfig.LayerOverrides) ----
        extra = {}
        if args.act16 or args.weight16:
            from qbcompiler import BitConfig
            print("[3b] re-parse to enumerate operator names for 16bit override")
            ops = _parse_operators(wrapper, fd)
            act_names = _select_names(ops, args.act16)
            w_names = _select_names(ops, args.weight16)
            if args.act16_exclude:
                ex = [s.strip().lower() for s in args.act16_exclude.split(",") if s.strip()]
                before = len(act_names)
                act_names = [n for n in act_names if not any(s in n.lower() for s in ex)]
                print(f"    act16 exclude: {before} -> {len(act_names)} (removed {before - len(act_names)})")
            print(f"    total ops={len(ops)}  act16={len(act_names)}  weight16={len(w_names)}")
            extra["bit_config"] = BitConfig(
                layer_overrides=BitConfig.LayerOverrides(
                    activation_16bits=act_names, weight_16bits=w_names,
                )
            )
        if args.calib_data_path:
            from qbcompiler import CalibrationConfig
            calib = args.calib_data_path
            if os.path.isdir(calib) and os.path.exists(os.path.join(calib, "npy_files.txt")):
                calib = os.path.join(calib, "npy_files.txt")
            # output: 0=activation per-layer, 1=per-channel(정밀↑). method: 1=대칭, 3=zeropoint(비대칭)
            cc = CalibrationConfig(method=args.calib_method, output=args.calib_output, mode=1,
                                   max_percentile=CalibrationConfig.MaxPercentile(percentile=0.9999, topk_ratio=0.01))
            print(f"[4] mxq_compile(backend=torch) + calib: {calib} (method={args.calib_method}, output={args.calib_output}, et={args.use_et}, act16={args.act16}, weight16={args.weight16})")
            if args.use_et:
                from qbcompiler import EquivalentTransformationConfig as ET
                extra["equivalent_transformation_config"] = ET(
                    norm_conv=ET.NormConv(apply=True), qk=ET.Qk(apply=True),
                    ud=ET.Ud(apply=True), vo=ET.Vo(apply=True),
                )
            mxq_compile(**common, calib_data_path=calib, calibration_config=cc, **extra)
        else:
            print("[4] mxq_compile(backend=torch) + random calib")
            mxq_compile(**common, use_random_calib=True, **extra)
        print(f"[OK] saved {args.save}")


if __name__ == "__main__":
    main()
