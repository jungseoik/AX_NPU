"""
PE-Core-L14-336 vision encoder를 RoPE2D 정적화 후 ONNX로 export.

문제: PE의 RoPE2D는 forward마다 grid로 freq를 동적 계산(arange/expand/einsum)하고
attention마다 cos/sin을 계산한다. 이 동적 연산 때문에 qbcompiler가 ViT를 25개
서브그래프로 분할하여 단일 입출력 추론/calibration이 불가능했다.

해결(Qwen2-VL이 SDK에서 쓰는 방식과 동일): 입력 해상도(336)가 고정이면 RoPE freq도
고정이므로, cos/sin을 미리 계산해 상수로 박는다.
- update_grid()를 no-op으로 (freq 재계산 제거)
- Rope2D.__call__을 미리 계산한 cos/sin 상수를 쓰는 형태로 교체

사용:
    python export_pe_rope_static.py --out ./out/PE-Core-L14-336_vision_ropestatic.onnx
"""
import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "PE-Core-L14-336"
IMAGE_SIZE = 336
PATCH = 14


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "out", f"{MODEL_NAME}_vision_ropestatic.onnx"))
    ap.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    sys.path.insert(0, HERE)
    from export_pe_onnx import VisionWrapper, _add_packages_to_path
    _add_packages_to_path()
    from pia.ai.tasks.T2VRet.models.PE.perception_models.pe_core.vision_encoder import pe as pe_mod
    from pia.ai.tasks.T2VRet.models.PE.perception_models.pe_core.vision_encoder.rope import (
        Rope2D, rotate_half,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    print(f"[1] {MODEL_NAME} 로드")
    model = pe_mod.CLIP.from_config(MODEL_NAME, device="cpu", load_default_weights=True).float().eval()
    visual = model.visual
    rope = visual.rope

    print("[2] RoPE freq 계산 (고정 grid) + cos/sin 상수화")
    gh = gw = args.image_size // PATCH  # 24
    rope.update_grid("cpu", gh, gw)
    freq = rope.freq.detach().clone()           # (1, seq, dim)
    cos_c = freq.cos()
    sin_c = freq.sin()
    print(f"    freq shape: {tuple(freq.shape)}  (seq={freq.shape[1]})")

    # update_grid 무력화 (freq 재계산 제거)
    rope.update_grid = lambda *a, **k: None
    rope.cos_c = cos_c
    rope.sin_c = sin_c

    # Rope2D.__call__ 을 상수 cos/sin 사용으로 교체 (special method라 클래스 레벨 패치)
    def static_call(self, q, k):
        c = self.cos_c[:, None, :, :]
        s = self.sin_c[:, None, :, :]
        q = q * c + rotate_half(q) * s
        k = k * c + rotate_half(k) * s
        return q, k
    Rope2D.__call__ = static_call

    wrapper = VisionWrapper(model).eval()
    dummy = torch.randn(1, 3, args.image_size, args.image_size)

    print("[3] forward 검증")
    with torch.no_grad():
        out = wrapper(dummy)
    print(f"    output: {tuple(out.shape)}")

    print(f"[4] ONNX export → {args.out}")
    with torch.no_grad():
        torch.onnx.export(
            wrapper, dummy, args.out,
            input_names=["image"], output_names=["embedding"],
            opset_version=args.opset, do_constant_folding=True,
        )

    print("[5] onnxsim 단순화")
    import onnx
    from onnxsim import simplify
    m = onnx.load(args.out)
    ms, ok = simplify(m, overwrite_input_shapes={"image": [1, 3, args.image_size, args.image_size]})
    n_if = sum(1 for n in ms.graph.node if n.op_type == "If")
    sim_path = args.out.replace(".onnx", "_sim.onnx")
    onnx.save(ms, sim_path)
    size = os.path.getsize(sim_path) / 1024 / 1024
    print(f"    simplify ok={ok}  If노드={n_if}  → {sim_path} ({size:.1f}MB)")
    print("DONE")


if __name__ == "__main__":
    main()
