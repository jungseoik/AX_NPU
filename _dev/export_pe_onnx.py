"""
PE-Core-L14-336 vision encoder를 HuggingFace에서 받아 ONNX로 export하는 독립 모듈.

Product-AI-mono의 perception_encoder 모듈이 사용하는 모델(Meta Perception Encoder,
PE-Core-L14-336)의 vision encoder를 NPU 컴파일용 ONNX로 변환한다.
- 가중치 출처: HuggingFace `facebook/PE-Core-L14-336` (`fetch_pe_checkpoint` → hf_hub_download)
- export 대상: CLIP의 vision tower(`model.visual`), 입력 (B,3,336,336), 출력 (B,1024)
- dynamic batch 축 적용 → Product-AI-mono trt_export.py의 `..._vision_dynamic.onnx`와 동일 성격

perception_models 코드는 Product-AI-mono/packages 아래에 있으므로, 이 스크립트 기준
상대경로(../../Product-AI-mono/packages)를 sys.path에 추가해 재사용한다.
환경변수 PE_PACKAGES_PATH로 경로를 덮어쓸 수 있다.

사용:
    python export_pe_onnx.py --out ./out/PE-Core-L14-336_vision.onnx --device cpu
"""
import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PKG = os.path.normpath(os.path.join(HERE, "..", "..", "Product-AI-mono", "packages"))
MODEL_NAME = "PE-Core-L14-336"
IMAGE_SIZE = 336


def _add_packages_to_path():
    pkg = os.environ.get("PE_PACKAGES_PATH", DEFAULT_PKG)
    if not os.path.isdir(pkg):
        raise FileNotFoundError(
            f"perception_models 패키지 경로를 찾을 수 없습니다: {pkg}\n"
            f"PE_PACKAGES_PATH 환경변수로 Product-AI-mono/packages 경로를 지정하세요."
        )
    if pkg not in sys.path:
        sys.path.insert(0, pkg)
    return pkg


class VisionWrapper(torch.nn.Module):
    """CLIP에서 vision tower만 떼어내 (B,3,H,W) -> (B,D) 임베딩을 내는 래퍼."""

    def __init__(self, clip_model):
        super().__init__()
        self.visual = clip_model.visual

    def forward(self, image):
        return self.visual(image)


def main():
    ap = argparse.ArgumentParser(description="PE-Core vision encoder ONNX exporter")
    ap.add_argument("--model-name", default=MODEL_NAME)
    ap.add_argument("--out", default=os.path.join(HERE, "out", f"{MODEL_NAME}_vision.onnx"))
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--no-simplify", dest="simplify", action="store_false",
                    help="onnxsim 단순화(If 노드 제거) 비활성화")
    ap.set_defaults(simplify=True)
    args = ap.parse_args()

    pkg = _add_packages_to_path()
    print(f"[init] perception_models path: {pkg}")

    from pia.ai.tasks.T2VRet.models.PE.perception_models.pe_core.vision_encoder import pe as pe_mod

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    print(f"[1/3] {args.model_name} 로딩 (HF에서 가중치 다운로드)...")
    model = pe_mod.CLIP.from_config(args.model_name, device=args.device, load_default_weights=True)
    # 양자화는 컴파일러가 수행하므로 ONNX는 float32로 export
    model = model.float().eval()

    wrapper = VisionWrapper(model).to(args.device).eval()
    dummy = torch.randn(1, 3, args.image_size, args.image_size, device=args.device)

    print("[2/3] forward 검증...")
    with torch.no_grad():
        out = wrapper(dummy)
    print(f"      input  : {tuple(dummy.shape)}")
    print(f"      output : {tuple(out.shape)}  dtype={out.dtype}")

    print(f"[3/3] ONNX export → {args.out} (opset {args.opset})")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            args.out,
            input_names=["image"],
            output_names=["embedding"],
            dynamic_axes={"image": {0: "batch"}, "embedding": {0: "batch"}},
            opset_version=args.opset,
            do_constant_folding=True,
        )
    size_mb = os.path.getsize(args.out) / 1024 / 1024
    print(f"       exported ({size_mb:.1f} MB)")

    # PE의 RoPE/위치임베딩 동적 조건문이 ONNX If 노드로 남으면 qbcompiler가 컴파일하지 못한다.
    # onnxsim으로 고정 입력 shape 기준 단순화하여 If 노드를 상수 폴딩으로 제거한다.
    if args.simplify:
        import onnx
        from onnxsim import simplify

        print("[simplify] onnxsim 단순화 (If 노드 제거)...")
        model_onnx = onnx.load(args.out)
        model_sim, ok = simplify(
            model_onnx, overwrite_input_shapes={"image": [1, 3, args.image_size, args.image_size]}
        )
        n_if = sum(1 for n in model_sim.graph.node if n.op_type == "If")
        onnx.save(model_sim, args.out)
        sim_mb = os.path.getsize(args.out) / 1024 / 1024
        print(f"[simplify] ok={ok} | If 노드={n_if} | {args.out} ({sim_mb:.1f} MB)")
        if n_if:
            print("[simplify] 경고: If 노드가 남아 있습니다. NPU 컴파일이 실패할 수 있습니다.")

    print(f"[done] {args.out}")


if __name__ == "__main__":
    main()
