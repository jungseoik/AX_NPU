"""
PE vision encoder 출력 비교: 원본 PyTorch(pth) vs ONNX vs NPU(MXQ).

같은 입력을 각 백엔드에 통과시켜 출력 임베딩을 비교한다.
- 기준(reference)은 원본 PyTorch 출력.
- 핵심 지표: 코사인 유사도(1에 가까울수록 좋음), 평균/최대 절대오차.

각 백엔드는 환경에 존재할 때만 실행한다.
- PyTorch  : perception_models 필요 (Product-AI-mono). 기준값.
- ONNX     : onnxruntime 필요.
- NPU(MXQ) : qbruntime + NPU(/dev/aries0) 필요.

사용 예:
    # 컴파일 컨테이너(NPU 없음): pth vs onnx 비교
    python compare_backends.py --onnx ../pe_onnx_export/out/PE-Core-L14-336_vision_sim.onnx

    # NPU 호스트(런타임 설치 후): pth vs onnx vs npu 전체 비교
    python compare_backends.py \
        --onnx ../pe_onnx_export/out/PE-Core-L14-336_vision_sim.onnx \
        --mxq  ../pe_onnx_export/out/PE-Core-L14-336_vision.mxq
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PE_EXPORT_DIR = os.path.normpath(os.path.join(HERE, "..", "pe_onnx_export"))
MODEL_NAME = "PE-Core-L14-336"
IMAGE_SIZE = 336


def make_input(batch: int, image_size: int, seed: int = 0) -> np.ndarray:
    """전처리 완료를 가정한 정규화된 입력 텐서 (B,3,H,W) float32."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((batch, 3, image_size, image_size), dtype=np.float32)


def load_real_input(npy_dir: str, batch: int) -> np.ndarray:
    """전처리된 실제 이미지 npy((3,H,W))를 batch개 로드 → (B,3,H,W).

    calibration과 동일 분포(실제 이미지)로 정확도를 재야 의미가 있다.
    (더미 가우시안 입력은 calib 분포와 어긋나 정확도가 왜곡됨)
    """
    import glob
    files = sorted(glob.glob(os.path.join(npy_dir, "*.npy")))
    files = [f for f in files if not f.endswith("_files.txt")][:batch]
    if not files:
        raise FileNotFoundError(f"npy 없음: {npy_dir}")
    return np.stack([np.load(f) for f in files], axis=0).astype(np.float32)


def run_pytorch(x: np.ndarray, model_name: str, device: str):
    """원본 PyTorch vision encoder 출력 (기준값)."""
    import torch

    # pe_onnx_export의 모델 로딩 로직 재사용
    if PE_EXPORT_DIR not in sys.path:
        sys.path.insert(0, PE_EXPORT_DIR)
    from export_pe_onnx import VisionWrapper, _add_packages_to_path

    _add_packages_to_path()
    from pia.ai.tasks.T2VRet.models.PE.perception_models.pe_core.vision_encoder import pe as pe_mod

    model = pe_mod.CLIP.from_config(model_name, device=device, load_default_weights=True)
    wrapper = VisionWrapper(model.float().eval()).to(device).eval()
    with torch.no_grad():
        t = torch.from_numpy(x).to(device)
        out = wrapper(t).detach().cpu().numpy()
    return out.astype(np.float32)


def run_onnx(x: np.ndarray, onnx_path: str):
    """onnxruntime 출력."""
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    outs = []
    for i in range(x.shape[0]):
        o = sess.run(None, {in_name: x[i : i + 1]})[0]
        outs.append(np.asarray(o).reshape(-1))
    return np.stack(outs, axis=0).astype(np.float32)


def run_npu(x: np.ndarray, mxq_path: str, core_mode: str):
    """NPU(MXQ) 출력."""
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    from mxq_inference import MXQInference

    m = MXQInference(mxq_path, core_mode=core_mode)
    return np.asarray(m.infer(x)).astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1)
    b = b.reshape(-1)
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / denom)


def compare(name: str, ref: np.ndarray, other: np.ndarray):
    cos = np.mean([cosine(ref[i], other[i]) for i in range(ref.shape[0])])
    mae = float(np.mean(np.abs(ref - other)))
    maxd = float(np.max(np.abs(ref - other)))
    print(f"[{name:>14}] cos_sim={cos:.6f}  MAE={mae:.6e}  max|Δ|={maxd:.6e}")
    return cos


def main():
    ap = argparse.ArgumentParser(description="PE vision encoder 백엔드 출력 비교")
    ap.add_argument("--onnx", default=None, help="ONNX 경로 (없으면 ONNX 비교 생략)")
    ap.add_argument("--mxq", default=None, help="MXQ 경로 (없으면 NPU 비교 생략)")
    ap.add_argument("--model-name", default=MODEL_NAME)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    ap.add_argument("--core-mode", default="single")
    ap.add_argument("--cos-threshold", type=float, default=0.99,
                    help="이 값 미만이면 경고 (양자화 품질 기준)")
    ap.add_argument("--real-npy-dir", default=None,
                    help="실제 이미지 npy 디렉토리. 지정 시 더미 대신 실제 이미지로 비교 "
                         "(calibration 분포와 일치시켜야 정확)")
    args = ap.parse_args()

    if args.real_npy_dir:
        x = load_real_input(args.real_npy_dir, args.batch)
        print(f"입력: {x.shape} (실제 이미지 {args.real_npy_dir})")
    else:
        x = make_input(args.batch, args.image_size)
        print(f"입력: {x.shape} (정규화된 더미 텐서)")

    print("\n[ref] 원본 PyTorch 추론...")
    ref = run_pytorch(x, args.model_name, args.device)
    print(f"      출력 shape: {ref.shape}")

    results = {}
    if args.onnx:
        if os.path.exists(args.onnx):
            print("\n[onnx] onnxruntime 추론...")
            try:
                onnx_out = run_onnx(x, args.onnx)
                results["pth_vs_onnx"] = compare("pth vs onnx", ref, onnx_out)
            except ImportError:
                print("      onnxruntime 미설치 → ONNX 비교 생략")
        else:
            print(f"\n[onnx] 파일 없음: {args.onnx}")

    if args.mxq:
        if os.path.exists(args.mxq):
            print("\n[npu] NPU(MXQ) 추론...")
            try:
                npu_out = run_npu(x, args.mxq, args.core_mode)
                results["pth_vs_npu"] = compare("pth vs npu", ref, npu_out)
            except ImportError:
                print("      qbruntime 미설치 → NPU 비교 생략 (NPU 호스트에서 실행하세요)")
            except Exception as e:
                print(f"      NPU 추론 실패: {type(e).__name__}: {e}")
        else:
            print(f"\n[npu] 파일 없음: {args.mxq}")

    print("\n=== 요약 ===")
    if not results:
        print("비교 가능한 백엔드가 없습니다. --onnx / --mxq 를 지정하세요.")
        return
    ok = True
    for k, cos in results.items():
        status = "OK" if cos >= args.cos_threshold else "LOW"
        if cos < args.cos_threshold:
            ok = False
        print(f"  {k}: cos_sim={cos:.6f} [{status}] (기준 {args.cos_threshold})")
    print("\n결과:", "통과" if ok else "코사인 유사도가 기준 미만 — 양자화 품질 점검 필요")


if __name__ == "__main__":
    main()
