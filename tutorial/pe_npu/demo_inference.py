"""
PE-Core-L14-336 NPU 추론 데모 (튜토리얼).

예제 이미지들을 전처리 -> NPU(full: image→embedding 전부 NPU)로 비전 임베딩 추출 -> 두 가지로 검증:
  1) 이미지 간 코사인 유사도 매트릭스 (비슷한 이미지는 높고 다른 이미지는 낮은지)
  2) 원본 PyTorch(pth) 임베딩 대비 NPU 임베딩 cos (양자화 정확도, 0.99+ 기대)

전처리/모델 로딩은 모두 pe_npu 패키지를 사용한다.

사용 (컨테이너 안, LD_LIBRARY_PATH 설정 후):
    python demo_inference.py --images-dir ./images
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

# tutorial/pe_npu/ 의 부모(AX_NPU/AX_NPU)를 import 경로에 추가 -> import pe_npu 가능
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..")))

from pe_npu import preprocess_image, load_pe, MXQInferenceFull
from pe_npu.inference import DEFAULT_FULL_MXQ


def cos(a, b):
    a, b = a.reshape(-1), b.reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", default=os.path.join(HERE, "images"))
    ap.add_argument("--full-mxq", default=DEFAULT_FULL_MXQ)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.images_dir, "*.jpg")) +
                   glob.glob(os.path.join(args.images_dir, "*.png")))
    if not paths:
        raise SystemExit(f"이미지 없음: {args.images_dir} (download_images.py 먼저 실행)")
    print(f"이미지 {len(paths)}장:", [os.path.basename(p) for p in paths])

    x = np.stack([preprocess_image(p) for p in paths], axis=0)  # (N,3,336,336)

    # --- NPU 임베딩 (full: image→embedding 전부 NPU) ---
    npu = MXQInferenceFull(args.full_mxq)
    emb_npu = npu.infer(x)  # (N,1024)
    print("\nNPU 임베딩 shape:", emb_npu.shape)

    # --- 원본 PyTorch 임베딩 (정확도 비교 기준, 미패치) ---
    ref_model = load_pe("PE-Core-L14-336", mode="full", patch=False)
    with torch.no_grad():
        emb_pth = ref_model(torch.from_numpy(x)).numpy()

    # --- 1) 이미지 간 유사도 매트릭스 (NPU 임베딩) ---
    print("\n[이미지 간 코사인 유사도 (NPU 임베딩)]")
    names = [os.path.basename(p)[:14] for p in paths]
    print("            " + "  ".join(f"{n:>14}" for n in names))
    for i in range(len(paths)):
        row = "  ".join(f"{cos(emb_npu[i], emb_npu[j]):>14.3f}" for j in range(len(paths)))
        print(f"{names[i]:>12}  {row}")

    # --- 2) 원본 대비 NPU 정확도 ---
    print("\n[원본 PyTorch 대비 NPU 임베딩 정확도]")
    for i in range(len(paths)):
        print(f"  {names[i]:>14}: cos={cos(emb_pth[i], emb_npu[i]):.4f}")
    mean_cos = np.mean([cos(emb_pth[i], emb_npu[i]) for i in range(len(paths))])
    print(f"  평균 cos = {mean_cos:.4f}  ({'OK (>=0.99)' if mean_cos >= 0.99 else '점검 필요'})")


if __name__ == "__main__":
    main()
