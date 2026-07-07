"""데모(.py 옵션): PE 제로샷 분류 — live 텍스트 인코딩 × NPU 이미지 임베딩.

임의 텍스트 문자열을 PE 텍스트 인코더(CPU)로 즉석 인코딩하고, 이미지는 NPU(full NPU, MXQInferenceFull)로
임베딩해 코사인 유사도로 분류한다. (노트북 버전: demo_text_classification.ipynb 권장)

  image --(NPU)--> 1024 ┐
  prompts --(PE text, CPU)--> (N,1024) ┘  → cosine → argmax

토크나이저 = CLIP BPE(open_clip) = 공식 perception_models SimpleTokenizer와 동일(검증됨).
실행:
  conda activate pe_npu_host
  python demo_text_classification.py --images images/*.jpg \
      --prompts "a fire" "smoke" "a person who has fallen down" "a normal scene"
"""
import argparse, glob, os, sys
import numpy as np, torch
import open_clip
sys.path.insert(0, os.path.abspath("../.."))
import pe_npu
from pe_npu import load_pe, preprocess_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", default=sorted(glob.glob("images/*.jpg")))
    ap.add_argument("--prompts", nargs="+",
                    default=["a fire", "smoke", "a person who has fallen down", "a normal scene"])
    ap.add_argument("--full-mxq", default=None)
    args = ap.parse_args()
    if not args.images:
        sys.exit("이미지 없음. download_images.py 먼저 실행하거나 --images 지정.")

    npu = pe_npu.MXQInferenceFull(args.full_mxq) if args.full_mxq else pe_npu.MXQInferenceFull.from_hf()
    clip = load_pe("PE-Core-L14-336", mode="clip", patch=False)   # 텍스트 타워(CPU)
    tok = open_clip.get_tokenizer("ViT-L-14")

    with torch.no_grad():
        txt = clip.encode_text(tok(args.prompts, context_length=clip.context_length), normalize=True)
    ls = clip.logit_scale.exp()

    print("프롬프트:", args.prompts, "\n")
    for p in args.images:
        e = torch.from_numpy(np.asarray(npu.infer(preprocess_image(p)[None]))).float().reshape(-1)
        e = e / e.norm()
        sim = e @ txt.T
        probs = (ls * sim).softmax(-1)
        print(f"{os.path.basename(p):14s} → {args.prompts[int(sim.argmax())]}  "
              f"({float(probs.max()):.2f})  " + str([round(float(x), 2) for x in probs]))


if __name__ == "__main__":
    main()
