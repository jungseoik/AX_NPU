"""데모: PE 비전 임베딩(NPU) × 텍스트 프롬프트 임베딩 → 제로샷 분류 (retrieval).

PE는 CLIP 계열이라 "이미지 임베딩 vs 텍스트 프롬프트 임베딩"의 코사인 유사도로
분류 문제를 푼다. 여기서는 NPU가 이미지 임베딩을, 미리 인코딩된 text_features.json이
텍스트(프롬프트) 임베딩을 담당한다.

  image --(NPU hybrid)--> (1024) ─┐
                                  ├─ cosine 유사도 → 가장 높은 클래스로 분류
  prompts --(사전 인코딩)--> (N,1024) ┘

text_features.json = PE 텍스트 인코더로 오프라인 인코딩된 프롬프트 임베딩
(키: class/prompt/ID/feature). HF `PIA-SPACE-LAB/PE-Core-L14-336`에서 자동 다운로드.
(임의 문자열 프롬프트를 즉석 인코딩하려면 PE 텍스트 인코더+토크나이저가 필요 — 여긴 사전 인코딩본 사용.)

실행:
  conda activate pe_npu_host
  python demo_text_classification.py --images images/*.jpg
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(".."))
import pe_npu
from huggingface_hub import hf_hub_download

INDEX = {0: "normal", 1: "falldown", 2: "fire", 3: "smoke", 4: "smoking", 5: "esfalldown", 6: "elvfalldown"}


def load_text_features(repo="PIA-SPACE-LAB/PE-Core-L14-336", filename="text_features.json"):
    """프롬프트 텍스트 임베딩 로드 (없으면 HF 자동 다운로드). → (class_arr, prompt_arr, feat(N,1024))."""
    path = hf_hub_download(repo_id=repo, filename=filename)
    items = json.load(open(path))
    cls = np.array([it["class"] for it in items])
    prm = np.array([it.get("prompt") for it in items], dtype=object)
    feat = torch.tensor([it["feature"] for it in items], dtype=torch.float32)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return cls, prm, feat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", default=sorted(glob.glob("images/*.jpg")))
    ap.add_argument("--feat-mxq", default=None, help="로컬 MXQ. 미지정 시 HF에서 자동 다운로드(from_hf)")
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()
    if not args.images:
        sys.exit("이미지가 없습니다. `python download_images.py` 먼저 실행하거나 --images 지정.")

    print("[1] NPU 추론기 로드")
    model = pe_npu.MXQInferenceHybrid(args.feat_mxq) if args.feat_mxq else pe_npu.MXQInferenceHybrid.from_hf()

    print("[2] 텍스트 프롬프트 임베딩 로드")
    cls, prm, txt = load_text_features()
    print(f"    프롬프트 {len(cls)}개, 클래스 {sorted(set(int(c) for c in cls))}")

    print("[3] 이미지별 제로샷 분류\n")
    for img_path in args.images:
        x = pe_npu.preprocess_image(img_path)            # (3,336,336)
        emb = torch.from_numpy(np.asarray(model.infer(x[None]))).float().reshape(-1)
        emb = emb / emb.norm()
        sim = emb @ txt.T                                # (N,)

        # 클래스별 최고 유사도로 분류
        per_class = {}
        for c in set(int(v) for v in cls):
            mask = torch.from_numpy(cls == c)
            per_class[c] = float(sim[mask].max())
        pred = max(per_class, key=per_class.get)

        topv, topi = torch.topk(sim, args.topk)
        print(f"📷 {os.path.basename(img_path)}  →  예측: **{INDEX.get(pred,'?')}**")
        print("    클래스별 점수:", {INDEX.get(c, c): round(s, 3) for c, s in sorted(per_class.items(), key=lambda kv: -kv[1])})
        print("    Top 프롬프트:")
        for v, i in zip(topv.tolist(), topi.tolist()):
            print(f"      {v:.3f}  [{INDEX.get(int(cls[i]),'?')}] {str(prm[i])[:60]}")
        print()


if __name__ == "__main__":
    main()
