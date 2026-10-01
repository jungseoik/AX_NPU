#!/usr/bin/env python3
"""두 비전 mxq 의 출력을 같은 이미지로 직접 비교한다 — **출력 순서 확정의 표준 도구**.

각 mxq 는 **자기가 선언한 입력 순서**로 먹이고, 출력은 전 쌍 코사인을 찍는다.
가중치가 같으면 대각선이 1.0 에 가깝고, 학습본 vs 스톡이면 높되 1 미만이다.
전부 0 근처면 한쪽 산출물이 깨진 것이다.

★ **PyTorch 를 기준으로 쓰려던 방법은 폐기했다.**
  `model.get_image_features(pixel_values, grid)` 와 비전 mxq 출력은 코사인이 0 근처로
  나온다 — **벤더 자신의 mxq + 스톡 체크포인트로 대조해도 0 이었다**(전처리 경로가
  다르다). 즉 그건 산출물 문제가 아니라 기준 계산이 틀린 것이었다.
  기준이 필요하면 **이미 검증된 mxq**(벤더 산출물)를 쓸 것.

사용:
    python3 compare_vision_mxq.py --a <검증된 모델 디렉토리> --b <새 모델 디렉토리> \
        --image <이미지> --card 7
  행·열 모두에서 최대인 1:1 대응이 나오면 그게 출력 매핑이다
  (`vision_output_order.json` 에 적는다).
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch


def run(model_dir, image, size, card, H=2048):
    sys.path.insert(0, os.environ.get("NPU_PE_2STAGE_VENDOR_SRC", "/repro/src"))
    sys.path.insert(0, str(Path(__file__).parent))
    import qbruntime as qb
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
    from pia_vlm.support import load_support
    from pia_vlm.artifact import metadata
    from pia_vlm.runtime import core_config
    from artifact_order import order_vision_inputs

    md = Path(model_dir)
    config, _, visual = load_support(md)
    proc = Qwen3VLProcessor.from_pretrained(str(md))
    proc.image_processor.size = {"shortest_edge": size * size, "longest_edge": 4096 * 256}
    img = Image.open(image).convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "hi"}]}]
    x = proc(text=[proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)],
             images=[img], return_tensors="pt")
    vp = md / config.vision_config.mxq_path
    with torch.inference_mode():
        inp = order_vision_inputs(str(vp), visual._prepare_dynamic_npu_inputs(
            x["pixel_values"], x["image_grid_thw"]))
    acc = qb.Accelerator(card)
    m = qb.Model(str(vp), core_config("global8")); m.launch(acc)
    outs = [np.asarray(o, np.float32).reshape(-1, H) for o in m.infer(inp)]
    m.dispose()
    return metadata(str(vp), "modelOutputList"), outs


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True); p.add_argument("--b", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--size", type=int, default=224); p.add_argument("--card", type=int, default=0)
    a = p.parse_args()
    na, oa = run(a.a, a.image, a.size, a.card)
    nb, ob = run(a.b, a.image, a.size, a.card)
    short = lambda s: (s[:34] + "…") if len(s) > 35 else s
    print("\nA:", a.a); [print(f"   [{i}] {n}") for i, n in enumerate(na)]
    print("B:", a.b); [print(f"   [{i}] {n}") for i, n in enumerate(nb)]

    def cos(u, v):
        u, v = u.reshape(-1), v.reshape(-1)
        n = min(u.size, v.size); u, v = u[:n], v[:n]
        d = np.linalg.norm(u) * np.linalg.norm(v)
        return float(u @ v / d) if d else 0.0

    print("\n출력 전 쌍 코사인 (행=A, 열=B)")
    print("        " + "".join(f"{j:>9}" for j in range(len(ob))))
    for i, u in enumerate(oa):
        print(f"  A[{i}] " + "".join(f"{cos(u, v):9.4f}" for v in ob))
    print("\nA norms:", [round(float(np.linalg.norm(u)), 2) for u in oa])
    print("B norms:", [round(float(np.linalg.norm(v)), 2) for v in ob])
    print("=== COMPARE_DONE ===")


if __name__ == "__main__":
    main()
