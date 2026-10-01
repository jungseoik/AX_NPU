#!/usr/bin/env python3
"""비전 mxq 를 PyTorch 기준과 대조해 (a) rope 입력 형식과 (b) 출력 순서를 확정한다.

벤더 런타임은 출력 이름 접두(`model_deepstack_merger_list_<i>_`)로 매칭하는데
우리 export 는 그 이름이 안 나온다. 그리고 rope 입력은 단순 concat 이 아니라
`_build_vision_rotate_tensor()` 의 **인터리브(rotateTensor)** 형식이다.
둘 다 수치로 확인한다(벤더도 json 에 "validate numerically after recompilation").
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--checkpoint", required=True, help="PyTorch 기준용 전체 체크포인트")
    p.add_argument("--image", required=True)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--card", type=int, default=0)
    p.add_argument("--write", action="store_true", help="확정되면 vision_output_order.json 저장")
    a = p.parse_args()

    sys.path.insert(0, os.environ.get("NPU_PE_2STAGE_VENDOR_SRC", "/repro/src"))
    sys.path.insert(0, str(Path(__file__).parent))
    import qbruntime as qb
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
    from pia_vlm.support import load_support
    from pia_vlm.artifact import metadata
    from pia_vlm.runtime import core_config
    from artifact_order import order_vision_inputs

    md = Path(a.model)
    config, _, visual = load_support(md)
    H = int(config.text_config.hidden_size)
    proc = Qwen3VLProcessor.from_pretrained(str(md))
    proc.image_processor.size = {"shortest_edge": a.size * a.size, "longest_edge": 4096 * 256}
    img = Image.open(a.image).convert("RGB").resize((a.size, a.size), Image.Resampling.BICUBIC)
    msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "hi"}]}]
    x = proc(text=[proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)],
             images=[img], return_tensors="pt")

    vp = md / config.vision_config.mxq_path
    names = metadata(str(vp), "modelOutputList")
    print("mxq 출력 이름:", *[f"\n  [{i}] {n}" for i, n in enumerate(names)])

    # ---- 입력 두 형식 ----
    with torch.inference_mode():
        base = visual._prepare_dynamic_npu_inputs(x["pixel_values"], x["image_grid_thw"])
        # 우리 export 가 먹인 형식: 패처 compute_side_inputs 와 동일한 단순 concat.
        # (라이브러리 쪽 compute_side_inputs 는 2값만 돌려주므로 직접 만든다)
        rot = visual._rot_pos_emb(x["image_grid_thw"])
        _emb = torch.cat((rot, rot), dim=-1)
        cv, sv = _emb.cos(), _emb.sin()
    inp_a = order_vision_inputs(str(vp), base)                      # rotateTensor
    plain = torch.cat([cv.reshape(1, -1, cv.shape[-1]),
                       sv.reshape(1, -1, sv.shape[-1])], -1).float().numpy()
    inp_b = [plain if t.shape[-1] == plain.shape[-1] else t for t in inp_a]  # 단순 concat
    print("입력 shape:", [tuple(np.asarray(t).shape) for t in inp_a])

    # ---- PyTorch 기준 ----
    from transformers import Qwen3VLForConditionalGeneration
    rm = Qwen3VLForConditionalGeneration.from_pretrained(a.checkpoint, dtype=torch.float32).eval()
    with torch.inference_mode():
        emb, ds = rm.model.get_image_features(x["pixel_values"].to(torch.float32), x["image_grid_thw"])
    if isinstance(emb, (list, tuple)):
        emb = emb[0]
    ref = [np.asarray(emb, np.float32).reshape(-1, H)] + \
          [np.asarray(d, np.float32).reshape(-1, H) for d in ds]
    labels = ["merger", "ds0", "ds1", "ds2"]
    print("기준 shape:", [r.shape for r in ref])

    def cos(u, v):
        u, v = u.reshape(-1), v.reshape(-1)
        n = min(u.size, v.size); u, v = u[:n], v[:n]
        d = np.linalg.norm(u) * np.linalg.norm(v)
        return float(u @ v / d) if d else 0.0

    acc = qb.Accelerator(a.card)
    m = qb.Model(str(vp), core_config("global8")); m.launch(acc)
    best = None
    for tag, vin in (("A rotateTensor(런타임 기본)", inp_a), ("B 단순concat(우리 export)", inp_b)):
        outs = [np.asarray(o, np.float32).reshape(-1, H) for o in m.infer(vin)]
        mat = [[cos(r, o) for o in outs] for r in ref]
        print(f"\n[{tag}]  코사인 (행=기준, 열=NPU)")
        print("        " + "".join(f"{i:>10}" for i in range(len(outs))))
        for li, row in enumerate(mat):
            print(f"  {labels[li]:<6}" + "".join(f"{c:10.4f}" for c in row))
        order = [int(np.argmax(r)) for r in mat]
        score = min(mat[i][order[i]] for i in range(len(ref)))
        ok = len(set(order)) == len(order)
        print(f"  → order={order} 1:1={'예' if ok else '아니오'} 최저코사인={score:.4f}")
        if ok and (best is None or score > best[1]):
            best = (order, score, tag)
    m.dispose()

    if best is None or best[1] < 0.9:
        raise SystemExit("❌ 어느 형식으로도 기준과 맞지 않는다 — 비전 재컴파일을 다시 봐야 한다")
    order, score, tag = best
    print(f"\n확정: {tag} / output_order={order} / 최저코사인 {score:.4f}")
    if a.write:
        f = md / "vision_output_order.json"
        f.write_text(json.dumps({"output_order": order, "labels": labels,
                                 "rope_format": tag, "min_cosine": round(score, 6),
                                 "basis": "PyTorch 기준과 코사인 대조"}, indent=2) + "\n")
        print("저장:", f)
    print("=== VISION_ORDER_DONE ===")


if __name__ == "__main__":
    main()
