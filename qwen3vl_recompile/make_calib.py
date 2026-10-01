#!/usr/bin/env python3
"""② 신규 체크포인트로 캘리브레이션 활성값 생성

벤더가 준 `inputs/calibration/` 은 **스톡 가중치로 뽑은 활성값**이라 신규 학습본엔 못 쓴다
(decoder 쪽은 embed_tokens·비전 출력에 직접 의존). 여기서 새로 만든다.

텐서 정의는 벤더 `calibration/*.json` 의 `info.input names` 와 정확히 맞춘다:

  vision  : cos                  [1,1,-1,128]   ← compute_side_inputs() 의 cos|sin concat
            pos_embeds/reshape   [1,1,-1,1024]
            float_1_channel_last [1,1,-1,1536]  ← fold_pixel_values() 결과
  decoder : inputs_embeds/reshape      [1,-1,2048]
            deepstack_visual_embeds_0  [1,3,-1,2048]
            cos/reshape_0              [1,-1,256]

★ 비전 출력은 **PyTorch(fp32)로 계산**한다. NPU(vision.mxq)를 쓰면 "mxq 를 만들려고 mxq 가
  필요한" 순환이 생기고, fp32 쪽이 캘리브 기준으로도 더 정확하다.
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np, torch


def mblt_inputs(path: str) -> list[str]:
    """컴파일러 자신의 로더로 .mblt 의 **실제 입력 이름/순서**를 읽는다.

    캘리브 json 의 input names/shapes 는 이것과 정확히 같아야 한다. 안 그러면
    `quantizeFB failed. Input shapes given in Model and JSON are not same` 로 죽는다.
    벤더 캘리브 json 을 그대로 베끼면 안 된다 — 벤더 mblt 와 우리 mblt 는 입력 구성이
    다르다(예: 벤더 decoder 는 deepstack 이 1입력, 우리는 3입력으로 이미 분리).
    """
    from mblt.serialize import SerializeMeta
    md = SerializeMeta.get_model_dict(path)
    return [getattr(i, "name", str(i)) for i in md.inputs]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--images", required=True, help="캘리브 이미지 디렉토리")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mblt-dir", type=Path, required=True,
                   help="export_mblt.py 산출물 — 입력 스펙을 여기서 읽는다")
    p.add_argument("--n-decoder", type=int, default=115, help="벤더와 동일 기본값")
    p.add_argument("--n-vision", type=int, default=106)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--prompt", default="Describe what is happening in this image in detail.")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    import qbcompiler  # noqa: F401
    from qbcompiler.model_dict_new.parser.patcher.models.hf_models import qwen3vl as P
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from PIL import Image
    P.ensure_qwen3vl_classes_loaded(); P.get_patcher_cfg()

    # ★ 이전 실행 잔해가 섞이면 레코드 수와 npy 개수가 어긋난다 — 깨끗이 비우고 시작
    import shutil
    if a.out.exists(): shutil.rmtree(a.out)
    data = a.out / "data"; data.mkdir(parents=True, exist_ok=True)
    proc = AutoProcessor.from_pretrained(a.checkpoint)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        a.checkpoint, dtype=torch.float32).eval()
    cfg = model.config
    vis = P.VisionModelForQwen3VL(model.model).eval()
    embed = model.model.get_input_embeddings()
    L = len(cfg.vision_config.deepstack_visual_indexes)

    files = sorted(Path(a.images).glob("*.jpg")) + sorted(Path(a.images).glob("*.png"))
    if not files:
        p.error(f"이미지가 없다: {a.images}")
    random.Random(a.seed).shuffle(files)
    need = max(a.n_decoder, a.n_vision)
    files = (files * (need // len(files) + 1))[:need]
    print(f"  이미지 {len(set(files))}종 / 레코드 decoder {a.n_decoder}, vision {a.n_vision}", flush=True)

    tpl = proc.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": a.prompt}]}],
        tokenize=False, add_generation_prompt=True)

    v_names = mblt_inputs(str(a.mblt_dir / "vision.mblt"))
    d_names = mblt_inputs(str(a.mblt_dir / "decoder.mblt"))
    print(f"  vision  입력 {v_names}", flush=True)
    print(f"  decoder 입력 {d_names}", flush=True)

    idx = 0
    rows_v, rows_d = [], []
    shapes_v, shapes_d = None, None

    @torch.inference_mode()
    def tensors(f):
        """이름 → 텐서 매핑을 만든다. 키는 mblt 입력 이름과 맞춘다."""
        img = Image.open(f).convert("RGB").resize((a.size, a.size), Image.Resampling.BICUBIC)
        x = proc(text=[tpl], images=[img], return_tensors="pt")
        grid, ids = x["image_grid_thw"], x["input_ids"]
        pos_embeds, cos_v, sin_v = vis.compute_side_inputs(grid)
        folded = P.fold_pixel_values(x["pixel_values"].to(torch.float32))
        V = {"float_1_channel_last": folded.float(),
             "pos_embeds/reshape": pos_embeds.float(),
             "cos": cos_v.float(), "sin": sin_v.float()}

        hidden, ds_list = vis(folded, pos_embeds, cos_v, sin_v)
        T = ids.shape[1]
        emb = embed(ids).float().clone()
        mask = (ids == cfg.image_token_id)[0]
        emb[0, mask] = hidden.reshape(-1, emb.shape[-1]).float()
        D = {"inputs_embeds/reshape": emb}
        # deepstack: mblt 가 슬라이스별로 분리돼 있으면 레이어별로, 뭉쳐 있으면 스택으로
        full = torch.zeros(1, len(ds_list), T, emb.shape[-1])
        for k2, ds in enumerate(ds_list):
            full[0, k2, mask] = ds.reshape(-1, emb.shape[-1]).float()
        ds_keys = [n for n in d_names if "deepstack" in n]
        if len(ds_keys) == 1:
            D[ds_keys[0]] = full
        else:
            for k2, n in enumerate(ds_keys):
                D[n] = full[:, k2]
        for n in d_names:          # cos 등 나머지 입력이 있으면 채운다
            if n in D: continue
            if "cos" in n:
                pos = torch.arange(T).view(1, 1, T).repeat(3, 1, 1)
                c = P.CachedQwen3VLTextRotaryEmbedding(model.model.language_model.rotary_emb)
                c.set_rope(pos)
                D[n] = c.cos[..., torch.arange(T), :].float()
        return V, D

    for n, rows, names, which in [(a.n_vision, rows_v, v_names, "v"),
                                  (a.n_decoder, rows_d, d_names, "d")]:
        for k in range(n):
            V, D = tensors(files[k])
            src = V if which == "v" else D
            missing = [nm for nm in names if nm not in src]
            if missing:
                raise SystemExit(f"❌ mblt 입력 {missing} 에 해당하는 텐서를 못 만들었다. "
                                 f"make_calib.py 의 tensors() 를 보완할 것.")
            paths, shp = [], []
            for nm in names:                      # ★ mblt 입력 순서 그대로
                t = src[nm].contiguous()
                fn = f"{idx:04d}.npy"; np.save(data / fn, t.numpy())
                paths.append(f"data/{fn}"); shp.append(list(t.shape)); idx += 1
            rows.append(paths)
            if which == "v" and shapes_v is None: shapes_v = shp
            if which == "d" and shapes_d is None: shapes_d = shp
            if (k + 1) % 25 == 0: print(f"    {which} {k+1}/{n}", flush=True)

    for kind, rows, names, shp in [("vision", rows_v, v_names, shapes_v),
                                   ("decoder", rows_d, d_names, shapes_d)]:
        dyn = [[-1 if i and d not in (1, 3) else d for i, d in enumerate(s_)] for s_ in shp]
        (a.out / f"{kind}.json").write_text(json.dumps(
            {"info": {"input names": names, "input shapes": shp},
             "calib paths": rows}, indent=2) + "\n")
        print(f"  {kind}.json  레코드 {len(rows)}  입력 {names}", flush=True)
        print(f"              shapes {shp}", flush=True)
    print(f"  npy {idx}개 / {sum(f.stat().st_size for f in data.glob('*.npy'))/2**30:.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
