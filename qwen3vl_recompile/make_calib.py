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


def rotate_tensor(cos: "torch.Tensor", sin: "torch.Tensor") -> "torch.Tensor":
    """런타임이 NPU 에 실제로 넣는 **rotateTensor(peSize) 레이아웃**으로 cos/sin 을 합친다.

    캘리브는 "실제로 들어올 값"으로 떠야 한다. 런타임은 단순 concat 이 아니라
    인터리브된 표를 넣는다(`mblt_model_zoo` 의 `_build_vision_rotate_tensor()` /
    `MobilintQwen3VLRotaryEmbedding`):

        rt[0:dim:2]       =  cos[:half]
        rt[1:dim:2]       = -sin[:half]
        rt[dim:2dim:2]    =  sin[half:]
        rt[dim+1:2dim:2]  =  cos[half:]

    단순 concat 으로 캘리브를 뜨면 **채널별 양자화 범위가 다른 채널에 배정**된다.
    그래도 그래프는 멀쩡히 돌아가서(지연 동일) 출력만 조용히 열화한다 —
    스톡 가중치로 돌려도 벤더가 `no` 라고 답하는 걸 `Yes` 라고 답했다.
    """
    cos = cos.reshape(-1, cos.shape[-1]).float()
    sin = sin.reshape(-1, sin.shape[-1]).float()
    n, dim = cos.shape
    half = dim // 2
    pe = 2 * (((dim + 63) // 64) * 64)
    rt = torch.zeros(n, pe, dtype=torch.float32)
    rt[:, 0:dim:2] = cos[:, :half]
    rt[:, 1:dim:2] = -sin[:, :half]
    rt[:, dim:2 * dim:2] = sin[:, half:]
    rt[:, dim + 1:2 * dim:2] = cos[:, half:]
    return rt


def _graph(path: str):
    from mblt.serialize import SerializeMeta
    md = SerializeMeta.get_model_dict(path)
    return md.to_dict() if hasattr(md, "to_dict") else md


def _dim(x):
    """DimValue → 동적이면 -1, 아니면 정수."""
    if isinstance(x, dict):
        return -1 if x.get("dynamic") else int(x["value"])
    return -1 if getattr(x, "dynamic", False) else int(getattr(x, "value", x))


def decoder_calib_spec(path: str):
    """decoder 캘리브의 **이름·shape 를 mblt 에서 직접 읽는다**.

    구성 = 그래프 입력(InputOptions) 전부 + RoPE cos 상수(InputConstant) 1개.
    cos 가 왜 끼는지: 우리 mblt 입력은 4개인데 컴파일러는 5개를 요구한다
    (4개만 주면 `quantizeFB failed. The size of info ... are not same`).
    벤더도 mblt 2입력 → 캘리브 3개로 항상 rope 상수 하나가 더 붙는다.

    ★ 벤더 캘리브 json 을 베끼면 안 된다. 이름도 shape 도 export 경로마다 다르다:
        벤더  cos/reshape_0                      [1,-1,256]   (cos|sin 병합 1노드)
        우리  p_language_model_rotary_emb_cos_0  [1,16384,128](cos·sin 별도 노드)
      이름이 안 맞으면 quantizeFB 는 **개수만 보고 통과**시킨 뒤, 한참 뒤
      메모리 플래닝에서 `could not find input info in memory: ...` 로 죽는다.
    """
    d = _graph(path)
    g = d["subgraphs"][0]
    acts = g.get("activations") or d.get("activations")

    def spec(name):
        for a in acts:
            ad = a if isinstance(a, dict) else (a.to_dict() if hasattr(a, "to_dict") else vars(a))
            if ad.get("name") == name:
                return [_dim(x) for x in ad["shape"]]
        raise SystemExit(f"❌ 활성 `{name}` 을 mblt 에서 못 찾았다")

    names = list(d["inputs"])
    cos = [o["name"] for o in g["operators"]
           if (o.get("options") or {}).get("type") == "InputConstantOptions"
           and "cos" in o["name"]]
    if len(cos) != 1:
        raise SystemExit(f"❌ RoPE cos 상수를 1개로 특정하지 못했다: {cos}")
    names.append(cos[0])
    shapes = [spec(n) for n in names]
    # ★ rope 항목만은 **노드 shape 가 아니다**.
    #   노드 자체는 [1,16384,128] 상수지만(벤더도 동일), 캘리브가 요구하는 건
    #   cos|sin 을 이어붙인 [1,-1,256](동적 길이)다. 노드 shape 를 그대로 쓰면
    #   `quantizeFB failed. Input shapes given in Model and JSON are not same`.
    shapes[-1] = [1, -1, shapes[-1][-1] * 2]
    return names, shapes


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

    # ★★ 캘리브 규격은 mblt 입력을 그대로 나열하면 안 된다. 벤더 성공 사례 기준:
    #    · vision  3입력 — sin 을 따로 주지 않고 cos 에 cos|sin 을 **병합(128폭)**
    #    · 시퀀스 축은 -1 (export 의 dynamic_axes 와 짝) — 고정값이면 거부된다
    #    4입력으로 주면 `size of info ... not same`, 축을 고정하면 `Input shapes ... not same`.
    V_NAMES = ["cos", "pos_embeds/reshape", "float_1_channel_last"]
    V_SHAPES = [[1, 1, -1, 128], [1, 1, -1, 1024], [1, 1, -1, 1536]]
    d_names, D_SHAPES = decoder_calib_spec(str(a.mblt_dir / "decoder.mblt"))
    print(f"  vision  캘리브 {V_NAMES}", flush=True)
    print(f"  decoder 캘리브 {d_names}", flush=True)

    idx = 0
    rows_v, rows_d = [], []

    @torch.inference_mode()
    def tensors(f):
        img = Image.open(f).convert("RGB").resize((a.size, a.size), Image.Resampling.BICUBIC)
        x = proc(text=[tpl], images=[img], return_tensors="pt")
        grid, ids = x["image_grid_thw"], x["input_ids"]
        pe, cv, sv = vis.compute_side_inputs(grid)
        folded = P.fold_pixel_values(x["pixel_values"].to(torch.float32))
        N = folded.shape[-1]                       # 시각 패치 수
        V = [rotate_tensor(cv, sv).reshape(1, 1, N, 128),
             pe.reshape(1, 1, N, 1024).float(),
             folded.permute(0, 2, 3, 1).reshape(1, 1, N, 1536).float()]

        hidden, ds_list = vis(folded, pe, cv, sv)
        T = ids.shape[1]
        emb = embed(ids).float().clone()
        mask = (ids == cfg.image_token_id)[0]
        emb[0, mask] = hidden.reshape(-1, emb.shape[-1]).float()
        full = torch.zeros(1, len(ds_list), T, emb.shape[-1])
        for k2, ds in enumerate(ds_list):
            full[0, k2, mask] = ds.reshape(-1, emb.shape[-1]).float()
        D = {"inputs_embeds/reshape": emb}
        ds_keys = [n for n in d_names if "deepstack" in n]
        if len(ds_keys) == 1:
            D[ds_keys[0]] = full
        else:
            for k2, n in enumerate(ds_keys):
                D[n] = full[:, k2]          # 활성 스펙이 [1,-1,2048] 이라 랭크 3
        for n in d_names:
            if n in D: continue
            if "cos" in n:
                # 위 shape 주석대로 cos|sin 을 이어붙여 (1, T, 256) 로 준다.
                pos = torch.arange(T).view(1, 1, T).repeat(3, 1, 1)
                c = P.CachedQwen3VLTextRotaryEmbedding(model.model.language_model.rotary_emb)
                c.set_rope(pos)
                idxs = torch.arange(T)
                D[n] = rotate_tensor(c.cos[..., idxs, :], c.sin[..., idxs, :]).reshape(1, T, -1)
        return V, D

    for n, rows, which in [(a.n_vision, rows_v, "v"), (a.n_decoder, rows_d, "d")]:
        for k in range(n):
            V, D = tensors(files[k])
            if which == "v":
                picked, names = V, V_NAMES
            else:
                missing = [nm for nm in d_names if nm not in D]
                if missing:
                    raise SystemExit(f"❌ decoder 입력 {missing} 를 못 만들었다")
                picked, names = [D[nm] for nm in d_names], d_names
            paths = []
            for t in picked:
                fn = f"{idx:04d}.npy"; np.save(data / fn, t.contiguous().numpy())
                paths.append(f"data/{fn}"); idx += 1
            rows.append(paths)
            if (k + 1) % 25 == 0: print(f"    {which} {k+1}/{n}", flush=True)

    for kind, rows, names, shp in [("vision", rows_v, V_NAMES, V_SHAPES),
                                   ("decoder", rows_d, d_names, D_SHAPES)]:
        (a.out / f"{kind}.json").write_text(json.dumps(
            {"info": {"input names": names, "input shapes": shp},
             "calib paths": rows}, indent=2) + "\n")
        print(f"  {kind}.json  레코드 {len(rows)}  shapes {shp}", flush=True)
    print(f"  npy {idx}개 / {sum(f.stat().st_size for f in data.glob('*.npy'))/2**30:.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
