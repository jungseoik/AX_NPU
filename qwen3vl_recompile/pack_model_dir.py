#!/usr/bin/env python3
"""⑤ 런타임이 바로 쓰는 모델 디렉토리를 조립한다.

mxq 두 개만 갈아끼우면 되는 게 아니다. 호스트가 들고 있어야 하는 것들이 있고,
그중 하나는 **변환이 필요**하다.

★★ `embed_tokens` 는 **SpinR1 회전을 먹여서** 넣어야 한다.
   decoder 는 `advancedQuantization.EquivalentTransformation.SpinR1` (하다마드
   `rotation.pth`)로 컴파일돼 **회전된 기저**에서 동작한다. 벤더가 배포한
   `model.safetensors` 도 그래서 생 가중치가 아니라 `embed_tokens @ R` 이다
   (스톡 원본에 R 을 곱하면 벤더 파일과 **최대오차 0.0** 으로 일치한다).

   생 임베딩을 넣으면 기저가 어긋나는데, **그래프는 멀쩡히 돌고 지연도 같다.**
   증상은 로짓 스케일이 ~5배 눌려(no 47 → 9) 고빈도 토큰('\\n')에 밀리는 것뿐이라
   양자화 열화로 착각하기 쉽다. W8 로 올려도, OPTQ 를 켜도 낫지 않는다.

   `model.visual.pos_embed.weight` 는 회전 대상이 **아니다**(벤더 파일과 원본이 동일).

나머지(config/proxy/manifest/rotation)는 벤더 레퍼런스에서 가져온다. 특히
`config.json` 은 **벤더 것을 써야** 한다 — `architectures: MobilintQwen3VL…`,
`mxq_path`, `max_batch_size`, `core_mode` 같은 NPU 런타임 설정이 들어 있어서
학습본 config 로 덮으면 모델이 아예 안 뜬다.
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path

EMBED = "model.language_model.embed_tokens.weight"
POS = "model.visual.pos_embed.weight"
FROM_CKPT = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
             "generation_config.json")


def load_rotation(path: Path):
    import torch
    m = torch.jit.load(str(path), map_location="cpu")
    mats = {n: p for n, p in m.named_buffers()}
    mats.update({n: p for n, p in m.named_parameters()})
    cand = [v for v in mats.values() if v.dim() == 2 and v.shape[0] == v.shape[1]]
    if len(cand) != 1:
        raise SystemExit(f"❌ rotation.pth 에서 정사각 행렬을 특정하지 못했다: "
                         f"{[tuple(v.shape) for v in mats.values()]}")
    R = cand[0].float()
    err = float((R @ R.T - __import__("torch").eye(R.shape[0])).abs().max())
    if err > 1e-5:
        raise SystemExit(f"❌ rotation 이 직교가 아니다(오차 {err:.2e})")
    return R


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vendor", type=Path, required=True, help="벤더 model-reference 디렉토리")
    p.add_argument("--checkpoint", type=Path, required=True, help="학습본 전체 체크포인트")
    p.add_argument("--vision-mxq", type=Path, required=True)
    p.add_argument("--decoder-mxq", type=Path, required=True)
    p.add_argument("--rotation", type=Path, default=None, help="기본: 벤더 디렉토리의 rotation.pth")
    p.add_argument("--vision-order", type=Path, default=None,
                   help="compare_vision_mxq.py 로 확정한 vision_output_order.json")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    out = a.out
    if out.exists():
        shutil.rmtree(out)
    (out / "variants").mkdir(parents=True)

    # 1) 벤더 레퍼런스에서 구조/런타임 설정
    for f in a.vendor.iterdir():
        if f.is_file() and f.suffix in (".json", ".py", ".pth", ".txt"):
            shutil.copy2(f, out / f.name)
    # 2) 학습본에서 토크나이저류
    for n in FROM_CKPT:
        s = a.checkpoint / n
        if s.is_file():
            shutil.copy2(s, out / n)
    # 3) mxq
    shutil.copy2(a.vision_mxq, out / "vision.mxq")
    shutil.copy2(a.decoder_mxq, out / "decoder.mxq")
    if a.vision_order and a.vision_order.is_file():
        shutil.copy2(a.vision_order, out / "vision_output_order.json")

    # 4) ★ 지원 가중치 — embed_tokens 는 회전해서 넣는다
    R = load_rotation(a.rotation or (a.vendor / "rotation.pth"))
    src = a.checkpoint / "model.safetensors"
    with safe_open(str(src), "pt") as f:
        keys = set(f.keys())
        for k in (EMBED, POS):
            if k not in keys:
                raise SystemExit(f"❌ 체크포인트에 {k} 가 없다")
        emb = f.get_tensor(EMBED).float()
        pos = f.get_tensor(POS).float()
    if emb.shape[-1] != R.shape[0]:
        raise SystemExit(f"❌ hidden {emb.shape[-1]} 과 rotation {R.shape[0]} 이 다르다")
    rot = (emb @ R).contiguous()
    save_file({EMBED: rot, POS: pos.contiguous()}, str(out / "model.safetensors"),
              metadata={"format": "pt"})
    print(f"  embed_tokens {tuple(emb.shape)}  SpinR1 회전 적용 "
          f"(노름 {float(emb.norm()):.3f} → {float(rot.norm()):.3f})")
    print(f"  pos_embed    {tuple(pos.shape)}  회전 없음")
    print(f"  조립 완료: {out}")
    for n in ("vision.mxq", "decoder.mxq", "model.safetensors"):
        print(f"    {(out / n).stat().st_size:>13,} B  {n}")
    print("=== PACK_DONE ===")


if __name__ == "__main__":
    main()
