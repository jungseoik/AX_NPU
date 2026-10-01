#!/usr/bin/env python3
"""mblt 그래프 입력 규격 조회 — 캘리브 json 이 어긋날 때 쓰는 진단 도구

캘리브 json 의 `input names` / `input shapes` 는 **이 출력과 정확히 같아야** 한다.
벤더가 준 `inputs/calibration/*.json` 을 베끼면 안 된다. 이름이 export 경로마다 다르다:

    벤더  cos/reshape_0                      [1,-1,256]    (cos|sin 병합 1노드)
    우리  p_language_model_rotary_emb_cos_0  [1,16384,128] (cos·sin 별도 노드)

증상별 원인:
  · `quantizeFB failed. The size of info ... are not same`
        → **개수**가 다르다. decoder 는 그래프 입력 + RoPE cos 상수 1개 = 입력수+1.
  · `could not find input info in memory: <텐서>`
        → 개수는 맞고 **이름이 다르다**. quantizeFB 는 개수만 보고 통과시키므로
          20분쯤 뒤 메모리 플래닝에서야 터진다. 가장 헷갈리는 증상.

사용:
    python3 inspect_mblt.py <model.mblt> [--ops N]
"""
from __future__ import annotations
import argparse, collections


def graph(path: str):
    from mblt.serialize import SerializeMeta
    md = SerializeMeta.get_model_dict(path)
    return md.to_dict() if hasattr(md, "to_dict") else md


def dim(x):
    if isinstance(x, dict):
        return -1 if x.get("dynamic") else int(x["value"])
    return -1 if getattr(x, "dynamic", False) else int(getattr(x, "value", x))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mblt")
    p.add_argument("--ops", type=int, default=0, help="앞 N개 연산자도 출력")
    a = p.parse_args()

    d = graph(a.mblt)
    g = d["subgraphs"][0]
    acts = g.get("activations") or d.get("activations")
    spec = {}
    for x in acts:
        ad = x if isinstance(x, dict) else (x.to_dict() if hasattr(x, "to_dict") else vars(x))
        spec[ad.get("name")] = [dim(v) for v in ad["shape"]]

    print(f"# {a.mblt}")
    print(f"연산자 {len(g['operators'])}개")
    print("\n[그래프 입력]")
    for n in d["inputs"]:
        print(f"  {n:52s} {spec.get(n)}")
    print("\n[InputConstant — 캘리브에 cos 하나가 더 들어간다]")
    for op in g["operators"]:
        if (op.get("options") or {}).get("type") == "InputConstantOptions":
            print(f"  {op['name']:52s} {spec.get(op['name'])}")

    print("\n[캘리브 json 에 쓸 값]")
    names = list(d["inputs"]) + [op["name"] for op in g["operators"]
                                 if (op.get("options") or {}).get("type") == "InputConstantOptions"
                                 and "cos" in op["name"]]
    print(f'  "input names" : {names}')
    print(f'  "input shapes": {[spec.get(n) for n in names]}')

    c = collections.Counter((o.get("options") or {}).get("type") for o in g["operators"])
    print("\n[연산자 구성]")
    for t, n in c.most_common(12):
        print(f"  {str(t):36s} {n:5d}")
    if a.ops:
        print(f"\n[앞 {a.ops}개 연산자]")
        for i, o in enumerate(g["operators"][:a.ops]):
            op = o.get("options") or {}
            print(f"  [{i:3d}] {o['name']:54s} {str(op.get('type')):28s} "
                  f"ins={op.get('inputs')} outs={op.get('outputs')}")


if __name__ == "__main__":
    main()
