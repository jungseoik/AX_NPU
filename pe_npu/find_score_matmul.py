"""Name-agnostic detector: find attention-score MatMul nodes (the MatMul that feeds
a Softmax through transparent ops). Works on any model via the parsed .mblt graph,
so layer names changing between the isolated head and the full model don't matter.

Uses only the public qbcompiler release API (load_mblt_model).

출처: Mobilint(전근우) 기술지원 제공. attention pooling head의 INT8 양자화 붕괴(cos 0.69)
원인이 QK^T matmul의 outlier임을 규명하고, 그 노드 activation만 16-bit로 올리면 복구됨을
확인(cos 0.998). 레이어 이름이 모델마다 달라지므로 그래프 구조(MatMul -> ... -> Softmax)로
score matmul을 자동 탐지한다. → reports/vendor/mobilint_resolution_attn_pool.md 참고.
"""
from qbcompiler.model_dict.serialize import load_mblt_model

# ops that merely reshape/rescale the logits between MatMul and Softmax — walk through them
# ops between the score MatMul and the Softmax that merely rescale/reshape/mask the
# logits — walk through them. "Adding" covers the attention mask add
# (softmax(QK^T + mask)); its mask input dead-ends at a constant so only the score
# branch reaches a MatMul.
TRANSPARENT = {"MultiplyConstant", "Requantize", "Reshape", "Transpose",
               "HeaderView", "Identity", "Cast", "Adding"}


def _lt(op):
    return str(getattr(op, "layertype", "")).split(".")[-1]


def find_score_matmuls(mblt_path):
    md, _ = load_mblt_model(mblt_path)
    names = []
    for sg in md.subgraphs:
        ops = sg.operators
        # activation id -> producing op
        producer = {}
        for o in ops:
            for a in getattr(o, "outputs", []):
                producer[a] = o
        for o in ops:
            if _lt(o) != "Softmax":
                continue
            # BFS upstream from the softmax through transparent ops, following EVERY
            # input (an add has two: the scores branch reaches a MatMul, the mask
            # branch dead-ends at a constant). Collect every MatMul so reached.
            stack = list(getattr(o, "inputs", []))
            seen = set()
            while stack:
                prod = producer.get(stack.pop())
                if prod is None or id(prod) in seen:
                    continue
                seen.add(id(prod))
                t = _lt(prod)
                if t == "MatMul":
                    names.append(prod.name)
                elif t in TRANSPARENT:
                    stack.extend(getattr(prod, "inputs", []))
                # else: non-transparent, non-matmul -> stop this branch
    return sorted(set(names))


if __name__ == "__main__":
    import sys
    found = find_score_matmuls(sys.argv[1])
    print("score MatMul(s) to force 16-bit:")
    for n in found:
        print("  ", n)
