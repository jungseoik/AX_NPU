#!/usr/bin/env python3
"""NPU 가 고른 토큰과 yes/no 로짓을 찍는다 — 로짓이 죽었는지 보려는 진단."""
import argparse, os, sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--images", nargs="+", required=True)
    p.add_argument("--card", type=int, default=7)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--prompt", default="Is this image a dog? Answer with only 'yes' or 'no'. Do not output anything else.")
    a = p.parse_args()
    sys.path.insert(0, os.environ.get("NPU_PE_2STAGE_VENDOR_SRC", "/repro/src"))
    sys.path.insert(0, str(Path(__file__).parent))
    import artifact_order
    artifact_order.install(a.model)
    from pia_vlm.runtime import Classifier
    clf = Classifier(a.model, device=a.card, threads=8, prompt=a.prompt,
                     vision_mode="global4x2", text_mode="global8",
                     max_patches=4096, min_pixels=a.size * a.size)
    imgs = [Path(f).read_bytes() for f in a.images]
    out = clf.burst(imgs, (a.size, a.size), min(len(imgs), clf.compiled_batch), 128)
    tok = clf.processor.tokenizer
    print(f"  {'img':>4} {'token_id':>9} {'label':<16} {'yes_logit':>10} {'no_logit':>10}")
    for r in sorted(out["requests"], key=lambda z: z["request"]):
        print(f"  {r['request']:>4} {r['token_id']:>9} {r['label']!r:<16} "
              f"{r['yes_logit']:>10.3f} {r['no_logit']:>10.3f}")
    clf.close()
    print("=== DUMP_DONE ===")


if __name__ == "__main__":
    main()
