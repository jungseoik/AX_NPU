#!/usr/bin/env python3
"""④ NPU A/B — 기준 모델 vs 새로 만든 모델의 출력·지연 비교

판정은 **sha256 이 아니라 NPU 실측**으로 한다. 벤더가 두 군데서 못 박았다:
  TOOLCHAIN.json : "identical millisecond timings or identical MXQ bytes are not promised"
  VALIDATION.md  : "같은 소스와 옵션이 MXQ의 바이트 단위 재현이나 예제 정답 일치까지
                    보장하지는 않았습니다"

합격 기준:
  · 라벨이 **글자 단위로 동일**      ← 가장 중요
  · 지연 차이가 측정 편차(~1%) 안
  · 배치 정답률 동일
"""
from __future__ import annotations
import argparse, json, os, statistics, sys, time
from pathlib import Path


def bench(md: str, card: int, imgs, prompts, sizes, reps: int):
    sys.path.insert(0, os.environ.get("NPU_PE_2STAGE_VENDOR_SRC", "/repro/src"))
    from pia_vlm.runtime import Classifier
    r = {}
    clf = Classifier(md, device=card, threads=8, prompt=prompts[0],
                     vision_mode="global4x2", text_mode="global8",
                     max_patches=4096, min_pixels=min(sizes[0] * sizes[1], 65536))
    r["compiled_batch"] = clf.compiled_batch
    clf.burst([imgs[0]], sizes, 1, 128)
    r["single_ms"] = statistics.median(
        [(lambda t: (clf.burst([imgs[0]], sizes, 1, 128), (time.perf_counter() - t) * 1000)[1])(time.perf_counter())
         for _ in range(reps)])
    # 프롬프트 교차 라벨
    labels = []
    for pr in prompts:
        clf.prompt = clf.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": pr}]}],
            tokenize=False, add_generation_prompt=True)
        labels += [clf.burst([im], sizes, 1, 128)["requests"][0]["label"].strip() for im in imgs]
    r["labels"] = labels
    clf.prompt = clf.processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompts[0]}]}],
        tokenize=False, add_generation_prompt=True)
    for n in (8, 20):
        pl = [imgs[i % len(imgs)] for i in range(n)]
        r[f"batch{n}_ms"] = statistics.median(
            [(lambda t: (clf.burst(pl, sizes, min(n, clf.compiled_batch), 128),
                         (time.perf_counter() - t) * 1000)[1])(time.perf_counter()) for _ in range(reps)])
        o = clf.burst(pl, sizes, min(n, clf.compiled_batch), 128)
        r[f"batch{n}_labels"] = [x["label"].strip() for x in sorted(o["requests"], key=lambda z: z["request"])]
    clf.close()
    return r


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline", required=True, help="기준 모델 디렉토리 (기존 batch32)")
    p.add_argument("--candidate", required=True, help="새로 만든 모델 디렉토리")
    p.add_argument("--images", nargs="+", required=True)
    p.add_argument("--prompts", nargs="+", default=[
        "Is this image a dog? Answer with only 'yes' or 'no'. Do not output anything else.",
        "Is this image a cat? Answer with only 'yes' or 'no'. Do not output anything else."])
    p.add_argument("--card", type=int, default=0)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--out", type=Path)
    a = p.parse_args()

    imgs = [Path(f).read_bytes() for f in a.images]
    sizes = (a.size, a.size)
    res = {}
    for tag, md in [("baseline", a.baseline), ("candidate", a.candidate)]:
        print(f"\n########## {tag}  {md} ##########", flush=True)
        res[tag] = bench(md, a.card, imgs, a.prompts, sizes, a.reps)
        q = res[tag]
        print(f"  compiled_batch={q['compiled_batch']}  단건 {q['single_ms']:.1f}ms  "
              f"8건 {q['batch8_ms']:.1f}ms  20건 {q['batch20_ms']:.1f}ms", flush=True)

    b, c = res["baseline"], res["candidate"]
    print("\n" + "=" * 66)
    print(f"{'항목':<18}{'기준':>14}{'신규':>14}{'차이':>14}")
    print("-" * 66)
    for k, nm in [("single_ms", "단건 지연"), ("batch8_ms", "8건 지연"), ("batch20_ms", "20건 지연")]:
        d = (c[k] - b[k]) / b[k] * 100
        print(f"{nm:<18}{b[k]:12.1f}ms{c[k]:12.1f}ms{d:+13.1f}%")
    ok_all = True
    for k, nm in [("labels", "프롬프트 교차"), ("batch8_labels", "8건 라벨"), ("batch20_labels", "20건 라벨")]:
        same = b[k] == c[k]
        ok_all &= same
        n = len(b[k]); match = sum(x == y for x, y in zip(b[k], c[k]))
        print(f"{nm:<18}{'':>14}{match:>10}/{n:<3}{'✅ 완전동일' if same else '❌ 불일치':>14}")
    if not ok_all:
        for k in ("labels", "batch8_labels", "batch20_labels"):
            if b[k] != c[k]:
                print(f"\n  [{k}] 기준 : {b[k]}")
                print(f"  [{k}] 신규 : {c[k]}")
    print("\n  판정:", "✅ 동등" if ok_all else "❌ 출력이 다르다 — 원인 규명 필요")
    if a.out:
        a.out.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n")
    print("=== VERIFY_DONE ===", flush=True)


if __name__ == "__main__":
    main()
