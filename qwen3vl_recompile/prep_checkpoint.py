#!/usr/bin/env python3
"""⓪ 학습 체크포인트를 컴파일 환경(transformers 4.57.1)에 맞게 정규화

파인튜닝 도구(unsloth 등)가 최신 transformers 로 저장하면 config 스키마가 달라
컴파일 컨테이너(4.57.1)가 못 읽는다. 또 프로세서 파일이 빠져 있기도 하다.
원본을 건드리지 않고 **정규화 사본**을 만든다.

실제로 겪은 차이 (AX_NPU_VL3_2B, unsloth 2026.9.4 / transformers 5.5.0):
  · text_config.rope_parameters  ← 5.x 신규. 4.57 은 rope_scaling + rope_theta 를 본다
  · preprocessor_config.json 없음 ← 이미지 프로세서 설정. 스톡에서 보충
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path

# 4.57 이 모르는 키 (남아 있어도 보통 무해하지만 혼란 방지)
DROP = ("unsloth_fixed", "unsloth_version", "model_name")
# 프로세서 동작에 필요한데 학습본에 없을 수 있는 파일 — 스톡에서 보충
FROM_STOCK = ("preprocessor_config.json", "video_preprocessor_config.json",
              "vocab.json", "merges.txt")


def normalize(src: Path, stock: Path, out: Path) -> dict:
    if out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True)
    report = {"copied": [], "from_stock": [], "config_fixes": []}
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(f, out / f.name); report["copied"].append(f.name)
    for n in FROM_STOCK:
        if not (out / n).exists() and (stock / n).exists():
            shutil.copy2(stock / n, out / n); report["from_stock"].append(n)

    cfg = json.loads((out / "config.json").read_text())
    tc = cfg.get("text_config", {})
    # rope_parameters(5.x) → rope_scaling + rope_theta(4.57)
    rp = tc.pop("rope_parameters", None)
    if rp:
        theta = rp.pop("rope_theta", None)
        if theta is not None and tc.get("rope_theta") is None:
            tc["rope_theta"] = theta; report["config_fixes"].append("text.rope_theta")
        if tc.get("rope_scaling") is None and rp:
            tc["rope_scaling"] = rp; report["config_fixes"].append("text.rope_scaling")
        report["config_fixes"].append("text.rope_parameters 제거")
    for k in DROP:
        if cfg.pop(k, None) is not None: report["config_fixes"].append(f"{k} 제거")
    # export 는 캐시를 쓰지 않는다
    tc["use_cache"] = False
    cfg["text_config"] = tc
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True, help="학습 체크포인트")
    p.add_argument("--stock", type=Path, required=True, help="스톡 체크포인트(보충용)")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    r = normalize(a.src, a.stock, a.out)
    print(f"  복사 {len(r['copied'])}개")
    if r["from_stock"]:   print(f"  스톡에서 보충: {r['from_stock']}")
    if r["config_fixes"]: print(f"  config 수정  : {r['config_fixes']}")
    # 검증
    import transformers
    from transformers import AutoProcessor, AutoConfig
    c = AutoConfig.from_pretrained(str(a.out))
    AutoProcessor.from_pretrained(str(a.out))
    print(f"  ✅ transformers {transformers.__version__} 로 로드 확인")
    print(f"     hidden={c.text_config.hidden_size} vocab={c.text_config.vocab_size} "
          f"deepstack={c.vision_config.deepstack_visual_indexes}")


if __name__ == "__main__":
    main()
