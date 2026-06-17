"""
PE-Core-L14-336 vision encoder -> single-input/single-output MXQ (qbcompiler torch backend).
CLI: python -m pe_npu.compile

pe_model.apply_pe_patches()로 RoPE 상수화 / einops 제거 / attn_pool 분해 / abs_posemb 상수화
패치를 적용한 PE 모델을 qbcompiler가 그래프로 추적·INT8 양자화·컴파일한다.

검증된 사용 (hybrid 추론용 trunk MXQ, demo cos 0.997):
    python -m pe_npu.compile --mode compile --save ./out/pe_feat.mxq --feat-only \
      --calib-data-path ./calib_coco_hwc --calib-output 1 --device gpu

옵션:
  --feat-only : attn_pool 전 forward_features(1,577,1024)만 컴파일 (hybrid trunk, 권장)
  --pool-only : pool 후 proj 전까지
  (둘 다 없으면 full VisionWrapper -> 1024 임베딩)
  --mode parse : 컴파일 없이 operator 목록/타입만 확인 (16bit override 이름 추출용)
  --calib-data-path : calib npy 디렉토리(npy_files.txt) 또는 txt. 미지정 시 random calib
  --calib-output 0/1, --calib-method, --use-et, --act16/--weight16/--act16-exclude
"""
from __future__ import annotations

import argparse
import os

import torch

from .pe_model import IMAGE_SIZE, load_pe

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SAVE = os.path.join(HERE, "out", "pe_feat.mxq")


def _build_feed_dict(wrapper):
    """qbcompiler torch backend용 feed_dict 생성 (dummy (1,3,336,336))."""
    from qbcompiler.model_dict.parser.backend.torch.util import wrap_tensor
    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.no_grad():
        out = wrapper(dummy)
    return {"image": wrap_tensor("image", dummy)}, dummy, out


def _parse_operators(wrapper, fd):
    """ModelParser로 sg0를 만들고 (layertype, name) 리스트 반환."""
    from qbcompiler.model_dict.parser.parser import ModelParser
    parser = ModelParser(
        model=wrapper, backend="torch", target_device="aries2",
        yolo_decode_include=True,
    )
    parser.cfg.allocate_to_devices = True
    parser.cfg.split_supported_concat = True
    parser.parse(feed_dict=fd, save_subgraph_type=1, debug=False)
    md, _ = parser.get_md_wd(body_only=False)
    sg0 = md.subgraphs[0]
    ops = []
    for op in sg0.operators:
        lt = op.layertype.name if hasattr(op.layertype, "name") else str(op.layertype)
        ops.append((lt, op.name))
    return ops


def _select_names(ops, spec):
    """spec('all'|'none'|'sub1,sub2')에 맞는 operator 이름 리스트 반환."""
    if spec is None or spec == "none" or spec == "":
        return []
    if spec == "all":
        return [name for _, name in ops]
    subs = [s.strip().lower() for s in spec.split(",") if s.strip()]
    out = []
    for lt, name in ops:
        hay = (name + " " + lt).lower()
        if any(s in hay for s in subs):
            out.append(name)
    return out


def compile_pe(mode: str = "feat", save_path: str = DEFAULT_SAVE,
               calib_path: str = None, calib_output: int = 1, calib_method: int = 1,
               device: str = "cpu", use_et: bool = False,
               act16: str = None, weight16: str = None, act16_exclude: str = None,
               model_name: str = "PE-Core-L14-336", inference_scheme: str = "single",
               bit4: float = 0.0):
    """PE vision encoder를 MXQ로 컴파일.

    mode        : 'feat'(trunk, hybrid 권장) | 'pool' | 'full'.
    save_path   : 출력 .mxq 경로.
    calib_path  : calib npy 디렉토리(HWC, npy_files.txt 포함) 또는 txt. None이면 random calib.
    calib_output: 0=activation per-layer, 1=per-channel(정밀↑).
    device      : 'cpu' | 'gpu'.
    """
    wrap_mode = {"feat": "feat", "pool": "pool", "full": "full"}[mode]
    wrapper = load_pe(model_name=model_name, mode=wrap_mode, patch=True)

    fd, dummy, out = _build_feed_dict(wrapper)
    print(f"[sanity] {mode} output: {tuple(out.shape)}")

    from qbcompiler import mxq_compile
    common = dict(
        model=wrapper, backend="torch", feed_dict=fd, save_path=save_path,
        target_device="aries2", yolo_decode_include=True,
        inference_scheme=inference_scheme, device=device,
    )

    extra = {}
    if act16 or weight16:
        from qbcompiler import BitConfig
        print("[16bit] re-parse to enumerate operator names")
        ops = _parse_operators(wrapper, fd)
        act_names = _select_names(ops, act16)
        w_names = _select_names(ops, weight16)
        if act16_exclude:
            ex = [s.strip().lower() for s in act16_exclude.split(",") if s.strip()]
            before = len(act_names)
            act_names = [n for n in act_names if not any(s in n.lower() for s in ex)]
            print(f"    act16 exclude: {before} -> {len(act_names)}")
        print(f"    total ops={len(ops)}  act16={len(act_names)}  weight16={len(w_names)}")
        extra["bit_config"] = BitConfig(
            layer_overrides=BitConfig.LayerOverrides(
                activation_16bits=act_names, weight_16bits=w_names,
            )
        )

    if bit4 and bit4 > 0:
        # mixed-precision weight 양자화: weight의 bit4 비율을 4비트로, 나머지는 8비트.
        # (docs 미기재 실험적 기능. importance threshold로 중요 레이어는 자동 고비트 유지)
        from qbcompiler import BitConfig
        T = BitConfig.Transformer
        MP = T.model_fields["mixed_precision"].annotation
        extra["bit_config"] = BitConfig(
            transformer=T(mixed_precision=MP(apply=True, bit_4=bit4, bit_8=1.0 - bit4))
        )
        print(f"[mixed-precision] weight bit_4={bit4}  bit_8={1.0 - bit4}")

    if calib_path:
        from qbcompiler import CalibrationConfig
        calib = calib_path
        if os.path.isdir(calib) and os.path.exists(os.path.join(calib, "npy_files.txt")):
            calib = os.path.join(calib, "npy_files.txt")
        cc = CalibrationConfig(method=calib_method, output=calib_output, mode=1,
                               max_percentile=CalibrationConfig.MaxPercentile(percentile=0.9999, topk_ratio=0.01))
        print(f"[compile] calib={calib} (method={calib_method}, output={calib_output}, et={use_et})")
        if use_et:
            from qbcompiler import EquivalentTransformationConfig as ET
            extra["equivalent_transformation_config"] = ET(
                norm_conv=ET.NormConv(apply=True), qk=ET.Qk(apply=True),
                ud=ET.Ud(apply=True), vo=ET.Vo(apply=True),
            )
        mxq_compile(**common, calib_data_path=calib, calibration_config=cc, **extra)
    else:
        print("[compile] random calib")
        mxq_compile(**common, use_random_calib=True, **extra)
    print(f"[OK] saved {save_path}")
    return save_path


def parse_pe(mode: str = "feat", dump_names: str = None, model_name: str = "PE-Core-L14-336"):
    """컴파일 없이 operator 목록/타입만 확인 (parse 모드)."""
    wrapper = load_pe(model_name=model_name, mode=mode, patch=True)
    fd, dummy, out = _build_feed_dict(wrapper)
    print(f"[sanity] {mode} output: {tuple(out.shape)}")
    from qbcompiler.model_dict.parser.parser import ModelParser
    parser = ModelParser(model=wrapper, backend="torch", target_device="aries2",
                         yolo_decode_include=True)
    parser.cfg.allocate_to_devices = True
    parser.cfg.split_supported_concat = True
    parser.parse(feed_dict=fd, save_subgraph_type=1, debug=False)
    md, wd = parser.get_md_wd(body_only=False)
    sg0 = md.subgraphs[0]
    print(f"[OK] parse finished. subgraphs={len(md.subgraphs)}  sg0 ops={len(sg0.operators)}")
    print(f"    sg0 inputs={sg0.inputs} outputs={sg0.outputs}")
    lines = []
    for op in sg0.operators:
        lt = op.layertype.name if hasattr(op.layertype, "name") else str(op.layertype)
        lines.append(f"{lt}\t{op.name}")
    if dump_names:
        with open(dump_names, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[OK] dumped {len(lines)} operator names -> {dump_names}")
    else:
        from collections import Counter
        cnt = Counter(l.split("\t")[0] for l in lines)
        print("    layertype counts:", dict(cnt))


def main():
    ap = argparse.ArgumentParser(description="PE vision encoder -> MXQ 컴파일")
    ap.add_argument("--mode", default="parse", choices=["parse", "compile"])
    ap.add_argument("--save", default=DEFAULT_SAVE)
    ap.add_argument("--calib-data-path", default=None,
                    help="calib npy 디렉토리/txt (HWC). 미지정 시 random calib")
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--calib-output", type=int, default=1, choices=[0, 1])
    ap.add_argument("--calib-method", type=int, default=1, choices=[0, 1, 2, 3])
    ap.add_argument("--use-et", action="store_true")
    ap.add_argument("--act16", default=None)
    ap.add_argument("--weight16", default=None)
    ap.add_argument("--act16-exclude", default=None)
    ap.add_argument("--dump-names", default=None, help="parse 모드: operator 목록 덤프 파일")
    ap.add_argument("--scheme", default="single", choices=["single", "multi", "global4", "global8"],
                    help="코어 모드(컴파일시 고정): single(기본)|multi(4-batch)|global4|global8(단건 latency↓)")
    ap.add_argument("--bit4", type=float, default=0.0,
                    help="mixed-precision: weight의 이 비율을 4비트로(0~1, 실험적). 예 0.5")
    ap.add_argument("--feat-only", action="store_true",
                    help="attn_pool 전 forward_features(1,577,1024) (hybrid trunk, 권장)")
    ap.add_argument("--pool-only", action="store_true", help="pool 후 proj 전 출력")
    args = ap.parse_args()

    wmode = "feat" if args.feat_only else ("pool" if args.pool_only else "full")

    if args.mode == "parse":
        parse_pe(mode=wmode, dump_names=args.dump_names)
    else:
        compile_pe(mode=wmode, save_path=args.save, calib_path=args.calib_data_path,
                   calib_output=args.calib_output, calib_method=args.calib_method,
                   device=args.device, use_et=args.use_et,
                   act16=args.act16, weight16=args.weight16, act16_exclude=args.act16_exclude,
                   inference_scheme=args.scheme, bit4=args.bit4)


if __name__ == "__main__":
    main()
