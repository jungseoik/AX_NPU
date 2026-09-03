"""
PE-Core-L14-336 vision encoder -> single-input/single-output MXQ (qbcompiler torch backend).
CLI: python -m pe_npu.compile

pe_model.apply_pe_patches()로 RoPE 상수화 / einops 제거 / attn_pool 분해 / abs_posemb 상수화
패치를 적용한 PE 모델을 qbcompiler가 그래프로 추적·INT8 양자화·컴파일한다.

권장 사용 (full NPU, image->embedding 전부 NPU, 원본 대비 cos 0.99):
    python -m pe_npu.compile --mode compile --save ./out/pe_full.mxq \
      --calib-data-path ./calib_coco_hwc --calib-output 1 --device cpu   # 기본 cpu(GPU 없어도 OK), GPU면 gpu
    # 기본 mode=full이고 full에는 --qk16(attn_pool QK^T 16bit)이 자동 적용된다(없으면 attn_pool이 INT8서 붕괴, cos 0.46).

레거시 (hybrid trunk MXQ, +CPU pool head, cos 0.997):
    python -m pe_npu.compile --mode compile --save ./out/pe_feat.mxq --feat-only --calib-data-path ./calib_coco_hwc

옵션:
  (기본) full VisionWrapper -> 1024 임베딩. full엔 QK^T 16bit 자동(끄려면 --no-qk16, 실험용).
  --feat-only : attn_pool 전 forward_features(1,577,1024)만 (레거시 hybrid trunk)
  --pool-only : pool 후 proj 전까지 (진단용)
  --mode parse : 컴파일 없이 operator 목록/타입만 확인
  --calib-data-path : calib npy 디렉토리(npy_files.txt) 또는 txt. 미지정 시 random calib
  --calib-output 0/1, --calib-method, --calib-stats-save/--calib-stats-load,
  --use-et, --act16/--weight16/--act16-exclude
  --target-device : 컴파일 타겟 칩. 미지정 시 qbcompiler 버전으로 자동 판별
                    (1.1.x→aries2 / 1.2.x→aries-rb, 둘 다 Aries2 하드웨어)
"""
from __future__ import annotations

import argparse
import os
import time

import torch

from .pe_model import IMAGE_SIZE, load_pe
from .target_device import add_argument as _add_target_device_arg, resolve_target_device

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SAVE = os.path.join(HERE, "out", "pe_full.mxq")


def _build_feed_dict(wrapper):
    """qbcompiler torch backend용 feed_dict 생성 (dummy (1,3,336,336))."""
    from qbcompiler.model_dict.parser.backend.torch.util import wrap_tensor
    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.no_grad():
        out = wrapper(dummy)
    return {"image": wrap_tensor("image", dummy)}, dummy, out


def _parse_operators(wrapper, fd, target_device=None):
    """ModelParser로 sg0를 만들고 (layertype, name) 리스트 반환."""
    from qbcompiler.model_dict.parser.parser import ModelParser
    parser = ModelParser(
        model=wrapper, backend="torch",
        target_device=resolve_target_device(target_device, verbose=False),
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


def _split_exact_names(spec):
    """쉼표로 구분된 정확한 mblt operator 이름을 정규화한다."""
    if not spec:
        return []
    return sorted({name.strip() for name in spec.split(",") if name.strip()})


def _build_bit_config(quant=None, activation_16bits=None, weight_16bits=None,
                      bit4: float = 0.0):
    """정식 transformer bit preset과 layer override를 하나의 BitConfig로 만든다."""
    from qbcompiler import BitConfig

    if quant and bit4 and bit4 > 0:
        raise ValueError("--quant and legacy --bit4 cannot be combined")

    activation_16bits = sorted(set(activation_16bits or []))
    weight_16bits = sorted(set(weight_16bits or []))
    kwargs = {}
    T = BitConfig.Transformer

    if quant:
        activation_output = 8 if quant == "w4a8" else 16
        weight_default = 8 if quant == "w8a16" else 4
        kwargs["transformer"] = T(
            activation=T.Activation(
                query=8, key=8, value=8, head=8,
                output=activation_output, ffn=activation_output,
            ),
            weight=T.Weight(
                query=weight_default,
                key=weight_default,
                value=8,
                output=weight_default,
                ffn=weight_default,
                head=weight_default,
            ),
        )
    elif bit4 and bit4 > 0:
        MP = T.model_fields["mixed_precision"].annotation
        kwargs["transformer"] = T(
            mixed_precision=MP(apply=True, bit_4=bit4, bit_8=1.0 - bit4)
        )

    if activation_16bits or weight_16bits:
        kwargs["layer_overrides"] = BitConfig.LayerOverrides(
            activation_16bits=activation_16bits,
            weight_16bits=weight_16bits,
        )

    return BitConfig(**kwargs) if kwargs else None


def _build_optimizer_configs(optq=False, search_weight_scale=False):
    """Mobilint 예제와 동일한 OPTQ/SearchWeightScale 설정을 만든다."""
    configs = {}
    if optq:
        from qbcompiler import OptqConfig
        configs["optq_config"] = OptqConfig(
            apply=True,
            attributes=OptqConfig.Attributes(
                actOrder=True, blockSize=128, percDamp=0.01,
            ),
        )
    if search_weight_scale:
        from qbcompiler import SearchWeightScaleConfig
        configs["search_weight_scale_config"] = SearchWeightScaleConfig(
            apply=True,
            transformer=SearchWeightScaleConfig.Transformer(
                query=True, key=True, value=True, out=True, ffn=True,
            ),
        )
    return configs


def _build_calibration_config(method: int, output: int,
                              stats_save: str = None, stats_load: str = None):
    """Build calibration settings, optionally saving or reusing raw statistics."""
    from qbcompiler import CalibrationConfig

    if stats_save and stats_load:
        raise ValueError("--calib-stats-save and --calib-stats-load cannot be combined")
    kwargs = dict(
        method=method,
        output=output,
        mode=1,
        max_percentile=CalibrationConfig.MaxPercentile(
            percentile=0.9999,
            topk_ratio=0.01,
        ),
    )
    if stats_save or stats_load:
        kwargs["statistics"] = CalibrationConfig.Statistics(
            apply=True,
            save_path=stats_save or "",
            load_path=stats_load or "",
            # A load config compiles one MXQ per candidate percentile.  Keep
            # exactly the percentile already used by max_percentile above.
            percentiles=[0.9999],
            percentile_index=0,
        )
    return CalibrationConfig(**kwargs)


def _validate_calibration_strategy(optq: bool, stats_save: str = None,
                                   stats_load: str = None,
                                   calib_path: str = None) -> None:
    """Reject calibration-cache combinations that would be ignored or incomplete."""
    if optq and stats_load:
        raise ValueError(
            "--calib-stats-load cannot be combined with --optq: "
            "qbcompiler 1.1.2 statistics omit the OPTQ Hessian; "
            "run full calibration instead"
        )
    if (stats_save or stats_load) and not calib_path:
        raise ValueError(
            "--calib-stats-save/--calib-stats-load require --calib-data-path; "
            "statistics are unavailable with random calibration"
        )


def _stats_output_candidate(save_path: str) -> str:
    root, extension = os.path.splitext(save_path)
    return f"{root}_0.9999{extension}"


def _normalize_stats_output(save_path: str, stats_load: bool) -> str:
    """Normalize qbcompiler's percentile-suffixed stats-load output path."""
    if not stats_load:
        return save_path
    candidate = _stats_output_candidate(save_path)
    if os.path.isfile(candidate):
        os.replace(candidate, save_path)
        print(f"[calib-stats] normalized output {candidate} -> {save_path}")
    return save_path


def _detect_score_matmuls(wrapper, fd, target_device=None):
    """attention score MatMul(QK^T) 노드 이름을 그래프 구조로 자동 탐지.

    mblt_compile로 한 번 파싱한 .mblt에서 find_score_matmuls로 MatMul->...->Softmax
    경로를 찾는다. 레이어 이름이 모델/패키징마다 달라도 동작. (출처: Mobilint 기술지원)
    """
    import tempfile
    from qbcompiler import mblt_compile
    from .find_score_matmul import find_score_matmuls
    with tempfile.NamedTemporaryFile(suffix=".mblt", delete=False) as tf:
        mblt_path = tf.name
    mblt_compile(model=wrapper, mblt_save_path=mblt_path, backend="torch",
                 feed_dict=fd,
                 target_device=resolve_target_device(target_device, verbose=False))
    names = find_score_matmuls(mblt_path)
    os.remove(mblt_path)
    return names


def compile_pe(mode: str = "feat", save_path: str = DEFAULT_SAVE,
               calib_path: str = None, calib_output: int = 1, calib_method: int = 1,
               calib_stats_save: str = None, calib_stats_load: str = None,
               device: str = "cpu", use_et: bool = False,
               act16: str = None, weight16: str = None, act16_exclude: str = None,
               model_name: str = "PE-Core-L14-336", inference_scheme: str = "single",
               bit4: float = 0.0, qk16: bool = False, quant: str = None,
               a16: str = None, optq: bool = False,
               search_weight_scale: bool = False,
               target_device: str = None):
    """PE vision encoder를 MXQ로 컴파일.

    mode        : 'feat'(trunk) | 'pool' | 'full'(trunk+attn_pool, qk16와 함께 권장).
    save_path   : 출력 .mxq 경로.
    calib_path  : calib npy 디렉토리(HWC, npy_files.txt 포함) 또는 txt. None이면 random calib.
    calib_output: 0=activation per-layer, 1=per-channel(정밀↑).
    device      : 'cpu' | 'gpu'.
    qk16        : True면 attention score MatMul(QK^T)을 16bit로 자동 override.
                  attn_pool의 QK^T outlier로 인한 INT8 붕괴(cos 0.46)를 복구 → full 모델도
                  cos 0.99. (Mobilint 해결책, full 모드에서 권장)
    """
    _validate_calibration_strategy(
        optq=optq,
        stats_save=calib_stats_save,
        stats_load=calib_stats_load,
        calib_path=calib_path,
    )
    wrap_mode = {"feat": "feat", "pool": "pool", "full": "full"}[mode]
    wrapper = load_pe(model_name=model_name, mode=wrap_mode, patch=True)

    fd, dummy, out = _build_feed_dict(wrapper)
    print(f"[sanity] {mode} output: {tuple(out.shape)}")

    from qbcompiler import mxq_compile
    common = dict(
        model=wrapper, backend="torch", feed_dict=fd, save_path=save_path,
        target_device=resolve_target_device(target_device), yolo_decode_include=True,
        inference_scheme=inference_scheme, device=device,
    )

    extra = _build_optimizer_configs(
        optq=optq, search_weight_scale=search_weight_scale,
    )
    qk_names = []
    if qk16:
        qk_names = _detect_score_matmuls(wrapper, fd, target_device)
        print(f"[qk16] 16bit 대상 score MatMul {len(qk_names)}개: {qk_names}")
        if not qk_names:
            raise RuntimeError("qk16=True인데 score MatMul(MatMul->Softmax)을 못 찾음")
    act_names = _split_exact_names(a16)
    w_names = []
    if act16 or weight16:
        print("[16bit] re-parse to enumerate operator names")
        ops = _parse_operators(wrapper, fd, target_device)
        act_names.extend(_select_names(ops, act16))
        w_names = _select_names(ops, weight16)
        if act16_exclude:
            ex = [s.strip().lower() for s in act16_exclude.split(",") if s.strip()]
            before = len(act_names)
            act_names = [n for n in act_names if not any(s in n.lower() for s in ex)]
            print(f"    act16 exclude: {before} -> {len(act_names)}")
        print(f"    total ops={len(ops)}  act16={len(act_names)}  weight16={len(w_names)}")

    # exact --a16 + substring --act16 + 자동 qk16을 반드시 같은 override에 합친다.
    act_names = sorted(set(act_names) | set(qk_names))
    bit_config = _build_bit_config(
        quant=quant,
        activation_16bits=act_names,
        weight_16bits=w_names,
        bit4=bit4,
    )
    if bit_config is not None:
        extra["bit_config"] = bit_config
        print(f"[quant] preset={quant or 'legacy/default'}  "
              f"activation16={len(act_names)}  weight16={len(set(w_names))}")
    if bit4 and bit4 > 0:
        # 비공식 mixed_precision 경로는 기존 실측에서 no-op. 비교 재현용으로만 유지한다.
        print(f"[mixed-precision legacy/no-op] weight bit_4={bit4}  bit_8={1.0 - bit4}")

    if calib_path:
        calib = calib_path
        if os.path.isdir(calib) and os.path.exists(os.path.join(calib, "npy_files.txt")):
            calib = os.path.join(calib, "npy_files.txt")
        cc = _build_calibration_config(
            method=calib_method,
            output=calib_output,
            stats_save=calib_stats_save,
            stats_load=calib_stats_load,
        )
        print(f"[compile] calib={calib} (method={calib_method}, output={calib_output}, et={use_et})")
        if calib_stats_save:
            print(f"[calib-stats] save={calib_stats_save} percentile=0.9999")
        if calib_stats_load:
            print(f"[calib-stats] load={calib_stats_load} percentile=0.9999")
            stale_candidate = _stats_output_candidate(save_path)
            if os.path.isfile(stale_candidate):
                os.remove(stale_candidate)
        if use_et:
            from qbcompiler import EquivalentTransformationConfig as ET
            extra["equivalent_transformation_config"] = ET(
                norm_conv=ET.NormConv(apply=True), qk=ET.Qk(apply=True),
                ud=ET.Ud(apply=True), vo=ET.Vo(apply=True),
            )
        started = time.perf_counter()
        mxq_compile(**common, calib_data_path=calib, calibration_config=cc, **extra)
    else:
        print("[compile] random calib")
        started = time.perf_counter()
        mxq_compile(**common, use_random_calib=True, **extra)
    elapsed = time.perf_counter() - started
    actual_save_path = _normalize_stats_output(
        save_path,
        stats_load=bool(calib_stats_load),
    )
    size_bytes = os.path.getsize(actual_save_path)
    print(f"[OK] saved {actual_save_path}  size_bytes={size_bytes}  compile_seconds={elapsed:.1f}")
    return actual_save_path


def parse_pe(mode: str = "feat", dump_names: str = None, model_name: str = "PE-Core-L14-336",
             target_device: str = None):
    """컴파일 없이 operator 목록/타입만 확인 (parse 모드)."""
    wrapper = load_pe(model_name=model_name, mode=mode, patch=True)
    fd, dummy, out = _build_feed_dict(wrapper)
    print(f"[sanity] {mode} output: {tuple(out.shape)}")
    from qbcompiler.model_dict.parser.parser import ModelParser
    parser = ModelParser(model=wrapper, backend="torch",
                         target_device=resolve_target_device(target_device),
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
    stats_group = ap.add_mutually_exclusive_group()
    stats_group.add_argument(
        "--calib-stats-save",
        default=None,
        help="200장 calibration 통계를 저장해 동일 모델의 후속 compile에서 재사용",
    )
    stats_group.add_argument(
        "--calib-stats-load",
        default=None,
        help="저장된 calibration 통계를 로드해 activation calibration forward를 생략 "
             "(qbcompiler 1.1.2에서는 OPTQ Hessian이 없어 --optq와 병용 불가)",
    )
    ap.add_argument("--use-et", action="store_true")
    ap.add_argument("--act16", default=None)
    ap.add_argument("--weight16", default=None)
    ap.add_argument("--act16-exclude", default=None)
    ap.add_argument("--dump-names", default=None, help="parse 모드: operator 목록 덤프 파일")
    ap.add_argument("--scheme", default="single", choices=["single", "multi", "global4", "global8"],
                    help="코어 모드(컴파일시 고정): single(기본)|multi(4-batch)|global4|global8(단건 latency↓)")
    ap.add_argument("--bit4", type=float, default=0.0,
                    help="레거시 mixed_precision 비율(기존 실측 no-op, 비교 재현용). --quant와 조합 불가")
    ap.add_argument("--quant", choices=["w8a16", "w4a16", "w4a8"], default=None,
                    help="정식 transformer per-component bit preset")
    ap.add_argument("--optq", action="store_true",
                    help="Mobilint OPTQ(actOrder=True, blockSize=128, percDamp=0.01)")
    ap.add_argument("--search-weight-scale", action="store_true",
                    help="transformer query/key/value/out/ffn weight scale 탐색")
    ap.add_argument("--a16", default=None,
                    help="추가 activation 16bit 정확한 mblt op 이름(쉼표 구분); qk16과 합집합")
    ap.add_argument("--feat-only", action="store_true",
                    help="attn_pool 전 forward_features(1,577,1024)만 (레거시 hybrid trunk)")
    ap.add_argument("--pool-only", action="store_true", help="pool 후 proj 전 출력 (진단용)")
    ap.add_argument("--qk16", action="store_true",
                    help="attention score MatMul(QK^T) 16bit override. full 모드엔 자동 적용되므로 명시 불필요 "
                         "(호환용 유지). feat/pool에 쓰면 그 그래프의 score matmul도 16bit")
    ap.add_argument("--no-qk16", action="store_true",
                    help="full 모드에서 QK^T 16bit override 비활성화(실험용). 끄면 attn_pool이 INT8서 붕괴(cos 0.46)")
    _add_target_device_arg(ap)
    args = ap.parse_args()

    if args.quant and args.bit4 > 0:
        ap.error("--quant and legacy --bit4 cannot be combined")
    try:
        _validate_calibration_strategy(
            optq=args.optq,
            stats_save=args.calib_stats_save,
            stats_load=args.calib_stats_load,
            calib_path=args.calib_data_path,
        )
    except ValueError as exc:
        ap.error(str(exc))

    wmode = "feat" if args.feat_only else ("pool" if args.pool_only else "full")
    # full 모드는 QK^T 16bit가 필수(없으면 attn_pool 붕괴) → 기본 자동 적용. feat/pool(레거시·진단)은 명시(--qk16) 시만.
    qk16 = args.qk16 or ((wmode == "full") and (not args.no_qk16))
    if wmode == "full" and not qk16:
        print("[warn] full 모드 + --no-qk16: attn_pool이 INT8서 붕괴해 cos~0.46이 됩니다(실험용만).")

    if args.mode == "parse":
        parse_pe(mode=wmode, dump_names=args.dump_names,
                 target_device=args.target_device)
    else:
        compile_pe(mode=wmode, save_path=args.save, calib_path=args.calib_data_path,
                   calib_output=args.calib_output, calib_method=args.calib_method,
                   calib_stats_save=args.calib_stats_save,
                   calib_stats_load=args.calib_stats_load,
                   device=args.device, use_et=args.use_et,
                   act16=args.act16, weight16=args.weight16, act16_exclude=args.act16_exclude,
                   inference_scheme=args.scheme, bit4=args.bit4, qk16=qk16,
                   quant=args.quant, a16=args.a16, optq=args.optq,
                   search_weight_scale=args.search_weight_scale,
                   target_device=args.target_device)


if __name__ == "__main__":
    main()
