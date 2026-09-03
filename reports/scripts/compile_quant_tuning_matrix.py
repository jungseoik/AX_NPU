"""PE-Core-L14-336 → MXQ 매트릭스 컴파일: quant(W8A16|W4A16|W4A8_L5A16) × tuning(none|sws|optq|sws_optq).

- 모든 조합에 QK^T 16bit override(full 모드 필수) + calib COCO 200장.
- 비트 배치는 Mobilint 예제(reports/inquiries/04_vit_quantization_speed/examples/) 준수:
    W8A16       act(q8,k8,v8,head8,out16,ffn16) / weight 전부 8   (기본 배포본과 동일)
    W4A16       act 동일 / weight(q4,k4,v8,out4,ffn4,head4)
    W4A8_L5A16  act 전부 8 + outlier 상위 5개 텐서만 16bit / weight W4와 동일
- W4A8의 5개 텐서 = 기존 프로파일링(NPU_pe_quant_schemes.md §W4A8) 상위 5개의 mblt 이름
  (parse --dump-names로 확인, torch 경로 유지됨).

컨테이너(/workspace=repo root) 안에서:
    python pe_npu/out/compile_matrix.py --quant W4A16 --tuning sws_optq --scheme single \
        --calib download/calib_coco_hwc --save pe_npu/out/xxx.mxq
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, "/workspace")

A16_TENSORS_W4A8 = [  # ratio=max/p99.9 상위 5 (L3/L12/L10 c_proj, L12/L10 gelu)
    "visual_transformer_resblocks_3_mlp_c_proj",
    "visual_transformer_resblocks_12_mlp_c_proj",
    "visual_transformer_resblocks_10_mlp_c_proj",
    "visual_transformer_resblocks_12_mlp_c_fc/reshape/gelu_0",
    "visual_transformer_resblocks_10_mlp_c_fc/reshape/gelu_0",
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", required=True, choices=["W8A16", "W4A16", "W4A8_L5A16"])
    ap.add_argument("--tuning", required=True, choices=["none", "sws", "optq", "sws_optq"])
    ap.add_argument("--scheme", required=True, choices=["single", "multi", "global4", "global8"])
    ap.add_argument("--calib", required=True)
    ap.add_argument("--save", required=True)
    ap.add_argument("--device", default="gpu", choices=["cpu", "gpu"])
    from pe_npu.target_device import add_argument as _add_target_device_arg
    _add_target_device_arg(ap)
    a = ap.parse_args()

    from pe_npu.compile import _build_feed_dict, _detect_score_matmuls
    from pe_npu.target_device import resolve_target_device
    from pe_npu.pe_model import load_pe
    from qbcompiler import mxq_compile, BitConfig, CalibrationConfig
    from qbcompiler.configs import OptqConfig, SearchWeightScaleConfig

    t0 = time.time()
    wrapper = load_pe(model_name="PE-Core-L14-336", mode="full", patch=True)
    fd, dummy, out = _build_feed_dict(wrapper)
    log(f"model load+sanity {time.time() - t0:.0f}s — output {tuple(out.shape)}")

    t = time.time()
    qk_names = _detect_score_matmuls(wrapper, fd, a.target_device)
    log(f"qk16 detect {time.time() - t:.0f}s — score MatMul {len(qk_names)}개")
    if not qk_names:
        raise SystemExit("score MatMul 탐지 실패 — full 모드에 qk16 없이는 진행 금지")

    T = BitConfig.Transformer
    act16_extra = []
    if a.quant == "W8A16":
        act = T.Activation(query=8, key=8, value=8, head=8, output=16, ffn=16)
        wgt = T.Weight(query=8, key=8, value=8, output=8, ffn=8, head=8)
    elif a.quant == "W4A16":
        act = T.Activation(query=8, key=8, value=8, head=8, output=16, ffn=16)
        wgt = T.Weight(query=4, key=4, value=8, output=4, ffn=4, head=4)
    else:  # W4A8_L5A16
        act = T.Activation(query=8, key=8, value=8, head=8, output=8, ffn=8)
        wgt = T.Weight(query=4, key=4, value=8, output=4, ffn=4, head=4)
        act16_extra = A16_TENSORS_W4A8
    bit = BitConfig(
        transformer=T(activation=act, weight=wgt),
        layer_overrides=BitConfig.LayerOverrides(
            activation_16bits=sorted(set(qk_names) | set(act16_extra)), weight_16bits=[]),
    )

    tuning = {}
    if "optq" in a.tuning:
        tuning["optq_config"] = OptqConfig(apply=True, attributes=OptqConfig.Attributes(
            actOrder=True, blockSize=128, percDamp=0.01))
    if "sws" in a.tuning:
        tuning["search_weight_scale_config"] = SearchWeightScaleConfig(
            apply=True, transformer=SearchWeightScaleConfig.Transformer(
                query=True, key=True, value=True, out=True, ffn=True))

    calib = a.calib
    if os.path.isdir(calib) and os.path.exists(os.path.join(calib, "npy_files.txt")):
        calib = os.path.join(calib, "npy_files.txt")
    cc = CalibrationConfig(
        method=1, output=1, mode=1,
        max_percentile=CalibrationConfig.MaxPercentile(percentile=0.9999, topk_ratio=0.01))

    log(f"compile 시작: quant={a.quant} tuning={a.tuning} scheme={a.scheme} device={a.device}")
    t = time.time()
    mxq_compile(
        model=wrapper, backend="torch", feed_dict=fd, save_path=a.save,
        target_device=resolve_target_device(a.target_device), yolo_decode_include=True,
        inference_scheme=a.scheme, device=a.device,
        calib_data_path=calib, calibration_config=cc,
        bit_config=bit, **tuning,
    )
    log(f"mxq_compile {time.time() - t:.0f}s")
    sz = os.path.getsize(a.save) / 1e6
    log(f"TOTAL {time.time() - t0:.0f}s → {a.save} ({sz:.1f} MB)")


if __name__ == "__main__":
    main()
