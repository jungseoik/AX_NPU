"""
PE-Core-L14-336 vision encoder ONNX를 Mobilint NPU용 MXQ로 컴파일한다.

두 가지 모드:
  - 동작 검증: --calib-data-path 없이 실행 → use_random_calib=True (랜덤, 정확도 무의미)
  - 정확도용 : --calib-data-path 지정 → 실제 calibration 데이터로 양자화 범위 산출
               (calibration 데이터 준비는 prepare_calib.py 참고)

calibration 데이터는 prepare_calib.py가 만든 (3,336,336) float32 .npy 디렉토리.
PE는 normalize를 모델 밖에서 수행하므로, calib 데이터도 normalize 완료된 텐서여야 한다
(prepare_calib.py가 동일 전처리 적용).

사용:
    # 동작 검증(랜덤)
    python compile_pe.py --onnx ./out/PE-Core-L14-336_vision_sim.onnx --save ./out/pe.mxq
    # 실제 calibration
    python compile_pe.py --onnx ./out/PE-Core-L14-336_vision_sim.onnx \
        --save ./out/pe_coco.mxq --calib-data-path ./calib_coco --scheme all
"""
import argparse
import os

from qbcompiler import CalibrationConfig, mxq_compile


def main():
    ap = argparse.ArgumentParser(description="PE vision encoder -> MXQ compiler")
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--onnx", default=os.path.join(here, "out", "PE-Core-L14-336_vision_sim.onnx"))
    ap.add_argument("--save", default=os.path.join(here, "out", "PE-Core-L14-336_vision.mxq"))
    ap.add_argument("--scheme", default="single",
                    choices=["single", "multi", "global", "global4", "global8", "all"])
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    ap.add_argument("--calib-data-path", default=None,
                    help="실제 calibration npy 디렉토리. 미지정 시 랜덤 calibration(동작 검증)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)

    print(f"[compile] onnx={args.onnx}")
    print(f"[compile] save={args.save} scheme={args.scheme} device={args.device}")

    common = dict(
        model=args.onnx,
        save_path=args.save,
        backend="onnx",
        device=args.device,
        inference_scheme=args.scheme,
    )

    if args.calib_data_path:
        # ViT 등 다중 컴포넌트 모델은 directory calibration이 막히므로,
        # 디렉토리를 주면 내부 npy_files.txt(npy 경로 목록)를 자동으로 사용한다.
        calib_path = args.calib_data_path
        if os.path.isdir(calib_path):
            txt = os.path.join(calib_path, "npy_files.txt")
            if os.path.exists(txt):
                calib_path = txt
            else:
                print(f"[warn] {txt} 없음 → 디렉토리 방식 사용(다중입력 모델이면 실패할 수 있음)")
        # 실제 calibration: per-channel + percentile 클리핑(이상치 영향 완화)
        print(f"[compile] calibration mode: {calib_path}")
        calibration_config = CalibrationConfig(
            method=1,  # weight per-channel, activation per-layer
            output=0,
            mode=1,    # MaxPercentile
            max_percentile=CalibrationConfig.MaxPercentile(percentile=0.9999, topk_ratio=0.01),
        )
        mxq_compile(
            **common,
            calib_data_path=calib_path,
            calibration_config=calibration_config,
        )
    else:
        # 동작 검증: 랜덤 calibration (정확도 무의미)
        print("[compile] random calibration (동작 검증용, 정확도 무의미)")
        mxq_compile(**common, use_random_calib=True)

    print("[compile] done")


if __name__ == "__main__":
    main()
