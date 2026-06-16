"""
MXQInference — Mobilint NPU(qbruntime) 기반 추론 래퍼.

Product-AI-mono의 perception_encoder 모듈에서 사용하는 `TRTInference`(trt_load.py)를
NPU 버전으로 그대로 대체(drop-in)하기 위한 클래스.

TRTInference와 동일한 인터페이스를 제공한다:
    model = MXQInference(mxq_path)
    out = model(image_cuda)        # 또는 model.infer(image_cuda)
    # 입력 : torch.Tensor (B,3,336,336) float32  (전처리/normalize 완료된 텐서)
    # 출력 : torch.Tensor (B, 1024)

내부적으로는 qbruntime이 numpy를 다루므로 torch<->numpy 변환을 수행한다.
NPU의 입력 레이아웃(NCHW vs NHWC)과 배치 처리 방식은 MXQ 컴파일 설정에 의존하므로,
launch 시점에 모델의 입력 shape를 조회해 레이아웃을 자동 정렬한다.

주의: 이 모듈은 qbruntime(런타임 라이브러리)과 실제 NPU(/dev/aries0)가 있어야 동작한다.
컴파일 전용 환경(qbcompiler 컨테이너)에서는 import/실행되지 않는다.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np

try:
    import torch

    _HAS_TORCH = True
except ImportError:  # 런타임 호스트에 torch가 없을 수도 있음
    _HAS_TORCH = False

import qbruntime


_CORE_MODES = ("single", "multi", "global4", "global8")


class MXQInference:
    """qbruntime 기반 NPU 추론기. TRTInference와 동일 인터페이스."""

    def __init__(
        self,
        mxq_path: str,
        device_id: int = 0,
        core_mode: str = "single",
        clusters: Optional[Sequence] = None,
    ):
        """
        Args:
            mxq_path: 컴파일된 MXQ 파일 경로
            device_id: NPU 디바이스 번호 (/dev/aries{device_id})
            core_mode: "single" | "multi" | "global4" | "global8"
                       (MXQ가 해당 스킴으로 컴파일되어 있어야 함. "all"로 컴파일하면 모두 가능)
            clusters: multi/global 모드에서 사용할 클러스터 목록. None이면 기본값.
        """
        if core_mode not in _CORE_MODES:
            raise ValueError(f"core_mode는 {_CORE_MODES} 중 하나여야 합니다: {core_mode}")

        self.mxq_path = mxq_path
        self.core_mode = core_mode

        self.acc = qbruntime.Accelerator(device_id)
        self.mc = self._build_model_config(core_mode, clusters)
        self.model = qbruntime.Model(mxq_path, self.mc) if self.mc else qbruntime.Model(mxq_path)
        self.model.launch(self.acc)

        # 입력/출력 메타는 런타임 버전에 따라 조회 API가 다를 수 있어 방어적으로 시도
        self.input_shape = self._try_get_input_shape()

    @staticmethod
    def _build_model_config(core_mode: str, clusters: Optional[Sequence]):
        from qbruntime import Cluster, ModelConfig

        mc = ModelConfig()
        if core_mode == "single":
            # single은 ModelConfig 없이 기본(자동 8코어 single) 동작 → mc 미사용
            return None
        if core_mode == "multi":
            cl = clusters or [Cluster.Cluster0, Cluster.Cluster1]
            mc.set_multi_core_mode(list(cl))
            return mc
        if core_mode == "global4":
            cl = clusters or [Cluster.Cluster0]
            mc.set_global4_core_mode(list(cl))
            return mc
        if core_mode == "global8":
            mc.set_global8_core_mode()
            return mc
        return None

    def _try_get_input_shape(self):
        for attr in ("get_model_input_shape", "get_input_shape", "input_shape", "get_input_shapes"):
            fn = getattr(self.model, attr, None)
            if fn is None:
                continue
            try:
                return fn() if callable(fn) else fn
            except Exception:
                continue
        return None

    # ---- TRTInference 호환 인터페이스 ----
    def __call__(self, *args, **kwargs):
        return self.infer(*args, **kwargs)

    def infer(self, image):
        """
        Args:
            image: torch.Tensor (B,3,H,W) 또는 numpy.ndarray. 전처리 완료 가정.
        Returns:
            torch.Tensor (B, D)  (입력이 torch면), 아니면 numpy.ndarray
        """
        return_torch = _HAS_TORCH and isinstance(image, torch.Tensor)

        # torch -> numpy
        if return_torch:
            arr = image.detach().cpu().contiguous().numpy()
        else:
            arr = np.ascontiguousarray(image)

        arr = arr.astype(np.float32, copy=False)

        # 배치 차원 분리: qbruntime은 보통 단일 입력을 받으므로 배치를 순회한다.
        if arr.ndim == 4:
            outs = [self._infer_single(arr[i]) for i in range(arr.shape[0])]
            out = np.stack(outs, axis=0)
        elif arr.ndim == 3:
            out = self._infer_single(arr)[None, ...]
        else:
            raise ValueError(f"입력 차원은 3 또는 4여야 합니다: {arr.shape}")

        if return_torch:
            t = torch.from_numpy(out)
            return t.to(image.device) if image.is_cuda else t
        return out

    def _infer_single(self, chw_or_hwc: np.ndarray) -> np.ndarray:
        """단일 샘플 추론. NPU 입력 레이아웃에 맞춰 정렬한다."""
        x = self._match_layout(chw_or_hwc)
        result = self.model.infer(x)
        # qbruntime.infer는 보통 list를 반환하며 첫 출력이 임베딩
        if isinstance(result, (list, tuple)):
            result = result[0]
        return np.asarray(result).reshape(-1).astype(np.float32)

    def _match_layout(self, sample: np.ndarray) -> np.ndarray:
        """모델이 기대하는 입력 레이아웃(NHWC/NCHW)으로 단일 샘플을 정렬한다."""
        # 기본 입력은 (3,H,W)(CHW)로 들어온다고 가정. 모델 input_shape를 알면 그에 맞춘다.
        shp = self.input_shape
        if shp is not None:
            try:
                dims = list(shp[0]) if isinstance(shp[0], (list, tuple)) else list(shp)
                last = int(dims[-1])
                # 모델이 HWC(마지막 채널=3)를 기대하면 CHW -> HWC 변환
                if last == 3 and sample.shape[0] == 3:
                    return np.ascontiguousarray(np.transpose(sample, (1, 2, 0)))
            except Exception:
                pass
        return np.ascontiguousarray(sample)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="MXQInference 단독 추론 점검")
    ap.add_argument("--mxq", required=True, help="MXQ 파일 경로")
    ap.add_argument("--core-mode", default="single", choices=list(_CORE_MODES))
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--image-size", type=int, default=336)
    args = ap.parse_args()

    m = MXQInference(args.mxq, core_mode=args.core_mode)
    dummy = np.random.randn(args.batch, 3, args.image_size, args.image_size).astype(np.float32)
    out = m.infer(dummy)
    print("input :", dummy.shape)
    print("output:", out.shape, out.dtype)
