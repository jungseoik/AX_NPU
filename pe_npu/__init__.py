"""
pe_npu — PE-Core-L14-336 비전인코더 NPU(Mobilint ARIES) 컴파일/추론 패키지.

- pe_model    : 모델 로딩 + 컴파일 패치 (load_pe, apply_pe_patches, VisionWrapper 등)
- preprocess  : preprocess_image (resize 336 + normalize 0.5)
- calib       : prepare_calib / to_hwc (CLI: python -m pe_npu.calib)
- compile     : compile_pe / parse_pe (CLI: python -m pe_npu.compile)
- inference   : MXQInferenceHybrid (NPU trunk + CPU pool/proj)

inference는 qbruntime/NPU가 있어야 import되므로, NPU 없는 환경을 위해 최상위에서는
지연 import한다. `from pe_npu import MXQInferenceHybrid` 또는
`from pe_npu.inference import MXQInferenceHybrid` 둘 다 동작.
"""
from .preprocess import preprocess_image, IMAGE_SIZE
from .pe_model import (
    load_pe,
    apply_pe_patches,
    VisionWrapper,
    FeatWrapper,
    PoolWrapper,
)

__all__ = [
    "preprocess_image",
    "IMAGE_SIZE",
    "load_pe",
    "apply_pe_patches",
    "VisionWrapper",
    "FeatWrapper",
    "PoolWrapper",
    "MXQInferenceHybrid",
]


def __getattr__(name):
    # 지연 import: qbruntime(NPU 런타임) 미설치 환경에서도 import pe_npu가 성공하도록.
    if name == "MXQInferenceHybrid":
        from .inference import MXQInferenceHybrid
        return MXQInferenceHybrid
    raise AttributeError(f"module 'pe_npu' has no attribute {name!r}")
