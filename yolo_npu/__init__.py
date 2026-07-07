"""
yolo_npu — YOLO(11) 계열을 Mobilint ARIES NPU로 추론/컴파일.

- detect  : YOLONPU (이미지 → bbox), preprocess/postprocess/draw, COCO_NAMES
- compile : ultralytics 모델 → ONNX → MXQ (4 코어모드). CLI: python -m yolo_npu.compile

추론(detect)은 qbruntime+NPU만 있으면 되고, 컴파일(compile)은 qbcompiler+ultralytics 환경 필요.
모델(11n/11m/11l …)은 mxq 경로만 바꾸면 동일 코드로 동작한다.

    from yolo_npu import YOLONPU
    det = YOLONPU("yolo11m_single.mxq")
    boxes = det("street.jpg"); det.draw("street.jpg", boxes, "out.jpg")
"""
from .detect import (YOLONPU, detect_npu_devices, preprocess, postprocess,
                     letterbox, COCO_NAMES, IMG_SIZE)

__all__ = ["YOLONPU", "detect_npu_devices", "preprocess", "postprocess",
           "letterbox", "COCO_NAMES", "IMG_SIZE"]
