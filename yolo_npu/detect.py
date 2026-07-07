"""
YOLO(11) NPU 추론 — Mobilint ARIES. 이미지 → bbox.

모델(11n/11m/11l 등)은 **mxq 경로만 바꾸면** 됩니다 (전처리/후처리 동일):
    det = YOLONPU("yolo11m_single.mxq")
    boxes = det("street.jpg")            # [(x1,y1,x2,y2,conf,cls_id), ...]
    det.draw("street.jpg", boxes, "out.jpg")

- 전처리: letterbox 640 + BGR→RGB + /255 → HWC float32 (NPU 입력)
- NPU: image → (1,8400,84) [cx,cy,w,h + 80 class score]  (YOLO11 decode 포함)
- 후처리: conf 필터 → xywh→xyxy(letterbox 역변환) → 클래스별 NMS
요구: qbruntime + NPU(/dev/aries0), opencv, numpy.
"""
from __future__ import annotations

import os
import numpy as np
import cv2

IMG_SIZE = 640

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


def letterbox(img, size=IMG_SIZE, color=114):
    """비율 유지 리사이즈 + 패딩. 반환: (letterboxed HxWx3, ratio, (pad_w, pad_h))."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = (size - nw) // 2, (size - nh) // 2
    out = np.full((size, size, 3), color, np.uint8)
    out[dh:dh + nh, dw:dw + nw] = resized
    return out, r, (dw, dh)


def preprocess(img_bgr, size=IMG_SIZE):
    """BGR 원본 → NPU 입력 (size,size,3) float32 [0,1] RGB + 역변환 정보."""
    lb, r, (dw, dh) = letterbox(img_bgr, size)
    rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
    x = np.ascontiguousarray(rgb.astype(np.float32) / 255.0)
    return x, r, (dw, dh)


def _nms(boxes, scores, iou_thres):
    """클래스 무관 NMS (numpy). boxes: (N,4) xyxy. 반환: 유지 인덱스."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1); h = np.maximum(0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thres]
    return keep


def postprocess(out, r, pad, conf_thres=0.25, iou_thres=0.45):
    """NPU 출력 (1,8400,84) → [(x1,y1,x2,y2,conf,cls), ...] (원본 이미지 좌표)."""
    o = np.asarray(out)
    o = o[0] if o.ndim == 3 else o                      # (8400,84)
    xywh, scores = o[:, :4], o[:, 4:]                    # (N,4),(N,80)
    cls = scores.argmax(1)
    conf = scores.max(1)
    m = conf >= conf_thres
    if not m.any():
        return []
    xywh, conf, cls = xywh[m], conf[m], cls[m]
    # cxcywh(letterbox 640 좌표) → xyxy → 패딩 제거 → ratio 나눔 → 원본 좌표
    dw, dh = pad
    cx, cy, bw, bh = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
    x1 = (cx - bw / 2 - dw) / r; y1 = (cy - bh / 2 - dh) / r
    x2 = (cx + bw / 2 - dw) / r; y2 = (cy + bh / 2 - dh) / r
    boxes = np.stack([x1, y1, x2, y2], 1)
    # 클래스별 NMS (클래스마다 offset을 줘서 한 번에)
    det = []
    for c in np.unique(cls):
        idx = np.where(cls == c)[0]
        keep = _nms(boxes[idx], conf[idx], iou_thres)
        for k in keep:
            j = idx[k]
            det.append((float(x1[j]), float(y1[j]), float(x2[j]), float(y2[j]),
                        float(conf[j]), int(c)))
    det.sort(key=lambda d: d[4], reverse=True)
    return det


def detect_npu_devices():
    """장착된 NPU device id 목록(`/dev/ariesN`)을 정렬해 반환. 없으면 빈 리스트."""
    import glob
    ids = []
    for p in glob.glob("/dev/aries*"):
        s = os.path.basename(p)[len("aries"):]
        if s.isdigit():
            ids.append(int(s))
    return sorted(ids)


class YOLONPU:
    """YOLO NPU 추론기. mxq만 바꾸면 11n/11m/11l 등 동일하게 동작. 단일/멀티카드 지원.

    카드 선택 (device_ids):
      - None(기본) → 단일 카드(device_id).           예) YOLONPU("y.mxq")               # aries0
      - 리스트     → 지정 카드들.                      예) YOLONPU("y.mxq", device_ids=[0,1])
      - "auto"     → 장착된 NPU 전부 자동 사용.        예) YOLONPU("y.mxq", device_ids="auto")

    배치는 detect_batch([img,...])로: 카드 라운드로빈 + 멀티스레드 동기 infer(카드·코어 전부 활용).
    (주의: qbruntime async multi-in-flight는 출력이 깨져 쓰지 않는다 — PE와 동일 방침.)
    """

    def __init__(self, mxq_path, device_id=0, device_ids=None, num_threads=8,
                 conf_thres=0.25, iou_thres=0.45, names=None):
        import qbruntime  # 지연 import: 컴파일 env(qbruntime 없음)에서도 전처리 유틸 재사용 가능
        if device_ids == "auto":
            ids = detect_npu_devices()
            if not ids:
                raise RuntimeError("NPU(/dev/aries*)를 찾지 못했습니다.")
        elif device_ids is None:
            ids = [device_id]
        else:
            ids = list(device_ids)
        self.device_ids = ids
        self.accs, self.models = [], []
        for d in ids:
            acc = qbruntime.Accelerator(d)
            m = qbruntime.Model(mxq_path)
            m.launch(acc)
            self.accs.append(acc)
            self.models.append(m)
        self.n = len(self.models)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.names = names or COCO_NAMES
        from concurrent.futures import ThreadPoolExecutor
        self._pool = ThreadPoolExecutor(max_workers=max(1, num_threads) * self.n)

    @classmethod
    def from_hf(cls, model="yolo11m", scheme="single", repo_id=None, revision=None,
                device_id=0, device_ids=None, num_threads=8,
                conf_thres=0.25, iou_thres=0.45, names=None):
        """HF에서 미리 컴파일된 YOLO MXQ를 받아 추론기 구성 (qbruntime만 필요).

        model : yolo11n|yolo11m|yolo11l …    scheme: single|multi|global4|global8
        device_ids: None(단일)|리스트|"auto"(전체). PE의 from_hf(scheme=)와 대칭.
        """
        from . import assets
        mxq = assets.ensure_yolo_mxq(model=model, scheme=scheme,
                                     repo_id=repo_id or assets.HF_REPO, revision=revision)
        return cls(mxq, device_id=device_id, device_ids=device_ids, num_threads=num_threads,
                   conf_thres=conf_thres, iou_thres=iou_thres, names=names)

    def __len__(self):
        return self.n

    def _read(self, image):
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise FileNotFoundError(image)
            return img
        return image  # 이미 BGR ndarray

    def _infer(self, m, x):
        o = m.infer(x)
        return o[0] if isinstance(o, (list, tuple)) else o

    def __call__(self, image, conf_thres=None, iou_thres=None):
        """단일 이미지 → detections (첫 카드 사용)."""
        img = self._read(image)
        x, r, pad = preprocess(img, IMG_SIZE)
        o = self._infer(self.models[0], x)
        return postprocess(o, r, pad,
                           conf_thres if conf_thres is not None else self.conf_thres,
                           iou_thres if iou_thres is not None else self.iou_thres)

    def detect_batch(self, images, conf_thres=None, iou_thres=None):
        """이미지 리스트 → detections 리스트(입력 순서 보존).
        카드 라운드로빈 + 스레드 동기 infer로 전 카드·코어 활용."""
        ct = conf_thres if conf_thres is not None else self.conf_thres
        it = iou_thres if iou_thres is not None else self.iou_thres
        pre = [preprocess(self._read(im), IMG_SIZE) for im in images]   # CPU 전처리

        def run(i):
            x, r, pad = pre[i]
            o = self._infer(self.models[i % self.n], x)                 # 카드 i%n
            return postprocess(o, r, pad, ct, it)

        return list(self._pool.map(run, range(len(images))))            # 순서 보존

    def draw(self, image, detections, save_path=None):
        """detections를 이미지에 그려 반환(BGR ndarray). save_path 주면 저장."""
        img = self._read(image).copy()
        rng = np.random.default_rng(3)
        colors = rng.integers(0, 255, size=(len(self.names), 3)).tolist()
        for x1, y1, x2, y2, conf, c in detections:
            col = [int(v) for v in colors[c % len(colors)]]
            p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.rectangle(img, p1, p2, col, 2)
            label = f"{self.names[c] if c < len(self.names) else c} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (p1[0], p1[1] - th - 4), (p1[0] + tw, p1[1]), col, -1)
            cv2.putText(img, label, (p1[0], p1[1] - 3), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
        if save_path:
            cv2.imwrite(save_path, img)
        return img

    def __del__(self):
        try:
            if getattr(self, "_pool", None) is not None:
                self._pool.shutdown(wait=False)
            for m in getattr(self, "models", []):
                try:
                    m.dispose()
                except Exception:
                    pass
        except Exception:
            pass
