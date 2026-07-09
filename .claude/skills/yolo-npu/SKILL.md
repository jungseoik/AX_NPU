---
name: yolo-npu
description: ARIES NPU에서 YOLO11 객체탐지 추론 코드를 짤 때 쓴다. 이미지→bbox(+선택적 추적). yolo_npu 패키지(YOLONPU.load(model, scheme))로 HF의 미리 컴파일된 MXQ를 가져와 쓴다(단일/멀티카드 detect·detect_batch + ByteTrack 추적). "YOLO 추론", "NPU 객체탐지", "bbox 검출", "yolo11", "객체 추적/ByteTrack" 같은 요청에 사용.
---

# YOLO11 NPU 추론 (yolo_npu)

## 언제 쓰나

ARIES NPU에서 **이미지 → bbox**(객체탐지) 추론 코드를 짤 때. 선택적으로 **다중객체 추적(ByteTrack)** 까지.
PE-Core(이미지→임베딩, `pe_npu`)·Qwen3-VL(VLM)과는 별개 워크스트림이다.

- 패키지: **`yolo_npu/`** (`detect` / `compile` / `track` / `assets`)
- 따라하기: **`tutorial/yolo_npu/README.md`**, 노트북 `demo_yolo11_npu.ipynb`(검출) · `demo_track_yolo11_npu.ipynb`(추적)
- 실측/원리: `reports/performance/NPU_yolo11_coremode_batch.md`

## 기본 진입점 — `YOLONPU.load(model, scheme)`

운영/빠른 시작 권장. HF `PIA-SPACE-LAB/MXQ_NPU/yolo/<model>/<scheme>/<model>.mxq`를 먼저 받고, 없으면 컴파일 안내.

```python
from yolo_npu import YOLONPU

det = YOLONPU.load("yolo11m", "single")                     # 단일 카드
det = YOLONPU.load("yolo11l", "global4", device_ids="auto")  # 멀티카드(장착 NPU 전부)
det = YOLONPU.load("yolo11m", "single", local_mxq="yolo_out/yolo11m_single.mxq")  # 로컬 MXQ
# det = YOLONPU.from_hf(model="yolo11m", scheme="single")    # HF 강제
# det = YOLONPU("path/to/yolo11m_single.mxq")                # MXQ 직접 지정

boxes = det("street.jpg")                    # 이미지→검출 (conf_thres/iou_thres 옵션)
det.draw("street.jpg", boxes, "out.jpg")     # 시각화 저장
batch = det.detect_batch([img1, img2, ...])  # 배치(출력 무결성 검증됨)
```

- **모델**: `yolo11n | yolo11s | yolo11m | yolo11l …` — 이름만 바꾸면 동일 코드(=다른 mxq).
- **코어모드(scheme)**: `single | multi | global4 | global8` (PE와 동일 개념, 출력 동일·속도만 차이).
- **카드 선택(`device_ids`)**: `None`(단일) | 리스트 `[0,1]` | `"auto"`(전체). PE `from_hf(scheme=)`와 대칭.
- 유틸: `preprocess` / `postprocess` / `letterbox` / `COCO_NAMES` / `IMG_SIZE`도 export.

## 추적(ByteTrack) — 검출 NPU + 추적 CPU

```python
from yolo_npu import YOLONPU, ByteTrack, draw_tracks   # track은 lazy import(scipy 필요)
```

`ByteTrack`은 자체 경량 구현(`yolo_npu/track.py`). 프레임 루프에서 검출→추적 붙이는 **정확한 사용 예시는 `tutorial/yolo_npu/demo_track_yolo11_npu.py`** 를 그대로 따를 것(칼만+IoU 매칭 루프 포함).

## 컴파일 (직접) — 패치 0개

PE와 달리 표준 CNN이라 **모델 패치 없이** 컴파일된다. qbcompiler(docker `mblt_compiler`) + ultralytics 환경 필요.

```bash
python -m yolo_npu.compile --model yolo11m --schemes single       # ultralytics→ONNX→MXQ
# 산출물로 추론:  YOLONPU.load("yolo11m","single", local_mxq="yolo_out/yolo11m_single.mxq")
```

## 자산 경로 (HF)

- 같은 repo `PIA-SPACE-LAB/MXQ_NPU`의 `yolo/` 하위: `yolo/<model>/<scheme>/<model>.mxq` (+ `CALIBRATION.md`), fp32 원본 `yolo/<model>/<model>.onnx`.
- 로컬 `local_mxq=` 우선 → 없으면 HF 자동 다운로드(`assets.ensure_yolo_mxq`).

## 주의

- 추론(`detect`)은 qbruntime+NPU만 있으면 됨. 컴파일은 qbcompiler+ultralytics 별도 환경.
- 다채널: **카드당 1인스턴스 + 멀티스레드 동기 `detect`/`detect_batch`**(async 다중 in-flight 금지 — PE와 동일 원칙, `../npu-setup` 및 CLAUDE.md 다채널 동시성 규칙 참조).
- 정확도: 11m INT8 mAP 0.53(fp32의 96%). 모드×배치·1~7카드 스케일링은 위 report 참조.
