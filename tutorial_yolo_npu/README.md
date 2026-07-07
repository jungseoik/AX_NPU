# YOLO(11) NPU 추론 튜토리얼 — 컴파일부터 bbox까지

ARIES NPU에서 **YOLO11**(객체 탐지)을 추론한다. **모델(11n/11m/11l/…)은 mxq 경로만 바꾸면**
동일 코드로 동작한다 (전처리·후처리 동일). 패키지 = **`yolo_npu/`**.

- 추론: 이미지 → NPU → bbox. `YOLONPU("yolo11m_single.mxq")` 한 줄.
- 컴파일: ultralytics 모델 → ONNX → 4 코어모드 MXQ. `python -m yolo_npu.compile`.
- PE(비전인코더)와 달리 **YOLO는 패치 없이 그대로 컴파일된다** (표준 CNN 검출기).

> **PE 대비 요점**: PE-Core는 ViT+RoPE라 단일입출력 컴파일에 5개 모델 패치가 필요했지만,
> YOLO11은 qbcompiler가 바로 파싱(895→635 op 최적화)하고 `--yolo-decode`로 디코드까지 그래프에
> 포함한다. 컴파일 난이도가 훨씬 낮다.

---

## 1. 옵션 B — 이미 만든 MXQ로 추론만 (qbruntime만 필요)

```python
from yolo_npu import YOLONPU

det = YOLONPU("yolo11m_single.mxq")           # ← 모델 바꾸려면 이 경로만 변경
boxes = det("street.jpg")                     # [(x1,y1,x2,y2,conf,cls_id), ...]
det.draw("street.jpg", boxes, "out.jpg")      # bbox 그려 저장
for x1,y1,x2,y2,cf,c in boxes:
    print(det.names[c], round(cf,2), (int(x1),int(y1),int(x2),int(y2)))
```
- 입력: 이미지 경로 또는 BGR numpy(cv2). 내부에서 letterbox 640 + RGB + /255 처리.
- `conf_thres`/`iou_thres`로 임계값 조정. `names`로 커스텀 클래스명.
- 데모: `demo_yolo11_npu.ipynb`(인라인 시각화) / `demo_yolo11_npu.py`(스크립트).

## 2. 옵션 A — 직접 컴파일 (4 코어모드)

### 2-1. 컴파일 환경 (torch 2.7.1 매칭 필수 — qbcompiler mmc ABI)
```bash
conda create -y -n yolo_c python=3.10 && conda activate yolo_c
pip install "torch==2.7.1" "torchvision==0.22.1" "numpy<2" \
            ultralytics onnx onnxslim "onnxruntime>=1.19.2"
pip install --no-deps download/qbcompiler-1.1.2+aries2-py3-none-any.whl
```
> ⚠️ torch가 2.7.1이 아니면 `Fail to import mmc` / `undefined symbol` (컴파일 셋업 함정).
> ultralytics가 최신 torch를 끌어오므로 위 순서로 2.7.1을 **덮어써야** 한다.

### 2-2. 컴파일 (모델명만 바꾸면 됨)
```bash
# 정확도용 (실이미지 calib — PE와 동일하게 COCO val2017 200장 권장)
python -m yolo_npu.compile --model yolo11m --schemes single,multi,global4,global8 \
    --calib /path/to/coco/val2017 --calib-num 200 --out ./yolo_out

# latency만 볼 거면 calib 생략 (random calib, 정확도 무의미)
python -m yolo_npu.compile --model yolo11n --schemes single --out ./yolo_out
```
- `--model`: `yolo11n` / `yolo11s` / `yolo11m` / `yolo11l` / `yolo11x`. ultralytics가 자동 다운로드.
- 산출물: `yolo_out/<model>_<scheme>.mxq` (각 ~20MB). 이 경로를 `YOLONPU()`에 주면 끝.
- 컴파일은 호스트 CPU(기본)에서 ~3~4분/모드. NPU는 추론 전용.

## 3. 코어모드 선택 (1카드 실측, YOLO11m)

| 코어모드 | 단건 지연 | 고배치 ms/img | 용도 |
|---|---|---|---|
| **global4** | 9 ms | ~3.8 (최고 처리량) | **다채널 권장** |
| global8 | **7 ms** (최저) | ~4.8 | 단건/저채널 실시간 |
| single | 19 ms(1코어)→스레드로 ~4.8 | ~4.8 | 처리량(멀티스레드 필수) |
| multi | — | ~12 (열세) | 비권장 |

- 카드당 동시 병렬 = 코어 배분이 한계(single 8-way). 더 필요하면 **카드 추가**.
- 상세 실측: `../reports/performance/NPU_yolo11_coremode_batch.md`.

## 4. 정확도 검증 (mAP, COCO val2017)

우리가 컴파일한 NPU MXQ(INT8) vs fp32(onnxruntime) — **동일 val2017 300장, 동일 전처리/후처리**로
pycocotools 평가. `../reports/scripts/eval_yolo_map.py` 재현. 상세: `../reports/performance/NPU_yolo11_coremode_batch.md`.

| yolo11m | mAP@0.5:0.95 | mAP@0.5 |
|---|---|---|
| fp32 (baseline) | 0.5537 | 0.7136 |
| **NPU INT8** | **0.5315** | **0.6952** |
| 양자화 손실 | **−4.0%** | −2.6% |

→ INT8로 fp32 대비 **~96% mAP 유지** (INT8 검출기 전형 손실 범위). calib = COCO val2017 200장(PE와 동일 방침).

## 5. 파이프라인 요약

```
이미지(BGR) → [letterbox 640 + RGB + /255] → NPU(image→(1,8400,84)) → [conf필터+NMS+좌표역변환] → bbox
                    yolo_npu.preprocess              MXQ(yolo_decode 포함)      yolo_npu.postprocess
```
- NPU 출력 `(1,8400,84)` = 8400 anchor × (cxcywh + 80 class score), YOLO decode는 그래프에 포함.
- 전처리·후처리(NMS)는 CPU. NMS 포함 e2e 지연은 위 순수추론 + 수 ms.
