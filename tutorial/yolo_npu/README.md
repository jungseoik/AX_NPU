# YOLO(11) NPU 추론 튜토리얼 — 컴파일부터 bbox까지

ARIES NPU에서 **YOLO11**(객체 탐지)을 추론한다. **모델(11n/11m/11l/…)은 mxq 경로만 바꾸면**
동일 코드로 동작한다 (전처리·후처리 동일). 패키지 = **`yolo_npu/`**.

- **기본 동작: HF 먼저 → 없으면 컴파일.** `YOLONPU.load("yolo11m","single")` — HF
  `PIA-SPACE-LAB/MXQ_NPU/yolo/`에서 미리 컴파일된 MXQ를 받아 바로 추론(qbruntime만 필요).
  HF에 없는 모델/설정이면 직접 컴파일(§2)로 안내된다.
- 모델(11n/11m/11l)·코어모드는 **인자만 바꾸면** 동일 코드로 동작.
- PE(비전인코더)와 달리 **YOLO는 패치 없이 그대로 컴파일된다** (표준 CNN 검출기).

> **PE 대비 요점**: PE-Core는 ViT+RoPE라 단일입출력 컴파일에 5개 모델 패치가 필요했지만,
> YOLO11은 qbcompiler가 바로 파싱(895→635 op 최적화)하고 `--yolo-decode`로 디코드까지 그래프에
> 포함한다. 컴파일 난이도가 훨씬 낮다.

---

## 1. 기본 사용법 — HF에서 가져와 추론 (없으면 컴파일)

```python
from yolo_npu import YOLONPU

# ★ 기본: HF 먼저 읽고(있으면 다운로드) → 없으면 컴파일 명령 안내. qbruntime만 필요.
det = YOLONPU.load("yolo11m", "single")                       # 단일 카드
det = YOLONPU.load("yolo11l", "global4", device_ids="auto")   # 멀티카드
# (로컬에서 직접 컴파일한 mxq를 쓰려면)
det = YOLONPU.load("yolo11m", "single", local_mxq="yolo_out/yolo11m_single.mxq")
# (원한다면 HF 강제)  det = YOLONPU.from_hf(model="yolo11m", scheme="single")

boxes = det("street.jpg")                     # [(x1,y1,x2,y2,conf,cls_id), ...]
det.draw("street.jpg", boxes, "out.jpg")      # bbox 그려 저장
for x1,y1,x2,y2,cf,c in boxes:
    print(det.names[c], round(cf,2), (int(x1),int(y1),int(x2),int(y2)))
```

> `load()` 우선순위: **`local_mxq`(지정 시) → HF `yolo/<model>/<scheme>/<model>.mxq` → (없으면) 컴파일 명령 안내.**
> **HF 배포 구조** (PE와 같은 repo `PIA-SPACE-LAB/MXQ_NPU` 안 `yolo/` 하위, PE `<scheme>/pe_full.mxq` 패턴에 모델 레벨 추가):
> `yolo/<model>/<scheme>/<model>.mxq` (+ `CALIBRATION.md`) + `yolo/<model>/<model>.onnx`. 업로드: `setup/upload_yolo_to_hf.py`.
- 입력: 이미지 경로 또는 BGR numpy(cv2). 내부에서 letterbox 640 + RGB + /255 처리.
- `conf_thres`/`iou_thres`로 임계값 조정. `names`로 커스텀 클래스명.
- 데모: `demo_yolo11_npu.ipynb`(인라인 시각화) / `demo_yolo11_npu.py`(스크립트).

### 멀티카드 (NPU 여러 장) — 배치 분산

```python
from yolo_npu import YOLONPU, detect_npu_devices

# ① 단일 카드 (기본)                      aries0
det = YOLONPU("yolo11m_single.mxq")
# ② 특정 카드 지정                         aries0,1 만
det = YOLONPU("yolo11m_single.mxq", device_ids=[0, 1])
# ③ 장착된 NPU 전부 자동 사용
det = YOLONPU("yolo11m_single.mxq", device_ids="auto")
print(detect_npu_devices())              # [0,1,2,...] 감지된 카드

# 배치(이미지 리스트)는 detect_batch — 카드 라운드로빈 + 카드당 8스레드 동기 infer
results = det.detect_batch([img1, img2, ...])   # [[det...], [det...], ...] (입력 순서 보존)
```
- 단일 이미지는 `det(img)`, 다수 이미지는 `det.detect_batch([...])`. 카드가 많을수록 배치 처리량↑.
- 검증: 2장 분산 결과가 단일카드와 **동일**(라운드로빈이 출력 안 깨뜨림). `"auto"`가 `/dev/aries*` 전부 사용.

### 멀티카드 스케일링 (YOLO11m, 배치 64, NPU-only)

| 카드 | 지연 | 처리량 | 1장 대비 |
|---|--:|--:|--:|
| 1 | 323 ms | 198 img/s | 1.00x |
| **2** | **177 ms** | **362 img/s** | **1.83x** |
| 4 | 74 ms | 859 img/s | 4.34x |
| 7 | 60 ms | 1072 img/s | 5.42x |

> 순수 NPU 추론은 카드 수에 거의 비례. 단, **e2e(전처리+NMS 포함)는 고배치에서 CPU 전처리가 병목**
> (7장 e2e 312ms ≈ CPU 전처리 지배) → 전처리 병렬화가 다음 최적화 포인트(PE와 동일). 상세: `../../reports/performance/NPU_yolo11_coremode_batch.md`.

## 2. 직접 컴파일 (HF에 없는 모델/설정일 때) — 4 코어모드

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
- 상세 실측: `../../reports/performance/NPU_yolo11_coremode_batch.md`.

## 4. 정확도 검증 (mAP, COCO val2017)

우리가 컴파일한 NPU MXQ(INT8) vs fp32(onnxruntime) — **동일 val2017 300장, 동일 전처리/후처리**로
pycocotools 평가. `../../reports/scripts/eval_yolo_map.py` 재현. 상세: `../../reports/performance/NPU_yolo11_coremode_batch.md`.

| yolo11m | mAP@0.5:0.95 | mAP@0.5 |
|---|---|---|
| fp32 (baseline) | 0.5537 | 0.7136 |
| **NPU INT8** | **0.5315** | **0.6952** |
| 양자화 손실 | **−4.0%** | −2.6% |

→ INT8로 fp32 대비 **~96% mAP 유지** (INT8 검출기 전형 손실 범위). calib = COCO val2017 200장(PE와 동일 방침).

## 5. 객체 추적 (ByteTrack, 자체 경량)

검출은 NPU, 추적(칼만+IoU+헝가리안)은 CPU. `yolo_npu/track.py`(의존성 numpy+scipy).

```python
from yolo_npu import YOLONPU, ByteTrack, draw_tracks
det = YOLONPU("yolo11m_single.mxq"); trk = ByteTrack(fps=30)
# 프레임 루프:
boxes  = det(frame)                          # [(x1,y1,x2,y2,conf,cls),...]
tracks = trk.update(boxes)                   # [(x1,y1,x2,y2,track_id,conf,cls),...]
vis    = draw_tracks(frame, tracks, det.names)
```
- 데모(공개 샘플 영상→track ID 영상): `demo_track_yolo11_npu.ipynb` / `demo_track_yolo11_npu.py`.
- 실측: 596프레임 people 영상 **25 fps@1카드**(검출+추적+영상IO), 사람별 ID 유지 확인.
- ByteTrack 2단계 매칭(고신뢰→저신뢰)으로 가림에 강함. 파라미터 `track_thresh/match_thresh/track_buffer/fps`.
- 외형(ReID) 기반이 필요하면 ReID CNN도 NPU 컴파일해 붙일 수 있음(YOLO와 동일 흐름).

## 6. 파이프라인 요약

```
이미지(BGR) → [letterbox 640 + RGB + /255] → NPU(image→(1,8400,84)) → [conf필터+NMS+좌표역변환] → bbox
                    yolo_npu.preprocess              MXQ(yolo_decode 포함)      yolo_npu.postprocess
```
- NPU 출력 `(1,8400,84)` = 8400 anchor × (cxcywh + 80 class score), YOLO decode는 그래프에 포함.
- 전처리·후처리(NMS)는 CPU. NMS 포함 e2e 지연은 위 순수추론 + 수 ms.
