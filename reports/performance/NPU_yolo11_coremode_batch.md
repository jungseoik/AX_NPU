# YOLO11 NPU — 컴파일 · 코어모드 배치지연 · 정확도(mAP)

PE-Core(비전인코더) 외에 **YOLO11 객체탐지**를 ARIES NPU로 직접 컴파일·추론한 실측.
코드: `yolo_npu/` (compile/detect), 튜토리얼: `tutorial/yolo_npu/`.

## 1. 컴파일 — 패치 불필요

- 경로: ultralytics `yolo11*.pt` → ONNX(imgsz 640) → `mxq_compile(backend="onnx", yolo_decode_include=True)`.
- **PE와 달리 모델 패치 0개.** YOLO11은 표준 CNN 검출기라 qbcompiler가 바로 파싱(895→635 op),
  YOLO decode까지 그래프에 포함 → 출력 `(1,8400,84)` (anchor 8400 × [cxcywh+80 class]).
- INT8 MXQ, calib = COCO val2017 200장(PE와 동일 방침). 컴파일 시간(호스트 CPU):

| 모델 | 크기 | single 컴파일 |
|---|---|---|
| yolo11n | 6 MB | 133 s |
| yolo11s | 12 MB | 219 s |
| yolo11m | 23 MB | 229~279 s |
| yolo11l | 29 MB | 341 s |

- 4 코어모드(single/multi/global4/global8) 각각 `inference_scheme`로 생성.

## 2. 코어모드 × 배치지연 (yolo11m, NPU 1장, 배치 1→64)

1모델 + 8스레드 동기 infer(검증된 다채널 패턴), median of 3, 순수 NPU 추론(전/후처리 제외).

**배치 전체 처리 (ms)**
| 모드 | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|--:|--:|--:|--:|--:|--:|--:|
| single | 19 | 26 | 30 | 40 | 77 | 147 | 312 |
| multi | 24 | 31 | 51 | 98 | 194 | 384 | 785 |
| **global4** | 9 | 10 | 17 | 32 | 62 | 122 | **263** |
| global8 | **7** | 15 | 21 | 40 | 77 | 151 | 316 |

**채널당 (ms/img)** — 고배치 수렴: global4 3.8 / single·global8 4.8 / multi 12.

- **global4 종합 최적**(다채널): 단건 9ms + 고배치 3.8ms/img. **64채널 ~0.26s (≈245 img/s).**
- global8: 단건 7ms(최저) — 저채널 실시간. single: 8코어 채우면 ~4.8ms/img(멀티스레드 필수).
- 카드당 동시 병렬 = 코어 배분 한계(single 8-way, B=8부터 평평). 더 필요하면 카드 추가.
- 참고: PE(ViT-L) ~63ms/img 대비 YOLO11m ~4ms/img (약 15배 가벼움).

## 3. 멀티카드 스케일링 (yolo11m single, 배치 64)

카드에 라운드로빈 분산 + 카드당 8스레드 동기 infer(PE와 동일 패턴, `YOLONPU(device_ids=...)`).

**출력 무결성 검증** (`../scripts/verify_yolo_multicard_output.py`): 서로 다른 val2017 **40장**을
단일카드 기준 vs 2장·7장 분산 `detect_batch`로 위치별 비교(개수+클래스+좌표+conf) → **0/40 불일치**
(2장·7장 모두 완전 일치). 순서 보존 + 카드 간 오염 없음. async(multi-in-flight 깨짐) 대신
멀티스레드 동기 infer를 쓰는 이유.

| 카드 | NPU-only 지연 | 처리량 | 1장 대비 | e2e(전처리+NMS 포함) |
|---|--:|--:|--:|--:|
| 1 | 323 ms | 198 img/s | 1.00x | 635 ms |
| **2** | **177 ms** | **362 img/s** | **1.83x** | 371 ms |
| 4 | 74 ms | 859 img/s | 4.34x | 326 ms |
| 7 | 60 ms | 1072 img/s | 5.42x | 312 ms |

- **순수 NPU는 카드 수에 거의 비례** (2장 1.83x, 큰 배치일수록 선형에 근접).
- **e2e는 2장 이후 CPU 전처리 병목**(7장 e2e 312ms ≫ NPU 60ms) — 고배치 다음 최적화는 전처리 병렬화(PE와 동일 결론, `NPU_preprocess_parallel.md`).
- 사용: `YOLONPU(mxq, device_ids=[0,1])`(지정) / `device_ids="auto"`(전체) + `detect_batch([...])`.

## 4. 정확도 (mAP, COCO val2017 300장)

우리 NPU MXQ(INT8) vs fp32(onnxruntime), **동일 이미지·동일 전/후처리**, pycocotools.

| 모델 | | mAP@0.5:0.95 | mAP@0.5 |
|---|---|--:|--:|
| **yolo11m** | fp32 baseline | 0.5537 | 0.7136 |
| | **NPU INT8** | **0.5315** | **0.6952** |
| | 양자화 손실 | **−4.0%** | −2.6% |
| **yolo11l** | fp32 baseline | 0.5690 | 0.7240 |
| | **NPU INT8** | **0.5539** | **0.7112** |
| | 양자화 손실 | **−2.7%** | −1.8% |
| **yolo11s** | fp32 baseline | 0.5084 | 0.6750 |
| | **NPU INT8** | **0.4932** | **0.6602** |
| | 양자화 손실 | **−3.0%** | −2.2% |
| **yolo11n** | fp32 baseline | 0.4335 | 0.5827 |
| | **NPU INT8** | **0.4215** | **0.5728** |
| | 양자화 손실 | **−2.8%** | −1.7% |

→ INT8로 fp32 대비 **~96~97% mAP 유지** (INT8 검출기 전형 손실 1~5%). 4모델 모두 동일 파이프라인·calib(val2017 200장).

## 5. 재현

```bash
# 컴파일 (env: yolo_c = qbcompiler + torch 2.7.1 + ultralytics, tutorial/yolo_npu/README.md 2절)
python -m yolo_npu.compile --model yolo11m --schemes single,multi,global4,global8 \
    --calib <coco/val2017> --calib-num 200 --out ./yolo_out
# 추론/데모 (env: pe_npu_host)
python tutorial/yolo_npu/demo_yolo11_npu.py --mxq yolo_out/yolo11m_single.mxq --image bus.jpg
# mAP
python reports/scripts/eval_yolo_map.py <mxq> <onnx> <val2017> <instances_val2017.json> 300
```

*실측 ARIES2 (aries0), qbcompiler 1.1.2, qbruntime 1.2.0. latency는 calib 무관(random calib 측정), 정확도는 val2017 200장 calib.*
