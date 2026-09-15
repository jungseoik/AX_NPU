# [전처리 ④] pe_npu CPU 전처리 병목 제거 — ROI 크롭 torch→cv2, resize/normalize 분리

> 대상: Product-AI-mono `packages/pia_prod/AI/modules/pe_npu/` (브랜치 `aiprod-313-npu-버그-해결-전처리파트`, PR #610 / AIPROD-313)
> **적용 위치는 Product-AI-mono 의 `pe_npu` 서비스이고, 본 문서는 그 근거를 AX_NPU 지식베이스에 남긴 것이다**
> (전처리 ③과 동일한 구성). 측정 스크립트는 `reports/scripts/bench_pe_roi_*.py` · `verify_pe_*.py` 이며
> 두 레포 경로를 환경변수(`AX_NPU_ROOT` / `PIA_PACKAGES`)로 받아 자기완결적으로 돈다.
> 앞 문서: [`NPU_preprocess_3_cv2_decision.md`](NPU_preprocess_3_cv2_decision.md) — 거기서 채택한 cv2 resize가
> **실서비스 경로에서는 적용되지 않고 있었다**는 것이 이 문서의 출발점이다.

## 1. 요약

- 전처리 ③(cv2 resize)은 단계 단독 벤치로 검증됐지만 `_detect` 실경로에서는 동작하지 않았다. ROI 크롭 단계가
  프레임을 torch로 올려 넘겨서 `preprocess.py`의 cv2 분기 조건(`isinstance(img, np.ndarray)`)이 항상 거짓이었다.
- 그 ROI 단계 자체가 병목이었다. **62채널 720p에서 ROI 크롭만 2.6초**. ROI를 설정하지 않은 스트림도
  전체화면 폴리곤이 주입되어 같은 비용을 냈다.
- 세 가지를 고쳤다.
  1. **ROI 크롭** — numpy 유지 + `cv2.fillPoly` + 스레드풀, ROI 미지정 시 크롭 스킵 → 2602 ms → **4.6 ms**
  2. **전처리** — 장당 resize+normalize를 **스레드 resize + 배치 normalize 1회**로 분리 → 176 ms → **40 ms**
  3. **레이아웃** — 전처리가 NPU 네이티브 **NHWC**로 바로 내보내 CHW↔HWC 왕복 제거 → 추론 단계 **−61 ms**
- 62채널 e2e(8카드) **3422 ms → 569 ms (6.0배)**, 1 fps 기준 수용 채널 **18ch → 62ch 이상**.
  출력은 전처리·레이아웃 비트 동일, ROI는 미설정 시 비트 동일 / 폴리곤 시 경계 0.2%(임베딩 cos 0.9988).
- 덤으로 HF에 새로 올라온 `<quant>/<tuning>/<scheme>` 산출물을 env로 고를 수 있게 했다(기본값은 기존 배포본 그대로).

| 62ch / 720p / e2e 단일 타이머 | 수정 전 | ROI만 | +전처리 분리 | **+NHWC(최종)** |
|---|---:|---:|---:|---:|
| 8카드 · 빈 폴리곤 | 3421.9 ms | 798.8 ms | 690.5 ms | **572.4 ms** |
| 8카드 · 다각형 ROI | 5277.2 ms | 881.5 ms | 774.8 ms | **648.3 ms** |
| 4카드 · 빈 폴리곤 | 3893.3 ms | 1286.1 ms | 1149.9 ms | **1062.7 ms** |
| 4카드 · 다각형 ROI | 5734.5 ms | 1369.1 ms | 1237.3 ms | **1149.2 ms** |

> ⚠️ **배포 전 확인**: 이 수정으로 `PE_NPU_RESIZE=cv2`가 "실제로" 켜진다(그동안은 설정과 달리 torchvision이 돌고 있었다).
> COCO 실이미지 40장 기준 임베딩이 **cos 평균 0.9863 / 최소 0.9588**만큼 이동한다. 기존 수치를 그대로 유지하려면
> `PE_NPU_RESIZE=torchvision`. 자세한 내용은 §5.5.

## 2. 환경

| 항목 | 값 |
|---|---|
| NPU | Mobilint ARIES2 ×8 (`/dev/aries0~7`), 카드당 8코어. 드라이버 1.13.0 / qbruntime 1.2.0 |
| MXQ | full NPU `global4/pe_full.mxq` (W8A16, 현행 배포본) |
| CPU | 96 스레드 (torch intra-op 기본 48), RAM 125 GB, GPU 미사용(torch CPU 빌드) |
| 입력 | 합성 프레임 1280×720 / 1920×1080. ROI 미지정 / 8각형 ROI(프레임의 약 35%) 2종 |
| 측정 | 단계별 median(3~7회), 사전 warmup. e2e는 단일 타이머로 별도 검증 |

## 3. 파이프라인과 측정 지점

```
디코드(numpy HWC) → [BGR→RGB] → ROI 크롭 → resize 336 + normalize → NPU(image→embedding) → 텍스트 유사도
      업스트림        (기본 skip)   [CPU] ①        [CPU] ②                  [NPU] ③            [CPU] ④
```

`ServiceBase`의 추론 스레드는 1개이고 `_detect`는 ①→②→③을 순차 실행한다. 배치 e2e = ①+②+③(+④).
④(유사도)는 카드 수·ROI와 무관하고 62ch에서 1~2 ms 수준이라 이 문서에서는 다루지 않는다.

## 4. 원인

### 4.1 ROI 단계가 numpy를 torch로 올린다 → cv2 경로 무력화

`pe_npu/roi_manager.py`
```python
device_batch = torch.from_numpy(batch).to(DEVICE) if isinstance(batch, np.ndarray) else batch
```
`batch_crop_region`이 torch 분기를 타고 결과가 `Tensor`로 나온다. 그런데 `pe_npu/preprocess.py`의 cv2 분기는
`isinstance(img, np.ndarray)`를 요구하므로 **절대 성립하지 않는다.** `PE_NPU_RESIZE=cv2`가 기본값인데도
실제로는 torchvision(antialias) 경로로 돌고 있었다. (부수적으로, 정확도는 문서상 0.97이 아니라 0.99가 유지되고 있었다.)

### 4.2 `torch_crop_region`은 프레임 전체를 훑는다

`pia/vision/roi/roi_manager.py:316`
1. `crop = chw[:, y0:y1, x0:x1].clone()` — ROI가 전체화면이면 **풀프레임 복사**
2. `torch.meshgrid`로 H×W 좌표 격자 2개 생성 (720p = 92만 점 × float32)
3. **폴리곤 꼭짓점마다 파이썬 루프**를 돌며 ray-casting (꼭짓점당 92만 원소 비교/나눗셈/XOR)
4. `crop[c][~mask] = bg[c]` 를 채널별 boolean 인덱싱

비용이 `O(H·W·꼭짓점수)`다. 같은 일을 하는 numpy 구현(`fillPoly` + 슬라이스)은 프레임당 3 ms 미만이다.

### 4.3 ROI를 안 그려도, 직사각형이어도 전부 마스킹을 돈다

ROI는 UI에서 안 그려도 `polygonCoordinates` 필드 자체는 항상 들어온다(DTO 기본값 `[]`). 두 형태 모두 낭비다.

- **빈 폴리곤**: `get_roi_info()`가 전체화면 사각형 `[[0,0],[0,h],[w,h],[w,0]]`을 채워 넣어 4.2를 그대로 수행한다.
  결과는 원본과 동일한 픽셀이다.
- **직사각형**(UI가 미지정 시 프레임 네 모서리를 보내는 경우, 또는 사용자가 사각형으로 그린 경우):
  폴리곤 마스크가 bbox를 다 덮어 **지워지는 픽셀이 0개**인데도 `fillPoly` + 마스크 검사를 돈다.
  같은 no-op 마스킹을 `npu_intrusion`에서 이미 제거한 적이 있다
  ([`NPU_npu_intrusion_e2e_opt.md`](NPU_npu_intrusion_e2e_opt.md) §①).

### 4.4 전처리가 이미지 1장씩 torch를 호출한다

`preprocess_image(list)`는 resize → `torch.stack` → **배치 단위** dtype 변환/normalize 구조다. 그런데
`ParallelPreprocessor._preprocess_one`이 `preprocess_image([img])`를 **장당** 호출하는 바람에 배치 이점이 사라지고,
워커 스레드 안에서 작은 torch 연산이 단일코어로 돌았다. 96코어 서버에서 실제 사용 코어가 10개에 그쳤다.

### 4.5 왜 그동안 안 보였나

- 전처리 ③의 벤치 표에는 **ROI 컬럼이 없다**. BGR→RGB / resize / NPU / 유사도만 측정했다.
- 단계 단독 벤치는 numpy 프레임을 전처리에 직접 넣어 cv2 경로를 탔다. 실경로는 그 앞에 ROI가 붙는다.
- 같은 함정을 `npu_intrusion`에서는 이미 밟고 고쳤다
  ([`NPU_npu_intrusion_e2e_opt.md`](NPU_npu_intrusion_e2e_opt.md) §① ROIcrop 210→17 ms). pe_npu에는 반영되지 않았다.
- 이 코드는 pe_npu 최초 커밋(`dfe658ad`)부터 torch 변환이었다. **깨진 게 아니라 처음부터 적용된 적이 없다.**

## 5. 변경 내역

### 5.1 ROI 크롭 (`roi_manager.py`)

| 항목 | 변경 전 | 변경 후 |
|---|---|---|
| 자료형 | numpy → torch 업로드 | **numpy 유지** (뒤 전처리가 cv2 경로를 탄다) |
| 크롭 구현 | `torch_crop_region` (ray-casting) | `_np_crop_region` — bbox 슬라이스 + `cv2.fillPoly` 마스크 + 114 채움 |
| ROI 미지정 | 전체화면 폴리곤으로 동일 연산 | **크롭 스킵** (원본 뷰 그대로, 복사 0) |
| 직사각형 ROI | `fillPoly` + 마스크 검사 | **마스킹 생략** — `_is_axis_aligned_rect` 판정 후 bbox 슬라이스만 (출력 비트 동일) |
| 병렬화 | 순차 for 루프 | **스레드풀** (`PE_NPU_PREPROCESS_WORKERS` / `..._MIN_BATCH` 공유, 새 env 없음) |
| torch 배치 입력 | — | 기존 torch 경로로 폴백 |

`service.py`의 `__del__`에 ROI 풀 정리를 추가했다. 패딩 규약(폴리곤 밖 114 그레이)은 그대로 유지했다.

### 5.2 전처리 (`preprocess.py`, `parallel_preprocess.py`)

- `resize_hwc_uint8()` — HWC uint8 1장 cv2 resize (이미 336이면 그대로 반환).
- `normalize_batch(..., nhwc=)` — (B,H,W,3) uint8 → float32. dtype 변환·정규화를 **배치 1회 in-place**로.
- `is_fusable()` — cv2 백엔드 + HWC uint8 3채널 numpy 리스트일 때만 새 경로.
- `ParallelPreprocessor._fused()` — resize만 스레드로 돌려 미리 잡아둔 `(B,336,336,3)` uint8 버퍼에 채우고 normalize 1회.

`torchvision.TF.convert_image_dtype`는 `.to(float)`와 `.div(255)`가 각각 새 텐서를 만든다. in-place로 바꾸면
같은 결과에 62ch 기준 **76 ms → 26 ms**다. (이 한 줄이 fused 이득의 절반이다.)

### 5.3 레이아웃 — NHWC 직행 (`PE_NPU_OUTPUT_NHWC`, 기본 on)

MXQ의 입력은 원래 HWC다 — `qbruntime.get_model_input_shape()` → `[(336, 336, 3)]`, calib도 그래서 HWC로 만든다
(`pe_npu/calib.py`). 그런데 컴파일 추적은 NCHW(`compile.py`: `torch.randn(1,3,336,336)`)이고, 추론 래퍼는
**TRTInference 드롭인 호환**을 위해 입력 계약을 NCHW로 고정했다. 그래서 전처리가 CHW를 만들고 추론기가 다시 HWC로
되돌리는 왕복이 생겼다. cv2 전처리는 애초에 HWC를 만드는데도 말이다.

```
before:  cv2 resize (HWC) → permute (CHW) → transpose (HWC) → NPU
after :  cv2 resize (HWC) →           그대로           → NPU
```

- `MultiNPUInferenceFull.infer` / `MXQInferenceFull.infer`가 **레이아웃을 판별**한다:
  `shape[-1]==3 && shape[1]!=3` → HWC(그대로), 아니면 NCHW(기존대로 변환). 336≠3이라 모호할 수 없고,
  애매하면 기존 계약(NCHW)으로 해석한다. **기존 NCHW 호출자는 전부 그대로 동작한다**(zero-mask 등).
- 전처리는 fused 경로에서만 NHWC를 내고 폴백 경로(torchvision·torch 입력·그레이스케일)는 계속 NCHW를 낸다.
  레이아웃이 섞여도 판별식이 추론기 한 곳에 모여 있어 문제없다.

되돌리기 비용은 단순 memcpy가 아니라 **stride 336² 간격의 캐시 비친화적 복사**라 62ch에서 61~72 ms였다.

### 5.4 모델 선택 — 양자화/튜닝 (`PE_NPU_QUANT`, `PE_NPU_TUNING`)

HF `PIA-SPACE-LAB/MXQ_NPU`에 세 세대 경로가 공존한다.

| 경로 | 설명 |
|---|---|
| `<scheme>/pe_full.mxq` | 기존 배포본 (`W8A16/<scheme>`와 sha 동일한 같은 파일) |
| `<quant>/<scheme>/pe_full.mxq` | 양자화별 |
| `<quant>/<tuning>/<scheme>/pe_full.mxq` | 양자화 × 튜닝(SWS/OPTQ) — 신규 |

`engine/assets.py`에 `hf_mxq_path(scheme, quant, tuning)`을 추가하고 env 두 개로 노출했다.
**둘 다 기본 빈 값 = 기존 배포본**이라 지정하지 않으면 동작이 지금과 완전히 같다.
로컬 캐시도 `assets/model/<quant>/<tuning>/<scheme>/`로 분리되어 옛 mxq를 조용히 재사용하는 사고를 막는다.
잘못된 조합(`TUNING`만 지정, 오타)은 명시적 `ValueError`.

```bash
PE_NPU_QUANT=W8A16 PE_NPU_TUNING=sws_optq   # 크기·속도 동일, 원본 대비 cos 0.9937 -> 0.9946
PE_NPU_QUANT=W4A16 PE_NPU_TUNING=sws_optq   # 크기 -42%, 더 빠름, cos 0.9654
```

교체하면 임베딩이 바뀐다(같은 모델의 다른 빌드). 현행 대비 실이미지 40장 기준 **cos 평균 0.9967 / 최소 0.9942**.
근거: [`NPU_pe_quant_tuning_matrix_120.md`](NPU_pe_quant_tuning_matrix_120.md).

### 5.5 트레이드오프 / 주의점

| 항목 | 내용 |
|---|---|
| **resize 백엔드가 실제로 바뀐다** | 설정은 원래 cv2였지만 실경로는 torchvision이 돌고 있었다. 수정 후 설정대로 cv2가 켜지며 실이미지 40장 기준 임베딩이 **cos 평균 0.9863 / 최소 0.9588 / 0.99 미만 24장** 이동한다. 임계값 근처 프레임에서 알람이 달라질 수 있다. 기존 수치 보존은 `PE_NPU_RESIZE=torchvision`(전처리 62ch 40 ms → 167 ms, 그래도 62ch/8카드 1 fps는 유지) |
| fused 적용 범위 | cv2 백엔드 + numpy HWC uint8 3채널만. torchvision·torch 입력·그레이/RGBA·float 입력은 기존 경로 폴백(5종 동작 확인) |
| 출력 | fused·NHWC 모두 기존과 **비트 동일**. ROI 미지정 크롭도 비트 동일 |
| 메모리 | 62ch 피크가 오히려 감소. 기존 ≈168 MB(장당 float 84 MB + stack 사본 84 MB) → 신규 ≈124 MB, NHWC면 permute 사본이 빠져 ≈104 MB |
| crop이 원본 뷰 | ROI 미지정이면 crop이 입력 프레임의 **뷰**다(복사 0). 하류가 crop을 in-place로 고치면 원본이 오염된다. 현재 경로는 읽기만 하며, 원본 무손상을 테스트로 확인했다 |
| 반환 타입 변화 | `process_batches_with_roi`가 torch → numpy. pe_npu 내부에서만 소비되지만 이 클래스를 상속·재사용하면 확인 필요 |
| `PE_NPU_DEVICE=cuda` | 예전엔 crop이 GPU에서 돌았고 이제 CPU numpy다. 기본값 cpu(NPU 서버는 GPU 없음)라 운영 영향 없음 |
| 직렬 구간 | normalize가 메인 스레드 배치 1회로 모인다. 62ch 26 ms(torch 30여 코어)지만 채널이 수백 단위면 이 지점이 직렬 병목이 된다 |
| 워커 수 의미 변화 | 워커가 이제 cv2 resize만 담당한다. 62ch 기준 w=8 40.3 / w=16 49.4 / w=32 54.3 ms로 적은 쪽이 낫지만, 차이가 e2e의 1.3%이고 폴리곤 ROI 크롭은 16~32에서 더 좋아 기본값 16을 유지했다 |
| 스레드 | ROI 풀 +16(전처리 풀과 설정 공유). `service.__del__`에 정리 연결 |
| 프로세스 모드 | `PE_NPU_PREPROCESS=process`는 fused를 타지 않고 3배 느리다. 쓰지 말 것(§8.1) |

## 6. 출력 동등성

전처리(fused vs 기존 장당 경로): 8ch·62ch 모두 `torch.equal` **True**, 최대 절대오차 0.00000000.

레이아웃(NHWC vs NCHW): 같은 crop을 두 레이아웃으로 전처리해 NPU까지 태운 임베딩이 **비트 동일**(빈 폴리곤·다각형 모두). 폴백 3종(torch 입력·float 입력·작은 배치)과 zero-mask NCHW 호출도 정상 동작 확인.

ROI 크롭 — NPU 임베딩까지 비교(8채널, 랜덤 프레임 = 경계 픽셀 차이가 최대로 드러나는 조건):

| 입력 | shape 일치 | 픽셀 불일치 | 임베딩 cos (수정 vs 현행) |
|---|:--:|---:|---:|
| 1280×720 · 빈 폴리곤 | ✔ | **0.0000 %** | **1.000000** |
| 1280×720 · 직사각형 ROI | ✔ | **0.0000 %** | **1.000000** |
| 1280×720 · 다각형(8각) | ✔ | 0.2193 % | 0.998772 |
| 1920×1080 · 빈 폴리곤 | ✔ | **0.0000 %** | **1.000000** |
| 1920×1080 · 직사각형 ROI | ✔ | **0.0000 %** | **1.000000** |
| 1920×1080 · 다각형(8각) | ✔ | 0.1453 % | 0.998871 |

폴리곤은 `cv2.fillPoly`와 ray-casting의 경계 판정 규칙 차이로 외곽 1픽셀 라인만 어긋난다.

## 7. 단계별 실측

### 7.1 ROI 크롭 (ms, 62채널)

ROI는 안 그려도 필드가 들어오므로, 실제로 무엇이 오느냐에 따라 비용이 달라진다.

| ROI 형태 | 720p 수정 전 | **720p 수정 후** | 1080p 수정 전 | **1080p 수정 후** |
|---|---:|---:|---:|---:|
| 빈 폴리곤(미지정) | 2601.8 | **1.0** | 2884.5 | **1.0** |
| 전체화면 직사각형 | 4538.5 | **5.1** | 4714.6 | **6.0** |
| 부분 직사각형 | — | **5.5** | — | **6.0** |
| 다각형(8각) | 4538.5 | **97.4** | 4714.6 | **123.8** |

채널 스윕(720p):

| ch | 수정 전 (빈 폴리곤) | **수정 후** | 수정 전 (다각형) | **수정 후** |
|---:|---:|---:|---:|---:|
| 8 | 327.6 | **0.9** | 579.2 | **18.0** |
| 16 | 666.0 | **1.1** | 1193.5 | **32.7** |
| 32 | 1456.4 | **1.7** | 2440.4 | **47.0** |
| 48 | 2005.2 | **2.7** | 3464.2 | **68.0** |
| 62 | **2601.8** | **2.3** | **4538.5** | **82.2** |

- 빈 폴리곤은 크롭 자체를, 직사각형은 마스킹을 건너뛴다. 둘 다 출력이 **비트 동일**하다.
- 실제 마스킹이 필요한 건 다각형뿐이고, 그 경우만 경계 1픽셀 라인이 달라진다.
- **수정 전 비용은 화소수에 비례한다.** 1280×720(0.92 MP)과 1108×828(0.92 MP)이 거의 동일했고
  1920×1080(2.07 MP)에서 커졌다. "해상도 종류"가 아니라 "화소수 × 꼭짓점수"가 변수다.

### 7.2 전처리 (ms, 62ch, 720p, 빈 폴리곤)

| 방식 | 시간 | 평균 코어 |
|---|---:|---:|
| 기존 장당 호출, thread w=16 | 176.2 | 10.2 |
| 기존 장당 호출, thread w=32 | 166.8 | 11.1 |
| **fused, w=8** | **40.3** | 34.9 |
| **fused, w=16 (기본값)** | **49.4** | 30.2 |
| fused, w=32 | 54.3 | 30.3 |

내부 분해(62ch, w=8): resize 11.7 ms + normalize 26 ms. resize 백엔드 비교(장당 경로 기준, 62ch)는
cv2 145.9 ms vs torchvision 209.7 ms였다.

### 7.3 e2e (ms) — ROI + 전처리 + NPU, 720p · 빈 폴리곤

수정 전 / **최종(ROI+전처리+NHWC)**:

| ch | 1카드 | 4카드 | 8카드 |
|---:|---:|---:|---:|
| 1 | 134 / **123** | 134 / **123** | 134 / **124** |
| 8 | 867 / **503** | 498 / **133** | 495 / **131** |
| 16 | 1744 / **1007** | 1002 / **263** | 883 / **144** |
| 32 | 3632 / **2027** | 2138 / **537** | 1897 / **293** |
| 48 | 5267 / **3038** | 3028 / **798** | 2664 / **434** |
| 62 | 6770 / **3928** | 3911 / **1062** | 3424 / **569** |

채널당 한계 비용: 1카드 63.4 ms / 4카드 17.1 ms / **8카드 9.2 ms**.
ROI 폴리곤이 걸리면 62ch/8카드 648 ms, 1080p+ROI면 720 ms 선이다.

### 7.4 수용 채널 수 (720p, 빈 폴리곤)

| 목표 | 1카드 | 4카드 | 8카드 |
|---|---|---|---|
| 1 fps/채널 | 9 → **15 ch** | 16 → **59 ch** | 18 → **62 ch 이상** (62ch 569 ms, 여유 43%) |
| 2 fps/채널 | 4 → **8 ch** | 8 → **30 ch** | 8 → **55 ch** |

- **62채널 1 fps는 8카드로 충분**(569 ms). 4카드도 1062 ms로 거의 턱걸이(0.94 fps)까지 올라왔다.
- 8카드의 "62ch 이상"은 62ch까지가 실측이고 그 위는 외삽이다(마진 기준 ~100 ch).

## 8. 남은 최적화 여지

### 8.1 워커 수·프로세스 풀 — 더 쓸 게 없다

수정 전 코드로 62ch 워커 1~64 스윕: 200~250 ms에서 평평하고 96코어 중 10~13코어만 썼다. 원인은 스레드 수가 아니라
장당 torch 호출(§4.4)이었고, fused로 바꾼 뒤에는 30~35코어를 쓴다. 워커를 더 늘려도 이득 없다(§7.2).

- `cv2.setNumThreads` 1/2/4 → 227 / 213 / 226 ms. 차이 없음(336×336은 cv2 내부 병렬이 안 먹힘). 1 유지.
- 프로세스 풀: thread 16 = 226 ms vs **process 16 = 673 ms, process 32 = 767 ms**. 62프레임 IPC 왕복(170 MB)이 이득을 다 먹는다.
  spawn 방식이라 호출 측 `__main__`에 가드가 없으면 재귀 스폰까지 난다. **`PE_NPU_PREPROCESS=process`는 쓰지 말 것.**

### 8.2 (완료) CHW ↔ HWC 왕복 제거

§5.3에서 처리했다. 62ch/8카드 infer 단계 651 ms를 쪼개면 **CHW→HWC 되돌리기 72.4 ms** + 순수 NPU 호출 499 ms였고,
NHWC 직행으로 되돌리기가 0이 됐다(전처리도 permute 사본 4 ms 절약). e2e 690 → 572 ms.

### 8.3 전처리와 NPU 추론 겹치기 (62ch/8카드 −14 %) ★ 다음 후보

`_detect`는 CPU 전처리와 NPU 추론을 순차 실행한다. 다음 배치 전처리를 별도 스레드로 미리 도는 더블버퍼
프로토타입으로 819 → 704 ms. `ServiceBase`의 큐/스레드 구조를 건드려야 하고 배치 한 개만큼 지연이 늘어나는
트레이드오프가 있다. 8.2보다 뒤에 볼 것.

### 8.4 상류에서 줄이기

남는 CPU 비용은 resize(채널당 0.2 ms @720p, fused 기준)와 폴리곤 ROI 크롭(채널당 1.3 ms)이다. 둘 다 원본 화소수 비례다.
디코더에서 작게 받으면 그만큼 줄지만 **ROI 좌표계가 원본 기준이라 축소 배율만큼 ROI 좌표도 스케일**해야 한다.
수정 후 1080p↔720p e2e 차이는 62ch/8카드에서 70~80 ms라 더 이상 최우선 과제는 아니다.

## 9. 측정 함정 — torch OMP 간섭

수정 전 ROI 단계의 비용은 **같은 프로세스에서 스레드풀 전처리를 한 번이라도 돌렸는지에 따라 10배 이상 달라졌다.**

| 62ch, 720p, 빈 폴리곤 (동일 프로세스, 순서대로) | 시간 |
|---|---:|
| ① 프로세스 시작 직후 ROI만 반복 | 165.5 ms |
| ② 전처리(스레드 16)와 번갈아 실행 ← **실서비스 패턴** | **2668.9 ms** |
| ③ ②를 거친 뒤 다시 ROI만 반복 | 2748.4 ms |

`torch_crop_region`은 torch 텐서 연산이라 intra-op 스레드(이 호스트 기본 48개)로 병렬화된다. 깨끗한 상태에서는
48코어를 끌어 써서 165 ms지만, 파이썬 스레드풀에서 torch 연산이 호출되고 나면 이후 메인 스레드의 elementwise
병렬화가 무너지고 **원상 복구되지 않는다**(③). torch 스레드를 직접 줄여도 같은 방향이다:

| torch intra-op threads | 48 | 16 | 8 | 4 | 1 |
|---|---:|---:|---:|---:|---:|
| 수정 전 ROI 62ch | 169 ms | 224 ms | 308 ms | 568 ms | 990 ms |

**영향 범위 확인**: 이 간섭이 다른 단계까지 번지는지 확인하려고 유사도 matmul `(62,1024)@(1024,13449)`를
깨끗한 프로세스 / 장당 전처리 후 / fused 전처리 후로 나눠 쟀다 — **1.28 / 0.94 / 1.11 ms로 차이 없다.**
GEMM은 자체 스레딩(MKL)이라 영향을 받지 않고, 무너지는 것은 elementwise 계열이다. 즉 이 함정의 피해자는
수정 전 ROI 크롭 하나였고, ROI에서 torch를 걷어낸 지금은 **간섭원 자체가 사라졌다**(워커 스레드는 cv2만 돌린다).

남는 교훈은 측정 방법론 쪽이다. 실서비스는 매 배치 전처리를 돌리므로 ②가 실제 값이고, 이 문서의 수정 전 수치는
모두 ② 패턴 + 단일 타이머 e2e로 교차 검증했다. **단계를 격리해서 잰 숫자만으로 판단하면 안 된다** — 전처리 ③이
ROI를 빼고 측정해서 병목을 놓친 것과 같은 종류의 함정이다.

## 10. 재현

```bash
# 환경: conda pe_npu_host (qbruntime 1.2.0) + Product-AI-mono 체크아웃(pe_npu 수정 반영본)
conda activate pe_npu_host
cd <Product-AI-mono>
export PIA_PACKAGES=$PWD/packages
export MXQ=~/.cache/huggingface/hub/models--PIA-SPACE-LAB--MXQ_NPU/snapshots/<rev>/global4/pe_full.mxq

S=<AX_NPU>/reports/scripts
python $S/verify_pe_roi_crop_equiv.py   # §6  ROI 출력 동등성(픽셀 + 임베딩 cos)
python $S/verify_pe_fused_preprocess.py # §6  전처리 fused 비트 동일 + 폴백 5종 + e2e
python $S/bench_pe_roi_shapes.py        # §7.1 ROI 형태별(빈 폴리곤/직사각형/다각형) 크롭 비용
python $S/verify_pe_roi_rect_fastpath.py # §6  직사각형 fast-path 비트 동일성
python $S/bench_pe_roi_stages.py        # §7.1 ROI·resize 채널 스윕 (PE_NPU_RESIZE=cv2|torchvision)
python $S/bench_pe_roi_capacity.py      # §7.3 카드수(1/4/8) × 채널 스윕
python $S/bench_pe_roi_e2e.py           # §1, §9 단일 타이머 e2e + OMP 간섭 재현
python $S/bench_pe_cpu_headroom.py      # §8.1, §8.3 워커/cv2 스레드/프로세스 풀/파이프라이닝
python $S/verify_pe_nhwc_layout.py      # §5.3 NHWC 기본 on 검증(ROI 유무 × 폴백 × e2e)
python $S/bench_pe_transpose_cost.py    # §8.2 CHW→HWC 왕복 비용
```

각 스크립트는 `LegacyROI`(수정 전 torch 경로)를 자체 재현하므로, 수정 반영본 한 벌로 수정 전/후를 같은 세션에서 비교할 수 있다.
`bench_pe_cpu_headroom.py`는 프로세스 풀을 쓰므로 **`if __name__ == "__main__":` 가드를 제거하면 재귀 스폰이 난다**(§8.1).

## 11. 관련 문서

- [`NPU_preprocess_3_cv2_decision.md`](NPU_preprocess_3_cv2_decision.md) — cv2 resize 채택(이 문서에서 "실경로 미적용"으로 정정)
- [`NPU_preprocess_1_parallel.md`](NPU_preprocess_1_parallel.md) — 전처리 스레드/프로세스 병렬화 1차 시도
- [`NPU_npu_intrusion_e2e_opt.md`](NPU_npu_intrusion_e2e_opt.md) — 같은 ROI 크롭 병목을 침입 모듈에서 먼저 제거한 사례(210→17 ms)
- [`NPU_pe_multicard_62ch_full.md`](NPU_pe_multicard_62ch_full.md) — NPU 추론 단계의 카드 스케일링

---

*실측 2026-09-10, 8×ARIES2 / 96 CPU / qbruntime 1.2.0 / W8A16 global4. 코드: Product-AI-mono `aiprod-313-npu-버그-해결-전처리파트`.*
