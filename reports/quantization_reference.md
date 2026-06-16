# Mobilint ARIES NPU 양자화 레퍼런스 (보고용)

> 질문: "ARIES NPU는 bfloat16/float32 입력을 받던데, 그럼 float로 추론하는 것 아닌가?
> 꼭 양자화(INT8)해야 하나?"
> 답: **입력 텐서는 float로 받지만, NPU 내부 연산은 INT8 정수로만 수행된다. 양자화는 필수다.**

본 문서는 위 질문에 대한 근거와 이유를, 공식 문서 + 실제 컴파일 산출물(MXQ) 메타데이터로
정리한 것이다.

---

## 1. 결론

| 구분 | 내용 |
|------|------|
| **입력 텐서 dtype** | Float32 / Float16 / UINT8 / INT8 로 **받아줌** (사용자 편의) |
| **NPU 내부 연산 dtype** | **INT8 정수 전용** (가중치·activation 모두 양자화) |
| **양자화 필요 여부** | **필수.** float/bf16 네이티브 추론 경로는 존재하지 않음 |

"입력을 float로 받는다"와 "float로 연산한다"는 **다른 얘기**다. 런타임은 입력을 float로
받아주지만, NPU 하드웨어에 올리는 순간 INT8로 양자화하여 정수 연산기로 처리한다.

---

## 2. 근거 (Evidence)

### 2-1. 공식 문서: 지원 입력 데이터 타입

`docs/programming_guide.md`:
> "현재 지원되는 입력 데이터 타입은 `UINT8`, `INT8`, `float32` 입니다."

이는 **런타임 API가 받는 입력 텐서의 형식**을 말한다 (NDArray<float> 등). 연산 정밀도가
아니라 입구에서 받아주는 형식이다.

### 2-2. 실제 MXQ 메타데이터 (결정적 증거)

컴파일한 PE-Core-L14-336 vision encoder MXQ를 `mblt-mxqtool show`로 확인한 결과:

```
Format Version:   0x70000           # MXQv7
Hardware Version: Aries2
  Input Section #1: (ch-wise scale, symmetric)
    User DataType:  Float32, Float16, Int8      # ← 입력으로 받아주는 형식
    NPU DataType:   Int8                         # ← NPU 내부 연산 형식
    Scale:          0.007861, 0.007825, ... (size: 1024)   # 양자화 스케일 (채널별)
    Zero Point:     0, 0, 0, ... (size: 1024)              # 대칭 양자화 (zero-point 0)
```

- `User DataType`에 Float32/Float16이 있어 **입력은 float로 줄 수 있다.**
- 그러나 `NPU DataType`은 **`Int8` 단 하나** — 내부 연산은 INT8뿐이다.
- `Scale` / `Zero Point`가 존재한다는 것 자체가, 입력 float 값을 INT8 정수로 변환하는
  **양자화 매핑이 모델에 구워져 있다**는 증거다 (channel-wise, symmetric quantization).

> 즉 입력으로 들어온 float 값은 `q = round(x / scale)` 로 INT8로 변환된 뒤 NPU 정수
> 연산기에서 처리된다. float 그대로 곱셈/덧셈하는 경로는 없다.

---

## 3. 이유 (Why) — 하드웨어 설계

`docs/advanced_usage.md` / `docs/multicore.md`에 따르면 ARIES는 8개의 NPU 코어로 구성된
**정수 연산 가속기**다. NPU가 INT8 전용인 이유:

- **전력·면적 효율**: 정수 곱셈기(MAC)는 부동소수점 유닛보다 훨씬 작고 전력을 적게 쓴다.
  ARIES MLA100은 25W로 80 TOPS를 내는데, 이는 float 유닛을 빼고 정수 연산에 집중한 결과다.
- **엣지 AI 타깃**: 추론(inference) 전용 가속기는 학습용 GPU와 달리 float 정밀도가 불필요.
  INT8로도 충분한 정확도를 유지하면서 처리량·전력을 크게 개선한다.

대조적으로 GPU(예: RTX PRO 6000)는 float 유닛이 풍부해 bf16/fp16/fp32를 네이티브로 돌린다.
그래서 GPU에선 양자화가 선택, NPU에선 필수다.

---

## 4. 흔한 오해 정리

| 오해 | 사실 |
|------|------|
| "float32 입력을 받으니 float로 추론한다" | 입력 형식 ≠ 연산 형식. 내부는 INT8 |
| "`weight_dtype=float16` 옵션이 있으니 float 가능" | 그건 **컴파일/calibration 과정의 호스트 연산 정밀도**. NPU 실행 정밀도 아님 |
| "양자화는 정확도를 위해 선택적으로 하는 것" | NPU에 올리려면 **필수 관문**. 안 하면 실행 자체 불가 |
| "차원(shape)만 맞으면 된다 (TRT FP16처럼)" | TRT FP16은 calibration 불필요했던 것. INT8은 값 범위 매핑(calibration) 필요 |

---

## 5. 실무 함의

- float 모델을 그대로 돌리고 싶으면 → NPU가 아니라 **GPU/CPU**에서 원본 모델 실행.
- NPU 가속을 받으려면 → **INT8 양자화 컴파일(MXQ) 필수.**
- 정확도가 걱정이면 → "양자화를 안 하는" 게 아니라 "잘 하는" 방향:
  8bit 유지(4bit 지양), per-channel + percentile calibration, 실제 도메인 데이터로 calibration.

---

## 부록: 검증 재현 명령

```bash
# MXQ의 User/NPU DataType 직접 확인
mblt-mxqtool show <model.mxq> | grep -E "User DataType|NPU DataType|Scale|Zero Point"
```

근거 문서: `docs/programming_guide.md`, `docs/advanced_usage.md`, `docs/multicore.md`,
`compilation/_guides/01_about_quantization_config.KR.md`
