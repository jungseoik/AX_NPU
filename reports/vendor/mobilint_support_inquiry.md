# Mobilint 기술지원 문의 (초안)

> 목적: 커스텀 **RoPE2D 기반 ViT-L/14 vision encoder**를 단일 입력/단일 출력 MXQ로
> 컴파일하려 했으나, 24개 transformer block이 독립 서브그래프로 분할되어 실사용 추론이
> 불가합니다. 원인과 해결 방법을 문의합니다.
> (모델 세부/가중치는 비공개. 아키텍처 일반 특성만 기재합니다.)

---

## 1. 환경

| 항목 | 버전 |
|------|------|
| NPU | ARIES MLA100 PCIe (Aries2), 펌웨어 1.2.5 |
| 드라이버 | mobilint-aries-driver 1.13.0 |
| 런타임 | qbruntime 1.2.0 |
| 컴파일러 | qbcompiler 1.1.2 (`mobilint/qbcompiler:1.1-cuda12.8.1-ubuntu22.04`) |
| 호스트 | Ubuntu 24.04, x86_64 |

## 2. 대상 모델 (일반 특성만)

- 구조: **Vision Transformer (ViT-L/14)** — 24 transformer blocks, hidden 1024
- 입력: 단일 이미지 텐서 `(1, 3, 336, 336)` (NCHW), 정규화된 float32
- 출력: 단일 임베딩 `(1, 1024)` (attention pooling head)
- 토큰 수: 577 (24×24 patches + 1 cls)
- **position embedding: 2D RoPE (rotary)** — 표준 CLIP/DeiT의 absolute learned PE가 아님
- ONNX는 **입력 1개 / 출력 1개**로 정상 export됨 (onnx.checker 통과, onnxruntime 추론이
  원본 PyTorch와 cos=1.000000으로 일치)

## 3. 증상

위 ONNX를 `mxq_compile(backend="onnx", inference_scheme="single", use_random_calib=True)`로
컴파일하면 **컴파일은 성공**하나, 런타임에서 입출력이 분할되어 노출됩니다:

```python
m = qbruntime.Model("model.mxq"); m.launch(acc)
m.get_model_input_shape()
# → 25개: [(1,577,1024)] × 24  +  (336,336,3) × 1   ← 마지막만 실제 image
m.get_model_output_shape()
# → 25개: (1,1,1024)  +  (3,577,1024) × 24          ← 첫째만 최종 임베딩, 나머지는 block별 출력
```

- 단일 image 입력만으로 `infer()` 호출 시: `ValueError: Input shape is invalid.`
- 24개 중간 텐서를 zeros로 채워 25개를 모두 주면 추론은 되지만, 24개 입력이 결과에 영향을
  미칩니다 (같은 image에 24개만 zeros↔random 변경 시 첫 출력 cos=0.958).
  → 즉 24개 block 입력을 외부에서 올바르게 공급해야 하며, **단일 image→embedding 추론이 불가**합니다.

## 4. 시도한 것과 결과

| # | 시도 | 결과 |
|---|------|------|
| 1 | ONNX 직접 (`backend=onnx`) | 컴파일 OK, 입출력 25개 분할 |
| 2 | `mblt_compile`(ONNX→MBLT) 후 `mxq_compile`(MBLT) | 동일 — quantize 단계 `The size of modelInputNames is expected to be 1 but got 25` |
| 3 | `backend="torch"` 직접 컴파일 | FX trace 실패: RoPE 모듈 동적 호출에서 `NameError: module is not installed as a submodule` |
| 4 | **RoPE를 고정 해상도 cos/sin 상수로 치환** 후 ONNX (If 노드 0개로 단순화) | **여전히 25분할** → RoPE가 분할의 주원인이 아님을 확인 |
| 5 | `split_blocks`/`split_parts` 검토 | 분할을 늘리는 옵션이라 해당 없음 |
| 6 | Directory / npy-list calibration | `Directory calibration only supports single-input models. modelInputNames.size() = 25` |

ONNX 그래프 op 분포(단순화 후): Softmax 25, MatMul 150, LayerNormalization 51, Erf 25.
→ Softmax 25개 = 24 transformer block attention + 1 attention-pooling. 분할이 **block 경계**에서
일어나는 것으로 보입니다.

## 5. 비교 관찰

- **absolute position embedding** ViT(예: DeiT/FlexiViT, SigLIP vision tower)는 단일 입출력으로
  정상 컴파일됨 (Model Zoo 기준).
- **Qwen2-VL vision encoder**(RoPE 기반 ViT)는 qbcompiler SDK 내부의 전용 패치
  (`fx_hf_extensions/.../qwen2vl.py`의 `VisionModelForQwen2VL.set_grid_thw()`로 cos/sin을
  buffer 상수화 + attention/patch-embed 패치)를 통해 단일 입출력으로 컴파일됨.
- 즉 **RoPE ViT를 단일 입출력으로 만드는 경로가 "모델별 SDK 패치"에 의존**하는 것으로 이해됩니다.

## 6. 질문

1. ViT의 **24개 transformer block이 독립 서브그래프로 분할되어 입출력으로 노출되는 정확한 원인**은
   무엇입니까? (특정 연산 미지원? 메모리? attention 구조?) 컴파일 로그에는 분할/offload 메시지가
   출력되지 않습니다.
2. 이 모델(또는 일반적인 RoPE2D ViT-L/14)을 **단일 입력(image)/단일 출력(embedding) MXQ**로
   컴파일하는 **공식 방법 또는 컴파일 옵션**이 있습니까?
3. RoPE를 cos/sin 상수로 치환해도 분할이 동일한데, **분할을 유발하는 연산**을 특정해 주실 수 있습니까?
   (Softmax/attention, LayerNormalization, attention-pooling 중 무엇입니까?)
4. Qwen2-VL처럼 SDK에 **모델별 패치가 필요**한 사안입니까? 그렇다면 일반 CLIP/ViT(absolute PE가
   아닌 RoPE) vision encoder를 위한 패치 또는 가이드 제공이 가능합니까?
5. 분할된 형태(입출력 25개)를 그대로 두고 **런타임에서 단일 image→embedding으로 추론**할 수 있는
   API(서브그래프 자동 연결/파이프라인)가 있습니까?

## 7. 재현 정보 (요청 시 제공 가능)

- 단일 입출력 ONNX (RoPE 상수화 버전, op 분포 위와 동일) — 모델 가중치 제외하고 그래프 구조만
  공유 가능 여부 협의 필요.
- `mxq_compile` 호출 인자 및 전체 컴파일 로그.

---

*작성: 2026-06-15. 내부 분석 상세는 `reports/testing/test_results.md` 참조.*
