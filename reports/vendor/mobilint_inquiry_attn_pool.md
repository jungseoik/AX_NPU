# Mobilint 기술지원 문의 — Attention Pooling Head의 NPU(INT8) 양자화 정확도 붕괴

> ✅ **[해결됨]** 이 문의는 해결되었습니다. 원인 = QKᵀ matmul의 outlier, 해결 = 그 score matmul만
> 16bit override → full 모델도 NPU에서 cos 0.996. **→ [`mobilint_resolution_attn_pool.md`](mobilint_resolution_attn_pool.md)**
> (이 문서는 문의 당시 기록으로 보존.)

> **목적**: ViT-L/14 vision encoder를 ARIES NPU용 INT8 MXQ로 컴파일했는데, **24개 transformer block
> (trunk)은 INT8에서 정상 동작(cos ≥0.99)**하나, 마지막 **attention pooling head**를 INT8에 포함하면
> 출력 정확도가 붕괴(원본 대비 cos ≈ 0.46)합니다. 현재는 이 head만 **CPU(float)** 로 우회(hybrid)해
> cos 0.9987을 얻고 있으나, CPU 처리가 다채널에서 병목입니다. **원인과 NPU 포팅/해결 방법**을 문의합니다.
> (모델 가중치/ONNX는 비공개. 아키텍처 일반 구조와 측정치만 기재합니다. 필요 시 구조 공유 협의 가능.)

---

## 1. 환경

| 항목 | 버전 |
|------|------|
| NPU | ARIES MLA100 PCIe (Aries2), 펌웨어 1.2.5 |
| 드라이버 / 런타임 | mobilint-aries-driver 1.13.0 / qbruntime 1.2.0 |
| 컴파일러 | qbcompiler 1.1.2 |
| 모델 | CLIP 계열 ViT-L/14 vision encoder (hidden 1024, 24 transformer blocks, 입력 336×336, 토큰 577) |

## 2. 구성: trunk(NPU) + attention pooling head(현재 CPU)

```
입력 이미지 (1,3,336,336)
   │
   ├─ [trunk] 24 transformer blocks (self-attention, 577 tokens, hidden 1024)   ← NPU INT8: 정상 (cos ≥0.99)
   │     출력: 토큰 피쳐 (1, 577, 1024)
   │
   └─ [attention pooling head] (1,577,1024) → 임베딩 (1,1024)                    ← INT8 포함 시 붕괴(cos≈0.46)
```

trunk는 INT8로 잘 양자화됩니다. **문제는 head(attention pooling)** 입니다.

## 3. Attention Pooling Head — 연산 상세

학습된 **probe 토큰 1개**가 577개 토큰 피쳐에 **cross-attention**해서 1개 벡터로 풀링하는 구조입니다
(CLIP/Perceiver 계열의 attention pooling). 구조:

- **embed_dim = 1024, num_heads = 16 (head_dim 64), probe = 1개 토큰** (`nn.Parameter`, shape `(1,1,1024)`)
- forward (의미):
  ```
  q = probe                              # (1, 1, 1024)        ← query 길이 = 1 (고정 학습 파라미터)
  k = x ; v = x                          # (1, 577, 1024)      ← key/value = trunk 출력 577 토큰
  attn = MultiheadAttention(q, k, v)     # cross-attn, 16 heads → (1, 1, 1024)
  x = attn + MLP(LayerNorm(attn))        # residual MLP: Linear(1024→4096) → GELU → Linear(4096→1024)
  embedding = x @ proj                   # 최종 projection Linear (1024→1024)
  ```
- 연산 분해(컴파일 시 MultiheadAttention을 명시적으로 풀어 줌):
  `Linear(q) / Linear(k) / Linear(v)` → `scaled_dot_product_attention( softmax(qkᵀ/√64) · v )` →
  `out_proj(Linear)` → `LayerNorm` → `MLP(Linear+GELU+Linear)` → residual add → `proj(Linear)`
- **trunk의 self-attention과의 차이**: trunk는 577×577 self-attention(query 577개)인데, **이 head는
  query가 단 1개(probe), key/value만 577개인 cross-attention**입니다.

### 3-1. 연산 그래프 (시각 자료)

이 head만 따로 ONNX로 export(랜덤 가중치, 구조만)해 분석한 연산 흐름입니다.

![attention pooling 연산 흐름](assets/attn_pool_flow.png)

ONNX 그래프 op 분포(head 단독, opset 17):

| op | 개수 | 비고 |
|---|---:|---|
| MatMul | 9 | q/k/v/out_proj/mlp/proj Linear + attention 2개 |
| Add | 8 | bias / residual |
| Softmax | **1** | attention (key 577개 위에서) |
| LayerNormalization | 1 | MLP 직전 |
| Erf | 1 | GELU |
| Div / Sqrt / Mul | 2/3/4 | 스케일(1/√64) 및 정규화 |
| Reshape / Transpose | 5 / 4 | head 분할/병합 |

> **head 단독 ONNX 파일**(`attn_pool_head.onnx`, 구조만·실가중치 아님)을 [Netron](https://netron.app)에서
> 바로 열어 보거나 티켓에 첨부 가능합니다(요청 시 제공). trunk/실모델은 비공개 유지됩니다.

## 4. 증상 (측정)

| 구성 | 원본 PyTorch 대비 코사인 | 비고 |
|------|:---:|------|
| trunk만 NPU INT8 + head CPU(float) | **0.9987** | 현재 운영(hybrid). 정상 |
| trunk + head **전부 NPU INT8** | **≈ 0.46** | **붕괴** |

- patch 적용 후 float(양자화 前) 정확도는 1.000(무손실)이고, calibration 레이아웃/데이터도 점검했으나,
  **head를 INT8에 넣는 순간 0.46으로 떨어집니다.** → head의 INT8 양자화 자체가 원인으로 추정됩니다.
- trunk(self-attention 24층)는 INT8에서 멀쩡하므로, **attention pooling 특유의 무언가**(query 1개 구조,
  특정 연산의 activation 분포 등)가 INT8에 민감한 것으로 보입니다.

## 5. 영향 (CPU 우회의 비용)

head를 CPU로 돌려 정확도는 확보했으나, **다채널(배치) 처리 시 CPU attention pooling이 병목**입니다.
- 7대 NPU 분산 환경, 56채널(최대 배치) 기준 e2e ≈ 2.1s 중 **CPU attention pooling이 약 0.58s(≈28%)** 차지
  (채널별 직렬 처리). NPU trunk(0.57s)와 맞먹습니다. → head를 NPU로 올리면 병목 해소가 기대됩니다.

## 6. Mobilint Model Zoo 자체 조사 (문의 범위를 head로 좁힘)

`mblt-model-zoo 1.3.1` image_classification 카탈로그를 확인한 결과:

- **ViT-Large 트렁크는 이미 지원됨**: `ViT_Large_Patch16_224/384`, `ViT_L_16/32`, `DeiT3_Large_Patch16`,
  `FlexiViT_Large` 등. → **ViT-L transformer trunk(self-attention)는 NPU INT8 지원이 확인**되며,
  이는 저희 trunk가 cos 0.99로 정상 동작하는 것과 일치합니다.
- **그러나 이들은 ImageNet 분류 모델 = CLS-token pooling**입니다(분류 토큰 1개 → linear classifier).
  저희 모델은 **CLIP 계열 attention pooling head(probe cross-attention)** 로, **풀링 방식이 다릅니다.**
  Model Zoo에는 attention pooling head를 쓰는 vision 모델이 없는 것으로 보입니다.
- 또한 Zoo에는 **patch14가 없고**(patch16/32만), 저희 모델은 patch14입니다(이는 conv stem 차이로 head 이슈와는 별개).

→ 즉 **막히는 지점은 ViT trunk가 아니라 attention pooling head**(CLS-token과 다른 구조)로 명확히 좁혀집니다.

## 7. 질문

1. **attention pooling(probe 1개 cross-attention)** 구조를 ARIES NPU에서 **INT8로 정확도 손실 없이**
   지원/컴파일하는 것이 가능할까요? 가능하다면 권장 방법이 무엇일지 안내 부탁드립니다.
2. trunk의 self-attention은 INT8에서 정상인데 **이 head만 INT8에서 붕괴(cos 0.46)하는 원인**이
   무엇일지 짚어 주실 수 있을까요? (어떤 연산이 문제일까요 — single-query attention의 softmax? GELU?
   LayerNorm? activation outlier? probe 파라미터의 dynamic range? 중 짐작되는 부분이 있을까요?)
3. 이 head에 대해 **부분 정밀도(예: activation/weight 16bit override)** 를 적용하는 방법이 있을까요?
   (예: `compile_config`로 attention-pooling 레이어만 16bit로 지정) 그 경우 권장 설정은 어떻게 될까요?
4. **per-channel calibration / EquivalentTransformation(SmoothQuant류) / zeropoint 비대칭** 등
   양자화 옵션 중 이 구조에 효과적인 조합이 있을까요?
5. Model Zoo의 ViT-Large는 **CLS-token 분류**라 INT8이 지원되는데, **CLIP attention pooling head**는
   별도로 지원되지 않는 것일까요? (즉, 미지원 사유가 attention pooling 구조 자체일지 궁금합니다.)
6. Qwen2-VL처럼 특정 구조에 대한 **SDK 내부 모델별 패치**가 필요한 사안일까요? 그렇다면 일반 CLIP/ViT의
   attention pooling head를 위한 가이드나 패치를 받아볼 수 있을지도 함께 여쭙고 싶습니다.

## 8. 재현 정보 (요청 시 협의)

- 모델 가중치/ONNX는 비공개이나, **위 head 구조만 동일하게 재현한 최소 예제**(랜덤 가중치) 또는
  qbcompiler 호출 인자/컴파일 로그는 공유 가능 여부 협의하겠습니다.
- 측정에 쓴 calibration 설정(레이아웃 HWC, output 모드 등)도 함께 제공 가능합니다.

---

*작성 2026-06. 관련 자체 분석: `reports/design/SOLUTION_single_io_compile.md`(hybrid 해결),
`reports/performance/NPU_coremode_pipeline_e2e.md`(attention pooling CPU 병목 실측).*
