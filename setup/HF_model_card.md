---
license: apache-2.0
library_name: mobilint-aries
tags:
  - clip
  - vision-encoder
  - mobilint
  - aries
  - npu
  - int8
---

# PE-Core-L14-336 — Mobilint ARIES NPU (MXQ)

[Meta Perception Encoder **PE-Core-L14-336**](https://huggingface.co/facebook/PE-Core-L14-336)
(CLIP ViT-L/14, 336px) vision encoder를 Mobilint **ARIES** NPU용 **MXQ**로 컴파일한 산출물.

- **입력 → 출력**: image `(1,3,336,336)` → embedding `(1,1024)` — **전부 NPU에서 수행 (full NPU)**.
- **정확도**: 원본 PyTorch 대비 cosine **≈ 0.99** (COCO holdout 0.9905 / 도메인 0.9889).
- qbruntime만 있으면 추론 가능 (qbcompiler·원본 가중치 불필요).

## 핵심 — attention pooling head의 INT8

CLIP attention pooling head는 그냥 INT8로 양자화하면 **QKᵀ matmul의 outlier** 때문에 정확도가
붕괴한다(cos≈0.46). 해결: attention score MatMul(QKᵀ)만 **16bit**로 올린다(`--qk16`). 그러면
trunk + head 전체를 NPU에 올리고도 cos≈0.99를 얻는다. (Mobilint 기술지원 확인)

## 파일 구조 — 코어모드별

각 폴더에 동일 모델(동일 calib·정확도), `inference_scheme`만 다르다(레이턴시/처리량 특성 차이).

| 폴더 | scheme | 특성 |
|------|--------|------|
| `single/`  | single  | 8코어 독립 — async throughput 최선 |
| `multi/`   | multi   | 코어 클러스터 |
| `global4/` | global4 | 4코어가 1추론 분담 — 중간(8~16ch 안정) |
| `global8/` | global8 | 8코어 전부 1추론 — **단건 latency 최소(71ms)** |

- 각 폴더: `pe_full.mxq` + `CALIBRATION.md`(컴파일/calib 상세).
- 레거시(hybrid): 루트 `pe_feat.mxq`(NPU trunk만) + `pe_pool_head.pt`(CPU pool head).

## Calibration

- **COCO val2017 200장** (일반 객체/장면, 다양성). `pe_npu.preprocess_image`와 동일 전처리(336 resize + /255 + normalize 0.5), HWC.
- 양자화: INT8 + attention score MatMul(QKᵀ) 25개만 16bit. method=1, output=1(per-channel).

## 사용

```python
import pe_npu   # github.com/jungseoik/AX_NPU
m = pe_npu.MXQInferenceFull.from_hf(scheme="global8")   # 단건 latency 최소
emb = m.infer(pe_npu.preprocess_image("img.jpg")[None])  # (1, 1024)
```

- 코어모드 선택: 단건/저채널 실시간 → `global8`, 고채널 throughput → `single`/`global4`.
- 컴파일러·원본 가중치 없이 동작. NPU(`/dev/aries*`) + qbruntime + 인터넷(최초 1회)만 필요.

*컴파일 qbcompiler 1.1.2 (target aries2). full NPU, QKᵀ 16bit, cos≈0.99.*
