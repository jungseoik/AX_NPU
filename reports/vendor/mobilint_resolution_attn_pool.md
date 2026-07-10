# Attention Pooling Head INT8 붕괴 — 원인 규명 및 해결 (RESOLVED)

`mobilint_inquiry_attn_pool.md`로 문의했던 attention pooling head의 INT8 양자화 붕괴가
**Mobilint(전근우님) 기술지원으로 해결**되었다. 핵심: **attention score MatMul(QKᵀ) 한 노드의
activation을 16bit로 올리면 복구**된다. 이 문서는 원인·해결·우리 측 적용 결과를 기록한다.

## 타임라인
1. 증상: full 모델(trunk+head) NPU INT8 → 원본 대비 **cos 0.46**. trunk만 INT8(hybrid)은 0.997.
2. 격리: head만 떼어 INT8 컴파일해도 **cos 0.69**로 붕괴 재현 → 원인이 head임을 입증.
   (재현 패키지: `attn_pool_repro` — head_model + 실 trunk 출력 calib/test)
3. Mobilint 분석: QKᵀ matmul의 outlier가 원인. 그 노드만 16bit → **cos 0.69 → 0.998**.
4. 우리 적용: full 모델을 `--qk16`으로 컴파일 → **CPU head 없이 순수 NPU**로 cos ≈ 0.99 (아래).

## 원인 — QKᵀ matmul의 outlier

attention `softmax(QKᵀ/√d) · V`에서 **QKᵀ matmul** 한 곳이 범인:

1. trunk 출력 토큰에 **outlier 채널** 존재 (전체 std ≈ 1.9인데 일부 채널 |max| ≈ 123).
2. → QKᵀ attention logit 범위가 **[2.4, 1045.8]**까지 벌어짐.
3. → 이 logit을 INT8로 양자화하면 step ≈ 1045/127 ≈ **8.2**. softmax 직전 logit들의 미세한
   대소관계가 모두 뭉개짐.
4. → softmax(지수함수)가 오차를 증폭 → 출력 임베딩 cos 0.69로 붕괴.

**왜 per-layer 지표로 못 잡았나** (Mobilint testQuantize 기준):

| 노드 | 수정 전 cos / SQNR | 수정 후 cos / SQNR |
|------|:---:|:---:|
| QKᵀ matmul (자기 출력) | 0.9999 / 37.5 dB | 0.9999 / 38.7 dB |
| softmax 내부 (exp) | 0.2382 / **−24.6 dB** | 0.9962 / +21.2 dB |
| 최종 출력 | 0.6976 / 1.9 dB | 0.9986 / 25.4 dB |

QKᵀ matmul **자신**의 출력은 멀쩡해 보이고, 손상은 **하류 softmax**에서 드러난다. 그래서
per-layer 지표만으로는 원인 특정이 어렵고, module ablation으로 확인했다.

## 해결 — QKᵀ matmul activation만 16bit

- 해당 노드 activation을 16bit로 지정하면 logit이 보존되어(step ≈ 0.03) 복구.
- softmax / 그 직전 requantize는 **이미 16bit**였고, 그 앞 INT8 matmul이 원인이었으므로
  **matmul 출력을 16bit로 올리는 것**이 정확한 처방. head 전체 16bit는 불필요(노드 1개로 충분).
- API: `BitConfig(layer_overrides=BitConfig.LayerOverrides(activation_16bits=[score_matmul]))`.
  **컴파일러 소스 수정 불필요**, 공개 릴리스 API만 사용.

### 레이어 이름 자동 탐지 (`pe_npu/find_score_matmul.py`)
override 대상 이름은 모델/패키징마다 다르므로 하드코딩하지 않고 **그래프 구조로 탐지**:
> 출력이 (스케일·reshape·mask 등 transparent 노드만 거쳐) **Softmax로 들어가는 MatMul** →
> attention score matmul → 16bit 대상.

full 모델이면 attention마다 1개씩(trunk self-attn 24 + head cross-attn 1 = 25개) 잡힌다.

## pe_npu 적용

- `pe_npu/find_score_matmul.py` — score matmul 자동 탐지 (Mobilint 제공, vendor).
- `pe_npu/compile.py` — `compile_pe(..., qk16=True)` / CLI `--qk16`. mblt 파싱 → 탐지 →
  `mxq_compile(bit_config=...)` 2-pass.
- `pe_npu/inference.py` — `MXQInferenceFull`(image→embedding 전부 NPU, CPU head 없음).

```bash
# full 모델을 QKᵀ 16bit로 컴파일 (image -> embedding, CPU head 불필요)
python -m pe_npu.compile --mode compile --save out/pe_full.mxq \
  --calib-data-path <calib_hwc> --device cpu --qk16 --scheme single
```

## 결과 (현재 서버 ARIES2 실측)

| 구성 | cos(NPU, 원본 float) | head 처리 |
|------|:---:|------|
| hybrid (trunk INT8 + CPU pool) — 기존 | 0.997 | **CPU** |
| **full NPU (QKᵀ 16bit)** — 신규 | **0.99** (COCO holdout 0.9905 / 도메인 0.9889) | **NPU** |
| head 단독 INT8 (수정 전, 참고) | 0.69 | — |
| head 단독 QKᵀ16bit (수정 후, 참고) | 0.998 | — |

→ **CPU attn_pool 우회가 더 이상 불필요.** 고채널 CPU 병목(attn_pool) 제거. (벤치: `../performance/NPU_pe_hybrid_vs_full.md`)

## 참고 자료
- 문의 원본: `mobilint_inquiry_attn_pool.md`
- 재현/해결 패키지: `mobilint_reply/`(회신), `attn_pool_repro`(우리가 보낸 재현본)
- 자동 탐지기: `pe_npu/find_score_matmul.py`

*작성 2026-06. 해결: Mobilint 전근우님 기술지원.*
