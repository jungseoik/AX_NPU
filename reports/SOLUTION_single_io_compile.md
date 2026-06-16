# [해결] PE-Core-L14-336 단일 입출력 NPU 컴파일

> 25-서브그래프 분할 문제를 **해결**했다. SDK 수정 없이 사용자 코드(모델 패치 + `backend="torch"`)로
> 단일 입력(image)/단일 출력(embedding) MXQ 컴파일 + NPU 추론에 성공.

## 결과

| 항목 | 값 |
|------|-----|
| MXQ 입출력 | **입력 1개 `(336,336,3)` / 출력 1개 `(1,1,1024)`** (기존 25/25 → 1/1) |
| 파서 리포트 | Supported 385.49 GOPS / **Unsupported 0.00 GOPS** (전 연산 지원) |
| NPU 추론 | `/dev/aries0`에서 image 1장 → embedding 정상 출력 |
| 원본 대비 정확도 | pth vs NPU **cos 0.936** (random calib 기준), pth vs onnx 1.000 |
| 산출물 | `pe_npu/compile.py` (구 `pe_onnx_export/pe_torch_compile.py`, `_dev/`에 원본 보존), `pe_npu/out/pe_feat.mxq` |

## 왜 됐나 (핵심)

기존 ONNX 경로는 export 단계에서 RoPE/posemb의 동적 연산이 If/subgraph로 굳어져 25분할됐다.
**torch FX 경로(`backend="torch"`)로 가면 정적화 패치를 트레이스 시점에 직접 주입**할 수 있어
단일 그래프로 트레이스된다. (Qwen2-VL이 SDK 내부에서 쓰는 `VisionModelForQwen2VL` 기법을
PE에 수동 적용한 것.)

op 미지원이 아니었다 — qbcompiler는 101개 op(RoPE 포함) 전부 지원. 문제는 **그래프 트레이스 형태**였다.

## 적용한 5개 패치 (모두 SDK 무수정, 모델 객체에만 적용)

`pe_npu/pe_model.py`의 `apply_pe_patches()` 참조 (구 `pe_torch_compile.py`, 원본은 `_dev/`). grid 24×24(336/14) 고정 전제.

1. **RoPE 상수화 + nn.Module 호출 제거**
   - `Rope2D`는 nn.Module이 아닌 순수 클래스인데 내부에 `self.rope`(nn.Module)를 호출 →
     트레이서가 submodule 경로 못 찾음(`NameError: module is not installed as a submodule`).
   - 해결: freq를 상수화하고 `Rope2D.__call__`을 cos/sin 상수만 쓰는 버전으로 교체.
2. **einops 제거** — `rotate_half`/`SelfAttention`의 `einops.rearrange`가 FX MbltProxy 통과 불가
   (`Tensor type unknown to einops`) → reshape/permute 네이티브로 재구현.
3. **abs posemb 상수화** — `_sample_abs_posemb`의 `[None, ...]`(newaxis+ellipsis) 미지원
   (`NotImplementedError`) → grid 고정이라 상수 반환.
4. **qkv split 슬라이싱** — `proj.split(int)` 미지원(`TypeError`) → `proj[..., :E]` 슬라이싱.
5. **attention pooling repeat 제거** — `probe.repeat((batch,1,1))` 미지원 → batch=1 고정이라 probe 직접 사용.

## 재현

```bash
# 단일 입출력 MXQ 생성 (random calib, 동작검증)
docker exec -w /workspace/AX_NPU mblt_compiler \
  python -m pe_npu.compile --mode compile --save ./pe_npu/out/pe_feat.mxq --feat-only

# 트레이스만 검증 (subgraphs=1 확인)
docker exec -w /workspace/AX_NPU mblt_compiler \
  python -m pe_npu.compile --mode parse --feat-only
```

## ★ 정확도 해결 (cos 0.9972) — Hybrid: NPU trunk + CPU pool head

**최종 결론**: 무거운 24 transformer block(ViT trunk)은 NPU(INT8), 작은 attn_pool+proj head는
CPU(float)로 분리하면 **pth 대비 cos 0.9972** 달성. (full-NPU는 0.46)

### 인과 사슬 (단계별 진단, 실제 NPU 추론)
| 지점 | pth 대비 cos |
|------|-------------|
| 24블록 후 forward_features (pool 전) | **0.95** (ViT trunk는 INT8 정상) |
| attn_pool 후 / proj 전 | 0.45 (← attn_pool이 범인) |
| proj 후 (full-NPU 최종) | 0.46 (proj 무죄) |
| **NPU features + CPU pool/proj (hybrid)** | **0.9972** ✅ |

### 왜 attn_pool만 깨지나
- attn_pool은 577토큰을 **1토큰으로 압축**하는 cross-attention pooling head. 이 1-토큰 출력 경로가
  INT8 양자화에서 본질적으로 손실이 크다.
- 검증으로 배제된 것: MHA→SDPA 재구현(0.46 무변), 16bit override(no-op/파괴/컴파일실패),
  per-channel·SmoothQuant(0.46). → 양자화 설정/구조 변경으로는 해결 안 됨.
- 핵심: 이 head는 **연산량이 전체 ViT의 1% 미만**이라 CPU float로 돌려도 부담 없음.

### 구현 (재현)
```bash
# 1) NPU trunk: forward_features만 출력하는 MXQ (COCO calib, per-channel)
#    (-w /workspace/AX_NPU 에서 실행)
python -m pe_npu.compile --mode compile --save ./pe_npu/out/pe_feat.mxq --feat-only \
  --calib-data-path ./pe_npu/calib_coco_hwc --calib-output 1 --device gpu
# 2) 추론: NPU feat → CPU pool/proj (pe_npu.MXQInferenceHybrid: visual._pool + visual.proj, float)
#    데모: tutorial_pe_npu/demo_inference.py
#    결과: pth_full vs hybrid cos = 0.997 (4장 [0.995,0.999,0.998,0.997])
```

### NPU 단독은 불가 — hybrid가 유일 정답 (조사로 확정)
Model Zoo/SDK 전수 조사 결과, **attention pooling을 NPU에서 정확도 유지한 사례가 없음**:
- SigLIP MAPHead: 모빌린트 포팅에서 head 제외(feature map만). (`model_zoo/.../siglip/modeling_siglip.py:23-30`)
- Qwen2/2.5/3-VL: 토큰압축이 MLP PatchMerger(attention pooling 아님) → 적용 불가. (`qbcompiler/.../qwen3vl.py:150-168`)
- MiniCPM-V resampler(동일 MHA pooling): 저수준 폴백 안티패턴, 미검증. (`.../remote_code/minicpmv.py:304`)
- SDK가 `MultiheadAttention=None`(미바인딩), 1-query attention 1급 지원 0건. (`converter.py:2444`)
→ INT8 정확도 보장 경로는 **CPU 분리뿐**. SDK 구조적 한계라 일반적.

### CPU 분리 오버헤드 (실측) — 무시 가능
| 항목 | 값 |
|------|-----|
| NPU features (24블록, 1코어 기준) | 284.7 ms |
| **CPU pool+proj head** | **2.22 ms (0.8%)** |
| features 전송 | 2.25 MB, 단방향 1회 (round-trip 아님) |
→ host-device 전송/파이프라인 버블 영향 미미. 실시간(멀티스트림)은 NPU/CPU 파이프라인으로 흡수.

### 서비스 통합 방향
- `MXQInference`를 hybrid로: image→NPU(feat MXQ)→numpy(577,1024)→torch→`visual._pool`+`@proj`(CPU, 가중치만 로드)→임베딩.
- pool head 가중치(attn_pool MHA + proj)는 PE 체크포인트에서 추출해 CPU에 상주(작음).
- 운영 분포(CCTV) calib로 trunk 재컴파일하면 0.95→더 상승 여지.

---

## (참고) 이전 정확도 분석 — full-NPU 양자화 튜닝 시도들

| 검증 | 결과 |
|------|------|
| pth vs onnx | **1.000000** (export 무손실) |
| patch 적용(양자화前) vs 원본 pth | **1.000000** (5개 패치 무손실 — patch는 정확) |
| COCO calib npy 레이아웃 | **HWC `(336,336,3)`** 가 정답 (CHW로 주면 C++ abort, HWC는 정상) |
| NPU 입력 레이아웃 스윕 | HWC 0.50 > WHC 0.45 > BGR 0.43 (HWC 최선) |
| **pth vs NPU(INT8)** | **0.45~0.50** ← 양자화 손실 |

**0.45~0.50의 원인 = ViT의 INT8 양자화 민감성 (확정).** patch 무손실·onnx 1.0·레이아웃 최적화에도
0.50이므로, 남은 손실은 transformer activation의 INT8 **per-layer** 양자화 때문이다. attention/
layernorm activation은 outlier가 커서 per-layer INT8로는 손실이 크다(LLM/ViT 양자화의 알려진 난제).

### 다음 과제: 양자화 정확도 튜닝 (0.99+ 목표)
1. **activation per-channel** — `CalibrationConfig(output=1)` (현재 output=0=per-layer)
2. **민감 레이어 16bit override** — qwen2vl가 쓰는 `compile_config={"bit":{"layerOverrides":{"activation16Bits":[...]}}}`. attention/pooling을 16bit로.
3. **EquivalentTransformation(SmoothQuant류)** — `equivalent_transformation_config`로 activation outlier 완화
4. zeropoint 비대칭 — `CalibrationConfig(method=3)`
→ 위 조합을 실험해 최고 cos를 찾아야 함 (재컴파일 반복).

### 기타
- 제약: 입력 336×336 고정(grid 24×24). 다른 해상도는 freq/posemb 재계산 후 재컴파일.
- `attn_pool`의 `nn.MultiheadAttention`은 파서가 자동 분해(비치명적).
- 입력 레이아웃 = HWC `(336,336,3)`. `mxq_inference`가 CHW→HWC 자동 변환.

## 다음
1. **양자화 튜닝**(위 1~4) → pth 대비 0.99+ ← 현재 핵심 블로커
2. `service.py`의 `TRTInference` → `MXQInference` 교체로 NPU 서비스 모듈 구성
