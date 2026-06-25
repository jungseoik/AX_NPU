# PE-Core-L14-336 NPU 양자화 정확도 튜닝 가이드 (재현용)

> **[SUPERSEDED 2026-06] full-NPU 붕괴(cos~0.46)의 원인은 attn_pool이 맞았고, 최종 해결은 QKᵀ 16bit.**
> 처음엔 hybrid(attn_pool을 CPU로 분리, cos 0.9987)로 우회했으나, 이후 Mobilint가 원인을 **attn_pool의
> QKᵀ matmul outlier**로 규명 → 그 **score matmul만 16bit**로 올리면 NPU에서도 정상(cos 0.998).
> 지금은 **full NPU**(image→embedding 전부 NPU, cos 0.99)가 기본이다.
> (이 문서가 "불필요"라 적었던 16bit override가 사실은 정답이었다 — 단, head 전체가 아니라
> **QKᵀ matmul 한 노드만**이 핵심.) 현재 정답 경로:
> [`../vendor/mobilint_resolution_attn_pool.md`](../vendor/mobilint_resolution_attn_pool.md).

> **경로 갱신(패키지 재편 후)**: 컴파일 스크립트 `pe_onnx_export/pe_torch_compile.py` →
> 패키지 `pe_npu/compile.py` (CLI `python -m pe_npu.compile`, `-w /workspace/AX_NPU`에서 실행).
> 옛 보조 스크립트(`compare_backends.py`, `mxq_inference.py`, `prepare_calib.py`,
> ONNX 경로 `pe_torch_compile.py`/`export_pe_onnx.py`)는 모두 `pe_npu/` 패키지로 흡수되며 폐기됨(git 히스토리 참조).
> calibration 데이터는 `pe_npu.calib`로 생성(`python -m pe_npu.calib --hwc`).
> 아래 옛 명령들은 이 매핑으로 치환해 읽을 것.

> 단일 입출력 컴파일은 해결됨(`SOLUTION_single_io_compile.md`). 이 문서는 **NPU 추론 정확도
> (pth 대비 cos)를 0.5 → 0.99+로 올리는** 후속 작업을, 다음 에이전트가 그대로 따라 재현할 수
> 있게 단계·명령·결과를 정리한 것이다.

## 0. 핵심 사실 (검증 완료)

- **무조건 전부 INT8이 아니다.** 레이어별로 16bit 유지 가능: `BitConfig.LayerOverrides`의
  `activation_16bits: List[str]`, `weight_16bits: List[str]`. → "민감 레이어를 그대로(고정밀) 두는" 방법.
- 양자화 정밀도/방식: `CalibrationConfig(output=0|1, method=1|3, mode=...)`.
- activation outlier 완화: `EquivalentTransformationConfig`(SmoothQuant/회전 — norm_conv, qk, ud, vo, hadamard 등).
- pth↔onnx=1.0, patch(양자화前)↔pth=1.0, 입력 레이아웃 HWC가 최적 → **정확도 손실은 순수 INT8 양자화에서 발생**(ViT activation 민감성).

## 1. 현재까지 결과 (실제 COCO 이미지 입력, batch=4, pth 대비)

| 설정 | cos |
|------|-----|
| random calib + 더미 입력 | 0.94 (분포 우연 일치, 무의미) |
| random calib + 실제 이미지 | 0.42 |
| COCO calib, percentile per-layer (method=1, output=0) | 0.46 |
| COCO calib, per-channel (output=1) | 0.46 |
| COCO calib, per-channel + **EquivalentTransformation**(norm_conv/qk/ud/vo) | 0.455 |

→ **calibration·per-channel·SmoothQuant(ET) 일반 기법 3종 모두 0.46에서 멈춤.** 양자화 설정 튜닝으론
한계. 남은 길은 **(B) 16bit override** 또는 **(D) 구조적 원인(attention pooling 분해)** 진단뿐.
ET가 효과 없는 건 PE wrapper가 HF transformer 구조가 아니라 ET 패치가 부분 적용됐을 가능성 있음(검증 필요).

## 1-B. 16bit override 실험 결과 (2026-06-15, 실제 NPU /dev/aries0, 실제 COCO 이미지 batch=4, pth 대비)

> 목표였던 **16bit override로 0.99+** 는 **달성 실패**. 그 과정에서 16bit override의 동작 특성과
> 병목 원인을 실측으로 규명했다. 아래 모든 수치는 컨테이너 안에서 실제 컴파일·NPU 추론한 값이다.

### layer_overrides 이름 공간

- `BitConfig.LayerOverrides.activation_16bits / weight_16bits` 의 이름은 **컴파일러 내부 operator 이름**
  (FX 노드 이름이 변형된 것)과 매칭된다. C++(`mmc.so`) 내부 `mQuantInfoMap` 키. 매칭 실패 시
  `"Layer {} from bit config not found in mQuantInfoMap, skipping"` 로 조용히 무시된다.
- 이름 목록 추출: `pe_torch_compile.py --mode parse --dump-names out/op_names.txt`
  → sg0.operators 533개. 타입 분포: Convolution 151, HeaderView 74, LayerNormalization 51,
  MatMul 50, Adding 50, StatefulRoPEWrapper 48, Transpose/Softmax/MultiplyConstant/Gelu 각 25 등.
- attention pooling은 마지막 영역: `inputconst_193/.../matmul...`(분해된 nn.MultiheadAttention),
  `visual_attn_pool_mlp_c_fc/c_proj`, 최종 출력 `add_96/squeeze/reshape/conv2d`.

### 측정 결과표

| 설정 (act16/weight16) | 16bit 레이어수 | 컴파일 | cos | 비고 |
|------|------|------|------|------|
| INT8 baseline (override 없음, per-channel) | 0 | OK | **0.4596** | 기준 (=기존 0.46 재현) |
| act16 = LayerNormalization | 56 | OK | **0.4596** | **완전 no-op** (mxq 바이트 동일, MAE 동일) |
| act16 = LN+Gelu+Adding+Softmax | 201 | OK | **0.0041** | **모델 파괴** (8/16bit 경계 스케일 불일치) |
| weight16 = 전체 Convolution | 151(W) | OK | **0.1473** | 악화 (W16/A8 정밀도 불일치) |
| act16 = 전체 Convolution | 151 | **FAIL** | — | `quantize failed. ch size is not matched 1 <-> 1024` |
| act16 = transformer blocks 전체 | ~522 | **FAIL** | — | 동일 ch size 에러 |
| act16 = all-except-attn_pool | 522 | **FAIL** | — | 동일 ch size 에러 |
| act16 = attn_pool 영역만 | 11 | **FAIL** | — | 동일 ch size 에러 |
| act16 = all (533) | 533 | **FAIL** | — | 동일 ch size 에러 (output=0/1 모두) |

### 결론 (실측 근거)

1. **부분 16bit activation override는 0.99 달성 불가.** 세 가지 양상만 관찰됨:
   - LayerNorm처럼 **no-op**(정확도 불변, mxq 바이트 동일), 또는
   - Softmax/Adding 포함 시 **8bit↔16bit 경계의 스케일 불일치로 출력이 깨짐**(cos 0.46→0.004), 또는
   - Convolution/MatMul 포함 시 **컴파일 자체가 실패**.
2. **일관된 전체(full) 16bit는 하드웨어 제약으로 차단됨**: `ch size is not matched 1 <-> 1024`.
   채널 차원 1인 레이어(attn_pool probe seq=1 / 최종 squeeze conv 등)가 16bit per-tensor/channel
   양자화와 충돌. calib output=0(per-layer)·output=1(per-channel) **양쪽 모두** 동일 실패.
   → 부분집합으로는 일관성이 깨지고, 전체로는 불가능 → **16bit override 경로는 막다른 길**.
3. **weight16도 악화**(cos 0.147). W16/A8 비대칭이 ViT에 불리.
4. 따라서 0.46의 병목은 **16bit로 끌 수 있는 "activation 양자화 정밀도"가 아니다.**
   LayerNorm을 16bit로 올려도 전혀 변하지 않는다는 점이 결정적 근거. 남은 유력 원인은
   **구조적 문제 — attn_pool의 nn.MultiheadAttention 자동 분해**(컴파일 로그
   `module ... MultiheadAttention is not registered ... try low-level function call`).
   이 분해 경로의 INT8화가 주범으로 의심되나, 해당 영역은 16bit로도 컴파일 불가라 override로 못 고친다.

### 재현 명령 (pe_torch_compile.py 신규 옵션)

```bash
cd /workspace/AX_NPU/pe_onnx_export
export LD_LIBRARY_PATH=/tmp/qbruntime_aries2-v4_v1.2.0_amd64/qbruntime/qbruntime/lib:$LD_LIBRARY_PATH

# operator 이름 덤프
python3 pe_torch_compile.py --mode parse --dump-names out/op_names.txt

# 16bit override 컴파일 (--act16 / --weight16 / --act16-exclude)
#   값: 'all' | 'none' | 쉼표구분 substring(이름 또는 layertype에 매칭)
python3 pe_torch_compile.py --mode compile --save ./out/pe_tune.mxq \
  --calib-data-path ./calib_coco_hwc --calib-output 1 --device gpu \
  --act16 "LayerNormalization" \
  [--act16-exclude "inputconst_193,attn_pool,add_96"] \
  [--weight16 "Convolution"]

# 정확도 (실제 이미지 필수). 컴파일은 calib 200장에 ~5-7분.
cd /workspace/AX_NPU/pe_npu_service
python3 compare_backends.py --onnx ../pe_onnx_export/out/PE-Core-L14-336_vision_sim.onnx \
  --mxq ../pe_onnx_export/out/pe_tune.mxq --real-npy-dir ../pe_onnx_export/calib_coco --batch 4
```

### 다음 후보 (16bit override 외)

- **attn_pool 분해 우회**: `pe_torch_compile.py`에서 `AttentionPooling.forward`를 이미 패치했듯,
  내부 `nn.MultiheadAttention`을 q/k/v/out **명시적 Linear + SDPA**로 직접 재구현해 분해를 통제
  (현재는 SDK 파서의 low-level 자동 분해에 맡겨짐). 분해 구조가 INT8에 적대적인지 검증.
- **단계별 출력 비교**: 패치된 wrapper에서 각 resblock 출력 / attn_pool 입력을 hook으로 뽑아
  pth vs NPU 중간 텐서 cos를 측정 → 손실이 누적되는 구간(블록 vs pooling)을 특정.
- **EquivalentTransformation 실효성 검증**: 0.455로 무효였음 → PE wrapper가 HF 구조가 아니라
  ET 패치가 미적용일 가능성. 적용 여부를 컴파일 로그로 확인.

---

## 2. 재현 환경

```bash
# NPU 연결 컨테이너 (없으면 pe_npu_service/run_npu_tests.sh의 1단계로 생성)
docker exec -w /workspace/AX_NPU/pe_onnx_export mblt_compiler bash -lc '...'
# 런타임 lib: export LD_LIBRARY_PATH=/tmp/qbruntime_aries2-v4_v1.2.0_amd64/qbruntime/qbruntime/lib:$LD_LIBRARY_PATH
# calibration 데이터(필수: HWC 레이아웃): pe_onnx_export/calib_coco_hwc/ (prepare_calib.py가 CHW로 만들면 transpose 필요)
#   CHW npy를 주면 C++ abort. (336,336,3) HWC npy + npy_files.txt 사용.
```

컴파일 스크립트: `pe_onnx_export/pe_torch_compile.py` (단일 입출력, calib 옵션 내장).
정확도 비교: `pe_npu_service/compare_backends.py --real-npy-dir ../pe_onnx_export/calib_coco --batch 4`
(반드시 `--real-npy-dir`로 **실제 이미지** 사용. 더미 입력은 calib 분포와 어긋나 왜곡됨.)

## 3. 다음 단계 (우선순위 순, 각 단계 재현 명령)

### A. EquivalentTransformation (SmoothQuant) — 레이어 이름 불필요, 먼저 시도
`pe_torch_compile.py`의 `mxq_compile(...)`에 추가:
```python
from qbcompiler import EquivalentTransformationConfig as ET
et = ET(norm_conv=ET.NormConv(apply=True), qk=ET.Qk(apply=True),
        ud=ET.Ud(apply=True), vo=ET.Vo(apply=True))
mxq_compile(**common, calib_data_path=calib, calibration_config=cc,
            equivalent_transformation_config=et)
```
ViT attention/FFN의 activation outlier를 가중치로 흡수 → INT8 손실 감소. PE wrapper에 적용되는지
먼저 확인(transformer 구조 가정 옵션이라 일부만 적용될 수 있음).

### B. 전체 activation 16bit — 정확도 "상한" 확인용
모든 레이어를 16bit로 두면 정확도 상한을 알 수 있다(속도/메모리는 희생).
```python
from qbcompiler import BitConfig
# 레이어 이름 목록 추출:
#   mblt-mxqtool show <mxq> | grep "Name:" → 컴파일러 내부 노드 이름 (예: add_96/squeeze/reshape/conv2d)
names = [...]  # 전체 또는 attention/pooling 관련
bit = BitConfig(layer_overrides=BitConfig.LayerOverrides(activation_16bits=names))
mxq_compile(**common, calib_data_path=calib, calibration_config=cc, bit_config=bit)
```
- 전체 16bit로 cos가 0.99+면 → 양자화가 원인 확정, 이후 C로 최소 16bit 집합 탐색.
- 전체 16bit로도 안 오르면 → 양자화 외 원인(예: attention pooling 분해) 재조사.

### C. 민감 레이어만 16bit (최소 집합 탐색)
B에서 정확도가 오르면, 어떤 레이어가 핵심인지 좁힌다.
- 후보 우선순위: **attention pooling(마지막, 출력 직접 영향) → 각 block의 attention(softmax 전후) → layernorm**.
- 레이어를 그룹별로 16bit 적용하며 cos 측정(이진 탐색식). 속도/정확도 트레이드오프 지점을 찾는다.

### D. attention pooling 정밀 검증
`attn_pool`의 `nn.MultiheadAttention`은 파서가 자동 분해(컴파일 로그에 "low-level function call" 경고).
이 부분이 0.46의 주범인지 의심됨 → 우선 16bit로 두거나, 분해 결과를 단계 출력으로 비교.

## 4. 검증 루프 (각 설정마다)

```bash
# 1) 재컴파일
python pe_torch_compile.py --mode compile --save ./out/pe_tune.mxq \
  --calib-data-path ./calib_coco_hwc --calib-output 1 --device gpu   # + 코드에 et/bit 추가
# 2) 정확도 (실제 이미지)
python compare_backends.py --onnx ../pe_onnx_export/out/PE-Core-L14-336_vision_sim.onnx \
  --mxq ../pe_onnx_export/out/pe_tune.mxq --real-npy-dir ../pe_onnx_export/calib_coco --batch 4
# → pth vs npu cos 기록. 0.99+ 목표.
```

## 5. 목표 달성 후

- 최종 설정(calib/bit/et)을 `pe_torch_compile.py` 기본값으로 고정 + 이 표에 최종 cos 기록.
- `service.py`의 `TRTInference` → `MXQInference`(pe_npu_service) 교체로 NPU 서비스 모듈 구성.
- 운영 분포(CCTV 프레임)로 calibration하면 실제 서비스 정확도는 더 오를 수 있음.
