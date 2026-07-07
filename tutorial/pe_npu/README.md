# PE-Core-L14-336 NPU 추론 튜토리얼 (처음부터 끝까지)

신규 서버에 ARIES NPU를 막 장착한 상태에서, **PE 비전인코더를 다운로드 -> NPU용 MXQ
컴파일 -> 추론 -> 정확도 확인**까지 따라 할 수 있는 전 과정. 모든 단계는 `pe_npu`
파이썬 패키지를 통해 수행한다 (`python -m pe_npu.*` 또는 `import pe_npu`).

- 대상 모델: PE-Core-L14-336 (Meta Perception Encoder, CLIP ViT-L/14) vision encoder
- 결과: 이미지 -> 1024-d 임베딩. 원본 PyTorch 대비 코사인 유사도 **0.99**
- 구조: **image→embedding 전부 NPU (full NPU)**. trunk 24 block + attn_pool head 모두 NPU INT8이되,
  attn_pool의 QKᵀ matmul만 16bit로 올려(`--qk16`) 양자화 붕괴를 피한다.
  (원인·해결: `../../reports/vendor/mobilint_resolution_attn_pool.md`. 예전 hybrid 방식은 attn_pool을 CPU로 뒀음)

> 모든 명령은 `mblt_compiler` 컨테이너 안에서 실행하며, `docker exec`로 감싸면 된다.
> 패키지 위치는 `/workspace/AX_NPU/`(= 호스트 `AX_NPU/AX_NPU/`)이고, 여기서 `import pe_npu`가 된다.
> 런타임 라이브러리 경로는 추론/컴파일 시 매번:
> `export LD_LIBRARY_PATH=/tmp/qbruntime_aries2-v4_v1.2.0_amd64/qbruntime/qbruntime/lib:$LD_LIBRARY_PATH`

> **자기완결(self-contained)**: PE 모델 코드는 `pe_npu/pe_vendor/`에 복사(vendor)되어 있어
> 외부 레포(Product-AI-mono) 없이 이 레포만 clone하면 동작한다. 가중치만 HuggingFace
> `facebook/PE-Core-L14-336`에서 최초 1회 자동 다운로드된다.

---

## 사용 방식 2가지 (먼저 선택)

| | 옵션 A — 직접 컴파일 | 옵션 B — 가져와 쓰기 (빠름) |
|---|---|---|
| 흐름 | calib → 컴파일 → 추론 (아래 0~5 전체) | HF에서 미리 컴파일된 자산 받아 추론만 |
| 필요한 것 | **qbcompiler**(컴파일러) + 원본 PE 가중치 + GPU | **qbruntime**(런타임)만. 컴파일러·원본 가중치 불필요 |
| 용도 | 컴파일 실험, 커스텀 calib/해상도, 재현 | 운영·빠른 시작 (NPU만 있으면 바로 추론) |
| 산출물 출처 | 내가 컴파일 | `PIA-SPACE-LAB/MXQ_NPU` (HF) |

**MXQ는 aries2 아키텍처 바이너리라 어디서 컴파일하든 동일**하다. 그래서 한 번 컴파일해
HF에 올려두면(옵션 B) 다른 사람은 그냥 받아 쓰면 된다.

### 옵션 B 빠른 시작 (qbcompiler 불필요)
```bash
# 런타임 conda env만 있으면 됨 (setup/setup_conda_host.sh)
conda activate pe_npu_host
python -c "
import numpy as np, pe_npu
m = pe_npu.MXQInferenceFull.load(scheme='single')  # 로컬 있으면 사용→없으면 HF→그래도 없으면 컴파일 안내
x = pe_npu.preprocess_image('tutorial/pe_npu/images/cat1.jpg')
print(m.infer(x[None]).shape)                    # (1, 1024)
"
```
> **기본 진입점 `load()`**: 로컬 mxq → HF `<scheme>/pe_full.mxq` → (없으면) 컴파일 안내. (HF 강제는 `from_hf(scheme=)`.) YOLO `YOLONPU.load()`와 대칭.
> 옵션 B는 NPU(`/dev/aries0`) + qbruntime + 인터넷(최초 1회 HF 다운로드)만 필요하다. full NPU라 torch도 불필요.
> 직접 만든 자산을 HF에 올리려면: `python setup/upload_assets_to_hf.py` (pe_full.mxq 업로드).
> (레거시 hybrid를 쓰려면 `MXQInferenceHybrid.from_hf()` + `--legacy` 업로드.)

아래 0~5단계는 **옵션 A(직접 컴파일)** 기준이다. 옵션 B면 0(드라이버/런타임)만 하고 위 빠른 시작으로 가면 된다.

---

## 0. 사전 준비 (신규 서버)

### 0-1. NPU 하드웨어/드라이버 인식 확인
```bash
lspci -d 209f:                 # PCI에 Mobilint NPU 보이는지
lsmod | grep aries             # 드라이버 모듈
ls -al /dev/aries0             # 디바이스 노드 (★ 이게 있으면 인식 정상)
# 없으면: sudo bash ../../setup/prepare_host_for_npu.sh  (드라이버 설치)
#         그 후 sudo modprobe aries  또는 재부팅
bash ../../setup/check_npu.sh     # 한 번에 점검 (드라이버/디바이스/PCI/런타임)
```

### 0-2. 컴파일+추론 컨테이너 준비 (옵션 — GPU 머신용)

> **기본은 GPU 없이도 됨.** 추론은 NPU(0-3 호스트 conda)면 충분하고, 컴파일도 conda(`pe_compile`, CPU)로 가능
> (→ `.claude/setup-notes.md` 방법 A). 아래 도커는 **GPU 머신에서 편하게** 쓸 때의 옵션이다.
```bash
docker rm -f mblt_compiler 2>/dev/null
docker run -dit --ipc=host --name mblt_compiler \
  $( [ -e /dev/nvidia0 ] && echo --gpus all ) \   # GPU 있을 때만 (옵션)
  --device /dev/aries0:/dev/aries0 \               # NPU (멀티카드면 aries1.. 추가)
  -v /home/gpuadmin/Repo/seoik/AX_NPU:/workspace -w /workspace \
  mobilint/qbcompiler:1.1-cuda12.8.1-ubuntu22.04 /bin/bash
# 컴파일러 + 런타임 설치
docker exec mblt_compiler pip install /workspace/AX_NPU/download/qbcompiler-1.1.2+aries2-py3-none-any.whl
docker exec mblt_compiler bash -lc 'cd /tmp && tar xzf /workspace/AX_NPU/download/qbruntime_aries2-v4_v1.2.0_amd64.tar.gz && pip install /tmp/qbruntime_aries2-v4_v1.2.0_amd64/qbruntime/qbruntime/python/*cp310*.whl'
docker exec mblt_compiler pip install onnxruntime
```
(이미지 pull: `docker pull mobilint/qbcompiler:1.1-cuda12.8.1-ubuntu22.04`)

> GPU 없는 서버는 도커 대신 **conda 컴파일(방법 A)** 을 쓴다. NPU 장착 후 통합 실행: `bash ../../setup/run_npu_tests.sh`.

### 0-3. (대안) docker 없이 호스트 conda로 — **추론 검증됨 (cos 0.99, full NPU)**
docker는 필수가 아니다. NPU 드라이버/`libqbruntime.so`는 호스트에 있고, Python 3.10~3.12 conda
env만 만들면 호스트에서 동일하게 추론된다(호스트 base conda가 3.13이면 qbruntime wheel(cp38~cp312)이
안 맞으므로 전용 env를 만든다).
```bash
bash ../../setup/setup_conda_host.sh          # env(pe_npu_host, py3.11) + qbruntime + torch/einops/timm
conda activate pe_npu_host
cd /home/gpuadmin/Repo/seoik/AX_NPU/AX_NPU
python tutorial/pe_npu/download_images.py
python tutorial/pe_npu/demo_inference.py    # docker exec 없이 바로 → cos ≈ 0.99 (full NPU)
```
> 이 경우 아래 단계들의 `docker exec -w /workspace/AX_NPU mblt_compiler python X` 명령은
> conda env 활성화 후 `python X`로 그대로 대체하면 된다(컨테이너 경로 `/workspace/AX_NPU` =
> 호스트 `AX_NPU/AX_NPU`). 컴파일(2단계)도 동일하게 호스트에서 가능.

빠른 동작 확인 (패키지 import):
```bash
docker exec -w /workspace/AX_NPU mblt_compiler python -c "import pe_npu; print('pe_npu OK')"
```

---

## 1. Calibration 데이터 준비 (양자화용)

INT8 양자화는 실제 입력 분포로 "눈금"을 잡는 calibration이 필요하다. 도메인 데이터가 없으면
COCO val2017(공개)로 충분하다. NPU 입력 레이아웃은 **HWC**라 `--hwc`로 곧장 HWC npy를 만든다.

```bash
# COCO val2017 다운로드 (~778MB, 공개)
docker exec -w /workspace/AX_NPU mblt_compiler bash -lc '
  mkdir -p coco && cd coco && \
  wget -q http://images.cocodataset.org/zips/val2017.zip && \
  python -c "import zipfile; zipfile.ZipFile(\"val2017.zip\").extractall(\".\")" && rm val2017.zip'
# PE 전처리(336 + normalize 0.5) 적용한 HWC calibration npy + npy_files.txt 생성
docker exec -w /workspace/AX_NPU mblt_compiler \
  python -m pe_npu.calib --dataset coco --src ./coco/val2017 --num 200 \
    --out ./pe_npu/calib_coco_hwc --hwc
```

---

## 2. NPU용 MXQ 컴파일 (full NPU)

full 모델(trunk 24 block + attn_pool head)을 한 번에 NPU용 MXQ로 컴파일한다(`--qk16`). PE는
RoPE2D·attention pooling 때문에 일반 ONNX 경로로는 컴파일이 막혀, `pe_npu`가 모델에 전용 패치(5개)를
적용한 뒤 컴파일한다. `--qk16`은 attention score MatMul(QKᵀ)을 자동 탐지해 16bit로 올린다 —
attn_pool이 그냥 INT8에서 깨지던(cos 0.46) 문제를 막아준다.

```bash
# 기본: --device cpu (GPU 없는 NPU 서버). GPU 있으면 --device gpu 로 더 빠르게(옵션).
python -m pe_npu.compile --mode compile --save ./pe_npu/out/pe_full.mxq --qk16 \
  --calib-data-path ./pe_npu/calib_coco_hwc --calib-output 1 --device cpu
# (도커로 할 경우) docker exec -w /workspace/AX_NPU mblt_compiler bash -lc '... 위 명령 ...'
# → pe_npu/out/pe_full.mxq (약 327MB) 생성. "Compilation was successful." 확인.
# [qk16] 16bit 대상 score MatMul 25개: [...] 로그가 나오면 정상 (trunk 24 + head 1).
# 코어모드는 --scheme single(기본)|multi|global4|global8 — 출력 동일, 속도/메모리만 차이
#   (→ ../../reports/performance/NPU_coremode_benchmark.md)
```

> 검증된 `pe_full.mxq`가 이미 `pe_npu/out/`에 있으면 재컴파일 없이 4단계로 바로 갈 수 있다.
> (레거시 hybrid trunk만 만들려면 `--feat-only` — attn_pool 전까지, CPU pool과 함께 사용.)

---

## 3. 추론 클래스 (참고)

추론은 `pe_npu.MXQInferenceFull`이 제공한다 (image→embedding 전부 NPU,
`model(image) -> (B,1024)`, 기존 `TRTInference`와 인터페이스 동일). 직접 만들 필요 없다.

```python
import pe_npu
import numpy as np

model = pe_npu.MXQInferenceFull()              # 기본 MXQ = pe_npu/out/pe_full.mxq
x = pe_npu.preprocess_image("some.jpg")        # (3,336,336) float32
emb = model.infer(x[None])                     # (1, 1024) 비전 임베딩
```
> 레거시 hybrid가 필요하면 `pe_npu.MXQInferenceHybrid()` (NPU trunk + CPU pool, pe_feat.mxq+pool head).

### 코어모드 선택 (추론 시점, 컴파일 불필요)

코어모드(single/multi/global4/global8)는 **출력은 동일하고 속도·메모리만 다르다.** Mobilint가 4종을
미리 컴파일해 HF에 올려놔서, **모드를 바꾸려고 직접 컴파일할 필요가 없다** — `from_hf(scheme=...)`로 골라 받으면 된다.

```python
m = pe_npu.MXQInferenceFull.from_hf(scheme="global4")   # single | multi | global4 | global8
emb = m.infer(x[None])                                  # (B,1024), 결과는 모드 무관 동일
```

| 모드 | 단건(1장) | 56ch(7카드) | 권장 상황 |
|------|---:|---:|------|
| `global8` | **71ms** | 553ms | 단건/저채널(≤카드수) 실시간 |
| `global4` | 119ms | **488ms** | 중간~고채널 균형 최선 |
| `single` | 285ms | 532ms | 고채널 throughput(8코어 독립) |
| `multi` | 358ms | 1559ms | (비권장) |

> 모드를 골라 단건/멀티채널 지연을 **직접 비교**하는 실행 예제: **`demo_coremode.ipynb`**.
> 상세 실측: `../../reports/performance/NPU_coremode_benchmark.md`, `../../reports/performance/NPU_full_pipeline_e2e.md`.

---

## 4. 추론 (예제 이미지 + 유사도 확인)

### 4-A. Jupyter 노트북 (권장) — `demo_inference.ipynb`
셀 단위로 돌리며 **이미지·유사도 히트맵·정확도 막대그래프를 인라인으로** 바로 본다.
호스트 conda(0-3) 셋업이면 `LD_LIBRARY_PATH`도 불필요(libqbruntime이 ldconfig 등록됨).
```bash
conda activate pe_npu_host
cd tutorial/pe_npu
jupyter notebook demo_inference.ipynb      # 또는 VS Code/Jupyter Lab에서 열기
```
노트북 흐름: 0 환경체크 → 1 이미지 준비/미리보기 → 2 전처리+NPU추론 → 3 유사도 히트맵
→ 4 원본(pth) 대비 정확도 → 5 내 코드에서 쓰기. 커널은 `pe_npu_host` env 선택.

### 4-B. 스크립트 (비대화형/CI용) — `demo_inference.py`
시각화 없이 텍스트로 같은 결과를 출력. 컨테이너/원격 등 노트북 띄우기 어려운 환경용.
```bash
# 예제 이미지 다운로드 (공개 COCO 5장, 로그인 불필요)
python download_images.py
# 추론 데모 (호스트 conda면 그대로, 컨테이너면 docker exec + LD_LIBRARY_PATH로 감싼다)
python demo_inference.py
```

데모 출력(노트북·스크립트 공통):
- **이미지 간 코사인 유사도 매트릭스** — 비슷한 이미지는 높고 다른 이미지는 낮게 나오는지
- **원본 PyTorch 대비 NPU 임베딩 cos** — 양자화 정확도 (평균 0.99+ 면 정상, full NPU 검증값 0.99)

### 4-C. 텍스트 프롬프트 제로샷 분류 (live 텍스트) — `demo_text_classification.ipynb` (권장)
PE는 CLIP 계열이라 **이미지 임베딩 ↔ 텍스트 임베딩의 코사인 유사도로 분류**를 푼다.
**임의 텍스트 문자열을 직접 입력 → PE 텍스트 인코더가 즉석 인코딩**(사전 파일 아님)하고, 이미지는 NPU로 임베딩.
```bash
jupyter notebook demo_text_classification.ipynb     # (권장) 예시 이미지 + 예측 라벨 인라인 표시
python demo_text_classification.py --images images/*.jpg \
   --prompts "a fire" "smoke" "a person who has fallen down" "a normal scene"   # (옵션)
```
- 흐름: `image → (NPU) 1024d` × `prompts → (PE 텍스트 인코더, CPU) (N,1024)` → cosine → 최고 클래스
- 토크나이저 = CLIP BPE(`open_clip`) — **공식 `perception_models`의 `SimpleTokenizer`와 동일**(검증함). 이미지 정규화 0.5/0.5도 공식 transform과 동일.
- 노트북은 **예시 이미지를 인라인 표시**하고 각 이미지에 예측 라벨을 붙여 보여줌. 함수 `classify(img, prompts)`로 아무 문장이나 시도 가능.
- 검증: 도메인 프롬프트에서 잘 분류 (falldown→"넘어진 사람" 0.97, smoke→smoke 0.93). (PE-Core는 일반 객체 zero-shot은 약함 — 공식 모델 특성.)
- 참고: 운영 `pe_binary`/`pe_npu` 서비스는 **같은 원리**지만 프롬프트를 미리 인코딩(`text_features.json`)해 속도를 높인다.

---

## 5. 직접 코드에서 쓰기 (요약)

```python
import pe_npu
import numpy as np

model = pe_npu.MXQInferenceFull("/workspace/AX_NPU/pe_npu/out/pe_full.mxq", num_threads=8)
# image: 전처리된 (B,3,336,336) float32 numpy 또는 torch
x = np.stack([pe_npu.preprocess_image(p) for p in paths], axis=0)
emb = model.infer(x)   # (B, 1024) — 배치는 1모델+num_threads 스레드 sync로 8코어 활용(출력 정확)
```
> 다채널 처리량/모드 선택(single·global4·global8)과 올바른 동시성 패턴: `../../reports/performance/NPU_throughput_modes_correct.md`.
> (⚠️ `infer_async`를 한 모델에 여러 건 동시 제출하면 출력이 깨짐 — `MXQInferenceFull`은 안전한 스레드 sync를 씀.)

전처리는 `pe_npu.preprocess_image` (운영 `service.preprocess_image`와 동일:
RGB -> resize 336 bilinear -> /255 -> normalize 0.5).

---

## 6. 병렬 추론 — 멀티코어 · 멀티카드 (`demo_parallel.py`)

같은 배치를 3가지로 처리하며 처리량 비교. **코어(한 장 8개)는 async가 자동 병렬, 카드(여러 장)는 라운드로빈으로 직접 분산**.
```bash
jupyter notebook demo_parallel.ipynb       # (권장) 처리량 막대그래프 인라인
python demo_parallel.py --batch 32         # (옵션) 비대화형/CI
```
| 방식 | 설명 | 실측(7×ARIES, 32장) |
|------|------|------|
| sync | 1카드 blocking 루프 | 3.5 img/s |
| async | 1카드 `infer_async`+`set_async_pipeline_enabled` → 8코어 | 15.8 img/s (×4.6) |
| **multicard** | 전 카드 라운드로빈(async) | **77.9 img/s (×22.5)** |

- 위 표는 `single` 모드. **다른 코어모드는 재컴파일 없이** `MXQInferenceFull.from_hf(scheme=...)`로 바로 바꿔 쓴다(위 "코어모드 선택" 절). 모드별 비교 실행: `demo_coremode.ipynb`.
- 멀티카드 분산 상세/62채널 스윕: `../../reports/performance/NPU_multicard_62ch_benchmark.md`.
- 고채널에선 CPU 전처리가 병목 → 병렬화: `../../reports/performance/NPU_preprocess_parallel.md`.

---

## pe_npu 패키지 구성

| 모듈 | 역할 | CLI |
|------|------|-----|
| `pe_npu.pe_model` | 모델 로딩 + 컴파일 패치 (`load_pe`, `apply_pe_patches`) | - |
| `pe_npu.pe_vendor` | Meta PE vision encoder 코드 vendor 복사본 (외부 의존 제거) | - |
| `pe_npu.preprocess` | `preprocess_image` (resize 336 + normalize 0.5) | - |
| `pe_npu.calib` | calibration npy 생성 / HWC 변환 | `python -m pe_npu.calib` |
| `pe_npu.compile` | PE -> MXQ 컴파일 (`compile_pe`, `parse_pe`). `--qk16`=full NPU | `python -m pe_npu.compile` |
| `pe_npu.find_score_matmul` | attention score MatMul(QKᵀ) 자동 탐지 (`--qk16`이 16bit override) | - |
| `pe_npu.inference` | `MXQInferenceFull`(image→embedding 전부 NPU, 권장) / `MXQInferenceHybrid`(레거시). `.from_hf()` = 옵션 B | - |
| `pe_npu.export_pool_head` | (레거시) hybrid용 pool head 가중치 추출 (~53MB) | `python -m pe_npu.export_pool_head` |
| `pe_npu.assets` | HF에서 MXQ/pool head 다운로드 (옵션 B) | - |

## 튜토리얼 파일

| 파일 | 설명 |
|------|------|
**노트북 (기본 — 권장):**
| 파일 | 설명 |
|------|------|
| `demo_inference.ipynb` | 추론 데모 — 이미지/유사도 히트맵/정확도 시각화 |
| `demo_text_classification.ipynb` | **텍스트 프롬프트 제로샷 분류** — 클래스별 유사도 차트 |
| `demo_parallel.ipynb` | **병렬 추론** — 멀티코어(async) vs 멀티카드 처리량 막대그래프 |
| `demo_coremode.ipynb` | **코어모드 비교** — single/global4/global8 단건·멀티채널 지연 (`from_hf(scheme=)`, 재컴파일 불필요) |
| `multicore_benchmark.ipynb` | 멀티코어 처리량 벤치 — 동기 vs async vs 멀티스레딩 |

**스크립트 (옵션 — 비대화형/CI용, 노트북과 동일 기능):**
| 파일 | 설명 |
|------|------|
| `demo_inference.py` / `demo_text_classification.py` / `demo_parallel.py` | 위 노트북들의 텍스트 출력 버전 |
| `download_images.py` | 공개 COCO 예제 이미지 다운로드 |
| `images/` | 예제 이미지 (gitignore) |

상세 배경/원리: `../../reports/design/SOLUTION_single_io_compile.md`, `../../reports/quantization/quantization_reference.md`
