# PE-Core-L14-336 NPU 추론 튜토리얼 (처음부터 끝까지)

신규 서버에 ARIES NPU를 막 장착한 상태에서, **PE 비전인코더를 다운로드 -> NPU용 MXQ
컴파일 -> 추론 -> 정확도 확인**까지 따라 할 수 있는 전 과정. 모든 단계는 `pe_npu`
파이썬 패키지를 통해 수행한다 (`python -m pe_npu.*` 또는 `import pe_npu`).

- 대상 모델: PE-Core-L14-336 (Meta Perception Encoder, CLIP ViT-L/14) vision encoder
- 결과: 이미지 -> 1024-d 임베딩. 원본 PyTorch 대비 코사인 유사도 **0.997**
- 구조: 무거운 24 transformer block은 NPU(INT8), 작은 attn_pool head는 CPU(float) = **hybrid**
  (이유는 `../reports/SOLUTION_single_io_compile.md`. attn_pool은 NPU INT8에서 깨져서 CPU로 둠)

> 모든 명령은 `mblt_compiler` 컨테이너 안에서 실행하며, `docker exec`로 감싸면 된다.
> 패키지 위치는 `/workspace/AX_NPU/`(= 호스트 `AX_NPU/AX_NPU/`)이고, 여기서 `import pe_npu`가 된다.
> 런타임 라이브러리 경로는 추론/컴파일 시 매번:
> `export LD_LIBRARY_PATH=/tmp/qbruntime_aries2-v4_v1.2.0_amd64/qbruntime/qbruntime/lib:$LD_LIBRARY_PATH`

> **자기완결(self-contained)**: PE 모델 코드는 `pe_npu/pe_vendor/`에 복사(vendor)되어 있어
> 외부 레포(Product-AI-mono) 없이 이 레포만 clone하면 동작한다. 가중치만 HuggingFace
> `facebook/PE-Core-L14-336`에서 최초 1회 자동 다운로드된다.

---

## 0. 사전 준비 (신규 서버)

### 0-1. NPU 하드웨어/드라이버 인식 확인
```bash
lspci -d 209f:                 # PCI에 Mobilint NPU 보이는지
lsmod | grep aries             # 드라이버 모듈
ls -al /dev/aries0             # 디바이스 노드 (★ 이게 있으면 인식 정상)
# 없으면: sudo bash ../setup/prepare_host_for_npu.sh  (드라이버 설치)
#         그 후 sudo modprobe aries  또는 재부팅
bash ../setup/check_npu.sh     # 한 번에 점검 (드라이버/디바이스/PCI/런타임)
```

### 0-2. 컴파일+추론 컨테이너 준비 (NPU 연결)
```bash
docker rm -f mblt_compiler 2>/dev/null
docker run -dit --gpus all --ipc=host --name mblt_compiler \
  --device /dev/aries0:/dev/aries0 \
  -v /home/gpuadmin/Repo/seoik/AX_NPU:/workspace -w /workspace \
  mobilint/qbcompiler:1.1-cuda12.8.1-ubuntu22.04 /bin/bash
# 컴파일러 + 런타임 설치
docker exec mblt_compiler pip install /workspace/AX_NPU/download/qbcompiler-1.1.2+aries2-py3-none-any.whl
docker exec mblt_compiler bash -lc 'cd /tmp && tar xzf /workspace/AX_NPU/download/qbruntime_aries2-v4_v1.2.0_amd64.tar.gz && pip install /tmp/qbruntime_aries2-v4_v1.2.0_amd64/qbruntime/qbruntime/python/*cp310*.whl'
docker exec mblt_compiler pip install onnxruntime
```
(이미지 pull: `docker pull mobilint/qbcompiler:1.1-cuda12.8.1-ubuntu22.04`)

> NPU 장착 후 0-2 + 4단계를 한 번에 돌리려면 `bash ../setup/run_npu_tests.sh`.

### 0-3. (대안) docker 없이 호스트 conda로 — **추론 검증됨 (cos 0.997)**
docker는 필수가 아니다. NPU 드라이버/`libqbruntime.so`는 호스트에 있고, Python 3.10~3.12 conda
env만 만들면 호스트에서 동일하게 추론된다(호스트 base conda가 3.13이면 qbruntime wheel(cp38~cp312)이
안 맞으므로 전용 env를 만든다).
```bash
bash ../setup/setup_conda_host.sh          # env(pe_npu_host, py3.11) + qbruntime + torch/einops/timm
conda activate pe_npu_host
cd /home/gpuadmin/Repo/seoik/AX_NPU/AX_NPU
python tutorial_pe_npu/download_images.py
python tutorial_pe_npu/demo_inference.py    # docker exec 없이 바로 → cos 0.9973
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

## 2. NPU용 MXQ 컴파일

trunk(24 transformer block)를 NPU용 MXQ로 컴파일한다(`--feat-only` = attn_pool 전까지).
attn_pool은 CPU에서 처리하므로 trunk만 NPU로 보낸다. PE는 RoPE2D·attention pooling 때문에
일반 ONNX 경로로는 컴파일이 막혀, `pe_npu`가 모델에 전용 패치(5개)를 적용한 뒤 컴파일한다.

```bash
docker exec -w /workspace/AX_NPU mblt_compiler bash -lc '
  export LD_LIBRARY_PATH=/tmp/qbruntime_aries2-v4_v1.2.0_amd64/qbruntime/qbruntime/lib:$LD_LIBRARY_PATH
  python -m pe_npu.compile --mode compile --save ./pe_npu/out/pe_feat.mxq --feat-only \
    --calib-data-path ./pe_npu/calib_coco_hwc --calib-output 1 --device gpu'
# → pe_npu/out/pe_feat.mxq (약 314MB) 생성. "Compilation was successful." 확인.
```

> 검증된 `pe_feat.mxq`가 이미 `pe_npu/out/`에 있으면 재컴파일 없이 4단계로 바로 갈 수 있다.
> 컴파일 없이 operator 목록만 확인하려면 `--mode parse` (16bit override 이름 추출용).

---

## 3. 추론 클래스 (참고)

추론은 `pe_npu.MXQInferenceHybrid`가 제공한다 (NPU trunk + CPU pool을 묶어
`model(image) -> (B,1024)`, 기존 `TRTInference`와 인터페이스 동일). 직접 만들 필요 없다.

```python
import pe_npu
import numpy as np

model = pe_npu.MXQInferenceHybrid()            # 기본 MXQ = pe_npu/out/pe_feat.mxq
x = pe_npu.preprocess_image("some.jpg")        # (3,336,336) float32
emb = model.infer(x[None])                     # (1, 1024) 비전 임베딩
```

---

## 4. 추론 (예제 이미지 + 유사도 확인)

### 4-A. Jupyter 노트북 (권장) — `demo_inference.ipynb`
셀 단위로 돌리며 **이미지·유사도 히트맵·정확도 막대그래프를 인라인으로** 바로 본다.
호스트 conda(0-3) 셋업이면 `LD_LIBRARY_PATH`도 불필요(libqbruntime이 ldconfig 등록됨).
```bash
conda activate pe_npu_host
cd tutorial_pe_npu
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
- **원본 PyTorch 대비 NPU 임베딩 cos** — 양자화 정확도 (평균 0.99+ 면 정상, 검증값 0.997)

---

## 5. 직접 코드에서 쓰기 (요약)

```python
import pe_npu
import numpy as np

model = pe_npu.MXQInferenceHybrid("/workspace/AX_NPU/pe_npu/out/pe_feat.mxq")
# image: 전처리된 (B,3,336,336) float32 numpy 또는 torch
x = np.stack([pe_npu.preprocess_image(p) for p in paths], axis=0)
emb = model.infer(x)   # (B, 1024) 비전 임베딩
```

전처리는 `pe_npu.preprocess_image` (운영 `service.preprocess_image`와 동일:
RGB -> resize 336 bilinear -> /255 -> normalize 0.5).

---

## pe_npu 패키지 구성

| 모듈 | 역할 | CLI |
|------|------|-----|
| `pe_npu.pe_model` | 모델 로딩 + 컴파일 패치 (`load_pe`, `apply_pe_patches`) | - |
| `pe_npu.pe_vendor` | Meta PE vision encoder 코드 vendor 복사본 (외부 의존 제거) | - |
| `pe_npu.preprocess` | `preprocess_image` (resize 336 + normalize 0.5) | - |
| `pe_npu.calib` | calibration npy 생성 / HWC 변환 | `python -m pe_npu.calib` |
| `pe_npu.compile` | PE -> MXQ 컴파일 (`compile_pe`, `parse_pe`) | `python -m pe_npu.compile` |
| `pe_npu.inference` | `MXQInferenceHybrid` (NPU trunk + CPU pool) | - |

## 튜토리얼 파일

| 파일 | 설명 |
|------|------|
| `demo_inference.ipynb` | **(권장)** 추론 데모 노트북 — 이미지/유사도 히트맵/정확도 시각화 |
| `multicore_benchmark.ipynb` | 멀티코어 처리량 벤치 — 동기 vs async vs 멀티스레딩 (single 모드 8코어, x4.5) |
| `download_images.py` | 공개 COCO 예제 이미지 5장 다운로드 |
| `demo_inference.py` | 추론 데모 스크립트 (비대화형/CI용, 텍스트 출력) |
| `images/` | 다운로드된 예제 이미지 (gitignore) |

상세 배경/원리: `../reports/SOLUTION_single_io_compile.md`, `../reports/quantization_reference.md`
