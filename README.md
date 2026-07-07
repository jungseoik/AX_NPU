# AX_NPU — Mobilint NPU 비전인코더 추론

Mobilint **ARIES MLA100 PCIe Card**(Aries2)에서 **PE-Core-L14-336**(Meta Perception Encoder,
CLIP ViT-L/14) 비전인코더를 NPU로 추론하는 작업 공간. 원본 TensorRT 추론을 NPU로 대체한다.

호스트: Ubuntu + ARIES NPU 장착 서버. NPU 카드는 **1~N대** 구성 가능(`/dev/aries0`, `/dev/aries1`, …).
이 레포는 NPU 있는 여러 서버로 옮겨다니며 쓰는 것을 전제로 하며, 실제 CPU/GPU/NPU 개수는 서버마다 다르다 — 각 서버에서 직접 확인할 것.

이 레포는 **자기완결(self-contained)** — PE 모델 코드를 `pe_npu/pe_vendor/`에 vendor 복사해
외부 레포 없이 clone만으로 동작한다. (가중치만 HF `facebook/PE-Core-L14-336` 자동 다운로드)

---

## 상태

- **컴파일·추론 모두 동작. image→embedding 전부 NPU (full NPU).** trunk(24 block) + attn_pool head
  모두 NPU, 원본 pth 대비 **cos ≈ 0.99**. → `MXQInferenceFull`.
  - attn_pool은 그냥 INT8로 하면 QKᵀ matmul outlier로 깨졌는데(cos 0.46), 그 **score matmul만 16bit**로
    올리면 복구된다(컴파일 시 `--qk16`, Mobilint 해결책). → `reports/vendor/mobilint_resolution_attn_pool.md`
  - 레거시 **hybrid**(NPU trunk + CPU attn_pool, cos 0.997)는 `MXQInferenceHybrid`로 유지. full이
    CPU pool 병목을 제거한다 → `reports/performance/NPU_full_vs_hybrid.md`.
- 핵심 패키지 = **`pe_npu/`** (`python -m pe_npu.compile`, `import pe_npu`).
- **멀티카드/다채널**: 채널을 카드에 라운드로빈 분산(7대=56코어, `reports/performance/NPU_multicard_62ch_full.md`).
  ⚠️ **동시성 주의**: 한 모델에 `infer_async` 여러 건은 출력이 깨진다 — **카드당 1모델 + 멀티스레드 동기 `infer()`**
  (`MXQInferenceFull(num_threads=8)`)를 쓸 것. 모드/패턴 확정: `reports/performance/NPU_throughput_modes_correct.md`.

## 추론 2가지 방식

| | 옵션 A — 직접 컴파일 | 옵션 B — 가져와 쓰기 |
|---|---|---|
| 방법 | calib → `python -m pe_npu.compile --qk16` → 추론 | `pe_npu.MXQInferenceFull.from_hf("PIA-SPACE-LAB/MXQ_NPU")` |
| 필요 | **qbcompiler**(docker `mblt_compiler`) + 원본 가중치 | **qbruntime만** (컴파일러·가중치 불필요) |
| 용도 | 커스텀 calib/해상도·실험 | 운영·빠른 시작 |

MXQ는 aries2 바이너리라 어디서 컴파일하든 동일 → 한 번 컴파일해 HF에 올려두면(옵션 B) 받아 쓴다.

## 디렉토리

| 경로 | 내용 |
|------|------|
| `pe_npu/` | **핵심 패키지** — compile / inference / calib / preprocess / pe_model / export_pool_head / assets / pe_vendor |
| `tutorial/pe_npu/` | 따라하기 README + 추론 데모 노트북(`demo_inference.ipynb`) + 멀티코어 벤치(`multicore_benchmark.ipynb`) |
| `reports/` | 분석/원리 문서 (아래 인덱스) |
| `setup/` | 호스트 셋업 스크립트 (conda env, 드라이버, HF 업로드 등) |
| `deploy/vllm/` | Docker+vLLM(NPU) OpenAI 서빙 (docker compose, batch1) |
| `.claude/skills/` | `npu-setup`(신규 서버 세팅) 등 skill |
| `docs/` | **Mobilint 공식 SDK 문서**(벤더 원본) |
| `download/` | SDK 파일(드라이버/런타임/컴파일러). 비공개라 gitignore — 사람이 직접 배치 |

## 문서 인덱스 (핵심만 — **전체 인덱스는 [`reports/README.md`](reports/README.md)**)

| 문서 | 내용 |
|------|------|
| `tutorial/pe_npu/README.md` | **시작점** — 설치~calib~컴파일~추론 (옵션 A/B) 따라하기 |
| `.claude/skills/npu-setup/` | 신규 서버에서 `mobilint-cli status`까지 세팅 |
| `reports/vendor/mobilint_resolution_attn_pool.md` | ★ attn_pool INT8 붕괴 원인(QKᵀ outlier)·해결(16bit) → full NPU cos 0.99 |
| `reports/performance/NPU_throughput_modes_correct.md` | ★ **다채널 처리량·모드선택 + 올바른 동시성 패턴**(서비스 짤 때 필독) |
| `reports/performance/NPU_full_vs_hybrid.md` · `NPU_multicard_62ch_full.md` | full NPU 병목 제거 / 멀티카드 실측 |
| `reports/README.md` | **분석/벤치 전체 인덱스** (성능·양자화·설계·벤더·테스트) |
| `docs/README.md` | Mobilint 공식 SDK 문서 인덱스 (드라이버/런타임/컴파일러/멀티코어) |

---

## 빠른 시작

**아무것도 없는 서버: clone → HF 로그인 → 세팅 (SDK·모델 전부 HF에서 자동)**
```bash
# 0) HF 로그인 (조직 계정) — SDK/모델 다운로드에 필요. 이것만 사람이 해주면 됨.
huggingface-cli login          # 또는 export HF_TOKEN=hf_...

# 1) NPU 드라이버/런타임/CLI 설치 → mobilint-cli status
#    (SDK가 download/에 없으면 HF private 레포 sdk/aries2_v1.2.0/ 에서 자동 fetch)
sudo bash .claude/skills/npu-setup/setup_npu_cli.sh    # 세부: .claude/skills/npu-setup/SKILL.md

# 2) 파이썬 추론 환경 (qbruntime + torch 등)
bash setup/setup_conda_host.sh && conda activate pe_npu_host
```
> SDK 수동 다운만: `python setup/fetch_sdk_from_hf.py` → `download/` 채움.
> (HF `PIA-SPACE-LAB/MXQ_NPU`는 **private** — mxq + `sdk/<버전>/`(드라이버·런타임·컴파일러) 함께 관리.)

### 추론 실행
```bash
# A) 직접 컴파일 (qbcompiler — conda `pe_compile`(CPU 기본) 또는 docker(GPU 옵션))
#   python -m pe_npu.compile --mode compile --save pe_npu/out/pe_full.mxq --qk16 \
#     --calib-data-path <calib_hwc> --device cpu    # full NPU(권장). 기본 cpu, GPU면 --device gpu
#   # 셋업: .claude/setup-notes.md 방법 A(conda)/B(docker). --scheme 등은 --help

# B) 가져와 쓰기 (컴파일러 불필요)
python -c "import pe_npu, numpy as np; m=pe_npu.MXQInferenceFull.from_hf(); \
  print(m.infer(pe_npu.preprocess_image('tutorial/pe_npu/images/cat1.jpg')[None]).shape)"
```

> 컴파일은 NPU가 아니라 **호스트 CPU/GPU**(`--device`)에서 한다. NPU는 추론 전용.
> NPU는 **INT8 전용**(더 낮추는 bit4는 no-op 확인 — `reports/performance/NPU_batch_latency.md`).
