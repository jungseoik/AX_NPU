# AX_NPU — Mobilint NPU 비전인코더 추론

Mobilint **ARIES MLA100 PCIe Card**(Aries2)에서 **PE-Core-L14-336**(Meta Perception Encoder,
CLIP ViT-L/14) 비전인코더를 NPU로 추론하는 작업 공간. 원본 TensorRT 추론을 NPU로 대체한다.

호스트: Ubuntu + ARIES NPU 장착 서버. NPU 카드는 **1~N대** 구성 가능(`/dev/aries0`, `/dev/aries1`, …).
이 레포는 NPU 있는 여러 서버로 옮겨다니며 쓰는 것을 전제로 하며, 실제 CPU/GPU/NPU 개수는 서버마다 다르다 — 각 서버에서 직접 확인할 것.

이 레포는 **자기완결(self-contained)** — PE 모델 코드를 `pe_npu/pe_vendor/`에 vendor 복사해
외부 레포 없이 clone만으로 동작한다. (가중치만 HF `facebook/PE-Core-L14-336` 자동 다운로드)

---

## 상태

- **컴파일·추론 모두 동작.** trunk(24 block)=NPU INT8, attn_pool head=CPU float = **hybrid**,
  원본 pth 대비 **cos ≈ 0.997**(샘플셋에 따라 0.997~0.999).
- 핵심 패키지 = **`pe_npu/`** (`python -m pe_npu.compile`, `import pe_npu`).
- **멀티카드**: NPU 여러 대가 있으면 채널을 라운드로빈 분산해 처리량을 키운다(7대=56코어 실측,
  `reports/NPU_multicard_62ch_benchmark.md`). `MXQInferenceHybrid` 자체는 단일 카드(`device_id`)용.

## 추론 2가지 방식

| | 옵션 A — 직접 컴파일 | 옵션 B — 가져와 쓰기 |
|---|---|---|
| 방법 | calib → `python -m pe_npu.compile --feat-only` → 추론 | `pe_npu.MXQInferenceHybrid.from_hf("PIA-SPACE-LAB/MXQ_NPU")` |
| 필요 | **qbcompiler**(docker `mblt_compiler`) + 원본 가중치 | **qbruntime만** (컴파일러·가중치 불필요) |
| 용도 | 커스텀 calib/해상도·실험 | 운영·빠른 시작 |

MXQ는 aries2 바이너리라 어디서 컴파일하든 동일 → 한 번 컴파일해 HF에 올려두면(옵션 B) 받아 쓴다.

## 디렉토리

| 경로 | 내용 |
|------|------|
| `pe_npu/` | **핵심 패키지** — compile / inference / calib / preprocess / pe_model / export_pool_head / assets / pe_vendor |
| `tutorial_pe_npu/` | 따라하기 README + 추론 데모 노트북(`demo_inference.ipynb`) + 멀티코어 벤치(`multicore_benchmark.ipynb`) |
| `reports/` | 분석/원리 문서 (아래 인덱스) |
| `setup/` | 호스트 셋업 스크립트 (conda env, 드라이버, HF 업로드 등) |
| `.claude/skills/` | `npu-setup`(신규 서버 세팅) 등 skill |
| `docs/` | **Mobilint 공식 SDK 문서**(벤더 원본) |
| `download/` | SDK 파일(드라이버/런타임/컴파일러). 비공개라 gitignore — 사람이 직접 배치 |

## 문서 인덱스

| 문서 | 내용 |
|------|------|
| `tutorial_pe_npu/README.md` | **시작점** — 설치~calib~컴파일~추론 (옵션 A/B) 따라하기 |
| `.claude/skills/npu-setup/` | 신규 서버에서 `mobilint-cli status`까지 세팅 |
| `reports/SOLUTION_single_io_compile.md` | 단일 입출력 컴파일 + hybrid 정확도(0.997) 해결 |
| `reports/NPU_batch_latency.md` | 배치 지연·멀티코어·Multi 모드·bit4 양자화 한계 (실측) |
| `reports/NPU_multicard_62ch_benchmark.md` | 멀티카드(7×ARIES, 56코어) 62채널 분산 추론 지연 (실측) |
| `reports/NPU_preprocess_parallel.md` | 고채널 병목인 CPU 전처리 병렬화 (스레드/멀티프로세스) 벤치 |
| `reports/compile_benchmark.md` | 컴파일 시간 GPU vs CPU |
| `reports/quantization_reference.md`, `QUANT_TUNING_guide.md` | 양자화 배경 |
| `docs/` | 드라이버/런타임/컴파일러 설치, 멀티코어 등 공식 문서 |

---

## 빠른 시작

```bash
# 1) 호스트 conda 추론 환경 (qbruntime + torch 등)
bash setup/setup_conda_host.sh && conda activate pe_npu_host

# 2-A) 직접 컴파일 (qbcompiler, docker mblt_compiler 필요)
#   python -m pe_npu.compile --mode compile --save pe_npu/out/pe_feat.mxq --feat-only \
#     --calib-data-path <calib> --device gpu       # 옵션: --scheme/--bit4 등은 --help

# 2-B) 가져와 쓰기 (컴파일러 불필요)
python -c "import pe_npu, numpy as np; m=pe_npu.MXQInferenceHybrid.from_hf(); \
  print(m.infer(pe_npu.preprocess_image('tutorial_pe_npu/images/cat1.jpg')[None]).shape)"
```

> 컴파일은 NPU가 아니라 **호스트 CPU/GPU**(`--device`)에서 한다. NPU는 추론 전용.
> NPU는 **INT8 전용**(더 낮추는 bit4는 no-op 확인 — `reports/NPU_batch_latency.md`).
