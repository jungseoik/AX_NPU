# AX_NPU 프로젝트

Mobilint **ARIES MLA100 PCIe Card**(Aries2)에서 **PE-Core-L14-336 비전인코더**를 NPU로 추론.
호스트: Ubuntu + NPU 장착 서버. 이 레포는 NPU 있는 여러 서버로 옮겨다니며 쓰는 것을 전제로 한다.

> 아래 결과(cos 0.997 등)를 **검증했던 테스트 환경** 스펙: Ubuntu / Core Ultra 9 285K(24T) / RTX PRO 6000 / NPU `/dev/aries0`.
> 이건 그 당시 한 서버에서 기록한 값일 뿐, 현재 작업 중인 서버 스펙과 다를 수 있다(CPU/GPU 유무/NPU 개수/OS). 실제 스펙은 각 서버에서 직접 확인할 것.

## 현재 상태

- **컴파일·추론 모두 동작.** PE trunk(24 block)=NPU INT8, attn_pool head=CPU float = **hybrid**, 원본 pth 대비 **cos 0.997**.
- **자기완결(self-contained)**: PE 모델 코드는 `pe_npu/pe_vendor/`에 vendor 복사 → 외부 레포(Product-AI-mono) 의존 없음. 가중치만 HF `facebook/PE-Core-L14-336` 자동 다운로드.
- 핵심 패키지 = **`pe_npu/`**.
- **멀티카드**: NPU 여러 대면 채널 라운드로빈 분산으로 처리량↑(7대=56코어 실측, `reports/performance/NPU_multicard_62ch_benchmark.md`). 고채널 병목은 CPU 전처리(`reports/performance/NPU_preprocess_parallel.md`). `MXQInferenceHybrid`는 단일 카드(`device_id`)용.

## pe_npu 패키지

| 모듈 | 역할 |
|------|------|
| `compile` | PE→MXQ 컴파일. **`python -m pe_npu.compile --help`** (옵션: `--feat-only`/`--scheme`/`--bit4`/`--calib-data-path`/`--device` 등) |
| `inference` | `MXQInferenceHybrid`(NPU trunk+CPU pool). `.from_hf()` = 미리 컴파일된 자산 사용 |
| `calib` / `preprocess` / `pe_model` / `export_pool_head` / `assets` / `pe_vendor` | calib 생성 / 전처리 / 모델 로딩·패치 / pool head 추출 / HF 다운로드 / vendor된 PE 코드 |

## 추론 2가지 방식

- **옵션 A(직접 컴파일)**: calib → `python -m pe_npu.compile --feat-only ...` → 추론. **qbcompiler**(docker `mblt_compiler`) 필요. 커스텀 calib/해상도·실험용.
- **옵션 B(가져와 쓰기)**: `MXQInferenceHybrid.from_hf("PIA-SPACE-LAB/MXQ_NPU")`. **qbruntime만** 있으면 됨(qbcompiler·원본 가중치 불필요). 운영·빠른 시작.

## 헷갈리지 말 것

- **컴파일은 NPU가 아니라 호스트 CPU/GPU(`--device`)에서** 한다. NPU는 추론 전용.
- **NPU는 INT8 전용.** 양자화를 더 못 낮춘다(bit4 mixed-precision = no-op 확인). → `reports/performance/NPU_batch_latency.md`
- 컴파일 = docker `mblt_compiler`(qbcompiler 1.1.2), 추론 = 호스트 conda `pe_npu_host`(qbruntime, py3.10~3.12) 또는 docker.
- SDK(`download/`)는 비공개라 gitignore — 사람이 직접 배치. MXQ/pool head도 gitignore(HF로 배포).

## 문서 라우팅

- **따라하기**(설치~컴파일~추론, 옵션 A/B): `tutorial_pe_npu/README.md`
- **Qwen3-VL(멀티모달 LLM) 추론**: `tutorial_pe_npu/README_VLM_qwen3.md` + `demo_vlm_qwen3.ipynb` + 헬퍼 `tutorial_pe_npu/vlm_npu.py` + skill `.claude/skills/qwen3-vl/`. 이미지+프롬프트→텍스트. PE-Core와 별개로, Mobilint가 올린 `mobilint/Qwen3-VL-*` MXQ를 표준 HF API(`AutoModelForImageTextToText`+`mblt-model-zoo`)로 그대로 가져와 씀(포팅 불필요). **코어모드=global8**(8코어 전부, 단일스트림 latency 최적화, max_batch_size=1). 설치 핀: `mblt-model-zoo==1.3.1` + `transformers>=4.57`. 출처: `mobilint-runtime-gui` 백엔드
- **신규 서버 NPU 세팅**: `.claude/skills/npu-setup/` (clone 후 `mobilint-cli status`까지)
- **분석/원리**:
  - `reports/design/SOLUTION_single_io_compile.md` — 단일 입출력 컴파일 + hybrid 정확도(0.997) 해결
  - `reports/performance/NPU_batch_latency.md` — 배치 지연/멀티코어/Multi 모드/bit4 양자화 한계 (실측)
  - `reports/performance/NPU_multicard_62ch_benchmark.md` — 멀티카드(7×ARIES=56코어) 62채널 분산 추론 지연 (실측)
  - `reports/performance/NPU_preprocess_parallel.md` — 고채널 병목인 CPU 전처리 병렬화 벤치
  - `reports/performance/compile_benchmark.md` — 컴파일 시간 GPU vs CPU
  - `reports/quantization/quantization_reference.md`, `reports/quantization/QUANT_TUNING_guide.md` — 양자화 배경
- Mobilint SDK 공식 문서: `docs/` (멀티코어 `docs/multicore.md` 등)

## Skill

`.claude/skills/npu-setup`(신규 서버 세팅), `.claude/skills/qwen3-vl`(Qwen3-VL VLM 추론 코드 작성), `mblt-model-zoo.md` / `mblt-sdk-tutorial.md`(해당 레포 작업 규칙).
