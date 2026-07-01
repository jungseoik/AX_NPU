# AX_NPU 프로젝트

Mobilint **ARIES MLA100 PCIe Card**(Aries2)에서 **PE-Core-L14-336 비전인코더**를 NPU로 추론.
호스트: Ubuntu + NPU 장착 서버. 이 레포는 NPU 있는 여러 서버로 옮겨다니며 쓰는 것을 전제로 한다.

> 아래 결과(cos 0.997 등)를 **검증했던 테스트 환경** 스펙: Ubuntu / Core Ultra 9 285K(24T) / RTX PRO 6000 / NPU `/dev/aries0`.
> 이건 그 당시 한 서버에서 기록한 값일 뿐, 현재 작업 중인 서버 스펙과 다를 수 있다(CPU/GPU 유무/NPU 개수/OS). 실제 스펙은 각 서버에서 직접 확인할 것.

## 현재 상태

- **컴파일·추론 모두 동작. image→embedding 전부 NPU (full NPU).** trunk 24 block + attn_pool head 모두 NPU. 원본 pth 대비 **cos 0.99**. → `MXQInferenceFull`.
  - **핵심 해결**: attn_pool은 그냥 INT8로 하면 QKᵀ matmul outlier로 깨졌는데(full-NPU cos 0.46), 그 **score matmul만 16bit**로 올리면 복구(Mobilint 해결책, 컴파일 시 `--qk16`). → `reports/vendor/mobilint_resolution_attn_pool.md`
  - **레거시 hybrid**(NPU trunk + CPU attn_pool, cos 0.997)는 `MXQInferenceHybrid`로 유지(비교/하위호환). full이 CPU pool 병목 제거 → `reports/performance/NPU_full_vs_hybrid.md`.
- **자기완결(self-contained)**: PE 모델 코드는 `pe_npu/pe_vendor/`에 vendor 복사 → 외부 레포(Product-AI-mono) 의존 없음. 가중치만 HF `facebook/PE-Core-L14-336` 자동 다운로드.
- 핵심 패키지 = **`pe_npu/`**.
- **멀티카드**: NPU 여러 대면 채널 라운드로빈 분산으로 처리량↑(7대=56코어, full NPU 재측정 `reports/performance/NPU_multicard_62ch_full.md`). full NPU면 고채널 병목은 CPU 전처리만 남음(`reports/performance/NPU_preprocess_parallel.md`). 추론기는 단일 카드(`device_id`)용 → 카드마다 하나씩.

## ★ 다채널 동시성 (반드시 지킬 것 — 안 그러면 출력 깨짐)

- **한 모델에 `infer_async` 여러 건 동시 제출 = 출력 깨짐**(async 파이프라인 1개 공유, N=1만 안전 → 첫 건만 맞고 나머지 0/garbage). **latency 측정엔 쓸 수 있어도 실제 출력엔 절대 쓰지 말 것.**
- **정확+고속 패턴 = 카드당 1모델 + 멀티스레드 동기 `infer()`.** 런타임이 동시 sync 호출을 코어에 안전 분배 → 출력 정확(cos 1.0) + 8코어 활용. `MXQInferenceFull(num_threads=8)`이 배치 추론에 내장. (multi-model 인스턴스는 처리량 동일·메모리만 낭비 → 불필요.)
- **모드 선택**(1카드 실측, 출력검증): 처리량=**global4(16 img/s)**/single, 단건 저지연=**global8(71ms)**, multi 비권장. **8장/62채널**: 카드당 1모델+8스레드, global4 기준 ≈130 img/s → ~0.5s.
- 상세·재현: **`reports/performance/NPU_throughput_modes_correct.md`** (동시성 패턴 규명 + 모드선택 확정).

## pe_npu 패키지

| 모듈 | 역할 |
|------|------|
| `compile` | PE→MXQ 컴파일. **`python -m pe_npu.compile --help`** (옵션: `--qk16`(full NPU, 권장)/`--feat-only`(trunk만)/`--scheme`/`--calib-data-path`/`--device` 등) |
| `inference` | `MXQInferenceFull`(image→embedding 전부 NPU, 권장) / `MXQInferenceHybrid`(레거시 NPU trunk+CPU pool). `.from_hf()` = 미리 컴파일된 자산 사용 |
| `find_score_matmul` | attention score MatMul(QKᵀ) 자동 탐지 → `--qk16`이 16bit override (Mobilint 제공) |
| `calib` / `preprocess` / `pe_model` / `export_pool_head` / `assets` / `pe_vendor` | calib 생성 / 전처리 / 모델 로딩·패치 / (레거시)pool head 추출 / HF 다운로드 / vendor된 PE 코드 |

## 추론 2가지 방식

- **옵션 A(직접 컴파일)**: calib → `python -m pe_npu.compile --qk16 ...`(full NPU) → 추론. **qbcompiler**(docker `mblt_compiler`) 필요. 커스텀 calib/해상도·실험용.
- **옵션 B(가져와 쓰기)**: `MXQInferenceFull.from_hf(scheme="single")`. **qbruntime만** 있으면 됨(qbcompiler·원본 가중치 불필요). 운영·빠른 시작.
  - HF `PIA-SPACE-LAB/MXQ_NPU`는 **코어모드 폴더별**: `single/` `multi/` `global4/` `global8/`(각 `pe_full.mxq` + `CALIBRATION.md`). `scheme=`로 선택. 단건 latency=global8, throughput=single/global4. (레거시 hybrid: 루트 `pe_feat.mxq`+`pe_pool_head.pt`)
  - 4모드 동일 calib(COCO val2017 200장), 전부 cos 0.99. 모드 선택: `reports/performance/NPU_full_pipeline_e2e.md`.

## 헷갈리지 말 것

- **컴파일은 NPU가 아니라 호스트 CPU/GPU(`--device`)에서** 한다. NPU는 추론 전용.
- **NPU는 INT8 전용.** 양자화를 더 못 낮춘다(bit4 mixed-precision = no-op 확인). → `reports/performance/NPU_batch_latency.md`
- 컴파일 = docker `mblt_compiler`(qbcompiler 1.1.2), 추론 = 호스트 conda `pe_npu_host`(qbruntime, py3.10~3.12) 또는 docker.
- SDK(`download/`)는 비공개라 gitignore — 사람이 직접 배치. MXQ/pool head도 gitignore(HF로 배포).

## 문서 라우팅

- **따라하기**(설치~컴파일~추론, 옵션 A/B): `tutorial_pe_npu/README.md`
- **Qwen3-VL(멀티모달 LLM) 추론**: `tutorial_pe_npu/README_VLM_qwen3.md` + `demo_vlm_qwen3.ipynb` + 헬퍼 `tutorial_pe_npu/vlm_npu.py` + skill `.claude/skills/qwen3-vl/`. 이미지+프롬프트→텍스트. PE-Core와 별개로, Mobilint가 올린 `mobilint/Qwen3-VL-*` MXQ를 표준 HF API(`AutoModelForImageTextToText`+`mblt-model-zoo`)로 그대로 가져와 씀(포팅 불필요). **코어모드=global8**(8코어 전부, 단일스트림 latency 최적화, max_batch_size=1). 설치 핀: `mblt-model-zoo==1.3.1` + `transformers>=4.57`. 출처: `mobilint-runtime-gui` 백엔드
- **신규 서버 NPU 세팅**: `.claude/skills/npu-setup/` (clone 후 `mobilint-cli status`까지)
- **분석/원리** (전체 인덱스는 `reports/README.md`):
  - `reports/vendor/mobilint_resolution_attn_pool.md` — ★ attn_pool INT8 붕괴 원인(QKᵀ outlier)·해결(score matmul 16bit) → full NPU cos 0.99
  - `reports/performance/NPU_throughput_modes_correct.md` — ★ 다채널 처리량·모드선택 (올바른 1모델+멀티스레드 sync 패턴, 출력검증) ← **다채널 서비스 짤 때 필독**
  - `reports/performance/NPU_full_vs_hybrid.md` — full NPU vs hybrid, CPU pool 병목 제거 실측
  - `reports/performance/NPU_full_pipeline_e2e.md` — full NPU 코어모드 4종 × 채널 스윕 단계별 (latency)
  - `reports/performance/NPU_multicard_62ch_full.md` — full NPU 멀티카드 1→62ch (비포와 동일 구조)
  - `reports/performance/NPU_1card_coremode_16ch.md` — 1장 코어모드×1~16ch 순수추론(슬롯 거동, latency)
  - `reports/design/SOLUTION_single_io_compile.md` — [비포] 단일 입출력 컴파일 + hybrid(0.997)
  - `reports/performance/NPU_batch_latency.md` — 배치 지연/멀티코어/Multi 모드/bit4 양자화 한계 (실측)
  - `reports/performance/NPU_multicard_62ch_benchmark.md` — [비포·hybrid] 멀티카드 62채널 (trunk만)
  - `reports/performance/NPU_preprocess_parallel.md` — 고채널 병목인 CPU 전처리 병렬화 벤치
  - `reports/performance/NPU_preprocess_uint8_offload.md` — 전처리 NPU 오프로드(uint8 입력) 실험: normalize는 폴딩되나 resize 불가라 전처리 이득 없음(정확도 0.99 유지). + 남은 최적화 정리
  - `reports/performance/NPU_preprocess_cv2_e2e.md` — 전처리 최적화 의사결정(e2e 기준): 리소스 원천=resize, torchvision→cv2(INTER_LINEAR) 전환으로 56ch e2e -25%·CPU 5배↓(정확도 0.99→0.97, opt-in), 워커16, 파이프라이닝 미채택
  - `reports/performance/compile_benchmark.md` — 컴파일 시간 GPU vs CPU
  - `reports/quantization/quantization_reference.md`, `reports/quantization/QUANT_TUNING_guide.md` — 양자화 배경
- Mobilint SDK 공식 문서: `docs/` (멀티코어 `docs/multicore.md` 등)

## Skill

`.claude/skills/npu-setup`(신규 서버 세팅), `.claude/skills/qwen3-vl`(Qwen3-VL VLM 추론 코드 작성), `mblt-model-zoo.md` / `mblt-sdk-tutorial.md`(해당 레포 작업 규칙).
