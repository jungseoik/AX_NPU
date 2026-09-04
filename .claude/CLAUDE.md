# AX_NPU 프로젝트

Mobilint **ARIES MLA100 PCIe Card**(Aries2)에서 딥러닝 모델을 NPU로 추론.
호스트: Ubuntu + NPU 장착 서버. 이 레포는 NPU 있는 여러 서버로 옮겨다니며 쓰는 것을 전제로 한다.

**다루는 모델 (워크스트림, 상세는 아래 문서 라우팅):**
- **PE-Core-L14-336 비전인코더** — 메인. 이미지→임베딩, 직접 컴파일(5패치)+full NPU. 패키지 `pe_npu/`.
- **YOLO11 객체탐지** — 이미지→bbox, 직접 컴파일(패치 0). 패키지 `yolo_npu/`.
- **Qwen3-VL(멀티모달 LLM)** — 이미지+프롬프트→텍스트, Mobilint MXQ 가져와 씀(포팅 불필요).

> 아래 결과(cos 0.997 등)를 **검증했던 테스트 환경** 스펙: Ubuntu / Core Ultra 9 285K(24T) / RTX PRO 6000 / NPU `/dev/aries0`.
> 이건 그 당시 한 서버에서 기록한 값일 뿐, 현재 작업 중인 서버 스펙과 다를 수 있다(CPU/GPU 유무/NPU 개수/OS). 실제 스펙은 각 서버에서 직접 확인할 것.

## 현재 상태

- **컴파일·추론 모두 동작. image→embedding 전부 NPU (full NPU).** trunk 24 block + attn_pool head 모두 NPU. 원본 pth 대비 **cos 0.99**. → `MXQInferenceFull`.
  - **핵심 해결**: attn_pool은 그냥 INT8로 하면 QKᵀ matmul outlier로 깨졌는데(full-NPU cos 0.46), 그 **score matmul만 16bit**로 올리면 복구(Mobilint 해결책, 컴파일 시 `--qk16`). → `reports/vendor/mobilint_resolution_attn_pool.md`
  - **레거시 hybrid**(NPU trunk + CPU attn_pool, cos 0.997)는 `MXQInferenceHybrid`로 유지(비교/하위호환). full이 CPU pool 병목 제거 → `reports/performance/NPU_pe_hybrid_vs_full.md`.
- **자기완결(self-contained)**: PE 모델 코드는 `pe_npu/pe_vendor/`에 vendor 복사 → 외부 레포(Product-AI-mono) 의존 없음. 가중치만 HF `facebook/PE-Core-L14-336` 자동 다운로드.
- 핵심 패키지 = **`pe_npu/`**.
- **멀티카드**: `MXQInferenceFull`이 단일/멀티 통합 — `device_id`(단일) / `device_ids=[..]`(지정) / `device_ids="auto"`(전 카드). 카드당 1모델 + **코어모드별 슬롯(single8/global4:2/global8:1)×카드** 스레드풀로 배치 자동 분산(출력 cos 1.0). 7대=56코어(`reports/performance/NPU_pe_multicard_62ch_full.md`), 고채널 병목은 CPU 전처리(`reports/performance/NPU_preprocess_1_parallel.md`).

## ★ 다채널 동시성 (반드시 지킬 것 — 안 그러면 출력 깨짐)

- **한 모델에 `infer_async` 여러 건 동시 제출 = 출력 깨짐**(async 파이프라인 1개 공유, N=1만 안전 → 첫 건만 맞고 나머지 0/garbage). **latency 측정엔 쓸 수 있어도 실제 출력엔 절대 쓰지 말 것.**
- **정확+고속 패턴 = 카드당 1모델 + 멀티스레드 동기 `infer()`.** 런타임이 동시 sync 호출을 코어에 안전 분배 → 출력 정확(cos 1.0). `MXQInferenceFull`이 코어모드별 슬롯(single8/global4:2/global8:1)×카드로 스레드풀 자동 구성 → 배치 주면 알아서 병렬. (multi-model 인스턴스는 처리량 동일·메모리만 낭비 → 불필요.)
- **모드 선택**(1카드 실측, 출력검증): 처리량=**global4(16 img/s)**/single, 단건 저지연=**global8(71ms)**, multi 비권장. **8장/62채널**: 카드당 1모델+8스레드, global4 기준 ≈130 img/s → ~0.5s.
- 상세·재현: **`reports/performance/NPU_pe_throughput_modes_full.md`** (동시성 패턴 규명 + 모드선택 확정).

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
  - **양자화 스킴 선택**: `from_hf(scheme="global4", quant="W4A16")` — 기본 `W8A16`(기존 배포본과 동일).
    HF 경로 `<quant>/<scheme>/pe_full.mxq`, 신규 경로 없으면 기존 최상위 경로로 자동 폴백.
  - HF `PIA-SPACE-LAB/MXQ_NPU`는 **코어모드 폴더별**: `single/` `multi/` `global4/` `global8/`(각 `pe_full.mxq` + `CALIBRATION.md`). `scheme=`로 선택. 단건 latency=global8, throughput=single/global4. (레거시 hybrid: 루트 `pe_feat.mxq`+`pe_pool_head.pt`)
  - 4모드 동일 calib(COCO val2017 200장), 전부 cos 0.99. 모드 선택: `reports/performance/NPU_pe_pipeline_e2e_full.md`.

## 헷갈리지 말 것

- **컴파일은 NPU가 아니라 호스트 CPU/GPU(`--device`)에서** 한다. NPU는 추론 전용.
- **양자화 비트폭은 조절된다(2026-09-01 정정).** 기존 MXQ는 순수 INT8이 아니라 **W8A16**(Activation 기본값이 output/ffn=16bit)이고,
  `BitConfig.Transformer.Weight(...=4)`로 **W4A16도 동작**한다(크기 −42%, 처리량 +43%, cos 0.9136 — 정확도 보정 옵션 검토 중).
  단 `mixed_precision`(비율 방식) 필드는 여전히 no-op. → `reports/performance/NPU_pe_quant_schemes.md`, `reports/performance/NPU_batch_latency.md`
- 컴파일 = docker qbcompiler 이미지, 추론 = 호스트 conda `pe_npu_host`(qbruntime, py3.10~3.12) 또는 docker.
  - **컴파일에 GPU 불필요** — 벤더가 버전별 `-cpu`/`-cuda` 이미지를 쌍으로 배포하고 코드가 CPU로 자동 폴백한다.
    GPU 없는 서버는 `mobilint/qbcompiler:1.1-cpu-ubuntu22.04`(SDK 1.0v) / `1.2-cpu-ubuntu22.04`(1.1v).
    이미지·요건은 `setup/sdk_versions.json` 기준, 조회는 `python setup/sdk_resolve.py --sdk <버전>`.
  - **SDK 번들 버전**: 1.0v(드라이버1.13/런타임1.2.0/컴파일러1.1.2, 기본) / 1.1v(1.14/1.4.0/1.2.0).
    신규 서버 세팅은 `bash setup/setup_all.sh --sdk 1.0|1.1` 한 줄 (스킬 `npu-setup`이 버전을 먼저 물어본다).
- SDK(`download/`)는 비공개라 gitignore — 사람이 직접 배치. MXQ/pool head도 gitignore(HF로 배포).

## 문서 라우팅

- **따라하기**(설치~컴파일~추론, 옵션 A/B): `tutorial/pe_npu/README.md`
- **Qwen3-VL(멀티모달 LLM) 추론**: `tutorial/pe_npu/README_VLM_qwen3.md` + `demo_vlm_qwen3.ipynb` + 헬퍼 `tutorial/pe_npu/vlm_npu.py` + skill `.claude/skills/qwen3-vl/`. 이미지+프롬프트→텍스트. PE-Core와 별개로, Mobilint가 올린 `mobilint/Qwen3-VL-*` MXQ를 표준 HF API(`AutoModelForImageTextToText`+`mblt-model-zoo`)로 그대로 가져와 씀(포팅 불필요). **코어모드=global8**(8코어 전부, 단일스트림 latency 최적화, max_batch_size=1). 설치 핀: `mblt-model-zoo==1.3.1` + `transformers>=4.57`. 출처: `mobilint-runtime-gui` 백엔드. **멀티카드**: 컴파일 모드는 global8 하나뿐이라 카드별 인스턴스로 **동시요청 분산**(`VLMPool(device_ids="auto")`, 카드 지정은 `config.vision/text.dev_no`). 실측(2B VQA 1토큰): 단건 ~180ms, 64동시 1장 12s→7장 2.2s → `reports/performance/NPU_qwen3vl_multicard_batch.md`
- **YOLO11 객체탐지 (컴파일~추론)**: `tutorial/yolo_npu/README.md` + `demo_yolo11_npu.ipynb` + 패키지 `yolo_npu/`(detect/compile). PE와 달리 **패치 0개**로 컴파일(표준 CNN, `yolo_decode_include`). 모델은 **mxq만 바꾸면**(11n/s/m/l). 단일/지정/`device_ids="auto"` 멀티카드 + `detect_batch`(출력 무결성 검증됨). **추적**: `ByteTrack`(자체 경량, `yolo_npu/track.py`) — 검출 NPU + 추적 CPU. mAP(11m INT8 0.53=fp32의 96%)·4모드×배치·1~7장 스케일링(64ch 198→1072 img/s): `reports/performance/NPU_yolo11_coremode_batch.md`. **기본 진입점 `YOLONPU.load(model, scheme)`** = HF `PIA-SPACE-LAB/MXQ_NPU/yolo/<model>/<scheme>/` 먼저 → 없으면 컴파일 안내
- **신규 서버 NPU 세팅**: `.claude/skills/npu-setup/` (clone 후 `mobilint-cli status`까지)
- **평가용 데이터셋**: `eval/README.md` + `eval/tta.py`. 실데이터는 git에 안 넣고 HF private에 zip으로 두고
  **토큰만 있으면 재현** — `export HF_TOKEN=... && python -m eval.tta download` → `eval/datasets/TTA_인증용/`(gitignore).
  현재 `TTA_인증용`(HF `PIA-SPACE/AX_NPU_TTA`, dataset·private): 이상행동 4종(falldown/fire/intrusion/smoke)
  영상 200 + 라벨 json 200 + 이벤트 clips 252, 총 2.2GB. 원천은 NAS192TB `10.128.30.36:/volume1/AI_data`의
  `TTA/TTA_인증용_재인코딩/`(재인코딩본을 씀). 갱신은 `pack` → `ZIP_SHA256` 반영 → `upload`.
- **분석/원리** (전체 인덱스는 `reports/README.md`):
  - `reports/vendor/mobilint_resolution_attn_pool.md` — ★ attn_pool INT8 붕괴 원인(QKᵀ outlier)·해결(score matmul 16bit) → full NPU cos 0.99
  - `reports/performance/NPU_pe_throughput_modes_full.md` — ★ 다채널 처리량·모드선택 (올바른 1모델+멀티스레드 sync 패턴, 출력검증) ← **다채널 서비스 짤 때 필독**
  - `reports/performance/NPU_pe_hybrid_vs_full.md` — full NPU vs hybrid, CPU pool 병목 제거 실측
  - `reports/performance/NPU_pe_pipeline_e2e_full.md` — full NPU 코어모드 4종 × 채널 스윕 단계별 (latency)
  - `reports/performance/NPU_pe_multicard_62ch_full.md` — full NPU 멀티카드 1→62ch (비포와 동일 구조)
  - `reports/performance/NPU_pe_1card_coremode_full.md` — 1장 코어모드×1~16ch 순수추론(슬롯 거동, latency)
  - `reports/design/SOLUTION_single_io_compile.md` — [비포] 단일 입출력 컴파일 + hybrid(0.997)
  - `reports/performance/NPU_pe_quant_tuning_compiler_version.md` — ★★ **OPTQ·SWS 열화는 qbcompiler 1.1.2 버그**(2026-09-03).
    1.2.0으로 컴파일하면 부호가 뒤집혀 **정확도가 올라간다**: W4A16 0.8795→**0.9654**, W8A16 0.9747→**0.9946**, W4A8+A16×5 0.8560→**0.9420**.
    튜닝 끈 베이스라인은 두 버전 동일(0.9110→0.9126) → 차이는 전적으로 튜닝 구현. 벤더 회신 수치(0.973/0.905)도 1.2.0에서 재현됨.
    `aries-rb`=Aries2(mxq v7 동일)라 **1.2.0 컴파일본이 드라이버 1.13에서 그대로 돈다** — 추론 코드 변경 0(`inference.py`에 버전 의존 없음, mxq만 교체).
    컴파일 타겟은 `pe_npu/target_device.py`가 `qbcompiler.__version__`으로 **자동 판별**(1.1.x→aries2 / 1.2.x→aries-rb, `--target-device`/`AX_NPU_TARGET_DEVICE`로 덮어쓰기 가능)
  - `reports/performance/NPU_pe_quant_tuning_matrix_120.md` — ★★ **1.2.0 매트릭스 24종 NPU 전수 검증(2026-09-04)**. 튜닝이 3개 양자화 모두에서 정확도 상승,
    **코어모드는 정확도 무관**(4모드 cos 동일, 지연만 다름). 배포 후보: **W8A16+튜닝 0.9946**(현행 0.9937과 크기·속도 동일 → 공짜 교체) / W4A16+튜닝 0.9654(−42%, +35%) / W4A8+튜닝 0.9420(가장 빠름).
    `multi` 모드는 20ch 5.7~7.5s로 쓰지 말 것. ★ 이전 W4A8 값 0.8932는 A16 5개 누락 산출물이었고 정정값이 0.9420
  - `reports/RUNBOOK_quant_matrix_120.md` — ★ **GPU 서버 재현 절차서**. clone→.env→SDK 1.1v→docker→calib→24조합(양자화3×튜닝2×**코어모드4**)→NPU 검증.
    `verify_quant_tuning_matrix.py --src-dir <dir>`가 **회귀 기준 24점**(표준 이미지셋 `download/coco/val2017` 앞 20장 기준)을 ±0.005로 자동 대조, 이탈 시 exit 1
  - `reports/performance/NPU_pe_quant_tuning_verify.md` — [1.1.2 한정] 양자화×튜닝 24종 NPU 실측(9/9 cos 악화). **위 문서에서 원인 규명·정정됨** — 배포 판단 근거로 쓰지 말 것. W4A8_L5A16은 cos 0.879로 보유·미배포
  - `reports/performance/NPU_pe_quant_schemes.md` — ★ 양자화 스킴(W8A16/W4A16/W4A8) × 코어모드 × 1~20채널 실측.
    배포 기본은 **W8A16**(cos 0.9936), W4A16은 크기 −42%·처리량 +13%·cos 0.9135, W4A8은 붕괴(미배포)
  - `reports/performance/NPU_pe_quant_tuning_compile.md` — [컴파일] 양자화 3종 × SWS/OPTQ 온오프 × single·global4 24종 GPU 스윕(조합당 5~8분). HF `<quant>/<tuning>/<scheme>/pe_full.mxq` + 루트 `TUNING_MATRIX.md`(컴파일 환경 표기). cos는 NPU 검증 대기
  - `reports/performance/NPU_batch_latency.md` — 배치 지연/멀티코어/Multi 모드/bit4 양자화 한계 (실측, §6 정정 포함)
  - `reports/performance/NPU_pe_multicard_62ch_hybrid.md` — [비포·hybrid] 멀티카드 62채널 (trunk만)
  - `reports/performance/NPU_preprocess_1_parallel.md` — 고채널 병목인 CPU 전처리 병렬화 벤치
  - `reports/performance/NPU_preprocess_2_uint8_offload.md` — 전처리 NPU 오프로드(uint8 입력) 실험: normalize는 폴딩되나 resize 불가라 전처리 이득 없음(정확도 0.99 유지). + 남은 최적화 정리
  - `reports/performance/NPU_preprocess_3_cv2_decision.md` — 전처리 최적화 의사결정(e2e 기준): 리소스 원천=resize, torchvision→cv2(INTER_LINEAR) 전환으로 56ch e2e -25%·CPU 5배↓(정확도 0.99→0.97, opt-in), 워커16, 파이프라이닝 미채택
  - `reports/performance/compile_benchmark.md` — 컴파일 시간 GPU vs CPU
  - `reports/quantization/quantization_reference.md`, `reports/quantization/QUANT_TUNING_guide.md` — 양자화 배경
- **Mobilint 문의 스레드**: `reports/inquiries/` — 번호가 문의 순서(클수록 최신). 인덱스 `reports/inquiries/README.md`.
  - ★ **대외 표기 규약**: 벤더 문의에는 모델 실명을 쓰지 않는다. PE-Core-L14-336 → **"PIA custom ViT-L/14"**,
    `pe_full.mxq` → `pia_full.mxq`, 체크포인트 → `vit_l14_336.pt`. **같은 모델이며 자산도 MD5 동일**이다.
    (우리 모델은 공개 CLIP과 다르다 — 일반 GELU + attention pooling head 보유)
  최신 04(ViT 양자화 W4A16/W4A8·uint8 입력·single+async 권고)는 위의 **"NPU는 INT8 전용(bit4=no-op)"** 및
  **"다채널 동시성 — async 다건 제출 금지"** 두 서술과 충돌 소지가 있다. 검증 전까지 기존 서술 유지, 확인 후 갱신.
- Mobilint SDK 공식 문서: `docs/` (멀티코어 `docs/multicore.md` 등)

## Skill

전부 `.claude/skills/<이름>/SKILL.md` 구조(팀 공유, git 추적).

- `npu-setup` — 신규 서버 NPU 환경 세팅(clone → `mobilint-cli status`까지).
- `qwen3-vl` — Qwen3-VL VLM(이미지+프롬프트→텍스트) 추론 코드 작성.
- `yolo-npu` — YOLO11 객체탐지(이미지→bbox, +ByteTrack 추적) 추론 코드 작성(`yolo_npu/`).
- `mblt-model-zoo` / `mblt-sdk-tutorial` — **서브모듈**(`mblt-model-zoo/`, `mblt-sdk-tutorial/`) 작업용. `paths:`가 각 서브모듈 서브트리로 스코프돼 그 안에서 파일을 만질 때만 발동(각 Mobilint 레포 작업 규칙). 서브모듈 동기화: `git submodule update --init`.
