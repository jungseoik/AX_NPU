# Mobilint 기술지원 문의 — Qwen3-VL-2B 배치·코어모드 서빙(vLLM) & NPU 1장 동시요청

> **한 줄**: **vLLM(OpenAI 호환 HTTP API)** 으로 `mobilint/Qwen3-VL-2B-Instruct`를 서빙하려는데,
> **NPU 1장에서 코어모드(single/multi/global4/global8) × 배치사이즈를 늘려가며 동시요청 처리량**을
> 측정하고 싶습니다. 현재 배포 MXQ는 **batch=1 / global8 고정**이고 공식 문서엔 컴파일·배치 안내가 없어,
> (A) 배치/모드별 MXQ 제공 또는 (B) 우리가 직접 컴파일할 calibration 데이터셋+코드 제공을 문의합니다.

---

## 1. 우리 목표 (배경 — 참고용, 답변 필요 X)
- 서빙 프레임워크 = **vLLM + `vllm-mblt` 플러그인** → OpenAI 호환 `/v1/chat/completions`.
- 최종적으로 **"NPU 1장당 Qwen3-VL 동시요청을 몇 개까지 받을 수 있는지"** 를 코어모드·배치별로 실측해
  배포 설계(카드당 채널 수)를 잡는 게 목표입니다. **이 측정·판단은 저희가 직접 진행**하며,
  이 문의는 그 측정을 하기 위한 **재료(배치/모드별 MXQ, 또는 컴파일 방법)** 를 받기 위한 것입니다.
- (참고) 별도로 저희는 **NPU 1장에서 CLIP 계열 비전모델과 Qwen3-VL을 공존**시키는 시나리오도 자체 테스트할
  예정입니다 — 이건 저희가 알아서 실험하는 부분이라 이번 문의에 포함하지 않습니다.

## 2. 하려는 테스트 (현재 서버: ARIES2 ×7, CPU+NPU)
- **NPU 1장**에서 Qwen3-VL-2B를 **코어모드 × 배치사이즈**를 바꿔가며 vLLM으로 서빙 →
  동시요청 N(1/2/4/8/…)을 보내 **총지연·처리량(req/s)·NPU 메모리**를 측정.
- vLLM이 이 배치/동시성을 **어떻게 스케줄링**하는지(연속 배칭, `max_num_seqs`, chunked prefill 등)도 확인.

### 예상 결과 테이블 (이런 형태로 채울 계획)
| core_mode | max_batch | 동시요청 1 | 2 | 4 | 8 | NPU mem(MB) | 비고 |
|-----------|----------:|----:|--:|--:|--:|------------:|------|
| single    | 1 / 4 / 8 | ? | ? | ? | ? | ? | 코어1/추론 → 슬롯 많음(처리량형?) |
| multi     | 1 / 4 / 8 | ? | ? | ? | ? | ? | 클러스터 |
| global4   | 1 / 4 / 8 | ? | ? | ? | ? | ? | 4코어/추론 |
| global8   | 1 / 4 / 8 | ? | ? | ? | ? | ? | 8코어/추론 → 단건 latency 최소 |
> 셀 = (총지연 ms / 처리량 req·img per s). max_batch>1은 batch-compiled MXQ가 있어야 측정 가능.

## 3. 현재 상황 + 문서에서 확인한 것 (문의 전 자체 검증 완료)
- 배포 `mobilint/Qwen3-VL-2B-Instruct`의 config: **`max_batch_size=1`, `core_mode=global8`(text/vision), target_clusters [0,1]**.
  → vLLM `max_num_seqs=1` → **동시요청 직렬 큐잉**(부하테스트: NPU mem 2.5GB 고정, 총지연 채널수에 선형).
- **vllm-mblt README (Runtime Tuning) 확인** — ⚠️ 여기가 처음 오해했던 부분:
  - `--model-loader-extra-config '{"max_batch_size":N}'` override는 **존재함**(`resolve_model_max_batch_size` → scheduler
    `max_num_seqs`로 반영, `mblt_platform.py`). "override 경로 없음"은 **틀린 서술이었음**.
  - 단, README는 실제 배치 실행을 **"batch-compiled MXQ"** 전제로 설명(`Llama-3.2-1B-Instruct-Batch32` 예시, README 174-191).
    → **batch=1 MXQ에 override만 줘서는 NPU 실배치 안 됨. 진짜 batch>1 = 배치 컴파일된 MXQ 필요** (이 결론은 유지·오히려 근거 강화).
  - `core_mode`도 extra-config override 문서화됨(README 137,160). 단 **예제가 텍스트 모델뿐** → VLM 적용 시 dev_no가
    `from_pretrained`로 새어 upstream `Qwen3VLForConditionalGeneration.__init__` TypeError(크래시). → Q4.
- **docs/multicore.md 확인**: ARIES=4모드(Single/Multi/Global4/Global8) 전부 지원, core_mode=컴파일 시 `inference_scheme`
  (모드별 별도 MXQ). 단 예제가 vision CNN(resnet50)이라 **VLM(언어 디코더 포함)에 4모드 다 되는지는 불명** → Q2.
- **Qwen3-VL 컴파일 자료 부재 확인**: `mblt-sdk-tutorial/compilation/vlm`은 **Qwen2-VL 전용**(language+vision 분리 컴파일
  레시피 있음), `mblt-model-zoo`/`vllm-mblt`는 추론 런타임만. Qwen3-VL 컴파일/배치 예제는 없음. → Q3.

---

## 4. 문의 사항

### Q1. (확인) Qwen3-VL 동시요청 증가 = batch-compiled MXQ 필요, 맞나?
- 문서 이해: batch=1 MXQ에 `--model-loader-extra-config '{"max_batch_size":N}'`을 줘도 스케줄러 `max_num_seqs`만
  오를 뿐 NPU 실배치는 안 되고, **동시요청 실질 증가엔 batch로 컴파일된 Qwen3-VL MXQ**(`Llama-...-Batch32`류)가 필요.
- 이 이해가 맞는지 확인 + batch 없이 `core_mode=single`(코어별 슬롯)으로 동시성 얻는 방식이 VLM에도 유효한지.

### Q2. (확인) Qwen3-VL(VLM)도 4개 코어모드 전부 컴파일 가능한가?
- 문서 이해: ARIES=4모드 지원, core_mode=컴파일 시 `inference_scheme`. 단 예제가 vision CNN이라 **VLM(언어 디코더)이
  single/multi/global4/global8 다 되는지 불명**(배포본 global8만). 제공/컴파일 가능 범위 + 모드별 특성 문의.

### Q3-A. 배치/모드별 Qwen3-VL-2B MXQ 제공 — 우선 희망
- `Llama-3.2-1B-Instruct-Batch32`처럼, **batch>1로 컴파일된 `Qwen3-VL-2B` MXQ**(예: Batch 4/8/16)를 주실 수 있을까요?
- 가능하면 **Q2에서 확인된 core_mode 변형**도 선택 가능한 형태(또는 각 모드 MXQ)면 위 테이블을 그대로 측정할 수 있습니다.
- VLM에서 배치가 이미지 입력에 주는 제약(현재 "초기 1이미지" 등)과 권장 batch/정확도도 함께 안내 부탁드립니다.

### Q3-B. 자체 컴파일용 데이터셋+코드 제공 — (A)가 번거로우면
- 저희가 직접 `Qwen3-VL-2B`를 **배치·모드별로 컴파일**할 수 있도록 **① calibration 데이터셋(또는 생성 스크립트)**
  **② Qwen3-VL 컴파일 코드/레시피**(vision+language, RoPE/그래프 패치, batch_size 지정, 패키징 규격)를 받을 수 있을까요?
- 참고: 저희 qbcompiler 1.1.2에 `qwen3vl` 파서는 있으나, Qwen2-VL 튜토리얼의 전용 자산을 Qwen3-VL로 옮기는 방법이 문서화돼 있지 않습니다.

## 5. vllm-mblt 0.1.0 — Qwen3-VL 서빙 중 확인한 것 (성격 다름, 구분)
1. **[확인된 버그] `config.vocab_size` AttributeError**: `mblt_worker._make_cached_sampling_state`(+`_pack_prompt_token_ids`)가
   `self.model.config.vocab_size` **직접 접근**. 배포 config는 top-level vocab_size 없음(=None), `text_config.vocab_size`(151936)에 있음 →
   이미지 요청 시 EngineCore 크래시. **근거**: `mblt-model-zoo/utils/benchmark_utils.py`엔 이미 `text_config.vocab_size` 폴백
   헬퍼(`_resolve_config_vocab_size`)+테스트가 있는데 vllm-mblt엔 미적용. → 우리가 폴백 패치해 구동 중(리포트 시 diff 첨부).
2. **[질문 — 버그로 단정 X] `--model-loader-extra-config` TypeError**: 로드시 core_mode/dev_no override 시도 → kwarg가
   `from_pretrained` 거쳐 upstream `Qwen3VLForConditionalGeneration.__init__`까지 새어 `unexpected keyword 'dev_no'`.
   원인: vllm-mblt 화이트리스트(dev_no/core_mode/target_clusters)를 텍스트 모델은 config property로 흡수하나, VLM 최상위
   config엔 그 property가 없음(vision/text sub-config에만). **→ VLM에서 로드시 레이아웃 override 올바른 방법이 뭔지 질문**(Q4).
   (참고: `dev_no`/`core_mode`는 load_model 화이트리스트(→from_pretrained)로 감. `max_batch_size`는 그 화이트리스트엔
   없지만 `mblt_platform.resolve_model_max_batch_size`가 extra-config에서 읽어 scheduler `max_num_seqs`로 반영 = override 자체는 됨.
   단 실배치는 batch-compiled MXQ 필요.)

## 6. 재현 정보 (요청 시)
- ⚠️ 정직성 주의: vocab_size 크래시는 **우리 Docker 서빙 구성 중 실제로 만남**(그래서 Dockerfile에 폴백 패치 존재).
  다만 **순정 `vllm serve` bare 명령으로 처음부터 돌려 raw traceback을 캡처한 기록은 없음** → 메일엔 "공식 명령으로도
  동일 발생할 것으로 판단"(코드·config 근거 추론)으로 서술. 단정("수정 없이 재현") 금지. NPU 반납으로 지금 bare 재현 불가.
- 제공 가능: (Docker 기반)크래시 로그, `text_config.vocab_size` 폴백 패치 diff, 버전 정보(vllm-mblt/mblt-model-zoo/qbruntime).
- (커스텀 Docker 구성·부하테스트 스크립트는 우리 것이라 재현엔 불필요 — 요청 시에만. NPU 확보되면 bare 재현 로그 준비 가능.)

*작성 2026-07. 관련: attn_pool 문의(해결됨 → ../../vendor/mobilint_resolution_attn_pool.md)와 별개 — 이건 Qwen3-VL 서빙 배치 건.*
