# Mobilint 기술지원 문의 — Qwen3-VL-2B 배치 서빙(vLLM) & NPU 1장 동시요청 테스트

> **한 줄**: **vLLM(OpenAI 호환 HTTP API)** 으로 `mobilint/Qwen3-VL-2B-Instruct`를 서빙하려는데,
> **NPU 1장에서 코어모드(single/multi/global4/global8) × 배치사이즈를 늘려가며 동시요청 처리량**을
> 측정하고 싶습니다. 현재 배포 MXQ는 **batch=1 / global8 고정**이고 공식 문서엔 컴파일·배치 안내가 없어,
> (A) 배치/모드별 MXQ 제공 또는 (B) 우리가 직접 컴파일할 calibration 데이터셋+코드 제공을 문의합니다.

---

## 1. 우리 목표 (배경)
- 서빙 프레임워크 = **vLLM + `vllm-mblt` 플러그인** → OpenAI 호환 `/v1/chat/completions`.
- **운영 시나리오 (NPU 1장 기준)**:
  1. **NPU 1장에 CLIP 계열 모델(비전 임베딩) + Qwen3-VL 공존** — 한 카드(8코어)를 나눠 쓸 때 각 모델을
     몇 코어/어느 모드로 배치하고 동시요청을 얼마나 받을 수 있는지.
  2. **NPU 1장에 Qwen3-VL 단독** — 동시요청(concurrency)을 배치로 얼마나 소화하는지.
- 즉 **"NPU 1장당 Qwen3-VL 동시요청 한계"** 를 모드·배치별로 실측해 배포 설계(카드당 채널 수)를 잡으려 합니다.

## 2. 하려는 테스트 (현재 서버: ARIES2 ×7, CPU+NPU)
- **NPU 1장**에서 Qwen3-VL-2B를 **코어모드 4종 × 배치사이즈**를 바꿔가며 vLLM으로 서빙 →
  동시요청 N(1/2/4/8/…)을 보내 **총지연·처리량(req/s)·NPU 메모리**를 측정.
- vLLM이 이 배치/동시성을 **어떻게 지원/스케줄링**하는지(연속 배칭, `max_num_seqs`, chunked prefill 등)도 확인.

### 예상 결과 테이블 (이런 형태로 채울 계획)
| core_mode | max_batch | 동시요청 1 | 2 | 4 | 8 | NPU mem(MB) | 비고 |
|-----------|----------:|----:|--:|--:|--:|------------:|------|
| single    | 1 / 4 / 8 | ? | ? | ? | ? | ? | 코어1/추론 → 슬롯 많음(처리량형?) |
| multi     | 1 / 4 / 8 | ? | ? | ? | ? | ? | 클러스터 |
| global4   | 1 / 4 / 8 | ? | ? | ? | ? | ? | 4코어/추론 |
| global8   | 1 / 4 / 8 | ? | ? | ? | ? | ? | 8코어/추론 → 단건 latency 최소 |
> 셀 = (총지연 ms / 처리량 req·img per s). max_batch>1은 batch-compiled MXQ가 있어야 측정 가능.

## 3. 현재 막힌 지점 (실측)
- 배포 `mobilint/Qwen3-VL-2B-Instruct`의 config: **`max_batch_size=1`, `core_mode=global8`(text/vision), target_clusters [0,1]**.
  → vLLM `max_num_seqs=1` → **동시요청 직렬 큐잉**(부하테스트: NPU mem 2.5GB 고정, 총지연 채널수에 선형).
- **모드/배치를 실행 중 바꿀 수 없음**:
  - `--model-loader-extra-config '{"core_mode":...,"max_batch_size":N}'` → dev_no/kwarg가 `from_pretrained`로
    전달되어 `TypeError: Qwen3VLForConditionalGeneration.__init__() got unexpected keyword 'dev_no'` → 엔진 크래시.
  - 배치는 애초에 **MXQ 컴파일 시 결정**되는 값이라 실행 중 조정 불가로 이해하고 있습니다.
- **공식 문서·튜토리얼 확인**: `mblt-sdk-tutorial/compilation/vlm`은 **Qwen2-VL 전용**(RoPE 패치 등), **Qwen3-VL
  컴파일·배치 예제는 없음**. `mblt-model-zoo`/`vllm-mblt`는 추론(런타임)만. → **Qwen3-VL 배치/모드 컴파일 방법이 없어 문의**드립니다.

## 4. 문의 (핵심 2가지)
### (A) 배치/모드별 Qwen3-VL-2B MXQ 제공 — 우선 희망
- `Llama-3.2-1B-Instruct-Batch32`처럼, **batch>1로 컴파일된 `Qwen3-VL-2B` MXQ**(예: Batch 4/8/16)를 주실 수 있을까요?
- 가능하면 **core_mode 변형(single/multi/global4/global8)** 도 선택 가능한 형태(또는 각 모드 MXQ)면 위 테이블을 그대로 측정할 수 있습니다.
- VLM에서 배치가 이미지 입력에 주는 제약(현재 "초기 1이미지" 등)과 권장 batch/정확도도 함께 안내 부탁드립니다.

### (B) 자체 컴파일용 데이터셋+코드 제공 — (A)가 번거로우면
- 저희가 직접 `Qwen3-VL-2B`를 **배치·모드별로 컴파일**할 수 있도록 **① calibration 데이터셋(또는 생성 스크립트)**
  **② Qwen3-VL 컴파일 코드/레시피**(vision+language, RoPE/그래프 패치, batch_size 지정, 패키징 규격)를 받을 수 있을까요?
- 참고: 저희 qbcompiler 1.1.2에 `qwen3vl` 파서는 있으나, Qwen2-VL 튜토리얼의 전용 자산을 Qwen3-VL로 옮기는 방법이 문서화돼 있지 않습니다.

## 5. (겸사) vllm-mblt 0.1.0 버그 2건 — Qwen3-VL 서빙 중 확인
1. **`config.vocab_size` AttributeError**: `mblt_worker._make_cached_sampling_state`가 `self.model.config.vocab_size`
   접근 → `MobilintQwen3VLConfig`는 vocab_size가 `text_config`에 있어 이미지 요청 시 EngineCore 크래시.
   (임시로 `text_config.vocab_size` 폴백 패치 사용 중.)
2. **`--model-loader-extra-config` TypeError** (위 3): VLM에서 런타임 레이아웃(core_mode/dev_no) override 방법을 알려주세요.

## 6. 재현 정보 (요청 시)
- Docker(compose)+vllm-mblt 서빙 구성, 720p 이미지 동시요청 부하테스트 스크립트, 크래시 로그 전문 제공 가능.

*작성 2026-07. 관련: attn_pool 문의(해결됨, ../../mobilint_resolution_attn_pool.md)와 별개 — 이건 Qwen3-VL 서빙 배치 건.*
