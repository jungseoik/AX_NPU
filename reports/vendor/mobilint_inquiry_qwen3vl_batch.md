# Mobilint 기술지원 문의 — Qwen3-VL-2B의 배치(batch>1) NPU 서빙 & vllm-mblt 이슈

> **목적**: `mobilint/Qwen3-VL-2B-Instruct`를 **vLLM(vllm-mblt)** 로 NPU 서빙 중인데, 현재 MXQ가
> **batch=1**이라 동시요청이 직렬 처리됩니다. **다채널(동시 여러 요청) 처리량**을 위해 batch>1이
> 필요합니다. 아래 (A) batch-compiled Qwen3-VL MXQ 제공 가능 여부, (B) 자체 컴파일 가이드,
> 그리고 (C) 확인된 vllm-mblt 버그 2건을 문의합니다.

## 1. 환경
| 항목 | 값 |
|------|----|
| NPU | ARIES MLA100 PCIe (Aries2), 7장, 펌웨어 1.2.5 |
| 런타임/컴파일러 | qbruntime 1.2.0 / qbcompiler 1.1.2 |
| 서빙 | vllm-mblt 0.1.0 (vllm 0.11.2), mblt-model-zoo[transformers] 1.5.1, Docker(CPU+NPU) |
| 모델 | `mobilint/Qwen3-VL-2B-Instruct` (image→text) |

## 2. 현황 (실측)
- Docker + vllm-mblt로 `vllm serve mobilint/Qwen3-VL-2B-Instruct` 정상 기동, OpenAI API 응답 OK.
- 그러나 config가 **max_batch_size=1** → vLLM `max_num_seqs=1`, 동시요청은 **직렬 큐잉**.
- 720p 이미지 + max_tokens=1로 동시 1/2/4/6 요청 부하테스트: **NPU 메모리 2548MB로 고정**(증가 없음),
  총지연은 채널 수에 선형 증가(직렬). 즉 배치가 안 되어 처리량이 안 오릅니다.

## 3. 배치>1 시도 결과 (전부 실패/불가)
1. **실행중 override**: `--model-loader-extra-config '{"dev_no":0,"max_batch_size":4}'` →
   `TypeError: Qwen3VLForConditionalGeneration.__init__() got an unexpected keyword argument 'dev_no'`
   (kwargs가 from_pretrained로 전달되어 wrapper __init__에서 터짐) → 엔진 크래시.
2. **자체 컴파일 조사**: `mblt-sdk-tutorial/compilation/vlm`은 **Qwen2-VL 전용**(RoPE 패치
   `CachedQwen2VLTextRotaryEmbedding`, 모델별 그래프 패치, SpinQuant rotation). **Qwen3-VL 예제·안내는 없음.**
   VLM language 컴파일에는 batch_size 노브도 노출돼 있지 않습니다(LLM 튜토리얼에만 `LlmConfig(batch_size)`).

## 4. 문의
### (A) batch-compiled Qwen3-VL-2B MXQ 제공 — 가장 희망
- `mobilint/Llama-3.2-1B-Instruct-Batch32`처럼, **batch>1로 컴파일된 `Qwen3-VL-2B` MXQ**를 제공해 주실 수 있을까요?
  (예: Batch4/8/16). 받으면 vllm-mblt에서 MODEL_NAME만 교체해 바로 배치 서빙이 됩니다.
- 권장 batch 값(정확도/메모리 균형)과, VLM에서 배치가 이미지 입력에 어떤 제약을 주는지도 알려주시면 좋겠습니다.

### (B) 자체 컴파일 가이드 (A가 어려우면)
- qbcompiler 1.1.2에 `qwen3vl` 파서가 있는데, **Qwen3-VL-2B를 batch>1로 컴파일하는 절차**를 안내해 주실 수 있나요?
  특히: ① Qwen3-VL용 RoPE/그래프 패치(Qwen2-VL의 `CachedQwen2VLTextRotaryEmbedding` 대응),
  ② **VLM language 모델에 batch_size를 지정하는 방법**, ③ vision+language MXQ 패키징/ config(max_batch_size) 규격.

### (C) vllm-mblt 0.1.0 버그 2건 (Qwen3-VL 서빙 시 확인)
1. **`config.vocab_size` AttributeError**: `mblt_worker._make_cached_sampling_state`가
   `self.model.config.vocab_size`에 접근하는데 `MobilintQwen3VLConfig`는 vocab_size가 `text_config`에 있어
   이미지 요청 시 EngineCore 크래시. (임시로 `text_config.vocab_size` 폴백 패치해 사용 중.)
2. **`--model-loader-extra-config` TypeError**: 위 3-1처럼 dev_no 등 kwarg가 from_pretrained로 전달되어
   Qwen3-VL wrapper `__init__`에서 실패. VLM에서 런타임 레이아웃(core_mode/dev_no) override는 어떻게 하나요?

## 5. 재현 정보
- Dockerfile/compose/부하테스트 스크립트 제공 가능(내부 레포 `deploy/vllm/`).
- 위 크래시 로그(스택트레이스) 전문 공유 가능.

*작성 2026-07. 관련: attn_pool 문의(해결)와 별개 — 이건 Qwen3-VL LLM/VLM 서빙 배치 건.*
