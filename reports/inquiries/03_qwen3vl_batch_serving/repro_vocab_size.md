# vocab_size 크래시 — 실제 재현 기록 (2026-07-03)

순정 `vllm-mblt 0.1.0` + 공식 README 명령으로 직접 재현·검증한 기록. (메일 첨부/근거용)

## 환경
- OS: Ubuntu 22.04.1 LTS (kernel 6.5.0-41-generic)
- CPU: Intel Xeon Gold 6526Y ×2 (64 threads)
- NPU: Mobilint ARIES ×7 (`/dev/aries0~6`, PCIe 209f:0000), driver 1.13.0, firmware 1.2.5, 16GB/카드
- SW: python 3.11 / `vllm==0.11.2` / **`vllm-mblt==0.1.0`(수정 없음)** / `mblt-model-zoo[transformers]==1.5.1` / `mobilint-qb-runtime==1.2.0`
- Model: `mobilint/Qwen3-VL-2B-Instruct`
- 실행: 컨테이너 내에서 README "Serve a VLM Model" 명령 그대로
  `vllm serve mobilint/Qwen3-VL-2B-Instruct --trust-remote-code` (max-model-len 자동 4096)

## 결과 매트릭스
| vllm-mblt | 요청 | top_k | 결과 |
|-----------|------|:-----:|------|
| 순정(unpatched) | 텍스트 "안녕" | 20(기본) | ✅ 정상 응답 |
| 순정(unpatched) | 이미지+텍스트(고양이 사진) | 20(기본) | ✅ 정상 응답(이미지 설명) |
| 순정(unpatched) | 텍스트 "안녕" | **-1** | 💥 **크래시** — EngineCore 종료 |
| 폴백 패치 | 텍스트 "안녕" | **-1** | ✅ 정상 응답 (수정 검증) |

**핵심 정정**: 트리거는 "이미지 요청"이 아니라 **`top_k <= 0`(top_k 비활성, 예: -1/0)** 이다.
기본 top_k(=모델 config 20)면 텍스트·이미지 모두 정상. `top_k<=0`이면 텍스트만으로도 크래시.

## 원인 (traceback 요지)
```
File ".../vllm_mblt/mblt_worker.py", line 1596, in execute_model
    cached_sampling_state=self._make_cached_sampling_state(
File ".../vllm_mblt/mblt_worker.py", line 1445, in _make_cached_sampling_state
    else self.model.config.vocab_size
File ".../transformers/configuration_utils.py", line 207, in __getattribute__
    return super().__getattribute__(key)
AttributeError: 'MobilintQwen3VLConfig' object has no attribute 'vocab_size'
```
- `_make_cached_sampling_state`:
  `top_k = int(sampling_params.top_k if sampling_params.top_k > 0 else self.model.config.vocab_size)`
  → `top_k <= 0`이면 `self.model.config.vocab_size` 접근. VLM config엔 top-level `vocab_size`가 없고
  `text_config.vocab_size`(151936)에만 있어 AttributeError.
- `_pack_prompt_token_ids`(line 1419)도 동일 접근 → batch 모델(max_batch_size>1)에서 추가로 문제 가능.

## 수정
- `self.model.config.vocab_size` → `getattr(config,"vocab_size",None) or config.get_text_config().vocab_size` 폴백.
- 근거: `mblt-model-zoo/utils/benchmark_utils.py`에 이미 동일 폴백(`_resolve_config_vocab_size`)+테스트 존재.
- 검증: 위 매트릭스 4행 — 패치 후 `top_k=-1` 정상.
