# Qwen3-VL-2B 자체 컴파일 시도 기록 (2026-07-03)

qbcompiler 1.1.2로 Qwen3-VL-2B를 직접 컴파일 가능한지 실측. **결론: 그래프(HF→MBLT) 단계는
문서화 안 된 deepstack 레시피를 역설계하니 오차 0으로 성공.** MXQ 양자화 레시피만 미확보.

## 환경
Ubuntu 22.04.1 / Xeon Gold 6526Y ×2(64T) / RAM 188G / **GPU 없음(CPU 컴파일)** /
qbcompiler 1.1.2 / **transformers 4.57.1**(튜토리얼 핀; 5.12.1이면 모델 구조 불일치로 실패) /
원본 가중치 `Qwen/Qwen2-VL-2B-Instruct`, `Qwen/Qwen3-VL-2B-Instruct`.

## 진행 단계와 결과
1. **파이프라인 검증 — Qwen2-VL 언어모델(2블록)**: `mblt-sdk-tutorial/compilation/vlm` 그대로.
   → ✅ MBLT 컴파일 + 검증 통과 (num_large_mape 0.00%, num_max_err 0). 25.5s. 툴체인 정상.
2. **Qwen3-VL 언어모델(2블록) 포팅**: qbcompiler의 `qwen3vl` 확장 클래스로 스왑
   (`Qwen3VLForConditionalGenerationWrapper`, `Projection`, `CachedQwen3VLTextRotaryEmbedding`,
   `Qwen3VL(Model)_get_image_feature(s)`).
   - 1차 실패: `_deepstack_process`의 `hidden_states[visual_pos_masks,:]=local_this`(masked scatter)를
     FX 트레이서가 못 다룸 → `UnboundLocalError`.
   - **해결(역설계)**: 확장에 이미 있는 두 함수를 배선 —
     `build_full_visual_embeds(deepstack_visual_embeds, visual_pos_masks)`로 sparse→dense[L,T,D] 변환 +
     `_deepstack_process`를 `patched_deepstack_process`(= `hidden_states + visual_embeds`)로 몽키패치.
     이 함수들은 qbcompiler 안 어디에서도 호출되지 않음 = **사용자가 배선해야 하는 미문서화 레시피**.
   → ✅ MBLT 컴파일 + 검증 통과 (**num_large_mape 0.00%, num_max_err 0**). 26.4s. MBLT 1587MB(2블록).

## 남은 것 (전체 배포용 MXQ까지)
- 언어모델 **전체 28블록** 컴파일(현재 2블록 스모크) — CPU라 수 시간 예상.
- **비전 인코더** 컴파일 — Qwen3-VL vision 패치(`PatchedQwen3VLVisionBlock`, `PatchedPatchMerger`,
  `VisionModelForQwen3VL`) 배선 필요(언어와 유사한 역설계 예상).
- **MBLT→MXQ 양자화** — `mxq_compile`에 calib(COCO 준비됨) + **`CompileConfig` 레시피**.
  ⚠️ **여기가 진짜 미지수**: Qwen2-VL은 `activation16Bits: ["inputs_embeds/reshape"]`(language),
  `["model_merger_fc2"]`(vision) + equivalentTransformation(QK/UD/SpinR1/SpinR2, HeadOutChRotation)를 씀.
  **Qwen3-VL용 16bit 레이어·transformation 값은 문서에 없음.** 잘못 잡으면 attn_pool 때처럼 양자화 붕괴 위험.
- config 패키징(model_type=mobilint-qwen3_vl, mxq_path, core_mode/batch) + 배포 MXQ와 정확도/거동 비교.

## 시사점 (→ Q3-B)
- **그래프 컴파일은 우리가 할 수 있음이 실증됨**(deepstack 레시피 역설계 성공, 오차 0).
- 자체 컴파일의 실질 병목 = **① Qwen3-VL용 CompileConfig 양자화 레시피(16bit 레이어) ② 비전/그래프 패치의
  공식 배선 절차** 두 가지. 이 둘만 Mobilint가 주면 배치·모드별 MXQ를 직접 뽑을 수 있음.
- 작업 재료: `~/vlm_compile_work/`(smoke 스크립트·로그·utils_q3.py).
