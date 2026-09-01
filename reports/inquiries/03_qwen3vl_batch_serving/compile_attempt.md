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

## 전체 컴파일 결과 (2026-07-03, "끝까지" 밀어본 결과)
3. **언어모델 전체 28블록 컴파일**: ✅ **오차 0** (num_ops=799, num_large_mape 0.00%, num_max_err 0.00%).
   90.9s, MBLT 6.5GB. → **언어 그래프는 완전히 우리가 컴파일 가능.**
4. **비전 인코더 전체 컴파일**: ⚠️ 그래프 생성·직렬화는 되나 **검증 불일치 81.53%**(128/157 op).
   `VisionModelForQwen3VL`(deepstack merger 포함) forward가 tuple `(hidden, deepstack_list[3])`을 반환하는데,
   출력 메타/deepstack merger 배선이 맞지 않아 수치가 안 맞음. 언어의 deepstack처럼 **비전 전용 배선 레시피가 더 필요**.
5. **MBLT→MXQ 양자화**: 착수 불가 지점 도달 — 두 벽:
   - (a) **calib 형식**: Qwen3-VL 언어 그래프는 입력 6개(inputs_embeds/attention_mask/position_ids/cache_position/
     deepstack_visual_embeds/visual_pos_masks) + `past_key_values`(DynamicCache) placeholder. multi-input calib
     디렉토리 형식은 있으나(sample_N/입력별 npy), 디코더+캐시 그래프의 calib 생성 절차가 미문서.
   - (b) **`CompileConfig` 레시피**: Qwen2-VL은 `activation16Bits ["inputs_embeds/reshape"]`(language),
     `["model_merger_fc2"]`(vision) + equivalentTransformation(QK/UD/SpinR1/SpinR2, HeadOutChRotation).
     **Qwen3-VL용 값은 문서에 없음.** 잘못 잡으면 attn_pool(cos 0.46) 때처럼 양자화 붕괴 → 공정한 비교 불가.

## 정리 (배포 MXQ 비교까지 가려면)
| 단계 | 우리 상태 |
|------|-----------|
| 언어 HF→MBLT (그래프) | ✅ 오차 0, 전체 28블록 |
| 비전 HF→MBLT (그래프) | ⚠️ 81% 불일치, 비전 배선 레시피 필요 |
| MBLT→MXQ (양자화) | ⛔ calib 절차 + activation16Bits 레시피 미문서 |
| 배포 MXQ와 비교 | ⛔ 위가 막혀 공정 비교 불가 |
→ **딱 Q3-B에서 요청하는 ①양자화 레시피 ②비전/그래프 패치 배선만 받으면 완주 가능.**

## 시사점 (→ Q3-B)
- **그래프 컴파일은 우리가 할 수 있음이 실증됨**(deepstack 레시피 역설계 성공, 오차 0).
- 자체 컴파일의 실질 병목 = **① Qwen3-VL용 CompileConfig 양자화 레시피(16bit 레이어) ② 비전/그래프 패치의
  공식 배선 절차** 두 가지. 이 둘만 Mobilint가 주면 배치·모드별 MXQ를 직접 뽑을 수 있음.
- 작업 재료: `~/vlm_compile_work/`(smoke 스크립트·로그·utils_q3.py).
