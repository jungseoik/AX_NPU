# 📮 Mobilint 문의 스레드 (시간순)

문의 1건 = 폴더 1개. **번호가 곧 문의 순서**이며, 번호가 클수록 최신이다.
보낸 문의(`INQUIRY.md`)와 받은 회신(`REPLY.md`)을 같은 폴더에 둔다.

| # | 폴더 | 시기 | 주제 | 상태 |
| --- | --- | --- | --- | --- |
| 01 | [`01_pe_single_io_compile/`](01_pe_single_io_compile/INQUIRY.md) | 2026-06 | PE 비전 인코더가 25개 서브그래프로 분할되는 문제 | **미발송** — 모델 레벨 5개 패치로 자체 해결 |
| 02 | [`02_attn_pool_int8/`](02_attn_pool_int8/INQUIRY.md) | 2026-06 | attention pooling head의 INT8 양자화 정확도 붕괴 | **해결** — 회신·분석은 [`../vendor/`](../vendor/mobilint_resolution_attn_pool.md) |
| 03 | [`03_qwen3vl_batch_serving/`](03_qwen3vl_batch_serving/README.md) | 2026-07 | Qwen3-VL-2B 배치·코어모드 서빙(vLLM), NPU 1장 동시요청 + vllm-mblt 버그 2건 | **미발송** |
| 04 | [`04_vit_quantization_speed/`](04_vit_quantization_speed/REPLY.md) | 2026-09 | ViT 비전 인코더 양자화 설정별 속도·정확도(W8A16/W4A16/W4A8), uint8 입력, 추론 패턴 | **회신 수신** — 재현 완료 |
| 05 | [`05_a16_selection_followup/`](05_a16_selection_followup/EMAIL.md) | 2026-09 | `select_a16.py` 요청 + A16 5개가 어느 체크포인트 기준인지 확인 | **발송 대기** ★최신 |

## 폴더 안 파일 규칙

| 파일 | 용도 |
| --- | --- |
| `INQUIRY.md` | 우리가 보낸(또는 작성한) 문의 본문 |
| `REPLY.md` | Mobilint 회신 정리 + 원문 |
| `examples/` · 그 외 | 회신 첨부 코드, 재현 기록 등 |

## ★ 대외 표기 규약 (벤더 문의 시)

**모델 실명을 대외 문의에 쓰지 않는다.** 정보 노출 방지를 위해 아래 표기를 쓴다 — **같은 모델이다.**

| 대외 표기 (벤더에게) | 실제 (내부) |
| --- | --- |
| `PIA custom ViT-L/14`, `PIA ViT-L/14 비전인코더`, `커스텀 PIA ViT` | **Perception Encoder — PE-Core-L14-336** (`pe_npu/`) |
| 첨부 파일명 `vit_l14_336.pt`, `pia_full.mxq` | `pe_full.mxq` (HF `PIA-SPACE-LAB/MXQ_NPU`) — **MD5 동일 확인됨** |

- 벤더에 보낸 `pia_ve_latency_headroom_full.zip` 의 MXQ 4모드는 HF 배포본과 바이트 단위로 같다
  (`single`: `bc022878c054aef2f6c673d91cb2e887`).
- 다만 **모델 구조는 숨기지 않았다** — 지연 문의 때 동봉한 README에 "표준 CLIP/ViT가 아니라
  2D RoPE 기반 ViT-L/14 + attention pooling head"라고 명시해 보냈다(31행).
- 벤더 회신 예제가 공개 CLIP(`openai/clip-vit-large-patch14-336`) 기준으로 작성됐을 가능성이 있어
  문의 05로 확인 중이다. 포팅 코드의 클래스명이 open_clip 계열이라 `CLIP` 으로 보이는 점이
  원인일 수 있다(`ve_lib` 는 미동봉).
- **주의**: 우리 모델은 공개 CLIP과 다르다 — 활성함수가 일반 GELU(공개 CLIP은 QuickGELU)이고,
  공개 CLIP에 없는 **attention pooling head**(`visual.attn_pool.*`)가 붙어 있다.

## 참고

- 02번의 회신 원문과 해결 분석은 다른 문서에서 널리 참조되고 있어 [`../vendor/`](../vendor/) 에 그대로 둔다
  (`mobilint_reply_email.md`, `mobilint_resolution_attn_pool.md`).
- 04번 회신은 현재 레포 서술 2건(**bit4 = no-op**, **async 동시 제출 금지**)과 충돌 소지가 있다.
  → [`04_vit_quantization_speed/REPLY.md`](04_vit_quantization_speed/REPLY.md) §4 참조.
