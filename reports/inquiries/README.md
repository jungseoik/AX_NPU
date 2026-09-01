# 📮 Mobilint 문의 스레드 (시간순)

문의 1건 = 폴더 1개. **번호가 곧 문의 순서**이며, 번호가 클수록 최신이다.
보낸 문의(`INQUIRY.md`)와 받은 회신(`REPLY.md`)을 같은 폴더에 둔다.

| # | 폴더 | 시기 | 주제 | 상태 |
| --- | --- | --- | --- | --- |
| 01 | [`01_pe_single_io_compile/`](01_pe_single_io_compile/INQUIRY.md) | 2026-06 | PE 비전 인코더가 25개 서브그래프로 분할되는 문제 | **미발송** — 모델 레벨 5개 패치로 자체 해결 |
| 02 | [`02_attn_pool_int8/`](02_attn_pool_int8/INQUIRY.md) | 2026-06 | attention pooling head의 INT8 양자화 정확도 붕괴 | **해결** — 회신·분석은 [`../vendor/`](../vendor/mobilint_resolution_attn_pool.md) |
| 03 | [`03_qwen3vl_batch_serving/`](03_qwen3vl_batch_serving/README.md) | 2026-07 | Qwen3-VL-2B 배치·코어모드 서빙(vLLM), NPU 1장 동시요청 + vllm-mblt 버그 2건 | **미발송** |
| 04 | [`04_vit_quantization_speed/`](04_vit_quantization_speed/REPLY.md) | 2026-09 | ViT 비전 인코더 양자화 설정별 속도·정확도(W8A16/W4A16/W4A8), uint8 입력, 추론 패턴 | **회신 수신** — 검증 미착수 ★최신 |

## 폴더 안 파일 규칙

| 파일 | 용도 |
| --- | --- |
| `INQUIRY.md` | 우리가 보낸(또는 작성한) 문의 본문 |
| `REPLY.md` | Mobilint 회신 정리 + 원문 |
| `examples/` · 그 외 | 회신 첨부 코드, 재현 기록 등 |

## 참고

- 02번의 회신 원문과 해결 분석은 다른 문서에서 널리 참조되고 있어 [`../vendor/`](../vendor/) 에 그대로 둔다
  (`mobilint_reply_email.md`, `mobilint_resolution_attn_pool.md`).
- 04번 회신은 현재 레포 서술 2건(**bit4 = no-op**, **async 동시 제출 금지**)과 충돌 소지가 있다.
  → [`04_vit_quantization_speed/REPLY.md`](04_vit_quantization_speed/REPLY.md) §4 참조.
