# Qwen3-VL 멀티카드 배치(동시요청) 지연 — VQA 1토큰

Qwen3-VL-2B를 ARIES NPU 여러 장에 나눠 **동시요청(배치)** 을 처리할 때, 카드수 × 배치별
**전부 응답받는 데 걸리는 총 지연**을 실측. 코드: `tutorial/pe_npu/vlm_npu.py`(`VLMPool`),
재현: `reports/scripts/bench_vlm_batch.py`.

## 설정
- 모델 `mobilint/Qwen3-VL-2B-Instruct` (mblt-model-zoo, global8 단일모드, `max_batch_size=1`).
- 태스크: 이미지 + "Is there a bus? Answer yes or no" → **`max_new_tokens=1`** (yes/no 1토큰). 답 "yes"(정답).
- 카드 지정: `config.vision_config.dev_no` / `text_config.dev_no` = 카드번호 (from_pretrained 인자 아님).
- 분산: 동시요청 B개를 N카드에 라운드로빈, **카드당 in-flight 1**(순차). 값 = B개 전부 끝난 wall-clock.
- 서버 NPU 7장(aries0~6) → **8장은 측정 불가**, 1/2/4/6/7로.

## 총 지연 (ms) — B개 동시요청 전부 응답까지
| 카드＼배치 | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 179 | 359 | 721 | 1478 | 2942 | 6058 | 11945 |
| 2 | 179 | 221 | 420 | 826 | 1639 | 3284 | 6590 |
| 4 | 179 | 221 | 256 | 459 | 867 | 1712 | 3352 |
| 6 | 180 | 209 | 256 | 482 | 679 | 1284 | 2372 |
| **7** | 180 | 218 | 251 | 454 | 691 | 1144 | **2179** |

## 처리량 (req/s)
| 카드＼배치 | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5.6 | 5.6 | 5.5 | 5.4 | 5.4 | 5.3 | 5.4 |
| 4 | 5.6 | 9.1 | 15.6 | 17.4 | 18.5 | 18.7 | 19.1 |
| 7 | 5.5 | 9.2 | 16.0 | 17.6 | 23.2 | 28.0 | 29.4 |

## 해석
- **단건(B=1) ≈ 180ms** — 카드 수 무관 상수. 요청 1개는 카드 1장이 처리(global8=한 카드 8코어 전부).
  멀티카드는 단건을 못 줄이고 **동시 처리량**을 올린다.
- **1장은 배치에 정비례**: B=64 → 64×~186ms ≈ 11.9s.
- **카드 늘리면 배치지연 ↓**: B=64 기준 1장 11.9s → **7장 2.18s (5.5x)**. 모델: `지연 ≈ ⌈B/N⌉ × ~186ms`.
- 저배치 6≈7은 granularity(⌈B/N⌉ 동일), 고배치(32/64)에서 7장 우위.
- 처리량 상한: 1장 5.4 → 7장 ~29 req/s. 1토큰이라 요청당 CPU/파이썬 글루 비중이 커 완전선형은 아님
  (출력 토큰 길어지면 NPU 비중↑ → 효율 개선).

## 사용 (VLMPool)
```python
from vlm_npu import VLMPool
pool = VLMPool("mobilint/Qwen3-VL-2B-Instruct", device_ids="auto")   # 장착 NPU 전부
pool.ask("img.jpg", "Is there a bus? yes/no", max_new_tokens=1)      # 단건(카드 자동배정)
pool.ask_batch([(img1, q1), (img2, q2), ...], max_new_tokens=1)      # 동시요청 분산, 순서 보존
```

*실측 ARIES2 7장, Qwen3-VL-2B, mblt-model-zoo 2.0.0 / transformers 4.57.1. dev_no 카드지정 실증.*
