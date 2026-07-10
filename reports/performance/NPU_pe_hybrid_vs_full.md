# [hybrid vs full] full NPU vs hybrid — CPU attn_pool 병목 제거 (QKᵀ 16bit)

> **[출력 정확성 주의]** 이 문서의 수치는 `infer_async`(같은 이미지)로 측정한 **latency**다 — 시간은 유효하나, `infer_async` multi-in-flight는 서로 다른 이미지에서 출력이 깨진다(N=1만 안전). **정확한 다채널 처리 패턴(1모델+멀티스레드 sync)과 출력검증 처리량**은 → [`NPU_pe_throughput_modes_full.md`](NPU_pe_throughput_modes_full.md).

attn_pool의 QKᵀ matmul만 16bit로 올린 **full MXQ**(`--qk16`, image→embedding 전부 NPU)와,
기존 **hybrid**(NPU trunk + CPU attn_pool)를 같은 서버·같은 입력으로 비교. CPU pool 병목이
사라지는지 실측. (배경: `../vendor/mobilint_resolution_attn_pool.md`)

- full : `[P]전처리(CPU)` → `[N]image→embedding(NPU 전부)`
- hybrid: `[P]전처리(CPU)` → `[T]trunk(NPU)` → `[Pool]attn_pool+proj(CPU, 채널별 직렬)`
- 측정: 7×ARIES async 분산, median of 5, 실제 영상 프레임. calib=COCO val2017 200장(레이턴시는 calib 무관, 최종 COCO 자산 기준 재측정).

## 1. 정확도 (원본 pth 대비 cos)

| 구성 | cos | head 처리 |
|------|:---:|------|
| hybrid (trunk INT8 + CPU pool) | 0.997 | CPU |
| **full NPU (QKᵀ 16bit)** | **0.99** (COCO holdout 0.9905 / 도메인 0.9889) | **NPU** |

→ full NPU가 hybrid와 **동급 정확도**. CPU 우회 없이도 정확.

## 2. 단계별·채널별 지연 (ms, 7카드 async)

| ch | P 전처리 | hyb T(NPU) | hyb Pool(CPU) | **hyb e2e** | full N(NPU) | **full e2e** | e2e 개선 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 51 | 285 | 8 | 344 | 286 | 337 | 2% |
| 4 | 222 | 287 | 18 | 526 | 287 | 509 | 3% |
| 7 | 660 | 288 | 27 | 976 | 287 | 947 | 3% |
| 8 | 324 | 312 | 28 | 664 | 311 | 635 | 4% |
| 16 | 1383 | 338 | 55 | 1776 | 339 | 1721 | 3% |
| 28 | 806 | 381 | 99 | 1285 | 369 | 1174 | 9% |
| 42 | 872 | 441 | 142 | 1455 | 429 | 1301 | 11% |
| **56** | 771 | 533 | **159** | 1463 | 522 | **1293** | **12%** |

## 3. 핵심 결론

1. **head를 NPU에 올려도 추론시간 추가 ≈ 0.** `full N ≈ hyb T` (56ch: 522 ≈ 533ms).
   attn_pool은 전체 연산의 ~0.8%라, full MXQ가 trunk-only MXQ와 사실상 같은 NPU 시간에 끝난다.
2. **CPU attn_pool 단계가 통째로 사라진다.** hybrid의 Pool은 채널별 직렬 CPU 연산이라
   1→56ch에서 8→159ms로 증가하는데, full에서는 이 단계 자체가 없다(NPU가 카드별 병렬 처리).
3. **e2e 개선은 채널이 늘수록 커진다**: 저채널 1% → 최대배치(56ch) **12%**. 고채널일수록 CPU
   pool 비중이 커지므로 제거 효과도 커진다.
4. **부가 이득(수치 외)**:
   - 추론 시 **torch / pe_vendor 의존성 제거** (full은 image→embedding NPU 한 번, CPU 연산 없음).
   - pool head 가중치(`pe_pool_head.pt`) 배포 불필요 → `pe_full.mxq` 하나로 끝.
   - CPU가 약하거나 동시성이 높은 환경일수록 hybrid의 CPU pool 병목이 더 크므로 full 이득이 더 커진다.

> 참고: 본 측정의 CPU pool(56ch 159ms)은 이전 hybrid 단독 측정(`NPU_pe_pipeline_e2e_hybrid.md`,
> 584ms)과 절대값이 다르다(서버 부하/측정 시점 차이). 상대 비교(full이 pool 단계를 제거)는
> 동일 런 back-to-back 측정이라 유효하다.

## 4. 재현
```bash
# full MXQ 컴파일 (QKᵀ 16bit)
python -m pe_npu.compile --mode compile --save pe_npu/out/pe_full.mxq \
  --calib-data-path <calib_hwc> --device cpu --qk16
# 벤치 (full vs hybrid)
conda activate pe_npu_host
python reports/scripts/bench_full_vs_hybrid.py pe_npu/out/pe_full.mxq
```
- 관련: [`../vendor/mobilint_resolution_attn_pool.md`](../vendor/mobilint_resolution_attn_pool.md)(원인·해결),
  [`NPU_pe_pipeline_e2e_hybrid.md`](NPU_pe_pipeline_e2e_hybrid.md)(hybrid 시절 CPU 병목 분석)

*작성 2026-06. full MXQ = QKᵀ 16bit override, 원본 대비 cos 0.99 (COCO holdout 0.9905 / 도메인 0.9889). 7×ARIES2 실측 (COCO 자산).*
