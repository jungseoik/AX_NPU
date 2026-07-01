# [애프터] full NPU 코어모드 × 채널 스윕 단계별 e2e — CPU attn_pool 병목 제거

> **[출력 정확성 주의]** 이 문서의 수치는 `infer_async`(같은 이미지)로 측정한 **latency**다 — 시간은 유효하나, `infer_async` multi-in-flight는 서로 다른 이미지에서 출력이 깨진다(N=1만 안전). **정확한 다채널 처리 패턴(1모델+멀티스레드 sync)과 출력검증 처리량**은 → [`NPU_throughput_modes_correct.md`](NPU_throughput_modes_correct.md).

`NPU_coremode_pipeline_e2e.md`(hybrid: `P→T(NPU trunk)→Pool(CPU attn_pool)→E`, 고채널 CPU 병목)의
**애프터판**. attn_pool의 QKᵀ matmul만 16bit로 올린 **full NPU**(image→embedding 전부 NPU)에서
같은 채널 스윕을 다시 측정. **Pool(CPU) 단계가 사라졌다.** (원인·해결: `../vendor/mobilint_resolution_attn_pool.md`)

- 단계: `[P]전처리(CPU)` → `[N]image→embedding(NPU 전부, 7카드 async)`  ← **Pool 없음**
- 4개 코어모드 MXQ(동일 COCO val2017 200장 calib + `--qk16`), median of 5, 실제 영상 프레임, 7×ARIES2.
- 정확도: 4개 모드 전부 원본 pth 대비 **cos 0.99** (COCO holdout 0.9905 / 도메인 0.9889; 모드는 스케줄링만 다름, 출력 동일).

## 1. NPU 추론 단계(N) — 모드별 (ms)

| ch | single | multi | global4 | global8 |
|---:|---:|---:|---:|---:|
| 1 | 286 | 360 | 119 | **71** |
| 4 | 287 | 360 | 119 | **71** |
| 7 | 287 | 361 | 119 | **71** |
| 8 | 310 | 386 | **123** | 141 |
| 16 | 338 | 750 | **241** | 211 |
| 28 | 369 | 784 | 246 | 281 |
| 42 | 431 | 1175 | **369** | 421 |
| 56 | 522 | 1567 | **491** | 561 |

- **≤7ch(실시간)**: **global8** 최저(71ms) — 8코어가 1프레임 분담.
- **8~16ch**: **global4**(123~241)가 평탄. global8은 8ch에서 2배 튐(71→141, 카드당 1슬롯).
- **고채널(42~56)**: **global4/single**이 우세. multi는 이 워크로드에서 비효율(1567ms).
- N 시간은 예전 trunk-only(T)와 거의 같다 → **head를 NPU에 올린 추가비용 ≈ 0**.

## 2. e2e 단계별 (ms, P 전처리 + N NPU)

| ch | single P/N/e2e | global4 P/N/e2e | global8 P/N/e2e |
|---:|---:|---:|---:|
| 1 | 86/286/**372** | 24/119/**143** | 47/71/**118** |
| 7 | 583/288/**871** | 163/119/**282** | 272/71/**344** |
| 16 | 1171/340/**1511** | 397/241/**638** | 745/211/**955** |
| 28 | 761/372/**1133** | 535/246/**781** | 1029/281/**1310** |
| 56 | 882/525/**1406** | 728/491/**1219** | 896/561/**1456** |

> hybrid 시절(`NPU_coremode_pipeline_e2e.md`)에는 여기 `Pool(CPU attn_pool)` 단계가 추가로 붙어
> 고채널에서 e2e의 큰 비중을 차지했다. full NPU에는 그 단계가 **아예 없다**.

## 3. 비포/애프터 — 병목 변화

| | 비포 (hybrid) | 애프터 (full NPU) |
|---|---|---|
| 단계 | P → T(NPU) → **Pool(CPU)** → E | P → **N(NPU)** |
| attn_pool | CPU float, 채널별 직렬 (고채널 병목) | NPU (추가비용 ≈0) |
| 56ch 병목 | CPU(전처리 + attn_pool) | **CPU 전처리만** |
| 의존성 | torch + pe_vendor(CPU 연산) | qbruntime만 (CPU 연산 없음) |
| 정확도 | cos 0.997 | cos 0.99 |

→ **이제 유일하게 남은 CPU 병목은 전처리(P)뿐**이다(`NPU_preprocess_parallel.md`로 병렬화 가능).
attn_pool은 NPU로 흡수되어 코어모드(global4/global8) 최적화가 e2e에 그대로 반영된다.

## 4. 모드 선택 가이드 (full NPU)

| 상황 | 권장 모드 | 근거 |
|------|-----------|------|
| 단건/저채널 실시간(≤7ch) | **global8** | 단건 71ms 최저 |
| 8~16ch | **global4** | 8ch서 평탄, global8의 2배 튐 회피 |
| 고채널 throughput(≥28ch) | **global4** 또는 **single**(async) | 56ch 491~522ms |
| multi | 비권장 | 이 워크로드서 비효율 |

## 5. 재현
```bash
conda activate pe_npu_host
python ../../scratchpad_repro/profile_full_modes.py \
  single:<single.mxq> multi:<multi.mxq> global4:<g4.mxq> global8:<g8.mxq>
```
- 모드별 MXQ: `python -m pe_npu.compile --mode compile --save pe_full_<mode>.mxq --qk16 --scheme <mode> --calib-data-path <calib_hwc>`
- 관련: [`NPU_coremode_pipeline_e2e.md`](NPU_coremode_pipeline_e2e.md)(비포·hybrid), [`NPU_full_vs_hybrid.md`](NPU_full_vs_hybrid.md), [`../vendor/mobilint_resolution_attn_pool.md`](../vendor/mobilint_resolution_attn_pool.md)

*작성 2026-06. full NPU(QKᵀ 16bit) 4모드 동일 COCO calib, cos 0.99 (COCO holdout 0.9905 / 도메인 0.9889). 7×ARIES2 실측 (COCO 자산). 레이턴시는 calib 무관.*
