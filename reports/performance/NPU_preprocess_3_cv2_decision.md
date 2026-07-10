# [전처리 ③ 채택: cv2] 전처리 최적화 의사결정 — e2e 기준, 리소스 원천, cv2 전환 (다채널)

> **결론**: CPU 전처리의 실제 비용은 **resize**(normalize·dtype 아님). resize는 NPU로 못 넘기므로
> (→ `NPU_preprocess_2_uint8_offload.md`), CPU에서 **torchvision → cv2(INTER_LINEAR)** 로 바꿔 줄였다.
> 다채널 e2e(전처리+NPU)에서 **56ch 736→552ms(-25%) + CPU ~26코어→~5코어**. 정확도는 0.99→0.97로
> 소폭 저하(antialias 차이)라 **cv2는 opt-in(`PE_NPU_RESIZE=cv2`), 기본은 torchvision**.
> 전처리-추론 파이프라이닝은 고채널서 -8ms뿐이라 **미채택**.
> (적용 위치는 Product-AI-mono `pe_npu` 서비스. 본 문서는 그 근거를 AX_NPU 지식베이스에 남긴 것. 실측 2026-06.)

PE 추론 서비스 전처리: `cv_bgr2rgb` → `ROI 크롭` → **resize 336 + normalize** → NPU. 병렬화 일반은
`NPU_preprocess_1_parallel.md`, uint8 오프로드 실험은 `NPU_preprocess_2_uint8_offload.md` 참고.

---

## 1. 왜 e2e로 의사결정했나 + 실제 리소스 원천

전처리만 보면 "느리다/빠르다"가 오해를 부른다. **추론(NPU)까지 합친 e2e**로 봐야 어디에 투자할지가 정해진다.

### 단계별 분해 (720p, N=28)
| 단계 | 시간 | 자원 |
|---|--:|---|
| ① BGR→RGB (cv2) | 27ms | CPU |
| ② resize + normalize | ~82ms | CPU |
| ③ **NPU 추론 (global4)** | **303ms** | NPU |

→ **720p e2e는 NPU-bound**(NPU 303ms ≫ 전처리 109ms). 저채널에선 전처리가 병목도 아니다.

### 리소스 원천 = resize (normalize/BGR 아님)
- **normalize(÷255, mean/std)는 작은 336×336 출력에 가하는 벡터연산이라 ~6ms로 저렴.** (uint8로 NPU에 폴딩해도 이득 ~0 — `NPU_preprocess_2_uint8_offload.md`에서 실증.)
- **진짜 비용은 resize**(원본 해상도 픽셀을 읽어 336으로 축소). 해상도에 비례:
  | N=56 전처리(현행 torchvision) | 720p | 1080p | 4K |
  |---|--:|--:|--:|
  | resize+normalize | ~100ms | ~110ms | ~465ms |
  | BGR→RGB | ~160ms | ~378ms | — |
- resize는 **NPU로 못 넘긴다**(공식도 CPU `stb_image_resize`). → CPU에서 더 빠른 resize로 바꾸는 수밖에.

---

## 2. 워커 vs 코어 (측정 해석용)
- **워커 = 스레드**(`ThreadPoolExecutor(W)`) = 우리가 만드는 동시 작업 수(소프트웨어).
- **코어** = 그 스레드가 실제로 도는 CPU(하드웨어). 본 문서의 "코어 수" = `CPU초/실시간` = 평균 실제 점유 코어.
- 병렬화 레벨 2개(곱해짐): **우리 풀(장 단위 병렬)** × **cv2 내부스레드(`setNumThreads`, 한 resize를 쪼갬)**.
  둘 다 켜면 오버서브스크립션 → cv2 내부는 1로 고정하고 **장 단위(우리 풀)로만 병렬**이 효율적(실측 §4).

---

## 3. resize 백엔드 비교 (720p, 56ch) — 속도·코어·정확도

| 방식 | 속도 | 코어 | 정확도(cos, 원본 대비) |
|---|--:|--:|--:|
| **torchvision** (현행, antialias bilinear) | 146ms | **26** | **0.99** |
| cv2 INTER_LINEAR (+스레드풀, setNumThreads=1) | **86~117ms** | **~5** | 0.973 |
| cv2 INTER_AREA (antialias 유사) | 540ms | 1.5 | 0.986 |

- **torchvision은 빠르지만 ~26코어**를 먹는다(CPU-only 부담). cv2 LINEAR은 **더 빠르면서 ~5코어**.
- 단 cv2 LINEAR은 **antialias가 없어** 큰 축소(720p/4K→336)서 정확도 0.99→**0.973**으로 소폭 저하. (근본은 보간 방식 차이.)
- antialias 맞춘 cv2 AREA는 정확도(0.986)는 낫지만 **느려서(540ms) 의미 없음**.

### 워커수 스윕 (cv2 LINEAR, setNumThreads=1, 56ch)
| 워커 | 속도 | 코어 |
|--:|--:|--:|
| 4 | ~132ms | 2.1 |
| 8 | ~131ms | 3.1 |
| **16** | **~86ms** | **4.8** |
| 32 | ~83ms | 5.5 |

→ **16이 스위트스폿**(>16은 +3ms뿐, 코어만 더). 그래서 기본 워커 = 16.

### cv2 내부스레드(setNumThreads) 스윕 — 거의 무효 (참고)
직렬 루프에서 setNumThreads 1→64로 올려도 320→243ms, 코어 ~1.7. **작은 resize는 cv2 내부 병렬이 안 먹힘** →
"장 단위 우리 풀"이 정답.

---

## 4. e2e 비교 (전처리+NPU, raw→임베딩, 720p, global4 7카드)

| N | A. 현행 torchvision(순차) | B. cv2(순차) | C. cv2+파이프라이닝 |
|--:|--:|--:|--:|
| 28 | 377ms | 324ms | 282ms |
| 56 | **736ms** | **552ms** | 544ms |

| 구간 절감 | 28ch | 56ch |
|---|--:|--:|
| cv2 전환 (A→B) | -53ms | **-184ms (-25%)** |
| 파이프라이닝 추가 (B→C) | -42ms | **-8ms** |

- **cv2 전환이 큰 이득**(56ch -184ms): 현행 torchvision이 CPU 26코어를 먹어 고채널서 느렸던 게 해소.
- **파이프라이닝은 한계 효용**: 정작 중요한 56ch에서 -8ms(NPU 지배 + 청킹 오버헤드가 상쇄). 복잡도 대비 무의미 → **미채택**.

---

## 5. 다채널 시 전처리 이점 (cv2 채택 효과)

| | 현행 torchvision | cv2(채택) | 효과 |
|---|---|---|---|
| 56ch 전처리 | 146ms / 26코어 | ~86~117ms / ~5코어 | 속도↑ + **CPU ~5배 절감** |
| 56ch e2e | 736ms | 552ms | **-25%** |
| CPU-only 영향 | 추론(~14스레드)과 26코어 경합 | ~5코어라 여유 | 다스트림·동시처리 유리 |
| 정확도 | 0.99 | 0.973 | 소폭 저하(수용 시) |

- **다채널일수록 이점이 커진다**: 채널이 많을수록 torchvision의 코어 경합이 심해지는데, cv2는 코어를 적게 쓰며 더 빠름.
- 저채널(≤7)은 전처리 자체가 작아 차이 미미.

---

## 6. 최종 결정

| 항목 | 결정 | 이유 |
|---|---|---|
| resize 백엔드 | **cv2 INTER_LINEAR opt-in** (`PE_NPU_RESIZE=cv2`), 기본 torchvision | 속도+CPU 이득 크나 정확도 0.97이라 기본은 보수적(0.99) 유지 |
| 기본 워커 | **16** | 스위트스폿(>16 효용 미미) |
| resize-skip 가드 | 적용 | 입력이 이미 336이면 resize 생략(정확도 무영향) |
| uint8 NPU 오프로드 | 미채택 | normalize는 싸고 resize는 NPU 불가 → 이득 없음(`NPU_preprocess_2_uint8_offload.md`) |
| 파이프라이닝 | 미채택 | 고채널서 -8ms, 복잡도 대비 무의미 |

### env (Product-AI-mono `pe_npu`)
```bash
PE_NPU_RESIZE=cv2            # 속도/리소스 우선 (cos~0.97, e2e -25%) / 기본 torchvision(0.99)
PE_NPU_PREPROCESS_WORKERS=0  # 0=auto(min(16,코어)). 큰 서버면 32로 상향 가능
```

*실측 2026-06. 추론=pe_npu_host(qbruntime), 7×ARIES2(Xeon Gold 6526Y 64T). 절대 수치는 서버 부하에 따라 변동, 상대 패턴(cv2<torchvision, resize가 비용, e2e NPU-bound)은 견고.*
