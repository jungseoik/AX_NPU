# 전처리 NPU 오프로드(uint8 입력) 실험 — 결과·장단점

> **한 줄 결론**: PE를 **uint8 입력으로 재컴파일**하면 normalize(÷255·mean/std)를 NPU(첫 conv)에
> 폴딩할 수 있고 **정확도도 ~0.99 유지**된다. 그러나 **전처리 시간 이득은 미미**하다 —
> normalize는 원래 싼 연산이고, 실제 비용인 **resize는 NPU로 못 넘기기** 때문.
> → 전처리 가속 목적의 uint8 재컴파일은 **실익 없음**. (실측 2026-06)

전처리 CPU 비용이 큰 것 같아 "전처리를 NPU로 넘길 수 있나"를 검증한 기록. 컴파일·calib 워크플로는
`../quantization/QUANT_TUNING_guide.md`, 전처리 병렬화는 `NPU_preprocess_parallel.md` 참고.

---

## 1. 무엇을 해봤나

목표: PE 전처리 3단계(**BGR→RGB / resize / normalize**)를 NPU로 오프로드해 CPU 부담 절감.

공식 문서(`docs/release_note.md` uint8 추론 지원, `docs/programming_guide.md` UINT8 입력)와
qbcompiler 1.1.2의 `Uint8InputConfig` / `PreprocessingConfig`를 이용해 다음을 컴파일·검증:

- `Uint8InputConfig(apply=True, division_factor=255)` → uint8 입력의 `/255`를 첫 conv 가중치에 폴딩
- `PreprocessingConfig(apply=True, auto_convert_format=True, pipeline=[{"op":"Normalize","mean":[0.5]×3,"std":[0.5]×3}])`
  → `(x-0.5)/0.5` normalize를 모델에 폴딩
- 기존 `--qk16`(attn_pool QKᵀ 16bit) + `single` scheme 유지
- calib: COCO val2017 200장을 **raw uint8 HWC(336,336,3)** 로 새로 생성(정규화 안 함)

컴파일 산출물: `pe_full_uint8.mxq`(327MB). 컴파일 로그에서 폴딩 확인:
`fuseUint8IntoConv | Divided weights by 255 for layer 'visual_conv1'` + `Fused uint8 bias compensation`.

## 2. 단계별 NPU 오프로드 가능 여부

| 전처리 | NPU 오프로드 | 방법 / 근거 |
|---|:--:|---|
| normalize (÷255, mean/std) | ✅ 가능 | uint8 입력 컴파일로 첫 conv에 폴딩(실측 확인) |
| dtype 변환 (uint8→float) | ✅ 가능 | uint8 입력이면 불필요 |
| BGR→RGB | △ 부분 | NPU 전용 연산 아님. 컴파일 채널순서 or 업스트림 RGB로 회피 |
| **resize** | ❌ **불가** | NPU resize 없음(공식 tutorial도 CPU `stb_image_resize`). **유일 회피=업스트림 pre-resize** |

## 3. 정확도 (uint8 MXQ vs 기존 float MXQ, single, 실제 이미지)

| 이미지 | cos | | 이미지 | cos |
|---|--:|---|---|--:|
| bus | 0.99414 | | falldown | 0.99392 |
| cat1 | 0.99505 | | fire | **0.98800** (최저) |
| cat2 | 0.99325 | | pizza | 0.99301 |
| dog | 0.99209 | | smoke | 0.99065 |

→ uint8 입력 양자화로 미세 손실만 추가, **~0.99대 유지**(float MXQ가 pth 대비 0.99이므로 uint8≈pth 0.98~0.99).

## 4. 전처리 시간 (float=resize+normalize vs uint8=resize만)

| 해상도/N | float (resize+norm) | uint8 (resize만) | 차이 |
|---|--:|--:|--:|
| 720p N=28 | 57ms | 51ms | -6ms |
| 720p N=56 | 102ms | 104ms | ~0 (노이즈) |
| 1080p N=28 | 53ms | 91ms | 오히려 + |
| 1080p N=56 | 84ms | 174ms | 오히려 + |

→ **normalize 제거 이득 ≈ 0~6ms.** uint8 경로는 HWC permute+numpy 변환 오버헤드로 고해상도에선 더 느려지기도.

## 5. 왜 이득이 없나

- **normalize는 작은 336×336 출력에 가하는 벡터 연산이라 원래 싸다(~6ms).** NPU로 폴딩해도 절감 미미.
- **진짜 CPU 비용은 resize**(원본 해상도 픽셀을 읽어 336으로 축소) — 이건 **NPU 불가**. uint8여도 그대로 CPU.
- 따라서 "전처리 비용이 크다"고 본 것의 실체는 resize(+고해상도 BGR→RGB)였고, uint8이 제거하는 normalize는 비용의 핵심이 아니었다.
- 참고: 720p e2e는 이미 **NPU-bound**(NPU 추론 303ms@28ch ≫ 전처리 109ms). 전처리는 720p에선 병목도 아님.

## 6. 장단점 정리

| | 장점 | 단점 |
|---|---|---|
| uint8 재컴파일 | normalize/float변환 NPU 흡수, 호스트→NPU 전송 4배↓, 정확도 0.99 유지 | **전처리 시간 이득 거의 없음**(resize가 진짜 비용), 재컴파일 53분, 별도 uint8 calib 필요, fire에서 0.988로 미세 저하 |

## 7. 결론

**전처리 가속 목적의 uint8 재컴파일은 권장하지 않는다.** 정확도는 유지되나 실익(전처리 단축)이 없다.
CPU 전처리를 줄이려면 **resize를 줄이는 방향**(업스트림 pre-resize / cv2 멀티스레드 resize / 저해상도 입력 /
전처리-추론 파이프라이닝)이어야 한다. → 남은 최적화는 `NPU_preprocess_parallel.md` 및 본 문서 §아래 참고.

## 8. pe_npu 레벨에서 남은 전처리 최적화 (resize 중심)

| 방법 | 기대효과 | 위치/난이도 | 비고 |
|---|---|---|---|
| **resize-skip 가드** (입력이 이미 336이면 resize 생략) | pre-resized 입력서 resize 비용 제거 | preprocess.py 한 줄 | 정확도 무영향 |
| **cv2.resize + `cv2.setNumThreads`** (torchvision 대체) | resize 자체 GIL-free 멀티스레드 가속 | preprocess.py | torchvision보다 빠를 수 있음(검증 필요) |
| **process 모드 자동화** (코어 많을 때) | 고채널 resize ~6x | parallel_preprocess.py(이미 opt-in) | spawn/IPC 비용 |
| **전처리-추론 파이프라이닝** | NPU 도는 동안 다음 배치 전처리 → e2e 단축 | service `_detect` 구조 변경 | 중간 난이도 |
| **워커수 코어 기준 자동 튜닝** | CPU-only 코어 적은 환경 오버서브스크립션 방지 | config/parallel_preprocess | 전처리풀+추론풀 합산 고려 |
| (업스트림) **pre-resize 336 RGB 공급** | 전처리 거의 0(NPU-bound) | 서비스 밖(디코더) | 가장 큰 이득, pe_npu 외부 |

*실측 2026-06. 컴파일=pe_compile(qbcompiler 1.1.2), 추론=pe_npu_host(qbruntime), 7×ARIES2. 스크립트: `scratchpad_repro/compile_uint8.py`, calib `scratchpad_repro/calib_uint8_hwc`.*
