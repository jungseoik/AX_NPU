# [04] ViT 비전 인코더 양자화·속도 개선 — Mobilint 회신

| 항목 | 내용 |
| --- | --- |
| 문의 순번 | **04** (현재 가장 최신) |
| 수신일 | 2026-09-01 |
| 발신 | Mobilint 임성훈 |
| 주제 | ViT 계열 비전 인코더의 양자화 설정별 속도·정확도, uint8 입력 컴파일, 추론 패턴 권고 |
| 첨부 | `examples/compile_w8a16.py`, `examples/compile_w4a16.py`, `examples/compile_w4a8_l5a16.py` |
| 상태 | **회신 수신 · 우리 쪽 검증 미착수** |

---

## 1. 회신 요지

- 우리가 측정한 조건은 **W8A16**(weight 8bit / activation 16bit)으로 이해했고, 목표 처리량은 24 imgs/s 이상으로 파악.
- 속도 개선을 위해 **W4A16 또는 W4A8** 양자화 설정을 고려할 수 있다.
- **ViT는 outlier가 있어 A8(activation 8bit)에서 성능 하락이 크다** ([arXiv:2309.16588](https://arxiv.org/abs/2309.16588)).
  → W4A8을 쓰려면 **일부 레이어만 A16으로 유지**하는 방식이 필요하다.
- 첨부 예제는 **mean/std 정규화를 모델 연산에 포함**시켜 **uint8 입력**으로 컴파일한다.
  → 12채널보다 많은 채널을 받을 때 **PCIe 전송 병목 완화**에 유리. 불필요하면 제거 가능.
- 추론은 **single 모드 + async 조합**이 더 빨랐다. 8개 코어를 쉴 틈 없이 쓰도록 NPU에 작업을 할당하고
  바로 빠져나오며, 전/후처리도 병렬로 처리하는 형태.
  → `infer_async` 사용 예: https://docs.mobilint.com/runtime/v1.3/kr/advanced_usage.html

## 2. Mobilint 측정 결과 (원문 표)

| 양자화 조건 | 세부 조건 | 속도 (imgs/s) | 정확도 (top-1, subsample) | cos 유사도 (원본 대비) |
| --- | --- | ---: | ---: | ---: |
| W8A16 | - | 16.6 | 78.00% | 0.995 |
| W4A16 | - | 23.6 | 77.80% | 0.973 |
| W8A8 | - | 22.7 | 33.00% | 0.678 |
| W4A8 | - | 29.4 | 21.90% | 0.647 |
| W4A8 | layer 5개만 A16 | 28.4 | 75.90% | 0.905 |

읽는 법:

- **A8 단독은 쓸 수 없다** — W8A8/W4A8의 top-1이 33%/21.9%로 붕괴. ViT outlier 때문이며, 우리가 attn_pool에서 겪은 현상(문의 02)과 같은 계열의 문제다.
- **W4A16이 균형점** — W8A16 대비 속도 1.42배(16.6 → 23.6)에 top-1 손실 0.2%p.
- **W4A8 + 5개 레이어 A16** 은 속도 1.71배에 top-1 −2.1%p, cos 0.905. 속도가 최우선일 때의 선택지.

## 3. 첨부 예제

대상 모델은 **CLIP ViT-L/14-336** 비전 인코더로, 우리 PE-Core-L14-336과 같은 계열이다.

| 파일 | 설정 | 비고 |
| --- | --- | --- |
| `examples/compile_w8a16.py` | W8A16 | 기준선 |
| `examples/compile_w4a16.py` | W4A16 + OPTQ | 균형점 |
| `examples/compile_w4a8_l5a16.py` | W4A8 + OPTQ, 5개 텐서만 A16 | `A16_TENSORS` 목록으로 지정. `--profile` 로 통계 확인 |

```bash
python compile_w8a16.py      --calib_images <jpg 폴더 경로>
python compile_w4a16.py      --calib_images <jpg 폴더 경로>
python compile_w4a8_l5a16.py --calib_images <jpg 폴더 경로> --profile
```

세 파일 모두 `--model-id`(기본 `openai/clip-vit-large-patch14-336`), `--size`(336), `--target-device`(`aries-rb`) 인자를 갖는다.
**컴파일이므로 qbcompiler 환경(docker `mblt_compiler`)에서 실행**한다.

## 4. ★ 이 회신이 우리 기존 기록과 충돌하는 지점

두 가지가 현재 레포 서술과 어긋난다. **재검증 전까지는 기존 서술을 그대로 두되, 이 항목을 근거로 확인이 필요하다.**

| 우리 기존 서술 | 이번 회신 | 확인해야 할 것 |
| --- | --- | --- |
| "NPU는 INT8 전용. 양자화를 더 못 낮춘다 — bit4 mixed-precision은 no-op" (`.claude/CLAUDE.md`, `../../performance/NPU_batch_latency.md` §bit4) | W4A16 / W4A8 예제와 실측치를 제시 | 우리가 시도한 `BitConfig.Transformer.mixed_precision`(weight 비율 혼합) 경로와, 예제가 쓰는 W4 설정 경로가 **같은 기능인지**. 다르다면 no-op 판정은 "그 API 한정"으로 범위를 좁혀야 한다 |
| "한 모델에 `infer_async` 여러 건 동시 제출 = 출력 깨짐 → 동기 `infer()` + 멀티스레드 사용" (`.claude/CLAUDE.md` ★ 다채널 동시성) | "single 모드 + async 조합이 더 빨랐다" | 벤더 권고 패턴이 **모델 인스턴스를 나눠 쓰는 형태인지**, 아니면 한 인스턴스에 다건 async인지. 후자라면 우리 실측(출력 깨짐)과 정면 충돌이므로 재현 확인 필요 |

## 5. 후속 액션 (미착수)

- [ ] 예제 3종을 우리 PE-Core-L14-336에 적용해 컴파일 → 속도·cos·정확도 실측 (기준: 현재 full NPU cos 0.99)
- [ ] W4A16이 우리 파이프라인에서도 속도 이득이 나오는지 확인 (벤더 기준 1.42배)
- [ ] uint8 입력 컴파일이 고채널 전처리 병목에 실제로 도움이 되는지 확인
      → 관련 선행 실험: [`../../performance/NPU_preprocess_2_uint8_offload.md`](../../performance/NPU_preprocess_2_uint8_offload.md) (normalize는 폴딩됐으나 resize 불가로 이득 없었음)
- [ ] single + async 패턴 재현 및 출력 무결성 검증 → 결과에 따라 `.claude/CLAUDE.md` 동시성 규칙 갱신
- [ ] bit4 no-op 판정 범위 재확인 → `../../performance/NPU_batch_latency.md` 갱신 여부 결정

---

## 부록. 회신 원문

> 안녕하세요. 모빌린트 임성훈입니다.
>
> 해당 모델에서 순수 NPU 계산 시간만 고려했을 때, 목표 처리량이 24 imgs/s 이상 되어야 하는 것으로 이해하였습니다.
>
> 그리고 Weight 8 Bit / Activation 16 Bit (이하 W8A16) 설정으로 양자화하여 속도를 측정하신 것으로 이해하였습니다.
>
> 속도 개선을 위해서는 W4A16 또는 W4A8 양자화 설정도 고려해볼 수 있습니다.
>
> vit 모델은 outlier가 존재하기 때문에 A8은 성능하락이 큽니다.
> (https://arxiv.org/abs/2309.16588)
>
> 관련해서 양자화 예제 전달드리며 저희쪽에서 측정한 결과도 공유드립니다.
>
> *(표 = 위 §2)*
>
> 전달해드리는 예제 파일은 아래처럼 사용해주시면 됩니다.
>
> mean/std 정규화가 모델 연산에 포함되어 uint8 입력으로 받도록 컴파일되는 예제입니다.
> 그러므로 추론시 uint8 형식의 데이터를 입력해야 합니다.
> uint8 입력을 사용시 추후 12채널보다 더 많은 채널을 받을 때, pcie 전송 병목을 완화할 수 있습니다.
> 불필요시 제거해주시면 됩니다.
>
> ```bash
> python compile_w8a16.py --calib_images jpg폴더 경로
> python compile_w4a16.py --calib_images jpg폴더 경로
> python compile_w4a8_l5a16.py --calib_images jpg폴더 경로 --profile
> ```
>
> 추론 쪽은 저희가 구현했을 땐, single mode + async 조합이 더 빨랐습니다.
> 싱글 모드로 8개의 코어를 쉴틈없이 사용할 수 있도록,
> npu에 작업을 할당하고 바로 빠져나올 수 있도록 하고 전/후처리 또한 병렬로 처리하는 형태입니다.
> qbruntime의 infer_async 사용 예시도 공유드립니다.
> https://docs.mobilint.com/runtime/v1.3/kr/advanced_usage.html
>
> 추가 문의는 언제든지 바로 연락주시면 도움을 드리겠습니다.
>
> 감사합니다.
