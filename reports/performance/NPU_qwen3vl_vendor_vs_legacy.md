# [실측] Qwen3-VL-2B — 벤더 최적화 빌드 vs 기존 HF 빌드

벤더 재현 패키지(`customer-capacity-repro`)를 도커로 돌려 기존 자산과 **같은 조건**에서 비교한 기록.
배경·용량표: [`../inquiries/06_qwen3vl_prefill_optimization/REPLY_2.md`](../inquiries/06_qwen3vl_prefill_optimization/REPLY_2.md)

| 항목 | 값 |
| --- | --- |
| 측정일 | 2026-09-23 |
| NPU | ARIES **카드 7 단일**, 드라이버 1.13 / 펌웨어 1.2.5 |
| CPU 격리 | **cpuset 8코어**(6,7,9,18,34,56,65,89) — 호스트 load 61.8, 운영 파드가 CPU 604% 사용 중 |
| 작업 | 720p 원본 이미지를 224×224 로 처리, 20장 동시, `max_new_tokens=1` |
| 프롬프트 | `Is this image a dog? Answer with only 'yes' or 'no'. Do not output anything else.` |
| 재현 구성 | `setup/vlm_repro/` (Dockerfile + 래퍼) |

## 1. 결과

| 빌드 | 실행 코드 | 스레드 | 20채널 완료 | 장당 | 배수 |
| --- | --- | ---: | ---: | ---: | ---: |
| **벤더 최적화** (batch32) | 벤더 `pia_vlm.Classifier` | 4 | **1107.8 ms** | 55.4 ms | **1.0×** |
| 기존 HF `7202ab5b` (batch1) | 우리 스택 (mblt-model-zoo 2.4.2) | 2 | 8126.4 ms | 406.3 ms | 7.3× 느림 |
| 기존 HF `7202ab5b` (batch1) | 〃 | 8 | 8447.4 ms | 422.4 ms | 7.6× 느림 |

- 벤더 빌드: rep2·rep3 가 1107.8 / 1105.5ms 로 **재현성 높음**(rep1 7494ms 는 콜드).
- 기존 빌드: 3회 모두 8100~8460ms 로 **일관** — 콜드가 아니라 실제 성능.
- **스레드 수는 무관했다**(2 ↔ 8 에서 차이 없음). 코어 수가 지배한다.

### 참고 — 벤더 자체 측정과의 거리

| | 20채널 224×224 |
| --- | ---: |
| 벤더 자체(유휴 i5-14600K, 4스레드) | **929.8 ms** |
| **우리 서버(벤더 패키지)** | **1107.8 ms** (1.19×) |

부하 있는 Xeon 서버에서 **1.19배**까지 좁혔다. i5-14600K 의 높은 단일코어 클럭과
호스트 경합을 감안하면 사실상 재현된 것으로 본다.

## 2. ★ 코어 수가 기존 빌드에 치명적이다

| 기존 HF 빌드 | 코어 | 20채널 |
| --- | --- | ---: |
| 2026-09-08 측정 | **96개 전부** | **3026 ms** |
| 2026-09-23 측정 | **8개 제한** | 8126 ms |

우리 스택은 전처리를 여러 코어로 흩어 돌기 때문에 코어를 묶으면 **2.7배** 느려진다.
반면 벤더 패키지는 **4스레드 전제 설계**라 코어 제한에 둔감하다.

→ **공정한 비교는 자원을 맞춘 8코어 기준이고, 거기서 7.3배다.**
코어를 넉넉히 준 9/8 값(3026ms)과 비교해도 **2.7배**다.

차이의 출처로 보이는 것 셋:

| 축 | 기존 | 벤더 |
| --- | --- | --- |
| 배치 | `max_batch_size=1` | **32** |
| 양자화 | text 1682.9MB | text 883.5MB (**크기 절반** — 더 낮은 비트폭으로 추정, 미확인) |
| 코어 배치 | vision·text 모두 `global8` | vision **`global4×2`** + text `global8` |

벤더 측정에서 `vision_stage_ms` 가 657~671ms 로 **전체 1107ms 의 약 60%** 였다.
비전 단계가 지배적이므로 그쪽 코어 배치가 특히 중요하다.

## 3. ★ 기존 mxq 는 벤더 코드로 못 돌린다 — 구조가 다르다

`--model` 만 바꿔 벤더 코드로 기존 mxq 를 돌리려 했으나 실패했다.

```
KeyError: 'model.visual.pos_embed.weight'
```

`model.safetensors` 의 키 구성이 다르다.

| 빌드 | 키 |
| --- | --- |
| 기존 HF | `model.language_model.embed_tokens.weight` **1개** |
| 벤더 | 위 + **`model.visual.pos_embed.weight`** 2개 |

**벤더 빌드는 비전 위치 임베딩을 CPU 로 분리**했고 기존 빌드는 mxq 안에 구워져 있다.
모델 경계가 달라 같은 실행 코드를 공유할 수 없다.
또한 `run_table.py` 는 batch32 경로를 전제하므로 batch1 인 기존 빌드로는 진입조차 안 된다.

→ **"동일한 파이썬 코드로 양쪽" 은 불가능하다.** 각자 자기 스택으로 재고
측정 조건(카드·cpuset·이미지·프롬프트)만 맞추는 것이 최선이다.

## 4. ★ HF 모델이 2026-09-23 갱신됐다

측정 중 우리 벤치가 최신 리비전을 자동으로 받아 실패하면서 발견했다.

| 리비전 | 날짜 | text mxq | 컴파일러 | 양자화 |
| --- | --- | ---: | --- | --- |
| `7202ab5b` | 2026-06-29 | 1682.9 MB | — | W8 |
| **`149444cc`** | **2026-09-23** | **887.6 MB** | **1.3.0.0** | **W4V8** |

- 파일명이 `Qwen3-VL-2B-Instruct_{text,vision}-W4V8.mxq` 로 바뀌었다.
- `Format 0x70000` / `Hardware Aries2` — 드라이버 1.13 호환.
- `max_batch_size` 는 여전히 **1**.
- **`mblt-model-zoo 2.4.2` 로는 로드 실패**:
  `ValueError: Qwen3-VL text MXQ must expose 2 ... or 3 ... inputs; got 5.`
  벤더 컨테이너의 `2.3.0+tta1` 도 구버전이라 해당 없음.

→ **새 HF 모델을 쓰려면 더 새로운 model-zoo 가 필요하다.** 벤더 확인 대상.

> **주의**: 우리 벤치가 리비전을 고정하지 않으면 자동으로 새 빌드를 받아 깨진다.
> `from_pretrained(..., revision=...)` 로 고정할 것.

## 5. 재현 방법

```bash
# 이미지 빌드(컨텍스트 = 재현 패키지)
docker build -f setup/vlm_repro/Dockerfile.runtime -t mblt_vlm_repro:rt \
    download/vendor/customer-capacity-repro

# 원본 MXQ 로 모델 조립
docker run --rm -v $PWD/download/vendor/customer-capacity-repro:/repro -w /repro mblt_vlm_repro:rt \
  python3.10 scripts/assemble_model.py --support inputs/support \
    --decoder reference/mxq/decoder-b32.mxq --vision reference/mxq/vision.mxq \
    --rotation inputs/rotation.pth --nonbatch-decoder reference/mxq/decoder-b1.mxq \
    --extended-vision reference/mxq/vision8192.mxq --output build/model-reference

# 용량표 재현 (CPUSET 필수 — 없으면 측정이 무의미하다)
CPUSET="6,7,9,18,34,56,65,89" NPU_DEVICES="7" \
  bash setup/vlm_repro/run_runtime.sh --model build/model-reference \
    --sizes 224x224 --repetitions 2 --output results/smoke2
```

**CPU 격리 없이는 측정이 무너진다**: cpuset 없이 돌리면 n=20 이 6940ms, n=21 이 1896ms 로
역전되는 등 값이 무의미해진다. 이 서버는 운영 파드가 전 코어를 쓴다.
