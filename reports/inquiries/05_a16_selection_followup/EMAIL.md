# [05] ViT 양자화 후속 — 재현 완료 보고 및 `select_a16.py` 요청

| 항목 | 내용 |
| --- | --- |
| 수신 | Mobilint 임성훈 님 (참조: 서현석 매니저님) |
| 관련 | 문의 04(ViT 비전 인코더 양자화·속도) 회신에 대한 후속 |
| 목적 | ① 재현 완료 보고 ② 1.1.2 에서의 튜닝 열화 현상 공유 ③ `select_a16.py` 요청 ④ A16 선정 기준 확인 |
| 상태 | 발송 대기 (초안 v2) |

> **내부 메모 ①**: 본 메일의 "저희 ViT-L/14 비전인코더 / PIA-VE"는 **Perception Encoder(PE-Core-L14-336)**
> 를 가리킨다. 대외 문의에서 모델 실명을 쓰지 않기 위한 표기이며 같은 모델이다.
> 앞선 지연 문의 때 **"표준 CLIP/ViT가 아니라 2D RoPE + attention pooling head"** 라는 구조 설명은
> 이미 전달했다(동봉 README 31행). (표기 규약: [`../README.md`](../README.md) 참조)
>
> **내부 메모 ②**: v1 초안([`EMAIL_v1_보류.md`](EMAIL_v1_보류.md))은 "재현이 안 된다 / 다른 모델로
> 측정하신 것 아니냐"는 질의였으나, **원인이 우리 쪽 컴파일러 버전(1.1.2)으로 밝혀져 폐기**했다.
> qbcompiler 1.2.0 으로 재컴파일하니 벤더 수치가 재현된다.
> → [`../../performance/NPU_pe_quant_tuning_compiler_version.md`](../../performance/NPU_pe_quant_tuning_compiler_version.md)

---

안녕하세요, PIA-SPACE 정서익입니다.

지난 ViT 비전 인코더 양자화 회신과 예제 감사히 받았습니다.
주신 설정으로 재현을 마쳤고, **결과가 잘 재현되어** 먼저 공유드립니다.

## 1. 재현 결과 — 주신 수치가 재현되었습니다

저희 인코더(`pia_ve_latency_headroom_full.zip` → `model/vit_l14_336.pt`)에 주신 설정을 적용해
NPU(Aries2)에서 측정한 결과입니다.

- calibration: COCO val2017 200장
- 정확도: 원본 PyTorch 임베딩 대비 코사인 유사도(20장 평균)
- A16 텐서: 예제와 동일한 기준(`ratio = max / p99.9` 상위 5개)을 저희 모델에 적용해 도출

| 조합 | 주신 값 | 저희 재현값 | 차이 |
| --- | ---: | ---: | ---: |
| W4A16 + OPTQ + SearchWeightScale | 0.973 | **0.9642** | −0.009 |
| W4A8 + A16 5개 + OPTQ + SearchWeightScale | 0.905 | **0.9420** | +0.037 |

W4A16 은 주신 값과 0.01 안쪽으로 일치하고, W4A8 은 저희 쪽이 조금 더 높게 나왔습니다.
차이는 calibration 데이터 구성이나 A16 텐서 선정 차이로 이해하고 있습니다.
크기·속도 이득도 함께 확인했습니다(W4A16 기준 크기 327MB → 188MB, 처리량 +13%).

**주신 방향이 저희 모델에서도 유효함을 확인**했고, W4A16 + 튜닝 조합을 배포 후보로
검토하려 합니다. 좋은 가이드 주셔서 감사합니다.

## 2. 공유드릴 현상 — qbcompiler 1.1.2 에서는 튜닝이 정확도를 **떨어뜨립니다**

재현이 처음에 되지 않았고, 원인을 추적한 결과 **컴파일러 버전 차이**였습니다.
혹시 이미 파악하고 계신 사항인지, 또는 저희 설정에 문제가 있는 것인지 확인차 공유드립니다.

동일한 모델·동일한 calibration·동일한 옵션으로, **컴파일러 버전만 바꿔** 측정한 값입니다.

| 양자화 (single) | 튜닝 없음 | + OPTQ + SWS (**1.1.2**) | + OPTQ + SWS (**1.2.0**) |
| --- | ---: | ---: | ---: |
| W8A16 | 0.9937 | 0.9747 ▼ | **0.9951** ▲ |
| W4A16 | 0.9110 | 0.8795 ▼ | **0.9642** ▲ |
| W4A8 + A16 5개 | 0.8790 | 0.8560 ▼ | **0.9420** ▲ |

**1.1.2 에서는 세 양자화 모두에서 튜닝이 정확도를 떨어뜨렸고, 1.2.0 에서는 모두 올라갑니다.**
튜닝을 끈 베이스라인은 두 버전이 동일했으므로(W4A16 기준 0.9110 → 0.9126, 노이즈 수준),
차이는 컴파일러 전반이 아니라 **튜닝(OPTQ/SearchWeightScale) 구현**에서 나오는 것으로 보입니다.

- 1.1.2 에서는 양자화 3종 × 튜닝 3종 = **9개 조합 전부**가 튜닝 없음보다 낮았습니다.
- 4bit 와 무관한 W8A16 에서도 동일하게 나타납니다(0.9937 → 0.9747).
- CPU 컴파일과 GPU 컴파일 양쪽에서 같은 패턴이었습니다.

여쭙고 싶은 것은 두 가지입니다.

1. 이 현상이 **1.1.2 의 알려진 문제이고 1.2.0 에서 수정된 것**이 맞을까요?
2. 그렇다면 튜닝 옵션 사용 시 **권장 최소 컴파일러 버전**을 1.2.0 으로 안내해 주시면
   저희 같은 사용자가 같은 혼선을 겪지 않을 것 같습니다.

참고로 `SearchWeightScaleConfig` / `OptqConfig` 의 **설정 필드 구성은 두 버전이 동일**했습니다
(`apply` + `transformer(query/key/value/out/ffn)`, `attributes(actOrder/blockSize/percDamp)`).
저희가 넘긴 값도 양쪽 동일합니다. 그래서 설정 인터페이스가 아니라 **내부 구현** 차이로 보고 있습니다.

## 3. `select_a16.py` 파일을 받을 수 있을까요?

받은 첨부(`vit_compile_examples.zip`)에는 아래 3개 파일이 들어 있었습니다.

```
compile_w8a16.py
compile_w4a16.py
compile_w4a8_l5a16.py
```

`compile_w4a8_l5a16.py` 안에 아래와 같이 두 번 언급되어 있는데, 해당 파일은 첨부에 없었습니다.

> `# (torch module, post-fusion mblt name). Run --profile to see the statistics`
> `# these were chosen from; select_a16.py derives them for another checkpoint.`
>
> `A16_TENSORS are post-fusion mblt names, specific to this checkpoint; use select_a16.py to derive`
> `them for another model.`

이번 재현에서는 저희가 예제의 기준(`ratio = max / p99.9`)을 그대로 구현해 A16 텐서를 도출했습니다만,
벤더 표준 방식과 어긋난 부분이 있는지 확인하고 싶습니다. 전달해 주실 수 있을까요?

## 4. A16 텐서 선정 기준 — 저희 모델은 프로파일 결과가 다릅니다

예제의 `A16_TENSORS` 는 아래와 같습니다.

```python
("vision_model.encoder.layers.12.mlp.fc2",           "add_76/reshape/quickgelu/conv2d"),
("vision_model.encoder.layers.11.mlp.fc2",           "add_70/reshape/quickgelu/conv2d"),
("vision_model.encoder.layers.12.mlp.activation_fn", "add_76/reshape/quickgelu"),
("vision_model.encoder.layers.11.mlp.activation_fn", "add_70/reshape/quickgelu"),
("vision_model.encoder.layers.9.mlp.fc2",            "add_58/reshape/quickgelu/conv2d"),
```

이 목록은 공개 `openai/clip-vit-large-patch14-336`(예제 기본 `--model-id`) 기준으로 보이며,
저희 모델과는 구조가 달라 그대로 적용되지 않았습니다.

| 항목 | 예제가 가리키는 구성 | 저희 `vit_l14_336.pt` |
| --- | --- | --- |
| 활성함수 | QuickGELU (`.../quickgelu/conv2d`) | **일반 GELU** — 컴파일 결과 op 533개 중 `quickgelu` 0개 |
| 모듈 명명 | `vision_model.encoder.layers.N.mlp.fc2` (HF CLIP) | `visual.transformer.resblocks.N.mlp.c_proj` (open_clip 계열) |
| 어텐션 풀링 헤드 | 없음 | **`visual.attn_pool.*` 존재** — probe(1,1,1024) + attn + layernorm + mlp |

그래서 예제와 **동일한 기준**으로 저희 모델을 직접 프로파일링해 상위 5개를 선정했고,
그 결과가 §1 의 재현값입니다. 다만 순위 분포가 예제와 꽤 다릅니다.

| 순위 | 저희 모델 텐서 | ratio |
| --- | --- | ---: |
| 1 | L3.mlp.c_proj | **74.9×** |
| 2 | L12.mlp.c_proj | 16.2× |
| 3 | L10.mlp.c_proj | 14.4× |
| 4 | L12.mlp.gelu | 13.0× |
| 5 | L10.mlp.gelu | 12.3× |

예제는 L9/L11/L12 중심인데 저희는 **L3 가 74.9× 로 압도적 1위**입니다.
이런 경우에도 단순히 상위 5개를 고르는 것이 맞을까요? 아니면 특정 레이어 구간(중후반)을
우선하는 등의 추가 기준이 있을까요?

## 5. 함께 여쭙고 싶은 것

1. §2 의 1.1.2 튜닝 열화가 알려진 문제인지, 1.2.0 이 권장 최소 버전인지
2. `select_a16.py` 전달 가능 여부 (§3)
3. A16 텐서 선정 시 **개수**(5개보다 늘리는 것)와 **선택**(어느 텐서냐) 중 무엇이 더 결정적인지 (§4)
4. 주신 표의 **전체 컴파일 설정**(OPTQ 파라미터, SearchWeightScale 적용 범위, uint8 입력·normalize
   폴딩 포함 여부, calibration 데이터셋 종류·장수) — 잔차 0.01 의 출처를 좁히는 데 도움이 될 것 같습니다

저희 모델 정의 코드(`ve_lib`)는 용량 문제로 동봉하지 못했는데, 검토에 필요하시면 바로 전달드리겠습니다.

바쁘신 중에 늘 상세히 답변 주셔서 감사합니다.

감사합니다.
PIA-SPACE 정서익 드림
