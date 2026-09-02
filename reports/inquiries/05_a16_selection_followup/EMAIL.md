# [05] ViT 양자화 후속 문의 — `select_a16.py` 요청 및 A16 선정 기준 확인

| 항목 | 내용 |
| --- | --- |
| 수신 | Mobilint 임성훈 님 (참조: 서현석 매니저님) |
| 관련 | 문의 04(ViT 비전 인코더 양자화·속도) 회신에 대한 후속 |
| 목적 | ① 회신에 언급된 `select_a16.py` 미첨부 → 요청 ② A16 5개가 어느 모델 기준인지 확인 |
| 상태 | 발송 대기 (초안) |

> **내부 메모**: 본 메일의 "저희 ViT-L/14 비전인코더 / PIA-VE"는 **Perception Encoder(PE-Core-L14-336)**
> 를 가리킨다. 대외 문의에서 모델 실명을 쓰지 않기 위한 표기이며 같은 모델이다.
> 다만 앞선 지연 문의 때 **"표준 CLIP/ViT가 아니라 2D RoPE + attention pooling head"** 라는 구조 설명은
> 이미 전달했다(동봉 README 31행). 즉 우리 쪽에서 CLIP이라고 안내한 적은 없다.
> (표기 규약: [`../README.md`](../README.md) 참조)

---

안녕하세요, PIA-SPACE 정서익입니다.

지난 ViT 비전 인코더 양자화 회신과 예제 감사히 받았습니다. 저희 모델에 적용해 재현해 본 결과와,
그 과정에서 확인이 필요한 점들을 여쭙습니다.

## 1. 재현 결과 공유

앞서 `pia_ve_latency_headroom_full.zip` 으로 보내드린 저희 인코더(`model/vit_l14_336.pt`, MXQ 4모드와
동일 자산)에 주신 설정을 적용했습니다. NPU 1장, calibration은 COCO val2017 200장, 정확도는 원본
PyTorch 임베딩 대비 코사인 유사도(20장 평균)입니다.

| 설정 | 크기 | cos | 처리량(20채널 배치) |
| --- | ---: | ---: | ---: |
| 기존 배포본 (컴파일러 기본값 + QKᵀ 16bit = W8A16) | 327.2 MB | **0.9936** | 15.94 img/s |
| **W4A16** (보정 옵션 없음) | 188.8 MB (−42%) | **0.9135** | 18.07 img/s (**+13%**) |
| W4A16 + SearchWeightScale 단독 | 188.8 MB | 0.8786 | — |
| **W4A16 + OPTQ + SearchWeightScale** (예제와 동일 구성) | 188.8 MB | **0.8611** | 22.38 img/s |
| W4A8 + A16 5개(저희 프로파일링 선정, 보정 없음) | 188.4 MB | 0.2609 | 20.21 img/s |
| **W4A8 + A16 5개(주신 목록) + OPTQ + SearchWeightScale** (예제와 동일 구성) | 188.4 MB | **0.3190** | 26.54 img/s |

### 확인된 것과 재현되지 않은 것

**W4A16이 동작한다는 점은 확인**했습니다. 크기 −42%, 처리량 +13~43%(코어모드별)로 주신 방향과
일치합니다.

다만 **정확도는 주신 수치가 재현되지 않았습니다.**

- W4A16: 주신 값 **cos 0.973** ↔ 저희 측 **0.8611**(예제와 동일 구성)
- W4A8 + A16 5개: 주신 값 **cos 0.905** ↔ 저희 측 **0.3190**(주신 A16 목록 + OPTQ + SWS)

더 특이한 점은 **보정 옵션을 켤수록 정확도가 내려간다**는 것입니다.

| W4A16 | cos |
| --- | ---: |
| 보정 없음 | **0.9135** (가장 좋음) |
| + SearchWeightScale | 0.8786 |
| + OPTQ + SearchWeightScale | **0.8611** (가장 나쁨) |

저희 환경에서는 OPTQ·SearchWeightScale이 오히려 역효과로 나옵니다. 예제 의도와 반대 방향이라
설정이나 전제에서 저희가 놓친 부분이 있는지 확인하고 싶습니다.

## 2. `select_a16.py` 파일을 받을 수 있을까요?

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

저희 체크포인트에 맞는 A16 텐서를 정확히 도출하려면 이 파일이 필요할 것 같습니다.
혹시 전달해 주실 수 있을까요?

## 3. 예제의 A16 5개는 어느 모델에서 도출된 것일까요?

예제 `A16_TENSORS` 는 아래와 같습니다.

```python
("vision_model.encoder.layers.12.mlp.fc2",           "add_76/reshape/quickgelu/conv2d"),
("vision_model.encoder.layers.11.mlp.fc2",           "add_70/reshape/quickgelu/conv2d"),
("vision_model.encoder.layers.12.mlp.activation_fn", "add_76/reshape/quickgelu"),
("vision_model.encoder.layers.11.mlp.activation_fn", "add_70/reshape/quickgelu"),
("vision_model.encoder.layers.9.mlp.fc2",            "add_58/reshape/quickgelu/conv2d"),
```

저희가 보내드린 체크포인트를 열어 대조해 보니 아래와 같은 차이가 있었습니다. 이 목록이
**공개 `openai/clip-vit-large-patch14-336` 기준으로 도출된 것은 아닌지 확인 가능할까요?**

| 항목 | 예제가 가리키는 구성 | 저희가 보내드린 `vit_l14_336.pt` (실제 확인) |
| --- | --- | --- |
| 활성함수 | **QuickGELU** (`.../quickgelu/conv2d`) | **일반 GELU** — 저희 컴파일 결과 op 이름 533개 중 `quickgelu` **0개**, 전부 `gelu` |
| 모듈 명명 | `vision_model.encoder.layers.N.mlp.fc2` (HF CLIP 규약) | `visual.transformer.resblocks.N.mlp.c_proj` (open_clip 계열 규약) |
| **어텐션 풀링 헤드** | 해당 키 없음(공개 CLIP은 class token + `ln_post` + `proj` 로 임베딩 생성) | **`visual.attn_pool.*` 존재** — `probe(1,1,1024)`, `attn.in_proj(3072,1024)`, `layernorm`, `mlp.c_fc/c_proj` |
| 예제 기본 모델 인자 | `--model-id openai/clip-vit-large-patch14-336` | — |

특히 **저희 모델에는 공개 CLIP에 없는 attention pooling head가 붙어 있습니다.**
앞선 지연 문의 때 동봉한 `README.md` 에도 아래와 같이 적어 보내드렸던 부분입니다.

> "이 모델은 **표준 CLIP/ViT가 아니라 2D RoPE 기반 ViT-L/14 + attention pooling head**라,
> 단일 입력/단일 출력 컴파일을 위해 모델 패치 5종(RoPE freq 상수화 / Rope2D einops-free 구현 /
> SelfAttention SDPA / AttentionPooling 분해 / abs_posemb 상수화)을 적용했습니다."

참고로 동봉한 `model/vit_l14_336.pt` 는 state_dict 형태이고, 키 이름 규약이 HF CLIP과
**전혀 겹치지 않습니다**(총 601개 키, 겹치는 키 0개).

| 구분 | 키 예시 |
| --- | --- |
| 저희 체크포인트 | `visual.transformer.resblocks.N.mlp.c_proj.weight`, `visual.attn_pool.probe`, `token_embedding.weight`, `logit_scale` |
| HF CLIP 규약 | `vision_model.embeddings.patch_embedding.weight`, `vision_model.encoder.layers.N.mlp.fc2.weight`, `visual_projection.weight` — 저희 파일에는 **모두 없음** |

그래서 이 파일을 `CLIPVisionModelWithProjection` 등 HF CLIP 클래스로 로드하려 하면 키가 하나도
맞지 않아 실패할 것으로 보입니다. 저희가 쓰는 모델 정의 라이브러리(`ve_lib`)를 용량 문제로 동봉하지
못한 탓에 로드가 어려우셨을 수도 있겠다는 생각이 듭니다. **필요하시면 모델 정의 코드를 별도로
전달드리겠습니다.**

만약 예제의 A16 목록이 공개 CLIP 기준으로 선정된 것이라면 저희 모델에 그대로 적용하기는 어려울 것
같아, 이 부분 확인 부탁드립니다.

참고로, 예제의 `--profile` 과 **동일한 기준**(`ratio = max / p99.9`, 추적 대상 mlp.fc1/fc2/
activation_fn·attn.out_proj·layer_norm)으로 저희 모델을 프로파일링한 결과는 아래와 같습니다.

| 순위 | 저희 모델 텐서 | ratio |
| --- | --- | ---: |
| 1 | L3.mlp.c_proj | **74.9×** |
| 2 | L12.mlp.c_proj | 16.2× |
| 3 | L10.mlp.c_proj | 14.4× |
| 4 | L12.mlp.gelu | 13.0× |
| 5 | L10.mlp.gelu | 12.3× |

예제는 L9/L11/L12 중심인데 저희 프로파일에서는 **L3가 74.9× 로 압도적 1위**로 나옵니다.
이 차이가 (a) 서로 다른 체크포인트를 보고 계신 것인지, (b) 선정 기준·집계 방식이 다른 것인지
알려주시면 감사하겠습니다.

## 4. 함께 여쭙고 싶은 것

1. `select_a16.py` 를 전달해 주실 수 있을까요? (§2)
2. **혹시 저희가 첨부드린 모델 파일(`pia_ve_latency_headroom_full.zip` → `model/vit_l14_336.pt`)로
   직접 컴파일·측정해 보신 것인지 확인 가능할까요?** 주신 표(W4A16 cos 0.973 / W4A8+A16 5개 0.905)가
   저희 재현값(0.8611 / 0.3190)과 큰 차이가 있어, 서로 다른 모델을 보고 있는 것이 아닌지 궁금합니다.
   §3의 구조 차이(활성함수·모듈 명명·attention pooling head)도 그 가능성을 시사하는 것 같습니다.
3. 측정에 사용하신 **qbcompiler 버전**을 알려주실 수 있을까요? 저희는 최신(1.2.0)과 이전(1.1.2)
   양쪽으로 시도하고 있는데, `SearchWeightScaleConfig` 필드가 두 버전 간에 달라져 있어
   버전 차이가 결과에 영향을 주는지 확인하고 싶습니다.
4. 주신 표(W4A16 cos 0.973 / W4A8+A16 5개 cos 0.905)의 **전체 컴파일 설정**을 알려주실 수 있을까요?
   OPTQ 파라미터, SearchWeightScale 적용 범위, uint8 입력·normalize 폴딩 포함 여부, calibration
   데이터셋 종류와 장수를 알면 저희 재현 결과와 직접 대조할 수 있습니다.
5. OPTQ·SearchWeightScale이 저희 환경에서는 정확도를 **떨어뜨립니다**(0.9135 → 0.8786 → 0.8611).
   적용 조건(레이어 범위 `apply_layers`/`exclude_layers`, calibration 데이터 성격, batch 크기 등)에
   저희가 놓친 전제가 있을까요?
6. A8에서의 붕괴가 A16 텐서 **개수 부족** 때문일 수 있을까요? 5개보다 늘리는 것이 권장되는지,
   아니면 개수보다 **어느 텐서를 고르느냐**가 결정적인지 의견 주시면 큰 도움이 되겠습니다.

바쁘신 중에 늘 상세히 답변 주셔서 감사합니다.

감사합니다.
PIA-SPACE 정서익 드림
