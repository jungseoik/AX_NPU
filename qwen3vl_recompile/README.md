# qwen3vl_recompile — 신규 Qwen3-VL-2B 체크포인트를 batch32 MXQ 로

학습한 Qwen3-VL-2B(동일 아키텍처) 체크포인트를 주면 **벤더와 같은 레시피로** NPU용
batch32 MXQ 를 만들고, 기존 batch32 와 출력을 대조한다.

```bash
bash run_all.sh --checkpoint /path/to/my-finetuned --out /data/mybuild
```

---

## 왜 필요한가

벤더 재현 패키지(`customer-capacity-repro`)는 **이미 export 된 `.mblt` 에서 시작**한다.
그 `.mblt` 는 스톡 `Qwen/Qwen3-VL-2B-Instruct`(rev `89644892`)에서 뽑은 것이라
신규 학습본에는 쓸 수 없다. 캘리브레이션도 **활성값**이라 가중치에 직접 의존한다.

이 패키지가 그 빠진 앞단(① export, ② 캘리브)을 채운다.

| 입력 | 가중치 종속? | 처리 |
|---|:---:|---|
| `.mblt` | ✅ | **①에서 새로 생성** |
| 캘리브레이션 활성값 | ✅ | **②에서 새로 생성** |
| `rotation.pth` | ❌ | 벤더 것 재사용 — 하다마드 행렬(직교오차 5.96e-08, 절댓값 고유 1개) |
| `configs/*.json` | ❌ | 벤더 것 재사용 |

---

## 파이프라인

```
체크포인트
   │
   ├─ ① export_mblt.py   →  vision.mblt (1.5GB) + decoder.mblt (6.6GB)     ~45초
   │
   ├─ ② make_calib.py    →  vision.json/decoder.json + npy 활성값          ~수분
   │
   ├─ ③ compile_mxq.py   →  vision.mxq / decoder-b32.mxq                   ~2시간
   │     (양자화 최적화: hessianQuant + searchWeightScale + 회전변환)
   │
   └─ ④ verify_npu.py    →  기존 batch32 와 라벨·지연 A/B                   ~수분
```

③이 오래 걸리는 건 컴파일이 아니라 **양자화 탐색** 때문이다. 레이어마다 헤시안을 구해
가중치를 하나씩 양자화하며 오차를 보정하고, 스케일 후보를 그리드 서치한다.

---

## ① export 에서 지켜야 하는 5가지

`mblt_compile()` 호출법은 벤더 문서에 없다. qbcompiler 안의 Qwen3-VL 패처
(`model_dict_new/parser/patcher/models/hf_models/qwen3vl.py`, 781줄)를 읽어 알아냈다.
**전부 시행착오로 확인한 것이라 바꾸지 말 것.**

1. **전체 모델을 통째로 넣으면 안 된다.** `Qwen3VLForConditionalGeneration` 을 그대로 주면
   FX 추적이 실패한다(`use_cache` 처리가 fx proxy 와 충돌).
   → 패처의 `VisionModelForQwen3VL` / `Projection` 을 **각각** 넣는다.
2. **`mblt_compile()` 은 `hf_config` 로더를 타지 않는다**(그건 `mxq_compile` 전용).
   → 모델 객체를 직접 만들어 넘긴다. `model=None` 이면 `'NoneType' has no '__name__'`.
3. **RoPE 는 호스트에서 선계산**한다. `CachedQwen3VLTextRotaryEmbedding` + `set_rope()`.
   안 하면 in-place setitem 때문에 `UnboundLocalError: 'ov'`.
4. **KV 캐시를 그래프 출력에서 뺀다.** `use_cache=False` + logits 만 반환하는 래퍼.
   안 하면 `NotImplementedError: HFProxyableDynamicCache`.
5. **deepstack 은 전체 시퀀스 길이(T)** 로 주고 `_deepstack_process` 를 덧셈으로 교체한다.
   HF 원본은 압축형(N_visual)을 masked-scatter 하는데 추적이 안 된다.
   안 하면 `tensor a (64) must match tensor b (83)`.

### 검증 결과 (스톡으로 확인)

| | 우리 | 벤더 | 차이 |
|---|---:|---:|---:|
| `vision.mblt` | 1543.7 MB | 1543.7 MB | **+0.00%** |
| `decoder.mblt` | 6579.9 MB | 6579.8 MB | **+0.00%** |

---

## batch32 에 `split_legacy_deepstack.py` 는 **불필요**하다

벤더 패키지의 그 스크립트는 **레거시 포맷 전용 어댑터**다. 벤더의 `decoder.mblt` 는 옛날
export 라 deepstack 이 1입력(`deepstack_visual_embeds_0`)으로 뭉쳐 있어서, batch32 용으로
3입력으로 쪼개야 했다. 신규 export 는 처음부터 분리돼 있다.

실제로 신규 mblt 에 돌리면 이렇게 거부한다:
```
This adapter is for the checked legacy format; re-export newer MBLT normally
```

| | 벤더 mblt | 우리 mblt |
|---|---|---|
| magic | `\x00\x00\x00\x00` | `MBLT` |
| 포맷 | legacy 33-byte footer | 신규 |
| split 필요 | 예 | **아니오** |

---

## ④ 판정 기준 — sha256 이 아니다

벤더가 두 군데서 못 박았다:

> `TOOLCHAIN.json`: "identical millisecond timings or **identical MXQ bytes are not promised**"
> `VALIDATION.md`: "같은 소스와 옵션이 MXQ의 바이트 단위 재현이나 예제 정답 일치까지
> 보장하지는 않았습니다"

실제로 스톡 재컴파일에서 mxq 가 65~734 바이트 달랐다(0.0002% 미만). 그래서 판정은:

- **라벨이 글자 단위로 동일** ← 가장 중요
- 지연 차이가 측정 편차(~1%) 안
- 배치 정답률 동일

> 참고: 스톡 재컴파일본은 벤더 원본과 단건 −0.7% / 8건 +0.3% / 20건 +0.6%,
> 라벨 완전 동일이었다. 이게 "동등"의 기준선이다.

---

## 요건

| | |
|---|---|
| GPU | **CUDA 12.8 NVIDIA** 필수 (벤더 스크립트가 하드 체크) |
| 검증 장비 | RTX PRO 6000 Blackwell SE 97GB ×2 (벤더와 동일 모델) |
| 디스크 | 80GB+ 여유 |
| 이미지 | py3.10 / torch 2.7.1+cu128 / qbcompiler **1.3.4+pia.capacity3** |
| 벤더 패키지 | `download/vendor/customer-capacity-repro` (configs, rotation.pth) |
| NPU | ④ 검증 단계에만 필요 (별도 서버 가능) |

컨테이너 빌드: `setup/vlm_repro/Dockerfile.compile` + `run_compile.sh`
절차서: `reports/RUNBOOK_qwen3vl_bs32_compile.md`

---

## 아직 검증 안 된 것

- **②·③을 우리 산출물로 끝까지** 돌린 적이 없다. ①은 스톡으로 입증됐고,
  ③은 **벤더 mblt + 벤더 캘리브**로 입증됐지만(2시간 7분, NPU 동등),
  **우리 mblt + 우리 캘리브** 조합은 미검증이다.
- 캘리브 활성값을 PyTorch(fp32)로 만든다. 벤더가 NPU 비전 출력을 썼는지는 모른다.
  fp32 쪽이 기준으로 더 정확하지만, 양자화 분포가 달라질 가능성은 있다.
- 파인튜닝으로 가중치 분포가 바뀌면 **양자화 정확도는 재검증이 필요하다**.
  PE-Core 에서 attn_pool 이 INT8 에서 cos 0.46 으로 무너진 전례가 있다.
