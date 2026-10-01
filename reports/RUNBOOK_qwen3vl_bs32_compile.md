# 런북 — GPU 서버에서 Qwen3-VL batch32 MXQ 재컴파일

> **목적**: 파인튜닝한 모델로 mxq 를 갈아끼우기 **전에**, 스톡 모델로 재컴파일 파이프라인이
> 도는지부터 확인한다. 가중치를 바꾸기 전에 파이프라인 검증이 순서다.
>
> 대상 산출물: `decoder-b1.mxq` / `decoder-b32.mxq` / `vision.mxq` / `vision8192.mxq`
> 관련: `reports/performance/qwen3vl/NPU_qwen3vl_bs32_server.md` (이 mxq 를 쓰는 검증서버)

---

## 0. 먼저 — **이 레포만 clone 해서는 재현되지 않는다**

| 필요한 것 | 레포에 있나 |
|---|---|
| 컨테이너 정의 `setup/vlm_repro/Dockerfile.compile`, `run_compile.sh` | ✅ 추적됨 |
| 이 런북 | ✅ |
| **벤더 재현 패키지 `customer-capacity-repro`** | ❌ **`download/` 는 gitignore** |

벤더 패키지는 **압축 9.8 GB / 해제 26 GB** 다(`download/vendor/customer-capacity-repro.tar.gz`).
git 으로 안 가므로 **별도로 옮겨야 한다.**

```bash
# NPU 서버 → GPU 서버 (예시)
rsync -avP --partial \
  download/vendor/customer-capacity-repro.tar.gz \
  <user>@<gpu-server>:/data/vendor/

# GPU 서버에서
mkdir -p ~/AX_NPU/download/vendor && cd ~/AX_NPU/download/vendor
tar xzf /data/vendor/customer-capacity-repro.tar.gz     # 26GB 로 풀린다
```

> HF 업로드는 권하지 않는다 — org private 쿼터가 이미 한계라 9.8 GB 가 또 막힌다
> (`NPU_qwen3vl_bs32_server.md` §8 참고).

**무결성 확인**(벤더 제공, 의존성 없이 돈다):
```bash
python3 scripts/verify_files.py      # 769/769 이어야 한다
```

---

## 1. 하드웨어 요건 — **GPU 없으면 못 한다**

벤더 `scripts/compile.py` 가 하드 블록한다:

```python
if not torch.__version__.startswith('2.7.1') or torch.version.cuda != '12.8':
    p.error('This compiler binary requires torch 2.7.1 with CUDA 12.8.')
if not torch.cuda.is_available():
    p.error('A CUDA GPU is required by this recorded recipe.')
```

**`device='cpu'` 로 우회하지 말 것.** 네이티브 모듈이 CUDA 빌드이고(`TOOLCHAIN.json` 이
sha256 까지 고정), 레시피를 바꾸면 "재현" 자체가 무의미해진다.
PE-Core 쪽 CPU 폴백(`-cpu` 이미지)은 **공개 qbcompiler** 얘기고 이 커스텀 휠과는 무관하다.

| | 벤더가 실제로 쓴 장비 (`VALIDATION.md`) |
|---|---|
| decoder / vision8192 | **RTX PRO 6000 Blackwell Server Edition** (VRAM 97,887 MiB) |
| vision (기본) | **H200 NVL** (143,771 MiB) |

> 벤더 주석: "실제 사용한 장비이며 **최소 VRAM 요구량 측정치는 아닙니다**."
> 즉 더 작은 GPU 로도 될 수 있으나 검증된 바 없다.

사내 GPU 서버는 **Xeon 6530P + RTX PRO 6000 × 2**(`RUNBOOK_quant_matrix_120.md` 기록) —
decoder 쪽은 벤더와 동일 장비다.

기타: 디스크 **80 GB 이상 여유** 권장(입력 20GB + split MBLT + 중간 산출물).

---

## 2. 절차

```bash
git clone <AX_NPU> && cd AX_NPU
# (0 절대로 벤더 패키지를 download/vendor/ 에 배치)

# 이미지 빌드 — 빌드 컨텍스트가 재현 패키지다
docker build -f setup/vlm_repro/Dockerfile.compile -t mblt_vlm_repro:compile \
    download/vendor/customer-capacity-repro

# 컴파일 (prepare 포함, 래퍼가 알아서 한다)
bash setup/vlm_repro/run_compile.sh --part all
#   --part decoder-b32   배치32 디코더만
#   GPUS=1 ...           GPU 지정
```

`run_compile.sh` 가 하는 일:
1. **GPU 존재 확인** (없으면 컨테이너 띄우기 전에 조기 종료)
2. `scripts/prepare.py` — 보정 경로 정규화 + **batch32 용 split MBLT 생성**
   (`split_legacy_deepstack.py` 가 legacy 2-input deepstack 을 쪼갠다)
3. `scripts/compile.py --part ...`

산출물은 `<repro>/build/*.mxq`.

---

## 3. 검증 — **바이트 동일을 기대하지 말 것**

대조군이 `<repro>/reference/mxq/` 에 있다(벤더가 실측에 쓴 원본 + `provenance.json` sha256).

그런데 벤더가 두 군데서 못 박았다:

> `TOOLCHAIN.json`: "Recompilation is measured separately; **identical millisecond timings or
> identical MXQ bytes are not promised.**"
>
> `VALIDATION.md` §교차시험: "같은 소스와 옵션이 MXQ의 바이트 단위 재현이나 **예제 정답 일치까지
> 보장하지는 않았습니다.**" — 실제로 벤더가 vision 을 재컴파일했을 때 차이가 나타났고,
> 원인(GPU 종류 / 수치 변동 / 컴파일러 구현)을 확정하지 못했다.

→ **sha256 대조는 참고일 뿐 합격 기준이 아니다.** 판정은 NPU 실측으로 한다:

```bash
# NPU 서버로 build/*.mxq 를 가져와 모델 디렉토리 조립 후
#   (decoder-b32 → decoder.mxq, decoder-b1 → variants/decoder-b1.mxq, vision.mxq,
#    + model.safetensors / rotation.pth / config.json 은 기존 것 그대로)
bash setup/vlm_repro/run_runtime.sh --sizes 224x224 --repetitions 2 --output results/recompiled
```

합격 기준: **정답 라벨 일치 + 지연이 기존 수준**
(2카드 20채널 짧은 프롬프트 ~740 ms — `NPU_qwen3vl_bs32_server.md` §1 표와 대조).

---

## 4. 파인튜닝 모델로 갈아끼울 때 (다음 단계)

★ **mxq 만 바꾸면 안 된다.** 런타임이 모델 디렉토리에서 따로 읽는 것들이 있다:

```python
# pia_vlm/support.py — load_support()
weights  = load_file(directory / "model.safetensors")
embed    = weights["model.language_model.embed_tokens.weight"]   # CPU측 토큰 임베딩
position = weights["model.visual.pos_embed.weight"]
```

파인튜닝하면 `embed_tokens` 가 바뀌는데, vocab·hidden 크기가 그대로면 **shape 검증을 통과해
에러 없이 옛 임베딩으로 돈다.** 조용히 틀린 답이 나오는 가장 위험한 형태다.

**같이 움직여야 하는 세트**: `decoder.mxq` + `variants/decoder-b1.mxq` + `vision.mxq`
+ `model.safetensors` + `rotation.pth` (+ 구조 바뀌면 `config.json`)

> 벤더 README: "**R1 행렬과 CPU embedding을 함께 유지해야 하며, 다른 export/회전 행렬을
> 섞으면 정답이 깨질 수 있습니다.**"

### 막히는 지점 — HF 체크포인트 → MBLT export

이 패키지는 **벤더가 미리 만든 `inputs/mblt/*.mblt` 에서 시작**한다. HF 체크포인트를 MBLT 로
export 하는 스크립트는 **패키지에 없다**(README: "모델 export … 는 필요하지 않습니다" = 범위 밖).
`TOOLCHAIN.json` 의 `hf_model` 도 스톡 `Qwen/Qwen3-VL-2B-Instruct` 고정 리비전이다.

**다만 컴파일러 휠 안에 경로가 있다** — 확인됨:
- `qbcompiler/model_dict_new/parser/patcher/models/hf_models/qwen3vl.py`
- `qbcompiler/model_dict/parser/backend/fx_hf_extensions/transformers/models/qwen3vl.py`
- public API 에 `mxq_compile_from_source`, `mxq_compile_from_mblt`

우리가 써 본 적은 없다. 안 되면 **벤더에 파인튜닝 체크포인트 export 를 문의**하는 게 빠르다
(어차피 batch32 는 커스텀 컴파일러 `1.3.4+pia.capacity3` 산출물이라 벤더 의존이 있다).

### 보정 데이터도 다시 봐야 한다

번들된 보정 레코드(decoder 115 / vision 106 / vision8192 113)는 **스톡 모델 기준**이다.
파인튜닝하면 활성값 분포가 바뀌므로 재보정이 필요하다.

★ **정확도는 반드시 재검증할 것.** PE-Core 에서 attn_pool 이 INT8 에서 cos 0.46 으로 무너졌다가
score matmul 만 16bit 로 올려 복구한 전례가 있다(`reports/vendor/mobilint_resolution_attn_pool.md`).
파인튜닝은 이런 함정을 새로 만들 수 있다.

---

## 5. 알려진 함정

- **컴파일 환경과 런타임 환경을 합치지 말 것.** torch 2.7.1+cu128 vs 2.9.0 — 벤더도 별도 가상환경 지시.
- **Python 3.10 고정.** 휠이 cp310 이고 `compile.py` 가 `sys.version_info[:2] != (3, 10)` 을 검사한다.
- **`QBCOMPILER_PIA_NATIVE`** 는 import 전에 정하고 한 프로세스에서 바꾸지 않는다
  (decoder/vision 이 서로 다른 네이티브 빌드라 `compile.py` 가 part 마다 새 프로세스를 띄운다).
- **공개 qbcompiler 1.3.0 은 이 용량 재현에 쓰면 안 된다.** 벤더 커스텀 `1.3.4+pia.capacity3` 이어야 한다
  (`reports/inquiries/06_qwen3vl_prefill_optimization/REPLY_2.md`).
