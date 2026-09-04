# 런북 — GPU 서버에서 qbcompiler 1.2.0 양자화 매트릭스 재현

> ## ✅ 실행 기록 (2026-09-04, GPU 서버 Xeon 6530P + RTX PRO 6000×2)
>
> - **§3 매트릭스 24조합 전부 컴파일 완료** — 4워커 병렬(잡마다 새 `--rm` 컨테이너, OMP 32스레드),
>   조합당 5~13분, 전체 벽시계 ~55분. §5-2 체크 통과: 24/24 로그에서 `aries-rb`·qk16 25개·(1,1024),
>   md5 24개 유일, 크기 정상(W8≈327-330MB / W4≈188-192MB). `mxqtool show` 는 이 서버에 런타임이
>   없어 **NPU 서버에서 확인 필요**.
> - **HF 업로드 완료** — `<quant>/<tuning>/<scheme>/pe_full.mxq` 24개 + 루트 `TUNING_MATRIX.md`
>   (철회 공지를 1.2.0 표로 교체, 컴파일 환경 명시). 로컬 원본: GPU 서버 `out/matrix_120/`.
> - **§2.5 검증 B·C 완료** (아래 각 섹션에 결과 기입). 검증 중 `validate_compile_cli.sh` 가
>   `--calib-data-path` 를 안 넘기는 버그 발견·수정(random calib으로 돌던 문제 — C 의 에러로 드러남).
> - **§4 NPU 검증 완료 (2026-09-04)** — HF 24종 전부 측정. 튜닝이 3개 양자화 모두에서 정확도 상승,
>   코어모드는 정확도 무관. 회귀 기준을 **24점 전부로 재앵커링**(표준 이미지셋 기준).
>   결과: [`performance/NPU_pe_quant_tuning_matrix_120.md`](performance/NPU_pe_quant_tuning_matrix_120.md).
>   W4A8 초기 기준값 0.8932 는 A16 누락 산출물이었고 정정값은 **0.9420**.
>   **남은 것**: C 의 stats-load 산출물(`out/validate_cli/pe_W4A16_none_single_statsload.mxq`)이
>   GPU 서버에만 있어 "통계 재사용 시 cos 동일" 확인이 미완. 그 파일만 넘겨주면 즉시 판정 가능.
> - ~~다음(NPU 서버)~~: §4 — `out/matrix_120/` 를 rsync 또는 HF에서 받아
>   `verify_quant_tuning_matrix.py` 로 24종 cos 측정(회귀 4점 자동 대조), B(0.8932)·C(0.9126 동일성) cos 확정,
>   `mxqtool show` 4항목 확인. 이후 §6 배포 판단(W8A16 sws_optq 우선).

> **왜 다시 돌리나**: 기존 24종은 qbcompiler **1.1.2** 로 컴파일했고, 그 버전의 OPTQ/SearchWeightScale
> 구현 문제로 튜닝본이 열화돼 있었다. 1.2.0 에서는 반대로 정확도가 올라간다.
> 근거: [`performance/NPU_pe_quant_tuning_compiler_version.md`](performance/NPU_pe_quant_tuning_compiler_version.md)
>
> 이 문서는 **GPU 서버에서 레포를 clone 한 상태에서 그대로 따라가면 재현되도록** 쓴 절차서다.

| 항목 | 값 |
| --- | --- |
| 대상 | PE-Core-L14-336 full NPU (`--qk16` 자동) |
| 컴파일러 | qbcompiler **1.2.0** (SDK 1.1v), docker `mobilint/qbcompiler:1.2-cuda12.8.1-ubuntu22.04` |
| 타겟 | `aries-rb` — **자동 판별되므로 지정 불필요** (`pe_npu/target_device.py`) |
| 조합 | 양자화 3 × 튜닝 2(`none`/`sws_optq`) × 코어모드 4 = **24** |
| 예상 시간 | GPU 조합당 5~8분 → 직렬 2~3시간 (CPU는 조합당 64분이라 비권장) |
| 검증 | 컴파일은 GPU 서버, **cos 측정은 NPU 서버**(2단계 분리) |

---

## 0. 왜 이 조합인가 — 축을 바꿨다

기존 24종은 `양자화 3 × 튜닝 4 × 코어모드 2` 였다. 튜닝이 되는지 몰랐을 때의 설계다.
지금은 답을 알기 때문에 축을 바꾸는 게 맞다.

- **튜닝 4종 → 2종**: `sws` 단독 / `optq` 단독은 배포 후보가 아니다. 둘 다 켠 게 셋 중 최선인 것은
  1.1.2·1.2.0 양쪽에서 일관됐다. `none` 은 베이스라인 대조용으로 반드시 필요하다.
- **코어모드 2종 → 4종**: 실제 배포는 single/multi/global4/global8 네 개다.
  1.2.0 측정은 아직 `single` 하나뿐이라 배포본을 갈아치울 근거가 없다.

---

## 1. 사전 준비 (GPU 서버)

```bash
git clone https://github.com/jungseoik/AX_NPU.git && cd AX_NPU
```

### 1-1. `.env` — 토큰

`.env` 는 gitignore 다. 직접 만든다.

```bash
cat > .env <<'EOF'
HF_TOKEN=hf_xxxxxxxxxxxx      # PIA-SPACE-LAB/MXQ_NPU (private) 접근용
SUDO_PASS=xxxxxxxx            # docker 명령에 sudo 필요한 경우만
EOF
```

### 1-2. SDK 1.1v 내려받기

SDK 바이너리는 비공개라 git 에 없다. HF 에서 받는다.

```bash
python setup/fetch_sdk_from_hf.py --sdk 1.1     # → download/sdk/v1.1/
python setup/sdk_resolve.py --sdk 1.1           # 해석 결과 확인
```

받아야 할 것 중 컴파일에 필요한 건 **`qbcompiler-1.2.0-py3-none-any.whl`** 하나다.
(드라이버·런타임은 NPU 없는 GPU 서버에서는 불필요)

### 1-3. docker 이미지

```bash
docker pull mobilint/qbcompiler:1.2-cuda12.8.1-ubuntu22.04     # 27.7GB
# GPU 없으면: mobilint/qbcompiler:1.2-cpu-ubuntu22.04          # 14.2GB (조합당 64분)
```

전제: NVIDIA 드라이버 + `nvidia-container-toolkit` (= `docker run --gpus all` 가능).
`nvidia-smi` 가 되는지 먼저 확인할 것.

### 1-4. calibration 데이터 — **1.1.2 측정과 동일해야 한다**

비교 가능성이 여기서 결정된다. COCO val2017 **앞 200장(sorted)** 을 써야 한다.

```bash
# COCO val2017 준비 (이미 있으면 생략)
mkdir -p download/coco && cd download/coco
wget http://images.cocodataset.org/zips/val2017.zip && unzip -q val2017.zip && cd -

python -m pe_npu.calib --dataset coco --src download/coco/val2017 \
                       --num 200 --out download/calib_coco_hwc --hwc
```

**확인**: `download/calib_coco_hwc/npy_files.txt` 가 200줄이어야 한다.

```bash
wc -l download/calib_coco_hwc/npy_files.txt      # → 200
```

---

## 2. 스모크 테스트 — 컴파일 걸기 전에 3분

24조합을 2~3시간 돌린 뒤 실패를 발견하면 낭비다. 먼저 짧게 확인한다.

```bash
bash setup/compile_in_docker.sh --sdk 1.1 --gpu --help 2>/dev/null   # 래퍼 동작
```

### 2-1. target_device 자동 판별 확인 ★

```bash
docker exec -w /workspace <컨테이너> python -c "
import qbcompiler
from pe_npu.target_device import resolve_target_device
print(qbcompiler.__version__, '→', resolve_target_device())"
```

**기대 출력**: `1.2.0 → aries-rb`

`aries2` 가 나오면 컴파일이 `ValueError: Unsupported device name: 'aries2'` 로 죽는다.
1.2.0 이 받는 이름은 `regulus-ra` / `aries-rb` / `regulus-rb` 뿐이고, 이 중 `aries-rb` 가 Aries2 다.

### 2-2. parse 로 그래프 추적 확인

```bash
docker exec -w /workspace <컨테이너> python -m pe_npu.compile --mode parse
```

**기대**: `[sanity] full output: (1, 1024)` + `[target] qbcompiler 1.2 → target_device=aries-rb`

---

## 2.5. 미검증 코드 경로 검증 — GPU 에서 20~25분

**GPU 에서는 조합당 5~8분이므로 아래를 먼저 털고 가는 게 이득이다.** CPU 에서는 건당 64분이라
부담스럽지만 GPU 라면 20분 남짓이다. 검증 방법이 깔끔한데, **이미 정답을 아는 조합을 다시 뽑아
숫자가 맞는지 보면 된다**(§5-1 회귀 기준).

### 한 줄로 돌리기

세 검증을 순서대로 돌리고, NPU 가 있으면 cos 대조까지 이어가는 스크립트가 있다.

```bash
# 컴파일 서버(GPU/CPU 무관) — 이미지·디바이스는 자동 선택
bash reports/scripts/validate_compile_cli.sh --sdk 1.1

# 일부만
bash reports/scripts/validate_compile_cli.sh --tests A

# NPU 서버에서 검증만 (qbruntime 있는 python 지정)
PYTHON=~/miniconda3/envs/pe_npu_host/bin/python \
  bash reports/scripts/validate_compile_cli.sh --verify-only --out out/validate_cli --device 0
```

**이미 만들어진 mxq 는 건너뛴다.** 그래서 한 서버에서 일부를 돌리고 `out/` 만 rsync 로 옮겨
다른 서버에서 이어 돌릴 수 있다. NPU 가 없는 서버면 mxq 만 만들고 검증 명령을 안내한다.
검증은 §5-1 회귀 기준을 자동 대조해 이탈 시 **exit 1** 이다.

아래는 그 스크립트가 실제로 무엇을 하는지, 그리고 왜 필요한지의 설명이다.

### 왜 필요한가

1.2.0 실측 4점과 기존 24조합은 각각 임시 하네스와 `compile_quant_tuning_matrix.py` 로 뽑았다.
**정식 CLI(`python -m pe_npu.compile --quant ... --optq ...`) 로 컴파일한 적이 없다.**
설정 구성은 두 컨테이너에서 확인했지만(비트 배치·OptqConfig·SearchWeightScaleConfig 필드 동일,
camelCase alias 정상), 실제 컴파일 산출물이 등가인지는 확인되지 않았다.

### 검증 A — 정식 CLI 경로 등가성 ★ → **완료 (2026-09-03, PASS)**

> **이미 검증됐다. GPU 서버에서는 B·C 만 돌리면 된다** (`--tests B,C`).
>
> qbcompiler 1.2.0 / CPU 로 아래 명령을 그대로 돌려 NPU 에서 cos 를 쟀다.
>
> | 항목 | 결과 |
> | --- | --- |
> | 타겟 자동 판별 | `qbcompiler 1.2 → target_device=aries-rb` ✓ |
> | qk16 score MatMul | 25개 ✓ |
> | 비트 프리셋 | `preset=w4a16 activation16=25 weight16=0` ✓ |
> | **cos** | **0.9642** — 회귀 기준과 완전 일치 ✓ |
> | 컴파일 | 5189.8s (86분, CPU) |
>
> **결론: `python -m pe_npu.compile --quant ... --optq --search-weight-scale` 는
> 임시 하네스·매트릭스 스크립트와 등가다.** 매트릭스를 CLI 로 돌려도 된다.
>
> **주의 — MD5 는 다르다.** 임시 하네스 산출물 188,491,097 bytes(md5 `7b039f39…`) vs
> CLI 산출물 188,491,103 bytes(md5 `6f2b5c1c…`). 차이 **6바이트는 저장 파일명 길이 차이와
> 정확히 일치**한다(`cli_W4A16_sws_optq_single.mxq` 29자 vs `W4A16_sws_optq_c120.mxq` 23자).
> mxq 가 save 경로를 내부에 담기 때문이다. **비트 구성·정확도는 동일하므로 md5 불일치를
> 문제로 보지 말 것** — 판정은 cos 와 크기로 한다.



```bash
python -m pe_npu.compile --mode compile \
  --quant w4a16 --optq --search-weight-scale \
  --scheme single --device gpu \
  --calib-data-path download/calib_coco_hwc \
  --save out/cli_W4A16_sws_optq_single.mxq
```

로그에서 확인:

| 항목 | 기대 |
| --- | --- |
| `[target]` | `qbcompiler 1.2 → target_device=aries-rb` |
| `[quant]` | `preset=w4a16  activation16=25  weight16=0` |
| `[qk16]` | score MatMul **25개** |
| `[OK]` | `size_bytes` ≈ 188,500,000 |

그다음 NPU 서버에서 cos:

```bash
python reports/scripts/verify_quant_tuning_matrix.py --src-dir <위 mxq 폴더> --device 0
```

**기대: cos 0.9642 (`✓ 기준 0.9642`).** 이 한 건이 맞으면 CLI 경로가 임시 하네스와 등가임이
증명되고, 이후 매트릭스를 CLI 로 돌려도 된다. 어긋나면 `compile_quant_tuning_matrix.py` 로만 간다.

### 검증 B — `--quant w4a8` 의 A16 주입 → **컴파일 측면 통과 (2026-09-04)**

> GPU 서버에서 아래 명령 그대로 실행: `[quant] preset=w4a8 activation16=30`(qk16 25 + A16 5) ✓,
> calib 200장 정상 주입, 산출물 188,065,891 bytes(≈기대 188.0MB). **cos 0.8932 는 NPU 검증 대기**
> (산출물: GPU 서버 `out/validate_cli/pe_W4A8_L5A16_sws_optq_single.mxq`).

**`--quant w4a8` 은 A16 텐서 5개를 자동으로 넣지 않는다.** `--a16` 으로 정확한 mblt 이름을
직접 넘겨야 한다(부분문자열 매칭인 `--act16` 을 쓰면 엉뚱한 텐서가 걸린다 — 과거 W4A8 cos 0.2609
오측정의 원인이 이것이었다).

```bash
python -m pe_npu.compile --mode compile --quant w4a8 \
  --a16 "visual_transformer_resblocks_3_mlp_c_proj,\
visual_transformer_resblocks_12_mlp_c_proj,\
visual_transformer_resblocks_10_mlp_c_proj,\
visual_transformer_resblocks_12_mlp_c_fc/reshape/gelu_0,\
visual_transformer_resblocks_10_mlp_c_fc/reshape/gelu_0" \
  --optq --search-weight-scale --scheme single --device gpu \
  --calib-data-path download/calib_coco_hwc --save out/cli_W4A8_sws_optq_single.mxq
```

`[quant] activation16=30` (qk16 25개 + A16 5개) 이어야 한다. **기대 cos 0.8932.**
`activation16=25` 면 A16 5개가 안 들어간 것이다.

> 이 5개 이름은 `compile_quant_tuning_matrix.py` 의 `A16_TENSORS_W4A8` 에 하드코딩돼 있다.
> 다른 체크포인트에서는 `--mode parse --dump-names` 로 다시 뽑아야 한다.

### 검증 C — calibration 통계 캐시 (`--calib-stats-save/load`) → **기능 동작 확인 (2026-09-04)**

> GPU 서버 실측: stats-save 컴파일 **424.4s** → stats-load 재사용 **185.6s (2.3배 단축)** ✓,
> 출력 경로 정상(`_0.9999` 잔재 없음, `_normalize_stats_output` 동작) ✓. 크기 차 10바이트는
> 저장 파일명 길이 차이(검증 A 와 동일 현상). **cos 0.9126 동일성만 NPU 검증 대기**
> (산출물: `out/validate_cli/pe_W4A16_none_single{,_statsload}.mxq`). 추가 발견:
> ① stats 는 파일이 아니라 **디렉토리**로 저장된다(~98KB). ② `--calib-stats-*` 는
> `--calib-data-path` 필수(없으면 에러 — 이 에러 덕에 validate 스크립트의 calib 누락 버그를 잡음).
> ③ GPU 메모리가 빠듯하면(당시 GPU0 에 vLLM 88GB 상주) stats-save 가 CUDA abort 로 죽는다 —
> 여유 GPU 에선 정상.

**전혀 검증되지 않은 기능이다.** 캘리브레이션이 컴파일 시간을 지배하므로(3개 병렬 CPU 실행에서
61분 시점에 아직 76%) 통계를 재사용할 수 있으면 매트릭스 전체가 크게 빨라진다.

```bash
# 1) 통계 저장하며 컴파일
python -m pe_npu.compile --mode compile --quant w4a16 --scheme single --device gpu \
  --calib-data-path download/calib_coco_hwc \
  --calib-stats-save out/calib_stats.bin --save out/stats_save.mxq

# 2) 통계 재사용
python -m pe_npu.compile --mode compile --quant w4a16 --scheme global4 --device gpu \
  --calib-data-path download/calib_coco_hwc \
  --calib-stats-load out/calib_stats.bin --save out/stats_load.mxq
```

확인할 것:

- [ ] 2) 가 1) 보다 **실제로 빨라지는가** (`compile_seconds` 비교)
- [ ] 산출물 cos 가 통계 없이 뽑은 것과 같은가 (W4A16 `none` 기준 **0.9126**)
- [ ] 출력 경로가 `out/stats_load.mxq` 로 정상화되는가
      — qbcompiler 가 `_0.9999` 접미사를 붙이는 quirk 를 코드가 rename 으로 보정한다(`_normalize_stats_output`).
        `out/stats_load_0.9999.mxq` 가 남아 있으면 그 보정이 실패한 것이다

> **제약**: `--calib-stats-load` 는 `--optq` 와 병용할 수 없다(코드가 막는다).
> 1.1.2 통계에 OPTQ Hessian 이 없어서다. 1.2.0 에서 이 제약이 풀렸는지도 함께 확인할 가치가 있다
> (풀렸다면 `_validate_calibration_strategy` 를 버전 조건부로 완화). 현재로선 튜닝 없는 조합에만 쓸 수 있다.

### 검증 결과에 따른 분기

| 결과 | 다음 |
| --- | --- |
| ~~A 통과~~ | **완료(2026-09-03)** — 매트릭스를 CLI 로 돌려도 된다 |
| B 통과 | `--quant w4a8 --a16` 조합을 CLI 로 써도 된다 |
| B 실패 | W4A8 은 `compile_quant_tuning_matrix.py` 로만 (A16 5개 하드코딩 보유) |
| C 통과 | `none` 조합 12개에 통계 재사용 → 시간 단축 |
| C 실패 | 그냥 전부 풀 캘리브레이션. 기능은 별도 수정 대상 |

~~**남은 것은 B·C 뿐이다.**~~ → **B·C 완료(2026-09-04, 위 각 섹션 결과 참조).** 남은 것은 NPU 서버의 cos 확정뿐.

---

## 3. 매트릭스 컴파일 (24조합)

```bash
mkdir -p out/matrix_120

for Q in W8A16 W4A16 W4A8_L5A16; do
  for T in none sws_optq; do
    for S in single multi global4 global8; do
      python reports/scripts/compile_quant_tuning_matrix.py \
        --quant $Q --tuning $T --scheme $S --device gpu \
        --calib download/calib_coco_hwc \
        --save out/matrix_120/pe_${Q}_${T}_${S}.mxq
    done
  done
done
```

컨테이너 안에서 돌린다(`/workspace` = repo root). `--target-device` 는 **지정하지 않는다** — 자동 판별된다.

### 병렬 실행 시 주의 ★

**스레드를 나누지 않고 병렬로 돌리면 오히려 느려진다.** 실측: 96코어에서 4개 병렬(각 176스레드)
= 704스레드 → 컨텍스트 스위치 6만/s, 건당 5배 느려짐. RAM 은 문제가 아니었다.

병렬로 돌리려면 잡당 스레드를 코어수/잡수 이하로 제한한다.

```bash
OMP_NUM_THREADS=30 MKL_NUM_THREADS=30 python reports/scripts/compile_quant_tuning_matrix.py ...
```

---

## 4. NPU 서버로 넘겨 cos 측정

컴파일 서버에 NPU 가 없으면 정확도를 못 잰다. mxq 를 NPU 서버로 옮긴다.

### 4-1. mxq 옮기기

```bash
# (NPU 서버에서) GPU 서버로부터 직접 받는다 — HF 왕복보다 빠르다
rsync -av <gpu-server>:~/AX_NPU/out/matrix_120/ ./out/matrix_120/
```

**파일명 규칙**: 검증 스크립트가 파일명(또는 경로)에서 조합을 읽는다. §3 의 루프가 만드는
`pe_<quant>_<tuning>_<scheme>.mxq` 형식을 유지하면 된다.
`<quant>/<tuning>/<scheme>/pe_full.mxq` 디렉토리 구조도 인식한다.

```
pe_W4A16_sws_optq_single.mxq   →  quant=W4A16  tuning=sws_optq  scheme=single
```

### 4-2. 검증 실행 (NPU 서버)

```bash
python reports/scripts/verify_quant_tuning_matrix.py \
  --src-dir out/matrix_120 --device 0 \
  --coco-dir download/coco/val2017 \
  --channels 1,4,8,12,16,20 \
  --out reports/assets/verify_matrix_120.json
```

- 참조 임베딩(원본 PyTorch)은 `--ref-cache`(기본 `/tmp/pe_ref_emb.npy`)에 캐시돼 재사용된다.
- **§5-1 의 회귀 기준값 4점을 스크립트가 자동으로 대조한다.** 기준과 ±0.005 넘게 벗어나면
  `⚠` 로 표시하고 **exit code 1** 로 끝난다 → CI/스크립트에서 바로 실패로 잡을 수 있다.
- 마지막에 마크다운 표를 출력하므로 리포트에 그대로 붙일 수 있다.

기대 출력 형태:

```
[done] W4A16      sws_optq  single   cos=0.9642 ...  ✓ 기준 0.9642
[done] W8A16      sws_optq  single   cos=0.9951 ...  ✓ 기준 0.9951
```

### 4-3. HF 업로드 (검증 통과 후에만)

```bash
python setup/upload_tuning_matrix_to_hf.py --src-dir out/matrix_120 --dry-run   # 먼저 확인
python setup/upload_tuning_matrix_to_hf.py --src-dir out/matrix_120
```

> ~~**HF 업로드 전 확인**: 루트 `TUNING_MATRIX.md` 는 현재 **철회 공지**로 바뀌어 있다~~
> → **완료(2026-09-04)**: 1.2.0 재컴파일 24종 업로드 + `TUNING_MATRIX.md` 를 철회 이력이 포함된
> 새 표로 교체했다(`setup/upload_tuning_matrix_to_hf.py` 가 생성). cos 검증 상태가 문서에 명시돼 있다.
> 배포 경로 `<quant>/<scheme>/` 와 루트 `<scheme>/` 는 건드리지 않았다 — 현행 배포본 그대로다.

---

## 5. 테스트 체크리스트 — 무엇이 맞아야 하는가

### 5-1. 회귀 기준값 — 24종 전부 실측 완료 (2026-09-04)

**`verify_quant_tuning_matrix.py` 의 `BASELINE_120` 에 24조합 전부가 들어 있고 자동 대조된다.**
기준 이미지 세트는 **`download/coco/val2017` 앞 20장(sorted)** — 표준 경로라 어느 서버에서든 동일하다.

| 양자화 | 크기 | 튜닝 없음 | +OPTQ+SWS |
| --- | ---: | ---: | ---: |
| W8A16 | 327 MB | 0.9937 | **0.9946** |
| W4A16 | 188 MB | 0.9175 | **0.9654** |
| W4A8 + A16×5 | 188 MB | 0.8884 | **0.9420** |

**코어모드 4종(single/multi/global4/global8)의 cos 는 소수 4자리까지 동일하다** — 코어모드는
정확도가 아니라 지연 특성만 바꾼다. 상세·채널 지연:
[`performance/NPU_pe_quant_tuning_matrix_120.md`](performance/NPU_pe_quant_tuning_matrix_120.md)

허용 오차 ±0.005. **회귀 판정은 cos 와 크기로만 한다.**

> **★ 참조 임베딩 캐시 주의**: cos 는 "이 이미지들"에 대한 값이다. `--ref-cache` 를 다른
> `--coco-dir` 로 재사용하면 **cos 가 조용히 무의미해진다**(실제로 겪었다 — W8A16 none 이
> 0.9937 대신 **0.3396**). 지금은 캐시에 이미지 목록을 함께 저장해 어긋나면 자동 재계산하지만,
> 다른 도구를 쓸 때도 이 함정을 기억할 것.

> **★ W4A8 은 A16 주입을 로그로 확인할 것**: `[quant] activation16=30`(qk16 25 + A16 5)이어야 한다.
> **25면 A16 5개가 빠진 것**이고 cos 가 0.94 → 0.89 로 떨어진다. 초기 기준값 0.8932 가 그 경우였다.

### 5-2. 반드시 확인할 것

- [ ] **`target_device` 가 `aries-rb` 로 자동 해석**되는가 (§2-1)
- [ ] `[qk16] score MatMul **25개**` 가 로그에 찍히는가 — 개수가 다르면 그래프가 달라진 것
- [ ] `[sanity] full output: (1, 1024)` — 출력 shape
- [ ] calib 파일이 **200줄**인가 (`wc -l npy_files.txt`)
- [ ] 크기가 위 표와 맞는가 — W8A16 ≈327MB / W4 계열 ≈188MB.
      **W4 인데 327MB 가 나오면 bit config 가 안 먹은 것**
- [ ] `mxqtool show` 결과가 아래와 같은가

```bash
mobilint-cli mxqtool show out/matrix_120/pe_W4A16_sws_optq_single.mxq
```

| 필드 | 기대값 | 의미 |
| --- | --- | --- |
| `Format Version` | `0x70000` (v7) | 1.1.2 산출물과 동일 → 기존 런타임 호환 |
| `Compiler Version` | `1.2.0.0` | 1.2.0 으로 빌드됐음 |
| `Hardware Version` | **`Aries2`** | `aries-rb` 로 컴파일해도 하드웨어는 Aries2 |
| `Core Mode` | 요청한 scheme | single/multi/global4/global8 |

- [ ] **추론 코드 변경 없이** 로드되는가 — `MXQInferenceFull` 그대로

```python
from pe_npu.inference import MXQInferenceFull
m = MXQInferenceFull(full_mxq_path="pe_W4A16_sws_optq_single.mxq", device_id=0)
```

### 5-3. 새로 얻어야 하는 정보

- [ ] **베이스라인 버전 무관성**을 W8A16·W4A8 에서도 확인 — 지금은 W4A16 하나만 확인됐다
      (0.9110 → 0.9126). W8A16 `none` 이 1.2.0 에서도 ≈0.9937 이어야 한다
- [ ] **코어모드 4종 전부**의 cos — 코어모드는 정확도에 영향이 없어야 하지만 검증된 바 없다
- [ ] 1~20채널 배치 지연 — 배포 결정용 ([`performance/NPU_pe_quant_schemes.md`](performance/NPU_pe_quant_schemes.md) 형식)

---

## 6. 배포 판단 재료

| 후보 | cos | 크기 | 비고 |
| --- | ---: | ---: | --- |
| **현행 배포본** W8A16 `none` | 0.9937 | 327 MB | 기준선 |
| W8A16 `sws_optq` | **0.9951** | 327 MB | 크기·속도 동일한데 정확도만 상승 → **사실상 공짜 교체** |
| W4A16 `sws_optq` | 0.9642 | 188 MB | 크기 −42%, 처리량 +13%. 정확도 −0.03 감수 가능하면 후보 |
| W4A8_L5A16 `sws_optq` | 0.8932 | 188 MB | 크기 이득이 W4A16 과 같은데 정확도만 낮다 → 배포 이유 없음 |

우선순위가 급한 것은 **W8A16 + sws_optq 4모드**다. 전체 24조합보다 이것만 먼저 뽑아
배포본을 갱신하는 편이 실익이 크다.

---

## 7. 알려진 함정

| 증상 | 원인 | 대처 |
| --- | --- | --- |
| `ValueError: Unsupported device name: 'aries2'` | 1.2.0 에 `aries2` 하드코딩 | 해결됨(자동 판별). 옛 스크립트를 쓰면 재발 |
| `ImportError: Fail to import mmc!!` | qbcompiler whl 에 네이티브 mmc 없음 / CPU torch 에 링크 실패 | 반드시 벤더 docker 이미지 사용 |
| `TypeError: only 0-dimensional arrays...` | NumPy 2.x | `numpy<2` (이미지 내부는 1.26) |
| `RuntimeError: No calibration tensors loaded` | `npy_files.txt` 의 경로가 컨테이너에서 안 보임 | 호스트와 **같은 경로**로 마운트 |
| 병렬인데 건당 5배 느림 | 잡당 176스레드 × N > 코어수 | `OMP_NUM_THREADS` 로 제한 (§3) |
| `RuntimeError: shape '[1, 1, 1024]' is invalid` | 패치된 참조 모델은 batch=1 전용 | cos 측정은 1장씩 루프 |
