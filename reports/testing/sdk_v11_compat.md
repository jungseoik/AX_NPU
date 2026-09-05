# SDK 1.1v 호환성 점검 — 기존 MXQ 자산·컴파일 경로

**결론: HF에 올려둔 컴파일 자산(mxq)은 SDK 1.1v(런타임 1.4.0)에서 재컴파일 없이 그대로 동작한다.**

측정일 2026-09-01 / 호스트: 드라이버 1.13, 펌웨어 1.2.5, NPU 카드 1장(`/dev/aries7`), GPU 없음.

## 0. ★ mxq × 실행환경 호환 매트릭스 (2026-09-04 갱신)

**어떤 컴파일러로 만든 mxq가 어떤 실행환경에서 도는가.** 나중에 판단 근거로 쓰려고 남긴다.

| mxq 빌드 ↓ / 실행환경 → | 드라이버 1.13 + 런타임 1.2.0 | 드라이버 1.13 + 런타임 1.4.0 | 드라이버 1.14 + 런타임 1.4.0 |
| --- | :---: | :---: | :---: |
| **qbcompiler 1.1.2** (target `aries2`, SDK 1.0v) | ✅ 검증 | ✅ 검증 (cos 0.9927) | ❔ 미검증 |
| **qbcompiler 1.2.0** (target `aries-rb`, SDK 1.1v) | ✅ 검증 (24종, 2026-09-04) | ❔ 미검증 | ❔ 미검증 |

**왜 되는가 — `aries-rb` 는 Aries2 의 새 이름일 뿐이다.**
두 빌드의 산출물 헤더가 같다:

```
Format Version:    0x70000      # v7, 두 빌드 동일
Hardware Version:  Aries2       # aries-rb 로 컴파일해도 Aries2
```

하드웨어도 포맷도 그대로라 **구 런타임이 신 컴파일러 산출물을 그냥 읽는다.**

**검증 내역**

| 항목 | 근거 |
| --- | --- |
| 1.2.0 빌드 × 드라이버 1.13/런타임 1.2.0 | 양자화3×튜닝2×코어모드4 = 24종 전부 정상, cos 회귀기준 일치 → [`../performance/NPU_pe_quant_tuning_matrix_120.md`](../performance/NPU_pe_quant_tuning_matrix_120.md) |
| 1.1.2 빌드 × 런타임 1.4.0 | 아래 §1 (cos 0.9927, 처리량 −2.3%) |
| 배치 출력 무결성 | 1.2.0 빌드 72/72 비트 동일 → [`../performance/NPU_pe_quant_tuning_matrix_120.md`](../performance/NPU_pe_quant_tuning_matrix_120.md) §7 |

**미검증 3칸의 이유**: 이 서버에 드라이버 1.14 를 올릴 수 없다. 운영 k8s 추론 파드가 카드를
물고 있어 커널 모듈을 교체하면 서비스가 끊긴다. §1 의 런타임 1.4.0 테스트도 드라이버는 그대로 두고
`LD_LIBRARY_PATH` 로 유저스페이스에서만 바꿔 확인한 것이다.

**실무 결론**: **SDK 를 1.1v 로 올리지 않아도 컴파일러만 1.2.0 을 쓰면 된다.**
튜닝 이득(W8A16 0.9937→0.9946, W4A16 0.9110→0.9654)을 드라이버 교체 없이 가져갈 수 있다.
1.1v 전체(드라이버 1.14 + 런타임 1.4.0)를 갖춘 서버가 생기면 나머지 3칸을 채울 것.

---

## 1. 기존 MXQ가 신규 런타임에서 동작하는가 — 동작한다

동일 파일(`PIA-SPACE-LAB/MXQ_NPU` → `single/pe_full.mxq`, PE-Core-L14-336 full)을 런타임만 바꿔 측정.
정확도 = 원본 PyTorch 임베딩 대비 코사인 유사도, 처리량 = 단일 카드 동기 infer(single 코어모드).

| 런타임 | 드라이버 | cos 평균 | 처리량 | 판정 |
| --- | --- | ---: | ---: | --- |
| 1.2.0 (SDK 1.0v) | 1.13 | 0.9936 (20장) | 15.73 img/s | 기준선 |
| **1.4.0 (SDK 1.1v)** | 1.13 | **0.9927 (8장)** | **15.37 img/s** | **동일 수준 — 재컴파일 불필요** |

- cos 차이 0.0009, 처리량 차이 2.3%로 측정 분산 범위. **MXQ 포맷 하위호환이 유지된다.**
- 검증 방식: 드라이버는 **교체하지 않고**(운영 파드가 카드 사용 중) 런타임 1.4.0을 tar에서 풀어
  `LD_LIBRARY_PATH` + cp311 wheel로 유저스페이스에서만 로드. 즉 **런타임 1.4.0 ↔ 드라이버 1.13 조합도 동작**.
- 따라서 `MXQInferenceFull.from_hf()` / `YOLONPU.load()` 흐름은 SDK를 올려도 그대로 쓸 수 있다.

> 드라이버 1.14로 올린 뒤의 조합은 미검증(모듈 교체가 운영 중단을 유발해 수행하지 않음).
> 신규 서버에서 1.1v 전체(드라이버 1.14 + 런타임 1.4.0)로 세팅할 때 재확인할 것.

## 1-b. ★ SDK 1.1v 전환 시 걸리는 것 — 타겟 디바이스 이름이 바뀌었다

qbcompiler **1.2.0은 `aries2` 타겟을 더 이상 받지 않는다.**

```
ValueError: Unsupported device name: 'aries2'.
Available devices: ['regulus-ra', 'aries-rb', 'regulus-rb']
```

| 컴파일러 | 인식하는 타겟 이름 |
| --- | --- |
| 1.1.2 (SDK 1.0v) | `aries2` |
| 1.2.0 (SDK 1.1v) | `aries-rb` / `regulus-ra` / `regulus-rb` — **`aries2` 없음** |

- `aries-rb` 가 우리 카드 계열이다. 최신 `mblt-sdk-tutorial` 예제가
  `--target-device {regulus-rb, aries-rb}`(기본 `aries-rb`)로 **ARIES / REGULUS 제품군**을 가른다.
- **`pe_npu/compile.py` 는 `target_device="aries2"` 를 4곳에 하드코딩**하고 있다.
  SDK 1.1v로 올리면 컴파일이 전량 실패하므로, 버전별 매핑이 필요하다(미적용).
- 부수 정황: 벤더가 보낸 양자화 예제의 기본값도 `--target-device aries-rb` 였다
  → **그 예제가 1.2.0 기준으로 작성됐을 가능성**. 문의 05의 "컴파일러 버전" 질문과 연결된다.

### 현재 배포본의 MXQ 정보 (`mobilint-cli mxqtool show`)

```
Format Version:     0x70000     → MXQ v7
Compiler Version:   1.1.2.0
Hardware Version:   Aries2
```

`docs/compatibility.md` 기준 런타임 1.0.0~latest 가 MXQv1~v7 을 지원하므로 현재 조합은 정상 범위다.
**1.2.0 산출물이 MXQ v8을 내면 런타임도 함께 올려야 한다** — 1.2.0 컴파일 성공 후 같은 명령으로 확인할 것.

## 2. 컴파일 이미지 — GPU 없어도 된다

벤더는 컴파일러 버전별로 `-cpu` / `-cuda` 이미지를 **쌍으로** 배포한다(Docker Hub `mobilint/qbcompiler`, 38개 태그).
컴파일 코드가 `torch.cuda.is_available()`로 CPU에 자동 폴백한다(`mblt-sdk-tutorial` 예제 README 명시).

| SDK 번들 | 컴파일러 | cpu 이미지 | cuda 이미지 |
| --- | --- | --- | --- |
| 1.0v | 1.1.2 | `qbcompiler:1.1-cpu-ubuntu22.04` (**14.2GB**) | `1.1-cuda12.8.1-ubuntu22.04` (**27.7GB**) |
| 1.1v | 1.2.0 | `qbcompiler:1.2-cpu-ubuntu22.04` | `1.2-cuda12.8.1-ubuntu22.04` |

호스트에 권장되는 이미지는 `python setup/sdk_resolve.py --sdk <버전>` 이 GPU 유무를 보고 알려준다.

### 왜 호스트 직접 설치는 안 되는가 (실측)

| 시도 | 결과 |
| --- | --- |
| 1.0v 컴파일러(1.1.2) 호스트 conda 설치 | `ImportError: Fail to import mmc!!` — whl에 mmc 네이티브 모듈이 아예 없다 |
| 1.1v 컴파일러(1.2.0) 호스트 conda 설치 + CPU torch | whl에 mmc는 있으나(`mmc.cpython-310-x86_64-linux-gnu.so`, 38MB) `libtorch_cuda.so` 없음으로 import 실패 |
| 〃 + pip CUDA torch 설치 시도 | `libcupti.so.12` 등 CUDA 라이브러리 체인을 계속 요구 → 중단 |
| **벤더 cpu 이미지(1.1-cpu)** | **mmc 정상 import** — 이 이미지 내부 torch가 **CUDA 빌드(2.7.1+cu128)** 라 링크가 충족된다 |

즉 `-cpu` 태그는 "GPU 없이 실행 가능"을 뜻하고, 이미지 안의 torch는 CUDA 빌드다.
**결론: 컴파일은 벤더 이미지로 한다. GPU 없는 서버는 `-cpu` 이미지(절반 크기)를 쓴다.**

## 3. 재현

```bash
python setup/sdk_resolve.py --list                     # 번들 목록·권장 이미지
python setup/fetch_sdk_from_hf.py --sdk 1.1            # 1.1v SDK 받기
# 런타임만 유저스페이스로 시험 (드라이버 교체 없이)
tar xzf download/sdk/v1.1/qbruntime_v1.4.0_amd64.tar.gz -C /tmp/rt14
pip install --force-reinstall --no-deps /tmp/rt14/qbruntime_v1.4.0_amd64/qbruntime/qbruntime/python/*cp311*.whl
export LD_LIBRARY_PATH=/tmp/rt14/qbruntime_v1.4.0_amd64/qbruntime/qbruntime/lib:$LD_LIBRARY_PATH
python -c "from pe_npu.inference import MXQInferenceFull; m=MXQInferenceFull.from_hf(scheme='single', device_id=7)"
```

- 관련: SDK 번들 정의 `setup/sdk_versions.json`, 세팅 `bash setup/setup_all.sh --sdk 1.0|1.1`
- 벤더 회신(양자화·컴파일 권고): [`../inquiries/04_vit_quantization_speed/REPLY.md`](../inquiries/04_vit_quantization_speed/REPLY.md)
