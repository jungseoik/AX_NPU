# SDK 1.1v 호환성 점검 — 기존 MXQ 자산·컴파일 경로

**결론: HF에 올려둔 컴파일 자산(mxq)은 SDK 1.1v(런타임 1.4.0)에서 재컴파일 없이 그대로 동작한다.**

측정일 2026-09-01 / 호스트: 드라이버 1.13, 펌웨어 1.2.5, NPU 카드 1장(`/dev/aries7`), GPU 없음.

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
