# Qwen3-VL 용량 재현 (Mobilint `customer-capacity-repro`)

벤더가 전달한 재현 패키지를 **호스트를 건드리지 않고** 돌리기 위한 도커 구성.

- 패키지 원본: `download/vendor/customer-capacity-repro/` (gitignore, 10GB tarball 해제본)
- 배경·결과 해석: [`../../reports/inquiries/06_qwen3vl_prefill_optimization/REPLY_2.md`](../../reports/inquiries/06_qwen3vl_prefill_optimization/REPLY_2.md)

## 왜 도커인가

| 계층 | 위치 | 교체 위험 |
| --- | --- | --- |
| 드라이버 1.13 | **호스트 커널 모듈** | ★ 교체 시 운영 k8s 추론 파드 전부 끊김 |
| 런타임 | 호스트 유저스페이스 (1.2.0) | 덮어쓰면 운영 영향 |
| 재현용 런타임 1.4.0 | **컨테이너** | 영향 없음 |

벤더 README 도 컴파일용·런타임용 **venv 분리**를 요구한다(Torch 2.7.1 vs 2.9.0).
호스트가 Python 3.10 이 아니고 운영 파드가 도는 서버라 도커가 맞다.
**드라이버는 호스트 것을 그대로 쓰고 `/dev/ariesN` 만 넘긴다** — 런타임 1.4.0 이
드라이버 1.13 에서 동작하는 것은 이미 확인했다(`reports/testing/sdk_v11_compat.md`).

## 이 서버에서 가능한 범위

| 단계 | 필요 | 가능 여부 |
| --- | --- | --- |
| **추론 재현** (제공된 MXQ 로 용량 측정) | MLA100 + 드라이버 | ✅ |
| 컴파일 재현 (MBLT → MXQ) | **CUDA 12.8 GPU** | ❌ GPU 없음 → GPU 서버 |

제공 MXQ 4개(`reference/mxq/`)는 `Format 0x70000(v7)` / `Hardware Aries2` 로
**드라이버 1.13 과 호환**된다(확인 완료). 컴파일 없이 바로 추론 대조가 가능하다.

## 사용

```bash
# 1) 이미지 빌드 (빌드 컨텍스트 = 재현 패키지)
docker build -f setup/vlm_repro/Dockerfile.runtime -t mblt_vlm_repro:rt \
    download/vendor/customer-capacity-repro

# 2) run_table.py 에 --device 인자 추가 (원본은 .orig 로 보존)
python setup/vlm_repro/apply_patch.py

# 3) 무결성 확인
bash setup/vlm_repro/run_runtime.sh --shell
  # 컨테이너 안에서: python3.10 scripts/verify_files.py

# 4) 스모크 (30회 용량검증 아님)
bash setup/vlm_repro/run_runtime.sh --sizes 224x224 --repetitions 2 --output results/smoke

# 5) 특정 해상도만
bash setup/vlm_repro/run_runtime.sh --sizes 224x224 320x320 --output results/sel
```

**카드 지정**: 기본 `1 7` (이 서버에서 비어 있는 카드. 나머지는 운영 파드 점유).

```bash
NPU_DEVICES="7" bash setup/vlm_repro/run_runtime.sh ...      # 7번만
NPU_DEVICES="1 7" RUN_DEVICE=7 bash ... run_runtime.sh ...   # 둘 다 넘기고 7번으로 실행
```

## `--device` 패치가 필요한 이유

`run_table.py` 는 `Classifier(device=0)` 로 **0번 카드 고정**이고 CLI 로 바꿀 수 없다.
이 서버는 0번을 운영 파드가 쓰고 있어 그대로는 못 돌린다.
`apply_patch.py` 가 `--device` 인자를 추가한다(되돌리기: `--revert`).

## 주의 — 벤더가 밝힌 한계

> 기존 표 전체의 대수를 재컴파일 모델로 재현한 상태는 아닙니다.
> 일부 행에서 간헐적 지연 초과, 64×64 에서 정답 불일치가 있습니다.

그래서 **비교 기준용 원본 MXQ 4개**를 함께 받았다. 재컴파일본은 바이트 일치나
동일 ms 를 보장하지 않는다. 먼저 **원본 MXQ 로 표를 대조**하는 것이 맞다.
