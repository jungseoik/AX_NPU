# PE-Core-L14-336 컴파일 시간 측정

MXQ 컴파일(NPU용 바이너리 생성)에 걸리는 시간을 환경/조건 명시해 기록한다.

## 먼저 — 컴파일은 NPU에서 하지 않는다

흔한 오해. **컴파일(MXQ 생성)은 호스트의 CPU 또는 GPU에서 qbcompiler가 수행**하고,
NPU(aries)는 **추론 전용**이다. `--device {cpu|gpu}`는 컴파일 과정(calibration/양자화 연산)을
어디서 돌릴지를 정하는 것이지, NPU에서 컴파일하는 게 아니다.

```
[원본 PE 모델] --(qbcompiler, 호스트 CPU 또는 GPU)--> [MXQ] --(qbruntime, NPU)--> 추론
                         ↑ 여기가 컴파일                          ↑ 여기가 NPU
```

## 측정 환경

| 항목 | 값 |
|------|-----|
| CPU | Intel Core Ultra 9 285K (24 threads) |
| RAM | 125 GiB |
| GPU | NVIDIA RTX PRO 6000 Blackwell (96 GB) |
| OS / 커널 | Ubuntu, Linux 6.8.0-41-generic |
| 컴파일러 | qbcompiler 1.1.2 (docker `mobilint/qbcompiler:1.1-cuda12.8.1-ubuntu22.04`) |
| 대상 | PE-Core-L14-336 vision encoder, **feat trunk**(24 transformer block), 385 GOPS |
| 코어 모드 | single, 입력 336×336 |

## 결과 — GPU vs CPU 컴파일 시간

| `--device` | 컴파일 소요 | 비고 |
|:---:|:---:|---|
| **gpu** | **66 초** | calibration 연산을 GPU에서 |
| **cpu** | **70 초** | 전부 CPU에서 |

(random calibration 1샘플 기준, wall-clock, 컴파일 성공 확인. "Compilation was successful")

### 분석: GPU 가속 효과가 작은 이유
- GPU(66s)와 CPU(70s) 차이가 **6%에 불과**하다.
- 컴파일 시간의 대부분은 **weight 양자화 + 그래프 컴파일/배치**(주로 CPU 작업)이고,
  GPU가 쓰이는 **calibration 연산 비중이 작다**. 특히 위 측정은 random calib 1샘플이라
  calibration 단계 자체가 짧다.
- → **실제 calibration 데이터가 많을수록(예: COCO 200장) GPU 효과가 커질 수 있다**(미측정).
  calib 샘플 수에 비례해 forward 연산이 늘고, 그 부분이 GPU 가속 대상이기 때문.

### 참고: 코어 모드별 (이전 측정, GPU)
| 컴파일 | 대략 소요 |
|---|---|
| feat trunk, single | ~66 초 |
| feat trunk, multi | ~70 초 (network load~export 구간 로그 기준 유사) |

## 결론 / 권고

- **컴파일은 1분 남짓**(PE feat trunk 기준). GPU/CPU 차이는 random calib에선 미미(6%).
- calibration 데이터가 많으면 GPU(`--device gpu`)가 유리할 수 있으니, 실 calib 컴파일은 GPU 권장.
- **컴파일은 한 번만 하면 된다** — MXQ는 aries2 바이너리라 재사용/배포 가능
  (옵션 B: HF `PIA-SPACE-LAB/MXQ_NPU`). 매번 컴파일할 필요 없음. (`reports/performance/NPU_batch_latency.md` 참고)

## 재현

```bash
docker exec -w /workspace/AX_NPU mblt_compiler bash -lc '
  export LD_LIBRARY_PATH=/tmp/qbruntime_aries2-v4_v1.2.0_amd64/qbruntime/qbruntime/lib:$LD_LIBRARY_PATH
  for DEV in gpu cpu; do
    t0=$(date +%s)
    python -m pe_npu.compile --mode compile --save /tmp/t_$DEV.mxq --feat-only --device $DEV >/tmp/log_$DEV 2>&1
    echo "$DEV: $(( $(date +%s) - t0 ))s  성공=$(grep -ac "successful" /tmp/log_$DEV)"
    rm -f /tmp/t_$DEV.mxq
  done'
```
