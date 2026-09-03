# 양자화 튜닝(OPTQ·SWS) — qbcompiler 1.1.2 vs 1.2.0

> **결론 먼저**: 이전 리포트 [`NPU_pe_quant_tuning_verify.md`](NPU_pe_quant_tuning_verify.md) 의
> **"OPTQ·SWS 는 쓰지 말 것"** 은 **qbcompiler 1.1.2 한정 결론이었고, 1.2.0 에서는 뒤집힌다.**
> 1.1.2 에서 튜닝은 정확도를 **떨어뜨렸고**, 1.2.0 에서는 **올린다**. 베이스라인(튜닝 없음)은
> 두 버전이 동일하므로, 차이는 전적으로 **튜닝 구현**에서 나온다.

| 항목 | 내용 |
| --- | --- |
| 측정일 | 2026-09-03 |
| 컴파일 | qbcompiler **1.2.0** (SDK 1.1v), docker `mobilint/qbcompiler:1.2-cpu-ubuntu22.04`, CPU |
| 타겟 | `--target-device aries-rb` (1.2.0 에는 `aries2` 이름이 없음) |
| 추론 | 드라이버 1.13 / 런타임 1.2.0, `/dev/aries7` |
| calib | COCO val2017 200장 (1.1.2 측정과 동일) |
| 정확도 | 원본 PyTorch 임베딩 대비 cos, 20장 평균 |
| 원자료 | [`../assets/pe_quant_tuning_compiler120.json`](../assets/pe_quant_tuning_compiler120.json) |

## 1. 버전별 비교 (single, OPTQ + SearchWeightScale)

| 양자화 | 1.1.2 튜닝없음 | 1.1.2 +OPTQ+SWS | **1.2.0 +OPTQ+SWS** | 1.1.2 튜닝효과 | **1.2.0 튜닝효과** |
| --- | ---: | ---: | ---: | ---: | ---: |
| W8A16 | 0.9937 | 0.9747 | **0.9951** | −0.019 | **+0.001** |
| W4A16 | 0.9110 | 0.8795 | **0.9642** | −0.032 | **+0.052** |
| W4A8 + A16×5 | 0.8790 | 0.8560 | **0.8932** | −0.023 | **+0.014** |

**튜닝 효과의 부호가 전 조합에서 뒤집힌다.**

## 2. 통제군 — 베이스라인은 두 버전이 같다

튜닝을 끈 상태로 1.2.0 에서 다시 컴파일해 비교했다.

| W4A16, 튜닝 없음 | cos | 크기 | 처리량 |
| --- | ---: | ---: | ---: |
| qbcompiler 1.1.2 | 0.9110 | 188.8 MB | 19.79 img/s |
| qbcompiler 1.2.0 | **0.9126** | 188.5 MB | 19.23 img/s |

차이 **+0.0016** — 측정 노이즈 수준이다. 즉 1.2.0 이 전반적으로 좋아진 것이 아니라,
**1.1.2 의 OPTQ/SearchWeightScale 이 제대로 동작하지 않았던 것**이다.
24조합에서 관측했던 "튜닝 켜면 9/9 악화"는 모델 특성이 아니라 그 버그를 24번 관측한 것이었다.

## 3. 1.2.0 실측 전체

| 양자화 | 튜닝 | 크기 | cos 평균 | cos 최저 | 처리량(single) |
| --- | --- | ---: | ---: | ---: | ---: |
| W8A16 | +OPTQ+SWS | 326.9 MB | **0.9951** | 0.9715 | 14.19 img/s |
| W4A16 | 없음 | 188.5 MB | 0.9126 | 0.8564 | 19.23 img/s |
| W4A16 | +OPTQ+SWS | 188.5 MB | **0.9642** | 0.9401 | 22.36 img/s |
| W4A8 + A16×5 | +OPTQ+SWS | 188.0 MB | 0.8932 | 0.7833 | 22.18 img/s |

## 4. 벤더 회신 수치와의 대조

문의 04 회신에서 벤더가 제시한 값이 **1.2.0 에서 재현된다**.

| 조합 | 벤더 제시 | 우리 1.1.2 | **우리 1.2.0** | 잔차 |
| --- | ---: | ---: | ---: | ---: |
| W4A16 +OPTQ+SWS | 0.973 | 0.8795 | **0.9642** | −0.009 |
| W4A8+A16×5 +OPTQ+SWS | 0.905 | 0.8560 | **0.8932** | −0.012 |

잔차 0.01 안쪽은 calibration 데이터 구성 차이로 설명되는 범위다.
**재현 실패의 원인은 우리가 쓰던 컴파일러 버전이었다.**

## 5. mxq 호환성 — `aries-rb` = Aries2

1.2.0 은 `aries2` 라는 타겟 이름을 받지 않는다(`ValueError: Unsupported device name`).
사용 가능한 이름은 `regulus-ra`, `aries-rb`, `regulus-rb` 이고, **`aries-rb` 가 Aries2 를 가리킨다.**

```
$ mobilint-cli mxqtool show W4A16_sws_optq_c120.mxq
Format Version:      0x70000        # v7, 1.1.2 산출물과 동일
Compiler Version:    1.2.0.0
Hardware Version:    Aries2         # aries-rb 로 컴파일했으나 하드웨어는 Aries2
```

따라서 **1.2.0 으로 컴파일한 mxq 가 드라이버 1.13 + 런타임 1.2.0 환경에서 그대로 로드·추론된다**(위 측정 전부가 그 환경).
SDK 버전을 올리지 않아도 컴파일러만 1.2.0 을 쓰면 이득을 취할 수 있다.

> **해결됨(2026-09-03)**: 이전에는 `pe_npu/compile.py` 가 `aries2` 를 하드코딩해(4곳) 1.2.0 에서 바로 실패했다.
> 지금은 **`pe_npu/target_device.py` 가 `qbcompiler.__version__` 으로 자동 판별**한다
> (1.1.x→`aries2` / 1.2.x→`aries-rb`). 우선순위: `--target-device` > 환경변수 `AX_NPU_TARGET_DEVICE` > 자동 감지 > `aries2` 폴백.
> `pe_npu/compile.py`·`yolo_npu/compile.py`·`reports/scripts/compile_quant_tuning_matrix.py` 전부 연결됐고
> 두 컨테이너에서 확인했다(1.1.2→aries2 / 1.2.0→aries-rb, 1.2.0 자체 `validate_target_device` 통과).
> **추론 쪽은 변경 없음** — `pe_npu/inference.py` 에는 버전·타겟 의존 코드가 없다.

## 6. 컴파일 비용

| 실행 형태 | 스레드 | 소요 |
| --- | ---: | ---: |
| 단독 | 176 | 3851s (64분) |
| 3개 병렬 | 30 × 3 | 약 3시간 (건당) |

이 서버에는 GPU 가 없어(nvidia-smi 실패) 전부 CPU 컴파일이며, 1.2-cuda 이미지도 받지 않았다.
GPU 서버에서는 조합당 5~8분이므로, 전체 재검증은 GPU 서버에서 도는 편이 훨씬 낫다.

## 7. 후속 과제

1. **1.2.0 기준 전체 매트릭스 재측정** — 이 리포트는 4조합만 확인했다. 배포 스킴을 다시 고르려면
   양자화 3종 × 튜닝 4종 × 코어모드 4종을 1.2.0 으로 다시 돌려야 한다(GPU 서버 권장).
2. **배포본 재검토** — W4A16+OPTQ+SWS 가 cos 0.9642 / 크기 −42% / 처리량 +13% 로
   실사용 후보가 되었다. 현행 배포본은 W8A16(0.9937).
   W8A16+OPTQ+SWS(0.9951) 는 현행보다 정확도가 더 높다.
3. ~~`pe_npu/compile.py` 의 `target_device` 인자화~~ → **완료**. `setup/sdk_versions.json` 에도
   버전별 매핑(`compile.target_device`)을 넣었다.
4. **재현 절차서**: [`../RUNBOOK_quant_matrix_120.md`](../RUNBOOK_quant_matrix_120.md) —
   GPU 서버에서 clone 후 그대로 따라가면 24조합이 재현되고, 검증 스크립트가 위 4점을 회귀 기준으로 자동 대조한다.
