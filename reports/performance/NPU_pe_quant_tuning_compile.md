# [컴파일] PE 양자화×튜닝 매트릭스 24종 GPU 컴파일 스윕

**결론 요약**
- **W8A16 / W4A16 / W4A8_L5A16 × 튜닝 4종(none/SWS/OPTQ/SWS+OPTQ) × single/global4 = 24조합 전부 컴파일 성공.**
  전량 HF `PIA-SPACE-LAB/MXQ_NPU`의 **`<quant>/<tuning>/<scheme>/pe_full.mxq`** 로 업로드
  (기존 배포 경로 `<quant>/<scheme>/`는 유지). 상세 표는 HF 루트 [`TUNING_MATRIX.md`](https://huggingface.co/PIA-SPACE-LAB/MXQ_NPU/blob/main/TUNING_MATRIX.md).
- **GPU가 정상이면 OPTQ+SWS 풀옵션도 조합당 5~8분** (calibration 200장 ≈ 3분).
  기존 CPU 서버 실측(단독 78분, +SWS 118분, +OPTQ 3~5h 예상 — `NPU_pe_quant_schemes.md` §4)이 GPU에서 해소됨.
- md5 24개 전부 상이 → 튜닝 옵션이 실제로 가중치를 바꿨다는 바이너리 수준 증거.
  (동일 조건 tuned vs plain 직접 비교에서도 md5 상이 확인)
- ~~cos 정확도는 미검증~~ → **검증 완료(2026-09-02)**: [`NPU_pe_quant_tuning_verify.md`](NPU_pe_quant_tuning_verify.md).
  결과: **모든 양자화에서 튜닝 없음이 최선**(OPTQ·SWS가 9/9 케이스에서 cos 악화), W4A8_L5A16은 붕괴가 아니라 **cos 0.8790**(앞선 0.2609는 A16 이름 지정 오류였다).
- (원문) cos 정확도는 미검증 — 이 서버(NPU 없음)에선 불가. NPU 서버에서
  `reports/scripts/bench_quant_schemes.py` 패턴으로 24종 검증 후 좋은 조합을 기존 경로로 승격할 것.
  관전 포인트: W4A16+SWS+OPTQ가 cos 0.9135(무보정)를 얼마나 회복하는지, SWS 단독 악화(0.8786)가 OPTQ 병용으로 반전되는지(문의 05 가설).

측정: 2026-09-02 / **NPU 없는 GPU 서버에서 컴파일만** 수행.

## 컴파일 환경

| 항목 | 값 |
|---|---|
| Host | Ubuntu 24.04 (kernel 6.8), Intel Xeon 6530P 128스레드, RAM 251GB |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition 96GB × 2, **driver 580.173.02** |
| Docker 이미지 | `mobilint/qbcompiler:1.1-cuda12.8.1-ubuntu22.04` (Ubuntu 22.04.5, Python 3.10.12) |
| 컴파일러 | **qbcompiler 1.1.2** (+aries2 whl, SDK 번들 **1.0v**) / torch 2.7.1+cu128 / numpy 1.26.0 |
| calib | COCO val2017 앞 200장(sorted), `pe_npu.preprocess.preprocess_image`(336², /255, mean·std 0.5), HWC float32 |
| CalibrationConfig | method=1, output=1(per-channel), mode=1, max_percentile(0.9999, topk 0.01) |
| 공통 | QKᵀ score MatMul 25개 16bit override(full 모드 필수), target aries2, `--device gpu` |
| 실행 | 컨테이너 4개 병렬(GPU0×1 — vLLM 88GB 상주와 공유, GPU1×3), 잡당 OMP 32스레드 |

## 조합 정의 (Mobilint inquiry 04 예제 설정 준수)

- **W8A16**: act(q8,k8,v8,head8,out16,ffn16) / weight 전부 8 — 기존 배포본과 동일 비트
- **W4A16**: act 동일 / weight(q4,k4,v8,out4,ffn4,head4)
- **W4A8_L5A16**: act 전부 8 + outlier 상위 5텐서만 16bit / weight W4와 동일.
  5텐서 = 기존 프로파일링(`NPU_pe_quant_schemes.md` §W4A8) 상위 5의 mblt 이름
  (`visual_transformer_resblocks_{3,12,10}_mlp_c_proj`, `..._{12,10}_mlp_c_fc/reshape/gelu_0` — parse `--dump-names`로 도출)
- **sws** = SearchWeightScale(apply, q/k/v/out/ffn 전부) · **optq** = OPTQ(actOrder, blockSize 128, percDamp 0.01)

## 컴파일 시간

| 조건 | 조합당 소요 |
|---|---:|
| **GPU 정상** (튜닝 유무 무관) | **5~8분** |
| CPU 폴백(아래 참조) | 50~60분 |
| (참고) 기존 CPU 96코어 서버 | 78분~3h+ |

주의 — 스윕 중 절반이 50~60분 걸렸는데 **설정 차이가 아니라 컨테이너가 도중 GPU 접근을 잃어
CPU 폴백**으로 돈 것(calibration 200장 47분 vs GPU 3분, 로그로 확인. `torch.cuda.is_available()=False`,
"No CUDA GPUs are available" — 장기 실행 컨테이너에서 cgroup 갱신 시 GPU 디바이스가 빠지는 알려진
nvidia-container 현상으로 추정). **산출물은 디바이스와 무관하게 동일 로직**이라 유효하다.
잡별 시간·md5 전체 표는 HF `TUNING_MATRIX.md` 참조.

## 재현

```bash
# 환경: qbcompiler cuda 이미지 + whl 설치 + calib 생성 (tutorial/pe_npu/README.md 참조)
python -m pe_npu.calib --dataset coco --src download/coco/val2017 --num 200 --out download/calib_coco_hwc --hwc

# 1조합 컴파일 (컨테이너 안, /workspace=repo root)
python reports/scripts/compile_quant_tuning_matrix.py \
  --quant W4A16 --tuning sws_optq --scheme single --device gpu \
  --calib download/calib_coco_hwc --save out/pe_W4A16_sws_optq_single.mxq

# HF 업로드 (24종 일괄 + TUNING_MATRIX.md 생성)
python setup/upload_tuning_matrix_to_hf.py --src-dir <mxq 디렉토리>

# 검증용 다운로드 (NPU 서버)
python -c "from huggingface_hub import hf_hub_download; \
  print(hf_hub_download('PIA-SPACE-LAB/MXQ_NPU', 'W4A16/sws_optq/single/pe_full.mxq'))"
```

관련: [`NPU_pe_quant_schemes.md`](NPU_pe_quant_schemes.md)(무보정 3스킴 NPU 실측),
[`../inquiries/04_vit_quantization_speed/REPLY.md`](../inquiries/04_vit_quantization_speed/REPLY.md)(벤더 예제),
[`../inquiries/05_a16_selection_followup/EMAIL.md`](../inquiries/05_a16_selection_followup/EMAIL.md)(SWS 단독 악화 문의)
