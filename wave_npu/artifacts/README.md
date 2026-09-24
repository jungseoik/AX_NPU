# artifacts/ — 결정 산출물 (git 추적)

실험의 **결론**만 둔다. 재생성 가능한 무거운 캐시는 `../cache/`(gitignore)에 있다.

| 파일 | 내용 |
|---|---|
| `mask_mean.npz` | 채택된 프롬프트 선택(16,125 → 13개) + 탐색 상태(`state` JSON 문자열) |
| `mask_percat.npz` | 카테고리별 독립 선택(참고 — 단일 마스크와 동점이라 미채택) |
| `combo_24fps.json` | 카테고리별 최종 규칙·평활창·임계값 (**배포 설정**) |
| `final_24fps.json` / `final_percat_24fps.json` | 최종 F1 (in-sample + held-out) |
| `pipeline_4cat.json` | 4종 스펙 파이프라인 결과 |
| `baseline.json` | 규칙 × 평활창 × 보정 전수 스윕 원자료 |

`cache/`는 다음으로 재생성한다(NPU 4장 기준 임베딩 22분 + 텍스트 6분 + 사람검출 4분):

```bash
PY=~/miniconda3/envs/pe_npu_host/bin/python
$PY -m wave_npu.embed --device-ids 1,3,4,5 --batch 128
$PY -m wave_npu.text
$PY -m wave_npu.person --fps 0 --device-ids 1,3,4,5
```
