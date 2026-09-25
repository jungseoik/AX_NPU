# wave_npu/docs — 이 워크스트림의 문서

`wave_npu` 는 **TTA 인증용 데이터셋 평가**를 위한 별도 워크스트림이다. 운영 추론 경로(`pe_npu`,
`yolo_npu`)와 목적이 다르므로 코드·스펙·산출물·문서를 전부 `wave_npu/` 안에서 관리한다.
레포 전역 인덱스(`reports/`)에는 넣지 않는다.

| 문서 | 내용 |
|------|------|
| [tta_event_f1.md](tta_event_f1.md) | ★ **정확도 근거** — 평가 프로토콜, 점수를 올린 요인, 과적합 검증, intrusion 처리, 원본 대조, 양자화 영향 |
| [tta_cert_runbook.md](tta_cert_runbook.md) | ★ **인증 시험 실행 절차서** — TTA-26-03361 규격 요약, `wave_npu.tta_cert` 사용법, 화면 표시 내용, 산출 파일, 시험 전 체크리스트 |

관련: [`../README.md`](../README.md) (패키지 개요·사용법·새 카테고리 추가 절차),
[`../artifacts/README.md`](../artifacts/README.md) (결정 산출물 목록).
