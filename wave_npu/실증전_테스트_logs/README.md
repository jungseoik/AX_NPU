# 실증 전 24시간 테스트 — 1·2차 추론기 로그 분석

> ## ⚠ 정정 고지 (2026-10-13)
>
> **본 문서의 집계는 별첨 `1차추론기 결과.zip` 범위(100건) 기준이며, 시험 구간 전체가 아님.**
> zip 은 수집이 시험 개시 37분 후(10-12 20:32)부터 이루어져 **102건 중 100건만** 수록되어 있음.
> 미수록 2건: `10-12 19:55:33 카메라 837(통과)`, `10-12 19:58:49 카메라 842(기각)`.
>
> | 항목 | 본 문서(zip 기준) | **정정(24h 전수)** |
> |---|---:|---:|
> | 1차 → 2차 전달 | 100건 | **102건** |
> | 2차 통과 | 7건 | **8건** |
> | 2차 기각 | 93건 | **94건** |
> | 차단율 | 93.0% | **92.2%** |
>
> 2차 통과 8건이 최종 쓰러짐 알람 8건과 일치함.
> **확정 수치는 `실증전_24h_에이징_시험_결과보고.pdf` 를 따를 것.**
>
> 아울러 아래 **「★ 결론」의 `오탐 8건 / 정탐 0건` 은 무효**임.
> 객체저장소 원본(1920px)으로 재판독한 확정 판정은 **정탐 2건 / 오탐 6건**(정탐은 카메라 837 2건)이며,
> 당초 판정은 256×192 썸네일만으로 내려 착석·와위 자세를 구분하지 못한 데 기인함.


대구도시철도 실증 투입 전, **NAS에 저장된 62채널 × 24시간 녹화영상**(채널당 1시간 단위 파일 24개)을
운영 2단계 파이프라인에 그대로 흘려보내고 남은 로그다. 사람이 연출한 상황이 아니라 **실제 역사 운영 영상**이다.

| | |
|---|---|
| 시험 구간 | **2026-10-12 19:55:33 ~ 2026-10-13 19:56:48** (24시간 01분) |
| 입력 | NAS 녹화영상 **62채널 × 24시간** (채널당 1시간 파일 × 24) |
| 1차 추론기 | PE-Core-L14-336 MXQ (NPU) — `npu_pe_2stage` |
| 2차 추론기 | Qwen3-VL-2B 학습본 MXQ (NPU) — 검증서버 VQA |
| 1차 후보 | **100건** (falldown 단일 카테고리, 12개 카메라) |
| 최종 경보 | **13건** = falldown 8 + intrusion_npu 5 |

> ⚠ 정답(GT)이 없는 실영상이다. **오탐은 셀 수 있지만 미탐(Recall)은 측정할 수 없다.**

---

## 1. 압축 풀면 뭐가 있나

### 1-1. 파일 목록

```
실증전_테스트_logs/
├── 1차추론기 결과.zip                    3.4 MB
└── 30.60_이벤트_24h_20261012-1013.csv    3.6 KB  (최종 이벤트 13건)
```

### 1-2. zip 내부 구조 — 305 항목 / 파일 200개

```
1차추론기 결과/
├── 20261012/falldown_ret/        ← MinIO 날짜 파티션
│   └── <카메라ID>_eomeom_<epoch_ms>/
│       ├── 00_<epoch_ms>.jpg     썸네일 256×192
│       └── verdict.json          2차 판정 결과
└── 20261013/falldown_ret/
```

**이벤트 100건 × (jpg 1장 + verdict.json 1개) = 200 파일.** 폴더명은 `<카메라ID>_eomeom_<발생시각 epoch ms>`.
카테고리 폴더는 **`falldown_ret` 하나뿐** — 24시간 동안 다른 카테고리는 1차 후보 자체가 안 나왔다.

> **⚠ 압축 해제 주의.** zip이 UTF-8 파일명을 쓰면서 UTF-8 플래그를 안 세워서, `unzip`으로 풀면 한글이 깨진다.
> 파이썬으로 `cp437` → `utf-8` 재해석해서 풀어야 한다.
>
> ```python
> import zipfile, pathlib
> z = zipfile.ZipFile('1차추론기 결과.zip')
> for i in z.infolist():
>     name = i.filename.encode('cp437').decode('utf-8')
>     dst = pathlib.Path('out') / name
>     if i.is_dir():
>         dst.mkdir(parents=True, exist_ok=True)
>     else:
>         dst.parent.mkdir(parents=True, exist_ok=True)
>         dst.write_bytes(z.read(i))
> ```

### 1-3. `verdict.json` 스키마

```json
{
  "response": {
    "request_id":  "526ec555-e662-4793-803e-c126f013f49a",
    "profile_key": "PENpu.binary",
    "verdict":     "confirmed",            // confirmed | denied
    "reason_code": "provider_confirmed",   // provider_confirmed | provider_denied
    "provider":    "local_vllm",
    "model":       "PIA-SPACE/AX_NPU_VL3_2B_mxq:ft/model",
    "latency_ms":  537
  },
  "decision":        "confirmed",   // 최종 게이트 결과
  "fallback_reason": "none",        // 2차 실패 시 폴백 사유
  "event_uuid":      "684aebac-...",// CSV video_minio_url 파일명과 동일 키
  "image_count":     1              // 2차에 넘긴 이미지 장수
}
```

`event_uuid`가 CSV의 `video_minio_url` 파일명과 같아서 **두 로그를 조인할 수 있다.**

### 1-4. CSV 컬럼

`event_id, start_kst, camera_id, camera_name, event_name, count, first_kst, last_kst, thumbnail_minio_url, video_minio_url`

`count`는 같은 이벤트에 누적된 발화 횟수, `first_kst~last_kst`가 이벤트 지속 구간이다.

---

## 2. 1차 검출기 (PE-Core-L14-336, NPU)

**모듈** `third_party/Product-AI-mono/packages/pia_prod/AI/modules/npu_pe_2stage`

| 항목 | 값 |
|---|---|
| 모델 | PE-Core-L14-336 비전인코더 → MXQ |
| 입력 | **336×336**, NHWC, cv2 resize |
| 양자화 / 코어모드 | W8A16 / **global4** (기본값) |
| 가중치 배포 | HF `PIA-SPACE-LAB/MXQ_NPU` |
| 방식 | 이미지 임베딩 × **텍스트 프롬프트 임베딩** 유사도 (CLIP 계열 zero-shot) |
| 클래스 | **8종** — `normal / falldown / fire / smoke / smoking / esfalldown / elvfalldown / climbingup` |
| 판정 알고리즘 | `METHOD=dist` — normal 분포와 이벤트 분포의 **히스토그램 IoU**가 임계 미만이면 후보 |
| IoU 임계 | 0.3 (`NPU_PE_2STAGE_IOU_THRESHOLD`, 카테고리별 오버라이드 가능) |
| 알람 확정 | **연속 3프레임**(`ALARM_DURATION_THRESHOLD=3`) |
| ROI | `PERoIManager` — 카메라별 폴리곤, cv2 `fillPoly` 크롭. ROI 미지정 시 전체 프레임 |
| 침입 | **별도 모듈(YOLO 사람검출).** 이 1차 8종에 포함되지 않는다 |

프레임마다 8종 텍스트 벡터와 비교해 후보를 뽑고, 연속 3프레임 유지되면 알람을 확정한 뒤 **2차 검증서버로 비동기(fire-and-forget) 위임**한다.

### 24시간 실측 결과

**1차 후보 100건 / 12개 카메라.** 발생률 `100 ÷ (62ch × 24h)` = **0.067 건/채널·시간** (채널당 약 15시간에 1건).

| 카메라 | 후보 | 2차 통과 | 통과율 |
|---|---:|---:|---:|
| **826** | **74** | **0** | **0.0%** |
| 819 | 6 | 4 | 66.7% |
| 814 | 5 | 2 | 40.0% |
| 844 | 5 | 0 | 0% |
| 827 / 860 | 각 2 | 0 | 0% |
| 825·836·837·842·850·854 | 각 1 | 1 (837) | — |

★ **후보의 74%가 826 한 대에서 나왔고, 전부 2차에서 걸러졌다.** 826은 역 대합실인데
**구석 벤치에 앉은 사람**을 계속 쓰러짐으로 집는다. 이웃 발화 간격 중앙값 12.3초로, 한 사람이 앉아 있는
동안 한 장면이 수십 건으로 불어난다(최대 16:41:48~16:46:24 사이 **22건**).
826을 빼면 후보는 100 → 26건, 2차 통과율은 7% → 27%가 된다.

---

## 3. 2차 검출기 (Qwen3-VL-2B 학습본, NPU)

| 항목 | 값 |
|---|---|
| 모델 | **`PIA-SPACE/AX_NPU_VL3_2B_mxq:ft/model`** (학습본 fine-tune MXQ) |
| provider | `local_vllm` — 검증서버 `POST /api/v1/validate`, 타임아웃 30s |
| profile_key | `PENpu.binary` — yes/no 1토큰 이진 판정 |
| 입력 | 썸네일 **1장**(`image_count: 1`), 256×192 |
| 적용 범위 | **화재·연기·쓰러짐 3종만.** 침입은 2차를 타지 않는다 |
| 판정 | 출력 첫 토큰이 `yes` → `confirmed`(경보 발사), 아니면 `denied`(폐기) |

### 쓰러짐 프롬프트 (운영 문안 그대로)

> Determine whether any person in the image has fallen.
> **Definition**: a fall means at least one person is **lying horizontally / parallel to the ground**.
> - `no` if a person is **bent over, crouched** toward the ground or an object (torso angled, not horizontal).
> - `yes` if at least one person is **lying horizontally**, regardless of whether it looks intentional.
> - `no` if every visible person is **upright or semi-upright**: standing, walking, **sitting (including sitting on the ground)**, crouching, squatting, kneeling.
> - Ignore intent and context entirely. Only horizontal lying = yes.
>
> Answer with exactly one word, "yes" or "no".

전문·화재·연기 문안은 `wave_npu/ax_package/ax_tta/config/vqa_prompts.yaml`
(출처 `PIA-SPACE-LAB/2stage_prompts`, variant `ax_npu`, snapshot `59c5c44`).

### 24시간 실측 결과

| 판정 | 건수 | 비율 |
|---|---:|---:|
| **denied** | **93** | 93.0% |
| **confirmed** | **7** | 7.0% |

- **폴백 0회** — `fallback_reason`이 100건 전부 `none`. 2차가 한 번도 안 죽었다.
- **지연** 최소 413 / 중앙 486 / 평균 487 / p95 552 / 최대 644 ms. confirmed(518ms)가 denied(485ms)보다 33ms 느린 정도로 정상 분포.
- **게이트 누수 0** — confirmed 7건은 CSV에 전부 등재, denied 93건은 **단 1건도 CSV에 없다**(event_uuid 대조).
- 100건 전부 동일 설정(`profile_key` / `provider` / `model`)으로 돌았다 → **학습본 mxq가 실환경에서 정상 가동 중**임이 확인된다.

---

## 4. 결과 — intrusion 제외 최종 카테고리

### 4-1. 최종 경보 13건의 구성

| 카테고리 | 건수 | 2차 검증 |
|---|---:|---|
| **falldown** | **8** | 받음 |
| ~~intrusion_npu~~ | 5 | 받지 않음(설계상 1차 종료) — **이 분석에서 제외** |

**침입을 빼면 24시간 동안 올라온 최종 카테고리는 `falldown` 하나뿐이다.**
`fire / smoke / smoking / esfalldown / elvfalldown / climbingup` 6종은 **1차 후보조차 0건**이었다.

### 4-2. falldown 최종 경보 8건 — 전수 판정

썸네일을 4배 확대해 전부 육안 확인했다.

| # | event_id | 발생시각(KST) | cam | 카메라명 | 장면 | 판정 |
|---|---|---|---|---|---|---|
| 1 | 1172442 | 10-12 19:55:33 | 837 | 1코너_회전 | *(1차 로그 수집 구간 밖 — 썸네일 없음)* | **오탐 추정** |
| 2 | 1172527 | 10-13 03:01:25 | 814 | 시점_지상부_EV1 | 카메라 바로 아래 **사람 머리만 화면 하단에 걸림**, 서 있음 | **오탐** |
| 3 | 1172584 | 10-13 08:44:34 | 819 | 상선_승강장_EV3 | 벽에 기대 **바닥에 앉은 사람**(다리 뻗음) | **오탐** |
| 4 | 1172585 | 10-13 08:44:47 | 819 | 상선_승강장_EV3 | 위와 동일 인물·동일 장면 | **오탐** |
| 5 | 1172586 | 10-13 08:44:59 | 819 | 상선_승강장_EV3 | 위와 동일 | **오탐** |
| 6 | 1172587 | 10-13 08:45:49 | 819 | 상선_승강장_EV3 | 위와 동일 | **오탐** |
| 7 | 1172608 | 10-13 19:13:27 | 814 | 시점_지상부_EV1 | #2와 같은 구도, **카메라 아래 사람 머리**(모자 착용) | **오탐** |
| 8 | 1172612 | 10-13 19:55:06 | 837 | 1코너_회전 | 젖은 바닥(청소 표지), **몸을 숙인 작업자** + 서 있는 사람 | **오탐** |

### ★ 결론

```
falldown 최종 경보   8건
  ├─ 진짜 쓰러짐      0건
  └─ 오탐             8건   (#1은 썸네일 부재로 추정, 나머지 7건은 확인)
```

- **오탐률(최종 경보 기준) = 8/8 = 100%.** 24시간 62채널 실영상에 실제 쓰러짐이 없었으므로 정탐 0은 당연하고,
  **문제는 "그럼에도 8건이 올라왔다"는 것**이다.
- 발생률로 보면 `8 ÷ 1,488 채널·시간` = **0.0054 건/채널·시간 ≈ 채널당 7.8일에 1건**. 운영 부하 자체는 낮다.
- 1차 기준으로 보면 후보 100건 중 **최소 100건이 오탐**(정탐 0) — 2차가 그중 93건을 막아 **차단율 93%**.

### 4-3. 오탐 3개 유형

| 유형 | 해당 | 원인 | 2차가 걸러야 했나 |
|---|---|---|---|
| **A. 카메라 바로 아래 인물** | 814 × 2 | 돔/어안 카메라 밑에 선 사람이 위에서 보면 납작하게 눌려 **수평 형상**이 된다 | 예 — 서 있는 사람이다 |
| **B. 바닥에 앉은 사람** | 819 × 4 | 벽에 기대 다리 뻗고 앉음 | 예 — 프롬프트가 *"sitting (including sitting on the ground)" = `no`* 라고 **명시**하는데 통과시켰다 |
| **C. 숙인 자세 / 작업자** | 837 × 2 | 젖은 바닥 청소 중 몸을 숙임 | 예 — 프롬프트가 *"bent over, crouched" = `no`* 라고 **명시** |

**세 유형 모두 프롬프트 규칙에 이미 `no`로 적혀 있는 자세다.** 즉 프롬프트 문안 부족이 아니라
**2차 모델이 자기 규칙을 못 지킨 것**이다.

### 4-4. 판정 흔들림 (재현성)

cam 819의 같은 인물·같은 장면이 **12초 간격으로 판정이 뒤집힌다.**

```
08:44:34  confirmed
08:44:47  confirmed
08:44:59  confirmed
08:45:11  denied      ← 같은 장면
08:45:49  confirmed
```

`reports/evaluation/NPU_tta4ax_pe_vs_vlm.md`에 적어둔 **"같은 설정도 실행마다 답이 0~4건 바뀐다"**가
실환경에서 그대로 재현됐다. 지금 `image_count: 1`이라 단일 프레임 1회 질의로 끝나는 구조라,
**N장 다수결**로 가면 이 흔들림은 상당 부분 흡수된다.

---

## 5. 참고 — intrusion (분석 제외분)

본 분석 범위 밖이지만 기록해 둔다. 5건 중 하나가 비정상이다.

```
1172443  10-12 20:12:04  cam 808 종점환기실  intrusion_npu  count 162  지속 23:44:44
```

**단일 이벤트가 24시간 가까이 162회 누적**됐다. 나머지 4건은 전부 `count 1 / 지속 0`이다.
상시 뭔가가 잡히거나 이벤트 종료 조건이 안 걸리는 것으로 보인다.
침입은 2차 VQA를 타지 않으므로 **걸러줄 장치가 현재 없다.**

---

## 6. 조치 우선순위

| 순위 | 대상 | 조치 | 기대효과 |
|---|---|---|---|
| 1 | **cam 826** | 벤치 구역 **ROI 제외** + 카메라별 **쿨다운**(동일 카메라 60초 내 재발화 억제) | 1차 후보 100 → 26건. 2차 호출이 1/4로 줄어든다. 가장 싸고 효과가 크다 |
| 2 | **cam 814 (유형 A)** | 어안 왜곡 구간(프레임 하단) ROI 제외. 프롬프트에 *"사람이 화면 가장자리에 일부만 보이면 `no`"* 추가 | 최종 오탐 2건 제거 |
| 3 | **2차 다수결** | `image_count` 1 → N (3~5장), 과반 `yes`일 때만 confirmed | 판정 흔들림 흡수. 819형 오탐 4건 중 상당수 제거 |
| 4 | **cam 808 intrusion** | 이벤트 종료 조건 점검 | count 162 → 정상화 |

전체를 쿨다운 기준(동일 카메라 60초)으로 묶으면 1차 후보 100건은 **36개 상황**으로 줄어든다.

---

## 7. 이 로그로 말할 수 있는 것 / 없는 것

**말할 수 있다**
- 2차 게이트가 1차 오탐의 93%를 차단하고, 누수는 0이다.
- 학습본 mxq가 실환경 24시간 연속으로 폴백 없이 돌았다(지연 p95 552ms).
- 최종 오탐 8건의 원인이 카메라 3대 / 자세 3유형에 집중돼 있다.

**말할 수 없다**
- **미탐(Recall).** 정답 라벨이 없는 실영상이라, 놓친 쓰러짐이 있었는지 알 방법이 없다.
- **확정적 오탐 판정.** 썸네일 256×192 1장만 보고 내린 판단이다. 2차 모델이 본 것과 같은 정보량이긴 하지만,
  확정하려면 CSV `video_minio_url`의 원본 mp4를 봐야 한다. 특히 #1은 썸네일 자체가 없어 **추정**이다.

---

## 부록 — 재현

```bash
cd wave_npu/실증전_테스트_logs
python3 - <<'PY'
import zipfile, pathlib
z = zipfile.ZipFile('1차추론기 결과.zip')
for i in z.infolist():
    dst = pathlib.Path('out') / i.filename.encode('cp437').decode('utf-8')
    dst.mkdir(parents=True, exist_ok=True) if i.is_dir() else (
        dst.parent.mkdir(parents=True, exist_ok=True), dst.write_bytes(z.read(i)))
PY
```

관련 문서
- 1차 모듈 — `third_party/Product-AI-mono/packages/pia_prod/AI/modules/npu_pe_2stage/{config.py,service.py,roi_manager.py}`
- 2차 프롬프트 — `wave_npu/ax_package/ax_tta/config/vqa_prompts.yaml`
- 2차 배치 서버 성능 — `reports/performance/qwen3vl/NPU_qwen3vl_bs32_server.md`
- PE 1차 vs VLM 2차 정확도 — `reports/evaluation/NPU_tta4ax_pe_vs_vlm.md`
- 학습본 재컴파일 — `reports/NPU_qwen3vl_mblt_export.md`
