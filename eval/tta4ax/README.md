# tta4ax — TTA_4종_AX 프레임 평가

`eval/datasets/TTA_4종_AX` 의 이벤트 구간 가운데 프레임으로 **PE 1차**와 **VLM 2차**를 비교한다.
결과·해석: `reports/evaluation/NPU_tta4ax_pe_vs_vlm.md`

## 순서

```bash
# ① 프레임 추출 (양성=이벤트 구간 가운데, 음성=클립 가운데) → /tmp/ax_eval/
python3 extract_frames.py

# ② VLM 서버 2대 기동 (카드/포트/크기는 환경에 맞게)
#    reports/evaluation/… 의 "방법" 절 참고

# ③ VLM 요청 (두 서버 동시) — 프롬프트는 prompts/ 사용
python3 run_eval.py 16

# ④ PE 1차 (제품 전처리·dist 판정 그대로)
python3 run_pe.py <카드번호>

# ⑤ 채점
python3 score_all.py     # 경로별 + 2단계 직렬
python3 score_size.py    # 입력 크기 비교
python3 variance.py      # 3회 반복 변동폭 (rep_<크기>_<회차>.json 필요)
```

## 주의

- **단일 실행으로 판단하지 말 것.** 같은 설정에서도 실행마다 답이 0~4건 바뀐다
  (배치 구성이 매번 다르다). 설정을 비교할 땐 3회 이상 돌려 범위가 겹치는지 본다.
- 프롬프트가 레포 `prompts.py` 와 다르므로 `/v1/chat/completions` 로 보낸다
  (`/api/v1/validate` 는 카테고리로 프롬프트가 고정된다).
- `dump_logits.py` 는 NPU 가 고른 토큰과 yes/no 로짓을 찍는다 — 출력이 이상할 때
  로짓 **크기**를 보면 기저 어긋남(스케일 붕괴)과 양자화 열화를 구분할 수 있다.
