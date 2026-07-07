#!/usr/bin/env bash
# NPU 장착 후 단계별 환경 점검 스크립트.
# 드라이버 → 디바이스 노드 → PCI 인식 → 펌웨어/유틸 → 런타임 순으로 확인한다.
# 각 단계 결과를 PASS/FAIL/SKIP으로 출력하고, 마지막에 다음 행동을 안내한다.
#
# 사용: bash check_npu.sh
set -u

pass() { echo "  [PASS] $1"; }
fail() { echo "  [FAIL] $1"; FAILED=1; }
skip() { echo "  [SKIP] $1"; }

FAILED=0
echo "================ NPU 환경 점검 ================"

echo "[1] 커널 드라이버 모듈 (aries)"
if lsmod 2>/dev/null | grep -q aries; then
  pass "aries 커널 모듈 로드됨"
else
  fail "aries 모듈 미로드 → 'sudo apt install mobilint-aries-driver' 후 재부팅"
fi

echo "[2] 디바이스 노드 (/dev/aries*)"
if ls /dev/aries* >/dev/null 2>&1; then
  pass "$(ls /dev/aries* | tr '\n' ' ')"
else
  fail "디바이스 노드 없음 → 드라이버 설치/재부팅 또는 카드 장착 확인"
fi

echo "[3] PCI 버스 인식 (vendor 209f)"
if command -v lspci >/dev/null 2>&1; then
  if lspci -d 209f: 2>/dev/null | grep -q .; then
    pass "$(lspci -d 209f: | head -1)"
  else
    fail "PCI에서 NPU 미인식 → 전원/슬롯 장착 상태 확인"
  fi
else
  skip "lspci 미설치"
fi

echo "[4] 펌웨어/상태 유틸 (선택)"
if command -v aries_flash_firmware >/dev/null 2>&1; then
  pass "aries_flash_firmware 존재 ('aries_flash_firmware status'로 펌웨어 확인)"
else
  skip "aries_flash_firmware 미설치 (mobilint_ctrl 패키지)"
fi
if command -v mobilint-cli >/dev/null 2>&1; then
  pass "mobilint-cli 존재 ('mobilint-cli status'로 드라이버/펌웨어 버전 확인)"
else
  skip "mobilint-cli 미설치 ('sudo apt install mobilint-cli')"
fi

echo "[5] 런타임 라이브러리 (qbruntime)"
if python3 -c "import qbruntime" 2>/dev/null; then
  VER=$(python3 -c "import qbruntime; print(getattr(qbruntime,'__version__','?'))" 2>/dev/null)
  pass "qbruntime import 성공 (version=$VER)"
else
  fail "qbruntime import 실패 → 'pip install mobilint-qb-runtime' 또는 런타임 설치"
fi

echo "=============================================="
if [ "$FAILED" -eq 0 ]; then
  echo "모든 필수 점검 통과. 다음 단계로 진행하세요:"
  echo "  1) 예제 이미지 다운로드:  (tutorial/pe_npu) python download_images.py"
  echo "  2) hybrid 추론 + 정확도 데모:"
  echo "     (tutorial/pe_npu) python demo_inference.py    # 평균 cos >= 0.99 기대"
  echo "  자세한 전 과정: tutorial/pe_npu/README.md"
else
  echo "일부 점검 실패. 위 [FAIL] 항목을 먼저 해결하세요. (setup-notes.md 추론 환경 절차 참고)"
fi
exit "$FAILED"
