#!/usr/bin/env bash
# 정식 컴파일 CLI(python -m pe_npu.compile) 경로 검증 — 어느 서버에서나 같은 명령으로.
#
# 검증 방법: **이미 정답을 아는 조합을 다시 뽑아 cos 가 맞는지 본다.**
# 절차·기대값 상세: reports/RUNBOOK_quant_matrix_120.md §2.5
#
#   A  정식 CLI 등가성        --quant w4a16 --optq --search-weight-scale   → cos 0.9642
#   B  --quant w4a8 A16 주입  --quant w4a8 --a16 "<5개>"                   → cos 0.9420
#   C  calibration 통계 캐시  --calib-stats-save → --calib-stats-load      → cos 0.9126 (양쪽 동일)
#
# GPU 서버면 조합당 5~8분(전체 20~35분), CPU 서버면 조합당 ~64분.
# 이미 만들어진 mxq 는 건너뛴다 → **다른 서버에서 이어 돌릴 수 있다**(rsync 로 out 디렉토리만 옮기면 됨).
#
# 사용:
#   bash reports/scripts/validate_compile_cli.sh                    # A,B,C 전부
#   bash reports/scripts/validate_compile_cli.sh --tests A           # A만
#   bash reports/scripts/validate_compile_cli.sh --out out/validate --sdk 1.1
#   bash reports/scripts/validate_compile_cli.sh --verify-only       # 컴파일 건너뛰고 NPU 검증만
#
# 검증에 쓸 파이썬은 qbruntime 이 있는 인터프리터여야 한다. 기본 `python` 이 아니면 지정한다.
#   PYTHON=~/miniconda3/envs/pe_npu_host/bin/python bash reports/scripts/validate_compile_cli.sh --verify-only
#
# NPU(/dev/aries*)가 있으면 컴파일 후 cos 검증까지 자동으로 이어간다.
# 없으면(컴파일 전용 GPU 서버) mxq 만 만들고, NPU 서버에서 검증하는 명령을 안내한다.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
cd "$REPO_ROOT"

OUT="out/validate_cli"
TESTS="A,B,C"
SDK_ARG=""
CALIB="download/calib_coco_hwc"
COCO="download/coco/val2017"
DEVICE_ID="${NPU_DEVICE:-0}"
PYTHON="${PYTHON:-python}"
VERIFY_ONLY=0
FORCE=""

while [ $# -gt 0 ]; do
  case "$1" in
    --out)     OUT="$2"; shift 2 ;;
    --tests)   TESTS="$2"; shift 2 ;;
    --sdk)     SDK_ARG="$2"; shift 2 ;;
    --calib)   CALIB="$2"; shift 2 ;;
    --coco)    COCO="$2"; shift 2 ;;
    --device)  DEVICE_ID="$2"; shift 2 ;;
    --gpu|--cpu) FORCE="$1"; shift ;;
    --verify-only) VERIFY_ONLY=1; shift ;;
    -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
    *) echo "알 수 없는 인자: $1"; exit 2 ;;
  esac
done

mkdir -p "$OUT"
A16_W4A8=$(python3 - <<'PY'
import re
s = open("reports/scripts/compile_quant_tuning_matrix.py").read()
m = re.search(r"A16_TENSORS_W4A8 = \[(.*?)\]", s, re.S)
print(",".join(re.findall(r'"([^"]+)"', m.group(1))))
PY
)
[ -n "$A16_W4A8" ] || { echo "A16 텐서 이름 추출 실패"; exit 1; }

has() { case ",$TESTS," in *",$1,"*) return 0 ;; *) return 1 ;; esac }

compile_one() {  # $1=파일명  $2...=pe_npu.compile 인자
  local name="$1"; shift
  local dst="$OUT/$name"
  if [ -f "$dst" ]; then
    echo "[skip] $name 이미 존재 ($(du -h "$dst" | cut -f1)) — 다른 서버에서 만든 것이면 그대로 쓴다"
    return 0
  fi
  echo "[compile] $name"
  # shellcheck disable=SC2086
  bash setup/compile_in_docker.sh ${SDK_ARG:+--sdk "$SDK_ARG"} $FORCE \
       --name "mblt_validate" "$@" --calib-data-path "$CALIB" --save "$dst" \
    || { echo "[FAIL] $name 컴파일 실패"; return 1; }
  [ -f "$dst" ] || { echo "[FAIL] $name 산출물 없음"; return 1; }
  echo "[ok] $name  $(stat -c %s "$dst") bytes"
}

if [ "$VERIFY_ONLY" = "0" ]; then
  [ -e "$CALIB/npy_files.txt" ] || { echo "calib 없음: $CALIB (RUNBOOK §1-4 참조)"; exit 1; }
  n=$(wc -l < "$CALIB/npy_files.txt")
  [ "$n" = "200" ] || echo "[warn] calib $n 장 — 회귀 기준은 200장 기준이다"

  # ── A: 정식 CLI 등가성 (기대 cos 0.9642)
  has A && compile_one pe_W4A16_sws_optq_single.mxq \
      --quant w4a16 --optq --search-weight-scale --scheme single

  # ── B: --quant w4a8 + A16 5개 주입 (기대 cos 0.9420, activation16=30)
  has B && compile_one pe_W4A8_L5A16_sws_optq_single.mxq \
      --quant w4a8 --optq --search-weight-scale --scheme single --a16 "$A16_W4A8"

  # ── C: calibration 통계 캐시 (양쪽 cos 0.9126 동일해야 함)
  if has C; then
    STATS="$OUT/calib_stats.bin"
    compile_one pe_W4A16_none_single.mxq \
        --quant w4a16 --scheme single --calib-stats-save "$STATS"
    if [ -f "$STATS" ] || ls "$OUT"/calib_stats* >/dev/null 2>&1; then
      compile_one pe_W4A16_none_single_statsload.mxq \
          --quant w4a16 --scheme single --calib-stats-load "$STATS"
      # _0.9999 접미사 보정이 실패하면 아래가 남는다
      for stray in "$OUT"/*_0.9999.mxq; do
        [ -e "$stray" ] && echo "[warn] 접미사 보정 실패 흔적: $stray (_normalize_stats_output 확인)"
      done
    else
      echo "[warn] C: 통계 파일이 생성되지 않았다 → --calib-stats-save 미동작"
    fi
  fi
fi

echo
echo "=== 산출물 ==="
ls -la "$OUT"/*.mxq 2>/dev/null || { echo "mxq 없음"; exit 1; }

if ls /dev/aries* >/dev/null 2>&1; then
  echo
  echo "=== NPU 검증 (device $DEVICE_ID) — 회귀 기준 자동 대조 ==="
  "$PYTHON" reports/scripts/verify_quant_tuning_matrix.py \
    --src-dir "$OUT" --device "$DEVICE_ID" --coco-dir "$COCO" \
    --channels 1,20 --out "reports/assets/validate_cli.json"
  rc=$?
  echo
  if [ "$rc" = "0" ]; then
    echo "✅ 전부 회귀 기준 일치"
    [ "$VERIFY_ONLY" = "0" ] && echo "   → 이번 실행에서 CLI 로 컴파일한 조합은 검증된 경로와 등가다"
    echo "   (--verify-only 였다면 '넘겨받은 산출물이 기준과 일치'라는 뜻이다)"
  else
    echo "⚠ 기준 이탈 — RUNBOOK §2.5 '검증 결과에 따른 분기' 참조"
  fi
  exit $rc
else
  cat <<EOF

=== 이 서버에는 NPU 가 없다(컴파일 전용) ===
mxq 를 NPU 서버로 옮겨 검증한다.

  # (NPU 서버에서)
  rsync -av <이 서버>:$REPO_ROOT/$OUT/ ./$OUT/
  PYTHON=<qbruntime 있는 python> \\
    bash reports/scripts/validate_compile_cli.sh --verify-only --out $OUT --device 0

기대값(RUNBOOK §5-1):
  pe_W4A16_sws_optq_single            cos 0.9654
  pe_W4A8_L5A16_sws_optq_single       cos 0.9420
  pe_W4A16_none_single(+_statsload)   cos 0.9175  (양쪽 동일해야 함)
EOF
fi
