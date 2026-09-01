#!/usr/bin/env bash
# 신규 서버 원샷 세팅 — clone 직후부터 'mobilint-cli status'까지.
#
#   1) 부트스트랩: python/huggingface_hub 확보 (없으면 conda env 생성)
#   2) SDK 다운로드: HF private 레포의 선택 버전 폴더 → 로컬 버전 경로
#   3) 드라이버 빌드/설치 + 런타임/CLI 설치 (sudo 필요)
#   4) 점검: 모듈/디바이스/PCI/status
#
# 사용:
#   bash setup/setup_all.sh --sdk 1.1                 # 권한 있으면 sudo 자동 사용
#   bash setup/setup_all.sh --sdk 1.0 --fetch-only    # 다운로드까지만(설치 X)
#   SUDO_PASS=... bash setup/setup_all.sh --sdk 1.1   # 비밀번호로 sudo 통과
#
# 필요한 것: HF_TOKEN(환경변수 또는 .env), NPU 카드 물리 장착, sudo 권한.

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
SDK_ARG="${SDK_VERSION:-}"
FETCH_ONLY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --sdk)        SDK_ARG="${2:-}"; shift 2 ;;
    --sdk=*)      SDK_ARG="${1#*=}"; shift ;;
    --fetch-only) FETCH_ONLY=1; shift ;;
    -h|--help)    sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "알 수 없는 인자: $1"; exit 2 ;;
  esac
done

say() { echo -e "$@"; }
die() { echo "  [ERROR] $1" >&2; exit 1; }

# .env 가 있으면 토큰/비번을 읽어온다 (git 미추적)
if [ -f "$REPO_ROOT/.env" ]; then
  set -a; . "$REPO_ROOT/.env"; set +a
  say "  [정보] .env 에서 자격증명 로드"
fi

# ---------------- 1. 부트스트랩 (python + huggingface_hub) ----------------
say "\n[1] 부트스트랩 — python / huggingface_hub"
PYBIN=""
for cand in "$HOME/miniconda3/envs/pe_npu_host/bin/python" "$(command -v python3)" "$(command -v python)"; do
  [ -n "$cand" ] && [ -x "$cand" ] || continue
  if "$cand" -c "import huggingface_hub" 2>/dev/null; then PYBIN="$cand"; break; fi
  [ -z "$PYBIN" ] && PYBIN_FALLBACK="$cand"
done
if [ -z "$PYBIN" ]; then
  PYBIN="${PYBIN_FALLBACK:-}"
  [ -n "$PYBIN" ] || die "python 이 없습니다. 'bash setup/setup_conda_host.sh' 로 conda env를 먼저 만드세요."
  say "  huggingface_hub 없음 → 설치 시도 ($PYBIN)"
  "$PYBIN" -m pip install -q huggingface_hub || die "huggingface_hub 설치 실패. 'bash setup/setup_conda_host.sh' 를 먼저 실행하세요."
fi
say "  python: $PYBIN"

# ---------------- 2. SDK 버전 해석 + 다운로드 ----------------
say "\n[2] SDK 번들 준비"
eval "$("$PYBIN" "$REPO_ROOT/setup/sdk_resolve.py" ${SDK_ARG:+--sdk "$SDK_ARG"} --shell)" || exit 1
say "  선택된 번들: $SDK_LABEL  (로컬 $SDK_DIR / HF $SDK_HF_FOLDER)"

if [ -z "$(ls $SDK_DIR/$DRIVER_GLOB 2>/dev/null | head -1)" ] || \
   [ -z "$(ls $SDK_DIR/$RUNTIME_GLOB 2>/dev/null | head -1)" ]; then
  [ -n "${HF_TOKEN:-}${HF_AUTH_TOKEN:-}" ] || say "  [WARN] HF_TOKEN 미설정 — 'huggingface-cli login' 되어 있으면 그대로 진행"
  "$PYBIN" "$REPO_ROOT/setup/fetch_sdk_from_hf.py" --sdk "$SDK_VERSION" || die "SDK 다운로드 실패 (HF 토큰/권한 확인)"
else
  say "  이미 로컬에 있음 — 다운로드 생략"
fi
"$PYBIN" "$REPO_ROOT/setup/sdk_resolve.py" --sdk "$SDK_VERSION"

[ "$FETCH_ONLY" = "1" ] && { say "\n[완료] --fetch-only: 다운로드까지만 수행."; exit 0; }

# ---------------- 3. 설치 (sudo) ----------------
say "\n[3] 드라이버 + 런타임 설치 (sudo)"
SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  if [ -n "${SUDO_PASS:-}" ]; then SUDO="echo $SUDO_PASS | sudo -S -E"
  elif sudo -n true 2>/dev/null;  then SUDO="sudo -E"
  else
    say "  [중단] sudo 권한이 필요합니다. 아래 중 하나로 진행하세요:"
    say "     · 프롬프트에서 직접:  ! sudo -E bash .claude/skills/npu-setup/setup_npu_cli.sh all --sdk $SDK_VERSION"
    say "     · .env 에 SUDO_PASS=... 추가 후 이 스크립트 재실행"
    exit 3
  fi
fi
eval "$SUDO bash '$REPO_ROOT/.claude/skills/npu-setup/setup_npu_cli.sh' all --sdk '$SDK_VERSION'" || die "설치 단계 실패"

say "\n[완료] 세팅 종료 — 위 'mobilint-cli status' 출력 확인."
say "        추론 파이썬 환경까지: bash setup/setup_conda_host.sh"
