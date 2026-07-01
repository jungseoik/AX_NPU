#!/usr/bin/env bash
# Mobilint ARIES NPU 호스트 세팅 — clone 후 'mobilint-cli status'까지 한 번에.
#
# 신규 서버에서 이 레포를 clone하고, download/ 에 SDK 3개 파일을 넣은 뒤 실행하면
#   드라이버 빌드/설치 -> 모듈 로드 -> 런타임+CLI(make install) -> 디바이스/상태 확인
# 까지 수행한다. 절대경로 하드코딩 없이 레포 위치를 동적으로 찾는다.
#
# 사전 준비(사람): download/ 에 아래 3개 파일을 넣어둘 것 (Mobilint 제공, 비공개)
#   - qbcompiler-*+aries2-py3-none-any.whl
#   - qbruntime_aries2-*_amd64.tar.gz
#   - mobilint-aries2-driver_*.tar.gz
# 그리고 NPU 카드가 PCIe 슬롯에 물리 장착되어 있어야 status가 뜬다.
#
# 사용:  sudo bash setup_npu_cli.sh            # 전체(드라이버+런타임+검증)
#        sudo bash setup_npu_cli.sh --runtime  # 런타임/CLI만 (드라이버 이미 설치된 경우)
#        bash setup_npu_cli.sh --check         # 설치/상태 점검만 (sudo 불필요)

set -u
MODE="${1:-all}"

# ---- 레포 루트 동적 탐색 (이 스크립트: <repo>/.claude/skills/npu-setup/) ----
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
DL="$REPO_ROOT/download"

say()  { echo -e "$@"; }
ok()   { echo "  [OK]   $1"; }
err()  { echo "  [FAIL] $1"; }
die()  { echo "  [ERROR] $1" >&2; exit 1; }

need_root() {
  [ "$(id -u)" -eq 0 ] || die "이 단계는 root 권한이 필요합니다. 'sudo bash $0 $MODE' 로 실행하세요."
}

find_one() {  # glob 패턴으로 파일 1개 찾기
  local f; f=$(ls $1 2>/dev/null | head -1); echo "$f"
}

# ---------------- 0. download/ SDK 파일 점검 (없으면 HF에서 자동 fetch) ----------------
fetch_sdk_if_missing() {
  mkdir -p "$DL"
  if [ -z "$(find_one "$DL/mobilint-aries2-driver_*.tar.gz")" ] || [ -z "$(find_one "$DL/qbruntime_aries2-*_amd64.tar.gz")" ]; then
    say "  [fetch] download/ 에 SDK 없음 → HF private 레포에서 받기 시도"
    local PY; PY=$(command -v python3 || command -v python)
    if [ -n "$PY" ] && "$PY" -c "import huggingface_hub" 2>/dev/null; then
      "$PY" "$REPO_ROOT/setup/fetch_sdk_from_hf.py" || say "  [WARN] HF fetch 실패 — 'huggingface-cli login'(또는 HF_TOKEN) 후 재시도 / 또는 수동 배치"
    else
      say "  [WARN] huggingface_hub 없음 → 'conda activate pe_npu_host' 후 'python setup/fetch_sdk_from_hf.py' 실행(또는 download/ 수동 배치)"
    fi
  fi
}

check_sdk_files() {
  say "[0] SDK 파일 점검 ($DL)"
  [ -d "$DL" ] || mkdir -p "$DL"
  fetch_sdk_if_missing
  DRIVER_TAR=$(find_one "$DL/mobilint-aries2-driver_*.tar.gz")
  RUNTIME_TAR=$(find_one "$DL/qbruntime_aries2-*_amd64.tar.gz")
  COMPILER_WHL=$(find_one "$DL/qbcompiler-*aries2-py3-none-any.whl")
  [ -n "$DRIVER_TAR" ]  && ok "driver:   $(basename "$DRIVER_TAR")"   || err "driver tar 없음 (mobilint-aries2-driver_*.tar.gz)"
  [ -n "$RUNTIME_TAR" ] && ok "runtime:  $(basename "$RUNTIME_TAR")"  || err "runtime tar 없음 (qbruntime_aries2-*_amd64.tar.gz)"
  [ -n "$COMPILER_WHL" ] && ok "compiler: $(basename "$COMPILER_WHL") (컴파일용, status엔 불필요)" || say "  [WARN] compiler whl 없음 (추론/상태엔 불필요, 컴파일 시 필요)"
  [ -n "$DRIVER_TAR" ] && [ -n "$RUNTIME_TAR" ] || die "드라이버/런타임 없음. HF 로그인 후 'python setup/fetch_sdk_from_hf.py' 실행(또는 download/ 수동 배치)."
}

# ---------------- 1. 드라이버 빌드/설치 ----------------
install_driver() {
  need_root
  say "\n[1] NPU 드라이버 빌드/설치 (tar 소스 직접 빌드)"
  command -v make >/dev/null || die "make/build-essential 미설치. 'apt install build-essential linux-headers-\$(uname -r)' 필요."
  local BUILD=/tmp/aries-driver-build
  rm -rf "$BUILD"; mkdir -p "$BUILD"
  tar xzf "$DRIVER_TAR" -C "$BUILD"
  local SRC; SRC=$(find "$BUILD" -mindepth 1 -maxdepth 1 -type d -name "aries-driver*")
  [ -n "$SRC" ] || die "드라이버 소스 디렉토리(aries-driver*)를 찾지 못했습니다."
  make -C "$SRC"
  [ -f "$SRC/aries.ko" ] || die "aries.ko 빌드 실패."
  make -C "$SRC" install
  depmod -a
  echo aries > /etc/modules-load.d/aries.conf   # 부팅 자동 로드
  ok "드라이버 설치 + 부팅 자동 로드 등록(/etc/modules-load.d/aries.conf)"
  modprobe aries 2>/dev/null && ok "modprobe aries 성공" || say "  [WARN] modprobe 실패 — NPU 카드 미장착이면 정상(장착 후 'sudo modprobe aries')"
}

# ---------------- 2. 런타임 + CLI 설치 ----------------
install_runtime() {
  need_root
  say "\n[2] 런타임 + mobilint-cli 설치 (make install)"
  local RT="${RUNTIME_TAR%.tar.gz}"
  [ -d "$RT" ] || tar xzf "$RUNTIME_TAR" -C "$DL"
  ( cd "$RT" && make install )   # libqbruntime.so* + /usr/local/bin/mobilint-cli 등
  ldconfig
  ok "libqbruntime + mobilint-cli 설치 (/usr/local/bin, /usr/local/lib)"
}

# ---------------- 3. 검증 ----------------
do_check() {
  say "\n[3] 설치/상태 점검"
  lsmod 2>/dev/null | grep -q aries && ok "aries 커널 모듈 로드됨" || err "aries 모듈 미로드 (sudo modprobe aries / 카드 장착 확인)"
  ls /dev/aries* >/dev/null 2>&1 && ok "디바이스 노드: $(ls /dev/aries* | tr '\n' ' ')" || err "디바이스 노드 없음 (카드 장착 + 모듈 로드 필요)"
  command -v lspci >/dev/null && { lspci -d 209f: 2>/dev/null | grep -q . && ok "PCI 인식: $(lspci -d 209f: | head -1)" || err "PCI 미인식 (전원/슬롯 확인)"; }
  if command -v mobilint-cli >/dev/null 2>&1; then
    ok "mobilint-cli 설치됨 → status 실행:"
    echo "  -------------------------------------------"
    mobilint-cli status 2>&1 | sed 's/^/  /'
    echo "  -------------------------------------------"
  else
    err "mobilint-cli 미설치 (위 [2] 런타임 설치 필요)"
  fi
}

# ---------------- 흐름 ----------------
say "================ Mobilint NPU 호스트 세팅 ================"
say "레포 루트: $REPO_ROOT"
case "$MODE" in
  --check)   do_check ;;
  --runtime) check_sdk_files; install_runtime; do_check ;;
  --driver)  check_sdk_files; install_driver; do_check ;;
  all|*)     check_sdk_files; install_driver; install_runtime; do_check ;;
esac

say "\n완료. NPU 상태가 안 뜨면: ① 카드 물리 장착 ② sudo modprobe aries ③ 펌웨어 'aries_flash_firmware status'."
say "추론(Python)까지 하려면: bash setup/setup_conda_host.sh 후 tutorial_pe_npu/README.md 참고."
