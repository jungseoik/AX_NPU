#!/usr/bin/env bash
# customer-capacity-repro 컴파일 컨테이너 실행 래퍼 (Qwen3-VL batch32 MXQ 재생성).
#
# ★ CUDA 12.8 GPU 가 있는 호스트에서만 돈다. NPU 서버(GPU 없음)에서는 실행 불가.
#
#   bash setup/vlm_repro/run_compile.sh --part all          # 4종 전부 (수 시간)
#   bash setup/vlm_repro/run_compile.sh --part decoder-b32  # 배치32 디코더만
#   GPUS=1 bash setup/vlm_repro/run_compile.sh --part vision
#   bash setup/vlm_repro/run_compile.sh --shell             # 셸만
#
# 산출물: <repro>/build/ (decoder-b1.mxq, decoder-b32.mxq, vision.mxq, vision8192.mxq)
# 대조군: <repro>/reference/mxq/  ← 벤더가 실측에 쓴 원본. provenance.json 에 sha256 있음.
#   ※ 벤더 TOOLCHAIN.json 주석: "identical millisecond timings or identical MXQ bytes are
#     not promised." — 바이트 동일을 기대하지 말고 **정확도/지연으로** 비교할 것.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
REPRO="${REPRO_DIR:-$REPO_ROOT/download/vendor/customer-capacity-repro}"
IMAGE="${IMAGE:-mblt_vlm_repro:compile}"
GPUS="${GPUS:-all}"

[ -d "$REPRO" ] || { echo "재현 패키지 없음: $REPRO"; echo "  → customer-capacity-repro.tar.gz 를 풀어야 한다(26GB)."; exit 1; }
[ -f "$REPO_ROOT/.env" ] && { set -a; . "$REPO_ROOT/.env"; set +a; }
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO=1
d() { if [ -n "$SUDO" ] && [ -n "${SUDO_PASS:-}" ]; then echo "$SUDO_PASS" | sudo -S docker "$@"; else docker "$@"; fi }

# GPU 가 실제로 보이는지 먼저 확인한다 — 없으면 컨테이너 안에서 벤더 스크립트가 죽는다.
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
  echo "❌ NVIDIA GPU 를 찾을 수 없다. 이 레시피는 CUDA 12.8 GPU 를 요구한다."
  echo "   (벤더 scripts/compile.py 가 torch.cuda.is_available() 를 하드 체크한다)"
  exit 1
fi
echo "[compile] GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd', ')"
echo "[compile] 이미지 $IMAGE | 패키지 $REPRO"
# docker 가 sudo 없이 되는 호스트에서 .env 의 SUDO_PASS 때문에 불필요하게 sudo 를 타지 않도록
docker ps >/dev/null 2>&1 && SUDO=""

TTY=(-i); [ -t 0 ] && TTY=(-it)
if [ "${1:-}" = "--shell" ]; then
  exec bash -c "$(declare -f d); d run --rm -it --gpus '$GPUS' --shm-size 16g \
    -v '$REPRO':/repro -w /repro '$IMAGE' bash"
fi

# 1) prepare — 보정 경로 정규화 + batch32 split MBLT 생성 (없으면 만든다)
echo "[compile] prepare (보정 경로 + batch32 split)"
d run --rm "${TTY[@]}" --gpus "$GPUS" --shm-size 16g -v "$REPRO":/repro -w /repro "$IMAGE" \
  python3.10 scripts/prepare.py

# 2) compile
echo "[compile] compile $*"
d run --rm "${TTY[@]}" --gpus "$GPUS" --shm-size 16g -v "$REPRO":/repro -w /repro "$IMAGE" \
  python3.10 scripts/compile.py "$@"

echo "[compile] 산출물:"
ls -la "$REPRO/build"/*.mxq 2>/dev/null | awk '{printf "  %8.1fMB  %s\n",$5/1048576,$9}' || echo "  (없음)"
