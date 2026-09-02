#!/usr/bin/env bash
# 벤더 qbcompiler 도커 이미지에서 MXQ 컴파일 — CPU 서버/GPU 서버 어디서나 같은 명령으로.
#
# 컴파일러는 호스트 직접 설치가 어렵다(mmc 네이티브 모듈 문제) → 벤더 이미지를 쓴다.
# 이미지는 SDK 번들 버전과 호스트 GPU 유무로 자동 선택된다(setup/sdk_versions.json 기준).
#   SDK 1.0v → qbcompiler:1.1-{cpu|cuda12.8.1}-ubuntu22.04
#   SDK 1.1v → qbcompiler:1.2-{cpu|cuda12.8.1}-ubuntu22.04
#
# ★ GPU 서버에서 쓰면 calibration·OPTQ 연산이 GPU로 가서 크게 빨라진다.
#   (OPTQ 없는 단순 컴파일은 GPU 이득이 작다 — reports/performance/compile_benchmark.md)
#   전제: NVIDIA 드라이버 + nvidia-container-toolkit 설치(=`docker run --gpus all` 가능)
#
# 사용:
#   bash setup/compile_in_docker.sh --calib-data-path download/calib_coco_hwc \
#        --save out/pe_w4a16.mxq --quant w4a16 --optq --scheme single
#   bash setup/compile_in_docker.sh --sdk 1.1 --cpu ...        # 이미지/디바이스 강제
#
# `--` 이후 인자는 그대로 `python -m pe_npu.compile --mode compile` 에 전달된다.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
SDK_ARG="${SDK_VERSION:-}"
NAME="${CONTAINER_NAME:-mblt_compiler}"
FORCE=""
PASS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --sdk)   SDK_ARG="$2"; shift 2 ;;
    --sdk=*) SDK_ARG="${1#*=}"; shift ;;
    --gpu)   FORCE="gpu"; shift ;;
    --cpu)   FORCE="cpu"; shift ;;
    --name)  NAME="$2"; shift 2 ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) PASS+=("$1"); shift ;;
  esac
done

[ -f "$REPO_ROOT/.env" ] && { set -a; . "$REPO_ROOT/.env"; set +a; }
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO=$( [ -n "${SUDO_PASS:-}" ] && echo "sudo -S" || echo "sudo" )
d() { if [ -n "${SUDO_PASS:-}" ] && [ -n "$SUDO" ]; then echo "$SUDO_PASS" | sudo -S docker "$@"; else $SUDO docker "$@"; fi }

PY="$(command -v python3 || command -v python)"
eval "$("$PY" "$REPO_ROOT/setup/sdk_resolve.py" ${SDK_ARG:+--sdk "$SDK_ARG"} --shell)"

# GPU 유무 판정: 강제 지정 > 실제 장치 존재
HAS_GPU=0
[ -e /dev/nvidiactl ] || [ -e /proc/driver/nvidia/version ] && HAS_GPU=1
[ "$FORCE" = "gpu" ] && HAS_GPU=1
[ "$FORCE" = "cpu" ] && HAS_GPU=0
if [ "$HAS_GPU" = "1" ]; then
  IMAGE="$DOCKER_IMAGE_CUDA"; GPU_FLAG="--gpus all"; CALIB_DEV="gpu"
else
  IMAGE="$DOCKER_IMAGE_CPU";  GPU_FLAG="";          CALIB_DEV="cpu"
fi

echo "[compile] SDK $SDK_LABEL | 이미지 $IMAGE | calibration device=$CALIB_DEV"
echo "[compile] 컨테이너 $NAME 준비"
d rm -f "$NAME" >/dev/null 2>&1 || true
# shellcheck disable=SC2086
d run -dit --ipc=host --name "$NAME" $GPU_FLAG \
  -v "$REPO_ROOT":/workspace \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -w /workspace "$IMAGE" bash >/dev/null

COMPILER_WHL_IN=$(basename "${COMPILER_PATH:-}")
if [ -n "$COMPILER_WHL_IN" ]; then
  REL="${COMPILER_PATH#"$REPO_ROOT"/}"
  echo "[compile] qbcompiler 설치: $COMPILER_WHL_IN"
  d exec "$NAME" pip install -q "/workspace/$REL" 2>&1 | grep -v WARNING || true
fi
d exec "$NAME" pip install -q onnxruntime 2>&1 | grep -v WARNING || true
d exec "$NAME" python -c "import qbcompiler.mmc" >/dev/null 2>&1 \
  && echo "[compile] mmc OK" || { echo "[compile] mmc import 실패 — 이미지/버전 확인 필요"; exit 1; }

echo "[compile] 실행: python -m pe_npu.compile --mode compile --device $CALIB_DEV ${PASS[*]}"
d exec -e HF_TOKEN="${HF_TOKEN:-}" -w /workspace "$NAME" \
  python -m pe_npu.compile --mode compile --device "$CALIB_DEV" "${PASS[@]}"
echo "[compile] 완료 — 컨테이너 $NAME 는 남겨둔다(재사용). 정리: docker rm -f $NAME"
