#!/usr/bin/env bash
# customer-capacity-repro 런타임 컨테이너 실행 래퍼.
#
# 호스트 드라이버(1.13)를 그대로 쓰고 지정한 NPU 카드만 넘긴다.
# 기본 카드는 이 서버에서 비어 있는 aries1 / aries7 이다(나머지는 운영 파드 점유).
#
#   bash setup/vlm_repro/run_runtime.sh --sizes 224x224 --repetitions 2 --output results/smoke
#   NPU_DEVICES="7" bash setup/vlm_repro/run_runtime.sh ... # 카드 지정
#   bash setup/vlm_repro/run_runtime.sh --shell             # 셸만 띄우기
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
REPRO="${REPRO_DIR:-$REPO_ROOT/download/vendor/customer-capacity-repro}"
IMAGE="${IMAGE:-mblt_vlm_repro:rt}"
DEVICES="${NPU_DEVICES:-1 7}"
# run_table.py 에 넘길 device 인덱스(첫 카드). setup/vlm_repro/apply_patch.py 로 추가된 인자.
RUN_DEVICE="${RUN_DEVICE:-$(echo $DEVICES | awk "{print \$1}")}"

[ -d "$REPRO" ] || { echo "재현 패키지 없음: $REPRO"; exit 1; }
[ -f "$REPO_ROOT/.env" ] && { set -a; . "$REPO_ROOT/.env"; set +a; }
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO=1
d() { if [ -n "$SUDO" ] && [ -n "${SUDO_PASS:-}" ]; then echo "$SUDO_PASS" | sudo -S docker "$@"; else docker "$@"; fi }

DEV_ARGS=()
for n in $DEVICES; do
  [ -e "/dev/aries$n" ] || { echo "없는 카드: /dev/aries$n"; exit 1; }
  DEV_ARGS+=(--device "/dev/aries$n:/dev/aries$n")
done
echo "[run] 이미지 $IMAGE | 카드: $DEVICES | 패키지 $REPRO"

if [ "${1:-}" = "--shell" ]; then
  # shellcheck disable=SC2086
  exec bash -c "$(declare -f d); d run --rm -it ${DEV_ARGS[*]} -v '$REPRO':/repro -w /repro '$IMAGE' bash"
fi
# shellcheck disable=SC2086
# --device 가 인자에 없으면 첫 카드로 채운다.
case " $* " in *" --device "*) EXTRA=() ;; *) EXTRA=(--device "$RUN_DEVICE") ;; esac
echo "[run] run_table.py --device ${EXTRA[1]:-(인자 지정)}"
d run --rm -it "${DEV_ARGS[@]}" -v "$REPRO":/repro -w /repro "$IMAGE" \
  python3.10 scripts/run_table.py "$@" "${EXTRA[@]}"
