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

# ★ qbruntime 은 /sys/class/aries 로 장치를 "열거"한 뒤 /dev/ariesN 을 연다.
#   일부 카드만 넘기면 열거 단계에서 실패한다(Acc_DeviceNotFound).
#   → 노드는 전부 넘기고 /sys 를 읽기전용으로 마운트하되, 실제로 여는 카드는
#     --device 인덱스로 하나만 고른다. privileged 는 쓰지 않는다.
DEV_ARGS=()
for n in 0 1 2 3 4 5 6 7; do
  [ -e "/dev/aries$n" ] && DEV_ARGS+=(--device "/dev/aries$n:/dev/aries$n")
done
DEV_ARGS+=(-v /sys:/sys:ro)
for n in $DEVICES; do
  [ -e "/dev/aries$n" ] || { echo "없는 카드: /dev/aries$n"; exit 1; }
done
echo "[run] 이미지 $IMAGE | 사용 카드: $DEVICES (열거용으로 전 노드 전달) | 패키지 $REPRO"
# TTY 가 없으면(-it 불가) 비대화 모드로 — nohup/백그라운드 실행 대응
TTY=(-i); [ -t 0 ] && TTY=(-it)
# CPU 격리: 벤더 기준 호스트는 유휴 상태 4스레드다. 이 서버는 운영 파드가 전 코어를 쓰므로
# cpuset 으로 최소한의 격리를 한다. CPUSET="" 로 끄면 전체 코어를 쓴다.
CPUSET="${CPUSET:-}"
CPU_ARGS=(); [ -n "$CPUSET" ] && CPU_ARGS=(--cpuset-cpus "$CPUSET")
[ -n "$CPUSET" ] && echo "[run] cpuset=$CPUSET"

if [ "${1:-}" = "--shell" ]; then
  # shellcheck disable=SC2086
  exec bash -c "$(declare -f d); d run --rm -it ${DEV_ARGS[*]} -v '$REPRO':/repro -v '$REPO_ROOT/setup/vlm_repro':/wrap:ro -w /repro '$IMAGE' bash"
fi
# shellcheck disable=SC2086
# 벤더 패키지를 수정하지 않기 위해 래퍼로 실행한다(무결성 검사 통과 유지).
case " $* " in *" --device "*) EXTRA=() ;; *) EXTRA=(--device "$RUN_DEVICE") ;; esac
echo "[run] device=${EXTRA[1]:-(인자 지정)}"
d run --rm "${TTY[@]}" "${CPU_ARGS[@]}" "${DEV_ARGS[@]}" -v "$REPRO":/repro \
  -v "$REPO_ROOT/setup/vlm_repro":/wrap:ro -w /repro "$IMAGE" \
  python3.10 /wrap/run_table_device.py "${EXTRA[@]}" "$@"
