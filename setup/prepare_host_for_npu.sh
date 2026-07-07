#!/usr/bin/env bash
# NPU 드라이버 설치 — APT 우회(tar 소스 직접 빌드) 버전.
#
# 이 호스트는 apt 업그레이드 보류가 511개(mesa/udev/dbus 등) 쌓인 stale 상태라,
# 'apt install mobilint-aries-driver'가 의존성 충돌로 막힌다(2026-06-15 확인).
# 시스템 전체 업그레이드(apt full-upgrade)는 GPU 워크스테이션에 영향이 크므로,
# 받아둔 드라이버 tar를 직접 빌드해 설치한다. (커널 모듈만 추가하므로 시스템 안전)
#
# 사전요건(확인됨): Secure Boot disabled, linux-headers/build-essential 설치, gcc-13.
# 사용: sudo bash prepare_host_for_npu.sh

set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # 레포 루트 동적 (<repo>/setup/)
DRIVER_TAR="$(cd "$HERE/.." && pwd)/download/mobilint-aries2-driver_v1.13.tar.gz"
BUILD_DIR=/tmp/aries-driver-build

echo "================ Mobilint 드라이버 설치 (tar 빌드) ================"

echo "[1/4] 소스 전개"
rm -rf "$BUILD_DIR"; mkdir -p "$BUILD_DIR"
tar xzf "$DRIVER_TAR" -C "$BUILD_DIR"
SRC=$(find "$BUILD_DIR" -mindepth 1 -maxdepth 1 -type d -name "aries-driver*")
echo "       $SRC"

echo "[2/4] 커널 모듈 빌드 (make)"
make -C "$SRC"
ls "$SRC"/aries.ko >/dev/null && echo "       aries.ko 빌드 완료"

echo "[3/4] 모듈 설치 + depmod (modules_install)"
make -C "$SRC" install
depmod -a

echo "[4/4] 부팅 자동 로드 등록"
echo aries > /etc/modules-load.d/aries.conf
echo "       /etc/modules-load.d/aries.conf"

echo "=================================================================="
echo "드라이버 설치 완료. 다음 순서:"
echo "  1) (전원 OFF) NPU 카드를 PCIe x8 슬롯에 장착"
echo "  2) 부팅 후: sudo modprobe aries   (또는 재부팅 시 자동 로드)"
echo "  3) 확인: bash check_npu.sh   (/dev/aries0 생성 여부)"
echo "  4) 펌웨어 최신화(선택): aries_flash_firmware update"
echo "  5) 추론은 컨테이너에서 (--device /dev/aries0). README.md 참고"
echo
echo "참고: 커널이 업데이트되면(uname -r 변경) 이 스크립트를 다시 실행해 모듈을 재빌드하세요."
echo "      (dkms 미설치 환경이라 자동 재빌드되지 않습니다.)"
