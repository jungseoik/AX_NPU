#!/usr/bin/env bash
# 호스트에 Mobilint 런타임 + CLI 설치 (mobilint-cli 등을 호스트에서 바로 쓰기 위함).
#
# 받아둔 런타임 tar의 'make install'을 실행한다. 설치되는 것:
#   - libqbruntime.so*  → /usr/local/lib/x86_64-linux-gnu  (+ ldconfig)
#   - mobilint-cli, mblt-status, mblt-benchmark, mblt-mxqtool, mblt-testinfer → /usr/local/bin
#   - 헤더 → /usr/local/include
# Python 바인딩은 건드리지 않으므로 호스트 conda(3.13)와 무관하다.
# (Python 추론은 cp313 wheel이 없어 호스트 불가 → 추론은 계속 컨테이너에서)
#
# 사용: sudo bash install_runtime_host.sh

set -e
DL=/home/gpuadmin/Repo/seoik/AX_NPU/AX_NPU/download
TAR=$DL/qbruntime_aries2-v4_v1.2.0_amd64.tar.gz
RT=$DL/qbruntime_aries2-v4_v1.2.0_amd64

echo "[1/2] 런타임 패키지 전개"
[ -d "$RT" ] || tar xzf "$TAR" -C "$DL"
cd "$RT"

echo "[2/2] make install (libqbruntime + mobilint-cli)"
make install

echo "완료. 확인:"
echo "  mobilint-cli status        # NPU 상태/펌웨어 버전 (nvidia-smi 류)"
echo "  mblt-mxqtool show <file>   # MXQ 정보"
