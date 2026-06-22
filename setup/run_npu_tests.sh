#!/usr/bin/env bash
# NPU 장착 후 통합 테스트 — 한 번에 추론 컨테이너 구성 + 추론 + 원본 비교까지 실행.
#
# 전제: NPU 카드 장착 + 드라이버 로드(/dev/aries0 존재) 상태.
#   드라이버가 안 올라와 있으면 먼저:  sudo modprobe aries
#
# 사용:  bash run_npu_tests.sh
#   (docker 명령을 쓰므로 docker 권한 필요. sudo 불필요하면 그대로, 필요하면 sudo로)

set -e
ROOT=/home/gpuadmin/Repo/seoik/AX_NPU
IMG=mobilint/qbcompiler:1.1-cuda12.8.1-ubuntu22.04
LDLIB=/tmp/qbruntime_aries2-v4_v1.2.0_amd64/qbruntime/qbruntime/lib
TUT=/workspace/AX_NPU/tutorial_pe_npu               # 컨테이너 내부 경로

step() { echo; echo "==================== $1 ===================="; }

step "[0/3] NPU 디바이스 확인"
if ! ls /dev/aries0 >/dev/null 2>&1; then
  echo "  /dev/aries0 없음 → 먼저 'sudo modprobe aries' (또는 카드 장착/재부팅) 후 다시 실행"
  echo "  상세 점검: bash setup/check_npu.sh"
  exit 1
fi
echo "  OK: $(ls /dev/aries*)"

step "[1/3] 추론 컨테이너 재생성 (NPU 기본, GPU 옵션)"
docker rm -f mblt_compiler 2>/dev/null || true
# GPU는 옵션: nvidia 디바이스가 있을 때만 --gpus all 추가. NPU(/dev/aries0)는 항상 매핑.
GPU_OPT=$([ -e /dev/nvidia0 ] && echo "--gpus all" || echo "")
docker run -dit $GPU_OPT --ipc=host --name mblt_compiler \
  --device /dev/aries0:/dev/aries0 \
  -v "$ROOT":/workspace -w /workspace "$IMG" /bin/bash
echo "  컴파일러/런타임 설치..."
docker exec mblt_compiler pip install -q /workspace/AX_NPU/download/qbcompiler-1.1.2+aries2-py3-none-any.whl
docker exec mblt_compiler bash -lc 'cd /tmp && tar xzf /workspace/AX_NPU/download/qbruntime_aries2-v4_v1.2.0_amd64.tar.gz && pip install -q /tmp/qbruntime_aries2-v4_v1.2.0_amd64/qbruntime/qbruntime/python/*cp310*.whl'
docker exec mblt_compiler pip install -q onnxruntime

step "[2/3] hybrid 추론 데모 (NPU trunk + CPU pool) + 원본 PyTorch 대비 정확도"
docker exec -w "$TUT" mblt_compiler python download_images.py
docker exec -w "$TUT" mblt_compiler bash -lc "
  export LD_LIBRARY_PATH=$LDLIB:\$LD_LIBRARY_PATH
  python demo_inference.py"

step "[3/3] 완료"
echo "  추론/정확도 테스트가 끝났습니다. 위 '평균 cos'(≥0.99 권장)를 확인하세요."
echo "  유사도가 낮으면: 실제 도메인 이미지로 calibration 후 재컴파일 (python -m pe_npu.compile)"
