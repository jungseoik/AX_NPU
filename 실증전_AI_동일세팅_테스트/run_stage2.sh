#!/usr/bin/env bash
# 2차 검출기 실행 — 운영 bs32 검증서버를 **있는 그대로** 띄우고 HTTP 로 묻는다.
#
# ENTRYPOINT 를 덮어쓰지 않는다. 덮어쓰면 VLMPool(카드 선택·배치 묶기·세마포어)을
# 건너뛰고 Classifier 를 직접 잡게 되어, 그 순간 "운영 동일" 이 아니게 된다.
#   계약: POST /v1/chat/completions, GET /health, 포트 8100
#   참조: third_party/.../pia_2stage/validation_server_npu_bs32/README.md
set -euo pipefail
cd "$(dirname "$0")"
AX=/home/gpuadmin/AX_NPU
NAME=ax-2stage-bs32

d() { echo "${SUDOPW:-}" | sudo -S "$@"; }
cleanup() { d docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT
cleanup

d docker run -d --name "$NAME" \
  --device /dev/aries0 --device /dev/aries1 --device /dev/aries2 --device /dev/aries3 \
  --device /dev/aries4 --device /dev/aries5 --device /dev/aries6 --device /dev/aries7 \
  -v /sys:/sys:ro --shm-size 16g -p 127.0.0.1:8100:8100 \
  -v "$AX/download/vendor/model-ft2":/model:ro \
  -e NPU_PE_2STAGE_VLM_MODEL=/model \
  -e NPU_PE_2STAGE_MODEL_AUTO_DOWNLOAD=false \
  -e NPU_PE_2STAGE_VLM_SIZE="${VLM_SIZE:-256x192}" \
  -e NPU_PE_2STAGE_VLM_DEVICE_IDS="${DEVICE_IDS:-auto}" \
  npu-pe-2stage-validation-bs32:latest >/dev/null

echo "서버 기동 대기 (8장 모델 로드에 1~2분)..."
for i in $(seq 1 240); do
  curl -sf http://127.0.0.1:8100/health >/dev/null 2>&1 && { echo "  준비됨 (${i}s)"; break; }
  d docker ps -q --filter "name=$NAME" 2>/dev/null | grep -q . || {
    echo "  [!] 컨테이너가 죽었다:"
    d docker logs "$NAME" 2>&1 | grep -vE 'rdkafka|^I[0-9]{4} ' | tail -25; exit 1; }
  sleep 1
done
curl -sf http://127.0.0.1:8100/health >/dev/null || { echo "  [!] health 실패"; exit 1; }

d docker logs "$NAME" 2>&1 | grep -E '\[entrypoint\]|VLMPool|준비 완료' | tail -8 || true
echo
~/miniconda3/envs/gov_vlm/bin/python stage2_client.py --workers "${WORKERS:-32}"
