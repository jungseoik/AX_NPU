#!/usr/bin/env bash
# 신규 Qwen3-VL-2B 체크포인트 → batch32 mxq 까지 한 번에.
#
#   bash run_all.sh --checkpoint /path/to/my-finetuned --out /data/mybuild
#
# 전제: GPU 서버(CUDA 12.8), 벤더 패키지 전개됨, mblt_vlm_repro:compile 이미지 빌드됨.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
VENDOR="${VENDOR:-$ROOT/download/vendor/customer-capacity-repro}"
IMAGE="${IMAGE:-mblt_vlm_repro:compile}"
IMAGES="${IMAGES:-$ROOT/download/coco/val2017}"
SIZE="${SIZE:-224}"
PARTS="${PARTS:-vision decoder-b32}"     # b1/vision8192 는 선택
CKPT=""; OUT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --checkpoint) CKPT="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --size) SIZE="$2"; shift 2;;
    --parts) PARTS="$2"; shift 2;;
    *) echo "모르는 인자: $1"; exit 1;;
  esac
done
[ -n "$CKPT" ] && [ -n "$OUT" ] || { echo "사용법: $0 --checkpoint <경로> --out <경로>"; exit 1; }
[ -d "$VENDOR" ] || { echo "벤더 패키지 없음: $VENDOR"; exit 1; }
command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null || { echo "❌ CUDA GPU 필요"; exit 1; }
mkdir -p "$OUT"
D=(--gpus all --shm-size 32g -v "$CKPT":/ckpt:ro -v "$VENDOR":/vendor:ro
   -v "$IMAGES":/images:ro -v "$OUT":/out -v "$HERE":/tool:ro)
R() { docker run --rm "${D[@]}" --entrypoint python3.10 "$IMAGE" "$@"; }

echo "═══ ① mblt export ═══ $(date +%T)"
R /tool/export_mblt.py --checkpoint /ckpt --out /out/mblt \
    --image /vendor/inputs/images/dog720.jpg --size "$SIZE"

echo "═══ ② 캘리브 생성 ═══ $(date +%T)"
R /tool/make_calib.py --checkpoint /ckpt --images /images --out /out/calib \
    --mblt-dir /out/mblt --size "$SIZE"

for part in $PARTS; do
  echo "═══ ③ compile $part ═══ $(date +%T)"
  R /tool/compile_mxq.py --part "$part" --mblt-dir /out/mblt \
      --calib-dir /out/calib --vendor /vendor --out /out/mxq
done

echo "═══ ④ 모델 디렉토리 조립 ═══"
M="$OUT/model"; mkdir -p "$M/variants"
for f in "$VENDOR"/inputs/support/*; do [ -f "$f" ] && cp -n "$f" "$M/" || true; done
cp -n "$CKPT"/{config.json,tokenizer*,vocab.json,merges.txt,*.safetensors} "$M/" 2>/dev/null || true
cp "$VENDOR/inputs/rotation.pth" "$M/" 2>/dev/null || true
[ -f "$OUT/mxq/decoder-b32/model.mxq" ] && cp "$OUT/mxq/decoder-b32/model.mxq" "$M/decoder.mxq"
[ -f "$OUT/mxq/vision/model.mxq" ]      && cp "$OUT/mxq/vision/model.mxq"      "$M/vision.mxq"
[ -f "$OUT/mxq/decoder-b1/model.mxq" ]  && cp "$OUT/mxq/decoder-b1/model.mxq"  "$M/variants/decoder-b1.mxq"
echo "  → $M"
ls -la "$M"/*.mxq 2>/dev/null | awk '{printf "    %8.1fMB %s\n",$5/1048576,$9}'

cat <<MSG

═══ 완료 $(date +%T) ═══
NPU 가 있는 서버에서 기존 batch32 와 비교:

  python3 verify_npu.py \\
      --baseline  <기존 model-reference 경로> \\
      --candidate $M \\
      --images <img1.jpg> <img2.jpg> --card 0
MSG
