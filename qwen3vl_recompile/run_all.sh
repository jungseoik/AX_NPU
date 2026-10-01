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

echo "═══ ④ 모델 디렉토리 조립 ═══ $(date +%T)"
# ★ 단순 복사로는 안 된다. 두 가지를 반드시 지켜야 한다(pack_model_dir.py 가 처리):
#   · config.json 은 **벤더 것**(NPU 런타임 설정). 학습본 config 로 덮으면 모델이 안 뜬다.
#   · embed_tokens 는 **rotation.pth 로 회전**해서 넣는다. 생으로 넣으면 컴파일·지연은
#     멀쩡한데 로짓만 ~5배 눌려 엉뚱한 토큰이 나온다.
REF="${REF:-$VENDOR/build/model-reference}"
[ -d "$REF" ] || { echo "❌ 벤더 레퍼런스 디렉토리 없음: $REF (REF=... 로 지정)"; exit 1; }
R /tool/pack_model_dir.py --vendor /vendor/build/model-reference --checkpoint /ckpt \
    --vision-mxq /out/mxq/vision/model.mxq \
    --decoder-mxq /out/mxq/decoder-b32/model.mxq \
    --out /out/model
[ -f "$OUT/mxq/decoder-b1/model.mxq" ] && cp "$OUT/mxq/decoder-b1/model.mxq" "$OUT/model/variants/decoder-b1.mxq"
M="$OUT/model"

cat <<MSG

═══ 완료 $(date +%T) ═══
NPU 가 있는 서버에서:

① 비전 출력 순서 확정 (이름 매칭이 안 되므로 **수치로** 정한다)
  python3 compare_vision_mxq.py --a <검증된 model-reference> --b $M \\
      --image <img.jpg> --card 0
  → 행·열 모두 최대인 1:1 대응을 $M/vision_output_order.json 에 적는다

② 기존 batch32 와 A/B
  python3 verify_npu.py --baseline <기존 model-reference> --candidate $M \\
      --images <img1.jpg> <img2.jpg> --card 0

  라벨이 어긋나면 dump_logits.py 로 **로짓 크기**를 먼저 본다 —
  순서는 맞는데 스케일만 눌렸으면 기저(회전) 문제, 순서가 흐트러졌으면 양자화 문제다.
MSG
