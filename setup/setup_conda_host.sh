#!/usr/bin/env bash
# 호스트 conda 환경에서 PE NPU 추론을 돌리기 위한 셋업 (docker 없이).
#
# docker가 필수가 아니다. NPU 추론에 필요한 것은:
#   - qbruntime (Python wheel)  + libqbruntime.so
#   - /dev/aries0 (드라이버, 호스트에 이미 설치)
#   - torch/einops/timm/huggingface_hub (CPU pool head + vendored PE 모델 코드)
# 호스트 conda(base)가 Python 3.13이면 qbruntime wheel(cp38~cp312)이 안 맞으므로,
# Python 3.10~3.12 전용 env를 따로 만든다.
#
# 사용:  bash setup_conda_host.sh [env이름(기본 pe_npu_host)] [python버전(기본 3.11)]

set -e
ENV="${1:-pe_npu_host}"
PYVER="${2:-3.11}"
DL=/home/gpuadmin/Repo/seoik/AX_NPU/AX_NPU/download
RT="$DL/qbruntime_aries2-v4_v1.2.0_amd64"

# qbruntime wheel은 cp{버전} 매칭 필요 (3.11→cp311)
CP="cp$(echo "$PYVER" | tr -d .)"
WHL=$(ls "$RT"/qbruntime/qbruntime/python/*"$CP"*.whl 2>/dev/null | head -1)
[ -z "$WHL" ] && { [ -d "$RT" ] || tar xzf "$DL/qbruntime_aries2-v4_v1.2.0_amd64.tar.gz" -C "$DL"; WHL=$(ls "$RT"/qbruntime/qbruntime/python/*"$CP"*.whl | head -1); }

echo "[1/4] conda env 생성: $ENV (python $PYVER)"
conda create -y -n "$ENV" python="$PYVER" >/dev/null

echo "[2/4] qbruntime 설치 ($CP wheel)"
conda run -n "$ENV" pip install -q numpy pillow "$WHL"

echo "[3/4] pool head + 모델 deps 설치 (CPU torch)"
conda run -n "$ENV" pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
conda run -n "$ENV" pip install -q einops timm huggingface_hub
conda run -n "$ENV" pip install -q open_clip_torch   # 텍스트 분류 데모(즉석 텍스트 인코딩, CLIP BPE 토크나이저)
conda run -n "$ENV" pip install -q matplotlib jupyter ipykernel   # 노트북 튜토리얼용

# (선택) Qwen3-VL 등 VLM 추론을 같은 env에서 쓰려면 WITH_VLM=1 로 실행.
#   mblt-model-zoo는 1.3.1로 핀(런타임 1.2.0 호환), transformers는 4.57(Qwen3-VL 지원)으로 올린다.
#   perception-models의 pillow/tokenizers 핀 충돌 경고가 뜨지만 PE-Core·VLM 둘 다 정상 동작(실측).
#   → tutorial/pe_npu/README_VLM_qwen3.md
if [ "${WITH_VLM:-0}" = "1" ]; then
  echo "[+] VLM(mblt-model-zoo 1.3.1 + transformers 4.57.1) 설치"
  conda run -n "$ENV" pip install -q "mblt-model-zoo[transformers]==1.3.1" "transformers==4.57.1"
fi

echo "[4/4] libqbruntime.so 확인"
if ldconfig -p 2>/dev/null | grep -q qbruntime; then
  echo "    libqbruntime.so 등록됨 (LD_LIBRARY_PATH 불필요)"
else
  echo "    ⚠️ libqbruntime.so 미등록 → 'sudo bash install_runtime_host.sh' 실행 필요"
  echo "       또는: export LD_LIBRARY_PATH=$RT/qbruntime/qbruntime/lib:\$LD_LIBRARY_PATH"
fi

echo "====================================================="
echo "완료. 호스트 conda에서 추론:"
echo "  conda activate $ENV"
echo "  cd /home/gpuadmin/Repo/seoik/AX_NPU/AX_NPU"
echo "  python tutorial/pe_npu/download_images.py"
echo "  python tutorial/pe_npu/demo_inference.py     # 원본 대비 cos 0.997"
echo "  # 코드: import pe_npu; m=pe_npu.MXQInferenceHybrid(); m.infer(x)"
