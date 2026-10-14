# -*- coding: utf-8 -*-
"""운영(macs-inference) 세팅 동결.

값의 출처는 전부 **살아있는 매니페스트**다. 임의로 고르지 않았다:
    kubectl get deploy -n macs-inference es-inference-node0-gpu1 -o json
    kubectl get deploy -n macs-inference es-pe-validation        -o json

여기 값을 바꾸면 "운영과 동일" 전제가 깨진다. 바꿀 땐 매니페스트도 같이 확인할 것.
"""
from __future__ import annotations

import os

# ── 경로 (전부 읽기 전용으로만 접근한다) ──────────────────────────────────────
AX = "/home/gpuadmin/AX_NPU"
MONO = f"{AX}/third_party/Product-AI-mono"
HERE = os.path.dirname(os.path.abspath(__file__))

# 데이터셋 — **참조만 한다. 절대 쓰지 않는다.**
DATASET = f"{AX}/eval/datasets/TTA_인증용"
CATEGORIES = ("falldown", "fire", "smoke")          # intrusion 제외(2차 검증 비대상)

# 카테고리별 평가에서 **제외**할 소속 폴더.
#   화재 영상은 화재만 본다 — fire 폴더 50건 전부가 smoke 라벨을 함께 달고 있어
#   연기 평가에 쓰면 모호하다. GT 를 0 으로 깔면(= 음성 취급) 실제로 연기가 보이는
#   화면이라 연기 검출이 전량 오탐으로 집계되어 정밀도가 부당하게 깎인다.
#   그래서 음성으로 쓰지 않고 **분모에서 뺀다**.
EVAL_EXCLUDE = {"smoke": ("fire",)}

# ── 1차 PE ───────────────────────────────────────────────────────────────────
# es-inference-node0-gpu1 env 그대로.
#   MODEL_PE_NPU_MXQ_PATH="" + QUANT/TUNING/SCHEME 조합 -> HF 캐시 경로로 해석된다.
#   (hostPath 로 마운트된 pe_w8a16_global4.mxq 는 env 가 비어 있어 참조되지 않는다)
PE_QUANT = "W4A16"
PE_TUNING = "sws_optq"
PE_SCHEME = "single"
PE_MXQ = (f"/home/models/huggingface/hub/models--PIA-SPACE-LAB--MXQ_NPU/snapshots/"
          f"f76e03d1f8529403bdd57f27eb4e847e602b650e/{PE_QUANT}/{PE_TUNING}/{PE_SCHEME}/pe_full.mxq")

# MODEL_PERCEPTION_ENCODER_TXT_FEATURE_PATH=/npu-assets/text_features.json 의 hostPath 원본
TEXT_FEATURES = f"{MONO}/assets/model/text_features.json"

PE_IMG_SIZE = (336, 336)        # pe_npu/config.py IMG_SIZE
PE_METHOD = "topk"              # PE_NPU_METHOD 미설정 -> config.py 기본값
PE_TOP_CANDIDATE = 13           # PE_TOP_CANDIDATE 미설정 -> 기본 13
# ★ 알람 FSM 은 **3창 중 2** 다. config.py 기본값(5창중3)이 아니다.
#   이 두 키는 Deployment env 가 아니라 **envFrom 의 ConfigMap** 에 있다:
#     kubectl get cm -n macs-inference inference-server-config -o json
#       PE_QUEUE_SIZE = 3 / PE_ALARM_DURATION_THRESHOLD = 2
#   env 만 보고 "미설정 -> 기본값" 으로 단정하면 틀린다(실제로 한 번 틀렸다).
PE_QUEUE_SIZE = 3               # ConfigMap inference-server-config
PE_ALARM_DURATION = 2           # ConfigMap inference-server-config
PE_JUDGE_MODE = "frame"         # PE_NPU_JUDGE_MODE=frame

# 입력 프레임률. node-gpu-model-mappings.json 의 PENpuService fps=1.
# ★ 알람 FSM(5창 중 3) 이 '초' 단위 의미를 가지려면 1fps 로 넣어야 한다.
#   24fps 로 넣으면 같은 FSM 이 0.2초 창이 되어 운영과 다른 동작이 된다.
FPS = 1

# 디코딩 병렬 워커 수. NPU 가 아니라 영상 디코딩이 병목이라(실측 NPU 2.7%) 여기를 늘린다.
DECODE_WORKERS = int(os.environ.get("DECODE_WORKERS", "24"))

# ── 2차 VQA ──────────────────────────────────────────────────────────────────
# 실증(DTRO) 로그 기준: model=PIA-SPACE/AX_NPU_VL3_2B_mxq:ft/model, resize 256x192, frames=1
# model-ft2 는 그 배포본과 sha256 일치를 확인한 로컬 사본이다.
VQA_IMAGE = "npu-pe-2stage-validation-bs32:latest"
VQA_MODEL_DIR = f"{AX}/download/vendor/model-ft2"
# ★ 2차 문안의 정식 출처는 2stage_prompts 의 **ax_npu** variant 다.
#   서버 prompts.py 해석 순서: ① VQA_VALIDATION_PROMPTS_YAML(명시)
#                            ② VQA_VALIDATION_PROMPTS_ASSET_DIR/<variant>
#                            ③ HF PIA-SPACE-LAB/2stage_prompts/<variant>
#   DEFAULT_VARIANT="ax_npu". 이 서버엔 로컬 자산이 없어 ③ 으로 풀린다.
#   ax_tta/vqa_prompts.yaml 은 내용이 같더라도 **출처가 아니다** — 갈라지면 조용히 틀려진다.
#   (서버 주석: "실제로 운영 문안이 레포 상수와 어긋난 채로 돌고 있었다")
VQA_PROMPT_VARIANT = os.environ.get("VQA_VALIDATION_PROMPT_VARIANT", "ax_npu")
VQA_PROMPTS_HF_REPO = "PIA-SPACE-LAB/2stage_prompts"


def resolve_prompts_yaml() -> str:
    """서버 prompts.py 와 동일한 우선순위로 prompts.yaml 경로를 찾는다."""
    import glob
    explicit = os.environ.get("VQA_VALIDATION_PROMPTS_YAML", "").strip()
    if explicit:
        if not os.path.isfile(explicit):
            raise FileNotFoundError(f"VQA_VALIDATION_PROMPTS_YAML 이 파일이 아님: {explicit}")
        return explicit
    asset_dir = os.environ.get("VQA_VALIDATION_PROMPTS_ASSET_DIR",
                               "assets/model/2stage_prompts")
    local = os.path.join(asset_dir, VQA_PROMPT_VARIANT, "prompts.yaml")
    if os.path.isfile(local):
        return local
    pat = (f"*/hub/models--{VQA_PROMPTS_HF_REPO.replace('/', '--')}"
           f"/snapshots/*/{VQA_PROMPT_VARIANT}/prompts.yaml")
    hits = sorted(glob.glob(os.path.join("/home/models/huggingface", pat[2:]))
                  + glob.glob(os.path.expanduser("~/.cache/huggingface/" + pat[2:])))
    if not hits:
        raise FileNotFoundError(
            f"{VQA_PROMPTS_HF_REPO}/{VQA_PROMPT_VARIANT}/prompts.yaml 을 찾지 못했다. "
            "HF 캐시에 없으면 받아두거나 VQA_VALIDATION_PROMPTS_YAML 로 지정할 것.")
    return hits[-1]


VQA_PROMPTS = resolve_prompts_yaml()
VQA_SIZE = (256, 192)           # (w, h) — secondary_validation.resize
VQA_FRAMES = 1                  # secondary_validation.frames

# ── 산출물 ───────────────────────────────────────────────────────────────────
OUT = f"{HERE}/out"
EVIDENCE = f"{HERE}/evidence"
