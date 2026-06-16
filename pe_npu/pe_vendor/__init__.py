"""Vendored Meta Perception Encoder vision encoder code.

이 디렉토리는 Meta `perception_models`(facebook/PE-Core)의 vision encoder 코드를
AX_NPU 레포에 그대로 복사(vendor)한 것이다. AX_NPU를 단독으로 clone해서 사용할 수
있도록 외부 레포(Product-AI-mono) 의존성을 제거하기 위함이다.

출처: perception_models/pe_core/vision_encoder/{pe.py, config.py, rope.py}
원본 라이선스: Copyright (c) Meta Platforms, Inc. and affiliates.
가중치는 여기 포함되지 않으며 HuggingFace `facebook/PE-Core-*`에서 자동 다운로드된다.

원본과의 유일한 차이: pe.py 내부 import를 절대경로(pia.ai...)에서 상대경로(.config/.rope)로 변경.
"""
