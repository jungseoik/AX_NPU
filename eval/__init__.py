"""eval — NPU 모델 평가용 데이터셋 관리.

실데이터는 git에 넣지 않고 HuggingFace Hub에 zip으로 두고, 토큰만 있으면
`python -m eval.tta download`로 어느 서버에서든 동일하게 재현한다.
"""
