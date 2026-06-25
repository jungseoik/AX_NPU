안녕하세요.

모빌린트 전근우입니다.

해당 모델의 성능 저하 원인 분석 및 성능 복구 완료하였습니다.
 ​attn_pool_repro_fixed.zip​ (암호: attn)

에러 원인은 Attention Module의 QK matmul에서 심한 outlier가 있어 양자화 error가 크게 발생한 것이었습니다.

해결 방법은, 해당 레이어의 output을 16bit로 변경하는 것입니다.
이를 통해 cos sim이 0.6976에서 0.9962로 크게 개선되었습니다.

compile_head_fixed.py 를 실행하여 직접 재현해보실 수 있습니다.
이 스크립트에 16bit 레이어를 지정하는 코드가 있습니다.
전체 layer 이름을 정확하게 입력해야 16bit layer를 지정할 수 있는데요.
full model에서는 보내주신 모델에서와 layer 이름이 다를 수 있습니다.
그래서 첨부드린 script에서는 QK matmul을 layer pattern으로 탐지해서, 16bit layer를 동적으로 지정하도록 해두었습니다.
참고하셔서 full model에 적용하시면 양자화 성능이 개선되실 것으로 예상됩니다.

그 밖에 자세한 내용은 첨부파일의 README.md에 기록해두었습니다.

추가적인 문의사항 있으시면 언제든지 회신 부탁드립니다.
감사합니다.

전근우 드림.