# 시작하기

이 섹션에서는 제품의 설치부터 제품 성능의 극대화를 위한 고급 기능들까지, 제품 사용에 필요한 내용들의 목록을 제공합니다.

## 모빌린트 NPU 사용 개요

모빌린트는 두가지 주력 NPU 칩 **ARIES**(AI 가속기)와 **REGULUS**(SoC)를 제공합니다. 사용중인 NPU 및 폼팩터에 따라 설치 과정이 달라질 수 있습니다.

아래에서 제품명을 클릭하여 제품의 시스템 요구사항과 시스템 내 설치 방법에 대한 안내를 확인할 수 있습니다.

### 하드웨어 소개

- [ARIES MLA100 PCIe Card](aries-mla100-pcie-card.md)
- [ARIES MLA100 Mobile Express Module (MXM)](aries-mla100-mobile-express-module-mxm.md)
- [ARIES MLX-A1 Edge AI PC](aries-mlx-a1-edge-ai-pc.md)
- REGULUS System-on-Module (SoM) *- 추가 예정입니다.*

### SDK 소프트웨어 소개

```{important}
- ARIES 기반 제품의 사용을 위해서는 반드시 **드라이버**와 **런타임 라이브러리**를 설치해야합니다.
- REGULUS 기기의 경우 필수적인 SDK 스택 (드라이버, 런타임 라이브러리)및 유틸리티가 사전 설치된 채로 제공됩니다. "Tutorials" 문서들을 바로 확인하시는 것을 권장합니다.
```

다음 설치 안내 문서들을 확인하여 NPU 제품 사용에 필수적인 SDK 소프트웨어 설치를 진행해주세요.

- [드라이버 설치](installing_driver.md)
- [펌웨어 업데이트](update_firmware.md)
- [런타임 라이브러리 설치](installing_runtime_library.md)
- [컴파일러 설치](installing_compiler.md)
- [호환성 정보 확인](compatibility.md)

```{note}
제공되는 SDK 파일 및 패키지에 접근하고 다운로드하려면 [다운로드 센터](https://dl.mobilint.com)의 계정이 필요합니다. 자세한 내용은 contact@mobilint.com 으로 문의해주세요.
```

NPU 작동에 필수적인 SDK qb를 모두 설치하였다면 튜토리얼 문서들을 확인하여 모빌린트 NPU에서 AI 추론을 성공적으로 실행하는 방법을 익힐 수 있습니다.

- [(Basic) ResNet50 모델 실행 예제](tutorial_resnet50.md)
- [NPU 프로그래밍 가이드](programming_guide.md)

다음으로, 모빌린트에서 제공하는 여러 편의성 도구들을 확인하세요. NPU의 상태 혹은 테스트시 도움이 될 유틸리티 도구들을 제공하고 있습니다. 그리고 오픈소스의 사전 학습(pre-trained) 모델을 바로 사용하고 싶으시면, Model Zoo에 방문하여 확인해주세요.

- [유틸리티 설치](installing_utility.md)
- [유틸리티 사용](utility_usage.md)
- [Model Zoo](model_zoo.md)

시스템 사용에 익숙해지면, 고급 기능 문서를 통해 NPU 활용을 더욱 최적화하거나 고급 튜토리얼을 참고할 수 있습니다.

- [고급 기능](advanced_usage.md)
- [NPU 멀티 코어 활용](multicore.md)

디바이스 플러그인을 설치하여 쿠버네티스 클러스터에 NPU 워크로드를 배포할 수 있습니다.

- [Mobilint Device Plugin](kubernetes_device_plugin.md)
