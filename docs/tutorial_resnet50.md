# ResNet50 모델 추론 예제

본 튜토리얼은 사전 컴파일된 **ResNet50** 모델을 제공된 런타임 패키지를 통해 모빌린트 NPU에서 추론을 하는 방법을 다룹니다.

다음의 예제를 포함하고 있습니다:
- **ARIES**기반 시스템들(MLA100 PCIe / MXM / MLX-A1) 사용 예제
- **REGULUS**의 사용을 위한 크로스 컴파일

## 요구 사항

시작하기 전에, 아래의 구성요소들이 시스템에 설치되어 있는지 확인해주세요:

- 모빌린트 NPU 하드웨어:
    - ARIES (MLA100 PCIe / MLA100 MXM / MLX-A1)
    - REGULUS SoC
- 드라이버
- 런타임 라이브러리

## ARIES 기반 폼팩터

1. 드라이버와 런타임 환경을 준비합니다.

2. [다운로드 센터](https://dl.mobilint.com)에서 런타임 라이브러리 패키지 파일을 다운로드합니다.

3. 다운로드 받은 런타임 패키지 압축을 해제하고 `resnet50` 폴더로 이동합니다.

    ```
    cd {YOUR_DOWNLOAD_DIR}/qb-runtime_aries2-v4_v{RUNTIME_VERSION_NUMBER}/qbruntime/resnet50
    ```

4. 해제한 패키지 내에는 아래와 같은 파일들이 포함되어 있습니다:

    ```
    qb-runtime_aries2-v4_v{RUNTIME_VERSION_NUMBER}/qbruntime/resnet50/
    ├── ILSVRC2012_val_00000001.JPEG  # 예시 이미지
    ├── resnet50.cc                   # C++ 추론 코드
    ├── resnet50.mxq                  # 컴파일된 Resnet50 model
    ├── resnet50.py                   # Python 추론 코드
    ├── stb_image.h                   # 이미지 로드를 위한 라이브러리
    └── stb_image_resize.h            # 이미지 처리를 위한 라이브러리
    ```

5. 예시 추론 코드를 실행합니다.

    - C++ 코드 (`resnet50.cc`)

        1. [관련 문서](programming_guide.md#c)를 따라 C++ 코드를 컴파일 합니다.

            ```bash
            gc++ -o resnet50 resnet50.cc -lqbruntime
            ```

        2. 컴파일된 실행 파일을 실행합니다.

    - Python 코드 (`resnet50.py`)

        1. 런타임 라이브러리 파이썬 패키지 `qbruntime`이 파이썬 라이브러리에 설치되어있어야 합니다. 설치는 [관련 문서](installing_runtime_library.md#pip)를 참조해주세요.

        2. 이미지 처리에 필요한 `opencv-python` 패키지를 설치해주세요.

            ```bash
            pip install opencv-python
            ```

        3. 예제 코드를 실행합니다.

            ```bash
            python resnet50.py
            ```


## REGULUS

```{note}
REGULUS는 드라이버및 런타임 라이브러리가 사전 설치된 채로 제공되어 추가적인 설치는 필요하지 않습니다.
```

REGULUS에서 동작하는 프로그램은 REGULUS 내의 ARM CPU에서 동작할 수 있도록 크로스 컴파일(cross-compile) 환경에서 빌드해야 합니다. 이후 REGULUS에서 실행할 수 있도록 업로드를 수행합니다.

1. 모빌린트 [다운로드 센터](https://dl.mobilint.com)를 통해 `regulus-release_vX.X.X.tar.gz` 파일을 다운로드 받습니다.

2. 파일 압축을 해제하고, 아래 처럼 `install-regulus-toolchain.sh` 실행 후 "enter"를 눌러 크로스 컴파일용 tool chain을 설치를 진행합니다..

    ```bash
    $ cd regulus-release_vX.X.X
    $ ./install-regulus-toolchain.sh
    # ==> type "enter"
    ```

    ```{note}
    위의 명령을 실행하면 "/opt/crosstools/mobilint/Y.Y.Y/X.X.X" 폴더를 생성합니다.
    ```

3. 아래 명령을 통해 크로스 컴파일 환경을 활성화 합니다.

    ```
    $ source /opt/crosstools/mobilint/Y.Y.Y/X.X.X/environment-setup-cortexa53-mobilint-linux
    ```

4. 데모 저장소에서 resnet50용 예제 파일을 다운로드하고 컴파일을 수행합니다.

    ```
    $ git clone https://github.com/mobilint/regulus-npu-demo.git
    $ cd regulus-npu-demo/image-classification-resnet50
    $ make
    ```

5. 생성된 프로그램을 REGULUS로 업로드하여 실행합니다.

    ```
    ./resnet50
    ```

## 에필로그: NPU 프로그램 구조 이해

ResNet50 예제 프로그램을 NPU에서 정상적으로 동작하는 것을 확인했다면, 이를 기반으로 자신만의 AI 애플리케이션을 개발할 수 있습니다.

아래 다이어그램은 NPU 응용 프로그램의 전형적인 구조를 보여주며, 런타임이 처리하는 부분과 사용자가 직접 개발해야하는 부분을 구분하여 보여줍니다.

![Typical structure of NPU App.](/res/image/structure.png "Typical structure of NPU App.")

이 ResNet50 예제는 **최소한**의 데모 구현입니다. 실제 제품 수준의 응용 프로그램에 가까워지기 위해서는 멀티 쓰레딩같은 고급 최적화 기법을 추가로 구현하는 것이 좋습니다.

```{seealso}
최적화 기법은 응용 프로그램 종류와 실행 환경에 따라 크게 달라질 수 있습니다. 보다 자세한 내용은 [고급 기능](advanced_usage.md) 문서를 참고하세요.
```
