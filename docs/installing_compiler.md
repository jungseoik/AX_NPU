# 컴파일러 설치

모빌린트의 NPU에서 사용자의 커스텀 딥러닝 모델을 사용하려면, 모델은 **MXQ (Mobilint ExeCUtable)** 포맷으로 컴파일해야 합니다. 모빌린트의 공식 컴파일러인 `qb Compiler`(또는 `qubee`, `qb`)는 이 컴파일 과정을 수행하며, 모델이 모빌린트의 하드웨어에서 최적의 성능으로 실행되도록 변환합니다. 컴파일러는 현재 모빌린트에서 제공하는 Docker 환경 내에 설치해야 합니다.

다음과 같은 경우에만 `qb Compiler`의 설치가 필요합니다:
- 사용자가 직접 학습한 모델을 사용하고 싶을 경우
- 해당 모델에 대한 MXQ 파일이 아직 제공되지 않은 경우

```{tip}
모빌린트에서 제공하는 **사전 컴파일 모델** ([Model Zoo](model_zoo.md) 또는 데모 패키지 내 포함)을 사용하는 경우, 컴파일러 설치가 필요 없습니다.
```

## 요구사항

- 모빌린트 공식 [다운로드 센터](https://dl.mobilint.com)에서 적절한 버전의 컴파일러 패키지를 다운로드합니다.
- 시스템 요구사항
    - 우분투 20.04.6 LTS 이상
    - (GPU 옵션 선택 시 필수) NVIDIA Graphics Driver 535.183.01 이상
- 필수 패키지
    - 도커
    - (GPU 옵션 선택 시 필수) nvidia-docker
    
    ```{seealso}
    Docker 설치에 대한 자세한 내용은 [Docker 공식 문서](https://docs.docker.com/desktop/)를 참조해주세요.
    ```

## 설치 과정

1. Docker 이미지를 다운로드하고 컨테이너를 생성합니다. 이때 **CPU-only 컴파일 환경**과 **GPU 가속 컴파일 환경**의 두 가지 환경 옵션이 제공됩니다.

    아래의 `{WORKING DIRECTORY}` 부분을 실제 작업 디렉토리 경로로 교체해주세요.

    ### CPU-only 컴파일 환경

    ```bash
    docker pull mobilint/qbcompiler:v{QUBEE_VERSION_NUMBER}-cpu
    cd {WORKING DIRECTORY}
    docker run -it --ipc=host --name {YOUR_CONTAINER_NAME} -v $(pwd):/workspace mobilint/qbcompiler:v{QUBEE_VERSION_NUMBER}-cpu /bin/bash
    ``` 

    ### GPU 가속 컴파일 환경

    ```bash
    docker pull mobilint/qbcompiler:v{QUBEE_VERSION_NUMBER}
    cd {WORKING DIRECTORY}
    docker run -it --gpus all --ipc=host --name {YOUR_CONTAINER_NAME} -v $(pwd):/workspace mobilint/qbcompiler:v{QUBEE_VERSION_NUMBER} /bin/bash
    ```

2. 다음 명령어를 실행하여 Docker 컨테이너 내에 `qb Comiler`를 설치합니다.

    ```bash
    docker cp {path/to/qubee.whl} {YOUR_CONTAINER_NAME}:/
    docker start {YOUR_CONTAINER_NAME}
    docker exec -it {YOUR_CONTAINER_NAME} /bin/bash
    cd /
    python -m pip install qubee-{QUBEE_VERSION_NUMBER}+aries2-py3-none-any.whl
    ```