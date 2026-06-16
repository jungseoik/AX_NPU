# 런타임 라이브러리 설치

이 섹션에서는 모빌린트 NPU에서 추론을 수행하기 위해 필요한 런타임 라이브러리 (`qb Runtime`) 설치 방법을 설명합니다.

```{attention}
- 런타임 설치를 진행하기 전에 반드시 [드라이버 설치](installing_driver.md)를 완료해야합니다.
- 런타임 설치를 진행하기 전에 호환되는 드라이버 및 펌웨어 버전을 [이곳](compatibility.md)을 통해 확인 후 설치해주세요.
```

## 목차

- Linux/Ubuntu

    - [방법 1: APT를 통한 설치](#apt)

    - [방법 2: 다운로드 센터를 통한 수동 설치](#manual-download-install)

        - [방법 2-1: 시스템 전역 설치](#system-wide-install)

        - [방법 2-2: 설치 없이 사용](#without-installation)
    
    - [방법 3: pip를 통한 설치](#pip)
    
- Windows

    - [다운로드 센터를 통한 수동 설치](#windows)

## Linux/Ubuntu

(apt)=
### 방법 1: APT를 통한 설치

이 절에서 다루는 설치 방법은 데비안 기반 OS들의 패키지 관리자 `apt`를 활용한 설치방법입니다.

#### 설치 절차

1. 아래 명령을 수행하여 모빌린트 APT 저장소를 시스템에 추가합니다.

    ```{tip}
    드라이버 설치 과정에서 이미 모빌린트 APT 저장소를 추가했다면, 이 과정는 건너뛰어도 됩니다.
    ```

    ```bash
    # Add Mobilint's official GPG key:
    sudo apt update
    sudo apt install ca-certificates curl
    sudo install -m 0755 -d /etc/apt/keyrings
    sudo curl -fsSL https://dl.mobilint.com/apt/gpg.pub -o /etc/apt/keyrings/mblt.asc
    sudo chmod a+r /etc/apt/keyrings/mblt.asc

    # Add the repository to apt sources:
    printf "%s\n" \
        "deb [signed-by=/etc/apt/keyrings/mblt.asc] https://dl.mobilint.com/apt \
        stable multiverse" | \
        sudo tee /etc/apt/sources.list.d/mobilint.list > /dev/null

    # Update available packages
    sudo apt update
    ```

2. apt 패키지 관리자를 통해 설치를 진행합니다.

    ```bash
    sudo apt install mobilint-qb-runtime
    ```

3. 설치를 확인합니다.

    ```bash
    dpkg -L mobilint-qb-runtime
    ```

#### Anaconda 환경에서의 문제 해결

아나콘다 가상 환경을 사용할 경우, 아나콘다의 `libstdc++.so`를 사용하게되어 해당 라이브러리 관련 오류가 발생할 수 있습니다.

```bash
../lib/libstdc++.so.6: version `GLIBCXX_3.4.32' not found
```

이러한 오류는 아나콘다와 함께 설치되는 libstdc++ 라이브러리의 버전 충돌로 인해 발생하는 문제입니다. 이는 사용하는 아나콘다 환경의 `libstdcxx-ng` 등의 라이브러리 버전을 업그레이드하여 해결해야 합니다. 아나콘다 버전 별로 업그레이드 하는 명령이 달라질 수 있기 때문에, 정확한 업그레이드 방법은 온라인에서 관련 정보를 검색하는 것을 권장합니다.

또한, 이러한 충돌을 방지하기 위해 시스템의 `libstdc++`을 사용하는 Python의 기본 제공 가상 환경 (`venv`)을 사용하는 것을 권장합니다.

(manual-download-install)=
### 방법 2: 다운로드 센터를 통한 수동 설치

#### 요구 사항

- 모빌린트의 공식 [다운로드 센터](https://dl.mobilint.com)에서 런타임 라이브러리를 다운받아야 합니다.

#### 설치 절차

1. 다운로드 센터에서 다운받은 런타임 라이브러리 파일의 압축을 해제합니다. 이때 `{RUNTIME_VERSION}` 은 설치할 런타임 라이브러리 버전에 대응되는 문자열입니다.

    ```bash
    tar -xvzf qb-runtime_aries2-v4_v{RUNTIME_VERSION}.tar.gz
    ```

    압축을 해제하면 아래와 같은 폴더 구조를 지니고 있습니다.

    ```bash
    qb-runtime_aries2-v4_v{RUNTIME_VERSION}
    ├── Makefile
    ├── qbruntime
    │   ├── qbruntime
    │   │ ├── include    # include path
    │   │ │   └── qbruntime
    │   │ ├── lib        # library path
    │   │ └── python
    │   └── resnet50
    └── mobilint-cli
    ```

이후 두가지 설치 방법이 존재합니다: **시스템 전역 설치** 혹은 **설치 없이 사용**.

(system-wide-install)=
#### 방법 2-1: 시스템 전역 설치

아래 명령을 통해 시스템 전역에 라이브러리를 설치합니다.

```bash
cd qb-runtime_aries2-v4_v{RUNTIME_VERSION}
sudo make install
```

위 명령으로 설치한 라이브러리는 `sudo make uninstall` 명령을 통해 삭제할 수 있습니다.

(without-installation)=
#### 방법 2-2: 설치 없이 사용

컴파일 과정에서는 `qb-runtime_aries2-v4_v{RUNTIME_VERSION}/qbruntime/qbruntime/include` 의 헤더파일 경로와 `qb-runtime_aries2-v4_v{RUNTIME_VERSION}/qbruntime/qbruntime/lib` 의 라이브러리 경로를 아래와 같이 명시해줍니다.

```bash
g++ -o {output_binary} {source_code} -I{path_to_include} \
    -L{path_to_library} -lqbruntime
```

이후 컴파일된 바이너리를 실행하는 과정에서는 동적 링커에게 런타임 라이브러리 (`*.so`)의 경로를 알려주기 위해 `LD_LIBRARY_PATH`에 해당 경로를 등록하여 사용합니다.

```bash
export LD_LIBRARY_PATH={path_to_library}
```

(pip)=
#### 방법 3: pip를 통한 설치

```{note}
pip를 통한 Ubuntu용 파이썬 라이브러리 패키지 설치는 런타임 라이브러리 **0.29.0** 버전부터 사용 가능합니다.
```

파이썬 라이브러리는 pip를 활용한 설치를 지원합니다. 아래 명령을 통해 파이썬 라이브러리 패키지를 설치합니다.

```bash
pip install mobilint-qb-runtime
```

## Windows

### 요구 사항

- 런타임 라이브러리 관련 파일을 모빌린트의 공식 [다운로드 센터](https://dl.mobilint.com)에서 다운로드 받습니다.

### 설치 절차

윈도우에서는 시스템 전역 설치는 지원되지 않고, **설치 없이 사용**하는 방법만 제공됩니다. 다운로드 받은 라이브러리를 활용해 **Visual Studio**에서 컴파일 시 활용하는 방법은 [C++ 컴파일 - Windows](programming_guide.md#compile-windows) 섹션을 참조해주세요.
