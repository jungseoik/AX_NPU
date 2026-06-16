# 호환성 확인

본 문서는 모빌린트에서 제공하는 소프트웨어들의 호환성 정보를 제공합니다. 시스템 구성 또는 업데이트시, 반드시 아래 표를 참조하여 상호 호환되는 버전을 사용해야합니다. 호환되지 않는 조합을 사용할 경우 정상 동작이 보장되지 않고, 예기치 않은 오류가 발생할 수 있습니다.

<style>
table.table {
  table-layout: fixed;
  width: 100%;
}
</style>

## Linux/Ubuntu

### 드라이버-런타임-펌웨어 호환성

| Driver Ver.   | Runtime Ver.    | Firmware Ver.  |
| ------------- | --------------- | -------------- |
| 1.0 ~ 1.10    | 0.28.0 ~ 0.30.1 | 1.1.1 ~ latest |
| 1.11 ~ latest | 1.0.0 ~ latest  | 1.1.1 ~ latest |

## Windows

### 드라이버-런타임-펌웨어 호환성

| Driver Ver.    | Runtime Ver.    | Firmware Ver.  |
| -------------- | --------------- | -------------- |
| 1.6.230        | 0.28.0 ~ 0.30.0 | 1.1.1 ~ latest |
| 1.7.1 ~ 1.7.5  | 0.30.1 ~ 0.30.1 | 1.1.1 ~ latest |
| 1.8.0 ~ latest | 1.0.0  ~ latest | 1.1.1 ~ latest |

## 런타임 - MXQ 호환성

모빌린트의 딥러닝 모델(`MXQ` 확장자)을 실행하기 위해서는, 해당 모델의 `MXQ` 버전에 따라 지원하는 런타임을 사용해야합니다. 각 런타임 버전과 `MXQ` 버전간의 호환성 정보는 다음과 같습니다. 

| Runtime Ver.    | MXQ Ver.      |
| --------------- | ------------- |
| 0.28.0 ~ 0.28.0 | MXQv1 ~ MXQv5 |
| 0.29.0 ~ 0.30.1 | MXQv1 ~ MXQv6 |
| 1.0.0  ~ latest | MXQv1 ~ MXQv7 |

## 버전 확인하는 방법

### 런타임 라이브러리

런타임 라이브러리를 [다운로드 센터](http://dl.mobilint.com)에서 다운로드 받았다면 다운로드 시에 안내된 버전 정보를 확인할 수 있습니다.

혹은, 런타임에서 제공하는 아래 메서드를 통해 버전 정보를 출력할 수 있습니다.

```cpp
// C++ 예제
std::cout << mobilint::getQbRuntimeVersion() << "\n";
```

```python
# Python 예제
print(qbruntime.__version__)
```

### 드라이버, 펌웨어

드라이버와 펌웨어 각각의 버전은 모빌린트에서 제공하는 유틸리티 툴을 활용하여 확인할 수 있습니다. 유틸리티 설치는 [이곳](installing_utility.md)를 참조해주세요.

- Linux/Ubuntu

  - Linux/Ubuntu 환경에 제공되는 유틸리티 `mobilint-cli`를 활용하여 드라이버, 펌웨어 버전을 확인할 수 있습니다.

  - `mobilint-cli status` 명령을 입력하여 아래의 붉은 박스에서 각각 드라이버, 펌웨어 버전을 확인할 수 있습니다.

  ![Version Check](/res/image/status_ver_check.png)

- Windows/Linux

  - Windows/Linux 환경 모두에 제공되는 유틸리티 `mobilint_ctrl`(GUI) 또는 `mobilint_ctrl_cli`(CLI)를 활용하여 드라이버, 펌웨어 버전을 확인할 수 있습니다.

  - [설치 문서](installing_utility.md)를 따라 다운로드 받은 `mobilint_ctrl` 또는 `mobilint_ctrl_cli`를 실행 후 아래와 같이 각각 드라이버, 펌웨어 버전을 확인할 수 있습니다.

  ![mobilint_ctrl](/res/image/mobilint_ctrl.png)

  ![mobilint_ctrl_cli](/res/image/mobilint_ctrl_cli.png)

### MXQ

```{caution}
현재 MXQ 버전을 확인하는 도구는 **Linux/Ubuntu** 환경을 대상으로만 제공됩니다.
```

- Linux/Ubuntu 용으로 제공되는 도구 `mobilint-cli` 를 통해 mxq의 버전을 확인할 수 있습니다.

- `mobilint-cli mxqtool show {mxq_path}` 명령을 수행하면, 아래와 같이 "Format Version" 정보가 출력됩니다.

  ```bash
  $ mobilint-cli mxqtool show resnet50_IMAGENET1K_V1.mxq
  Format Version:           0x60000
  Compiler Version:         0.10.0.0
  Hardware Version:         Aries2
  ...
  ```

  - 이때 Format Version 값의 가장 앞자리가 MXQ 버전에 대응됩니다. (위의 경우, 0x60000 이므로 MXQv6)
  