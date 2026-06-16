# 유틸리티 설치

모빌린트 NPU의 상태 확인, 벤치마크, MXQ 정보 확인 등의 기능을 제공하는 유틸리티를 제공합니다. 유틸리티를 설치하는 방법을 소개합니다.

## Linux 전용 유틸리티

- **mobilint-cli**: NPU 상태 확인, 벤치마크 실행, MXQ 정보 조회 등을 수행할 수 있는 커맨드라인 유틸리티.
- **mobilint-runtime-gui**: NPU를 이용해 다양한 AI 모델을 코드 없이 쉽게 실행·테스트할 수 있는 GUI 기반 유틸리티.

### mobilint-cli 설치

#### 방법 1: APT를 통한 설치

[드라이버 설치](installing_driver.md) 혹은 [런타임 라이브러리 설치](installing_runtime_library.md)에서 모빌린트의 apt 저장소를 시스템에 추가한 경우에, 아래 명령과 같이 apt를 통해 `mobilint-cli`를 설치할 수 있습니다.

```bash
sudo apt install mobilint-cli
```

#### 방법 2: 다운로드 센터를 통한 수동 설치

모빌린트 공식 [다운로드 센터](https://dl.mobilint.com)를 통해 제공되는 런타임 (qb Runtime)의 패키지 내에 `mobilint-cli`가 포함되어 있습니다.

다운로드한 디렉토리 내에서 `sudo make install` 명령을 통해 설치 시 런타임 라이브러리 및 `mobilint-cli`가 모두 설치됩니다.

자세한 사항은 [런타임 라이브러리 설치 문서](installing_runtime_library.md#manual-download-install)를 참고하세요.

### mobilint-runtime-gui 설치

[드라이버 설치](installing_driver.md) 또는 [런타임 라이브러리 설치](installing_runtime_library.md) 과정에서 모빌린트 APT 저장소를 추가한 경우, 아래 명령으로 GUI 도구를 설치할 수 있습니다.

```bash
sudo apt install mobilint-runtime-gui
```

## Windows, Linux 공용 유틸리티

다음 유틸리티는 Windows 및 Linux 환경에서 모두 사용할 수 있습니다.

- **mobilint_ctrl**: NPU의 상태를 확인하고, firmware 업데이트 하도록 GUI 유틸리티.
- **mobilint_ctrl_cli**: NPU의 상태를 확인할 수 있는 CLI 유틸리티.
- **aries_flash_firmware**: NPU firmware 를 업데이트 할 수 있는 CLI 유틸리티.

### Mobilint Ctrl 설치

#### Windows

모빌린트 [공식 다운로드 센터](https://dl.mobilint.com)에서 Windows용 패키지를 다운로드할 수 있습니다.

배포 파일: `mobilint_ctrl_{version}_windows.zip`

압축 해제 후 `mobilint_ctrl.exe` 파일을 더블클릭하여 실행할 수 있습니다.

#### Linux 설치

모빌린트 [공식 다운로드 센터](https://dl.mobilint.com)에서 Linux용 패키지를 다운로드할 수 있습니다.

공식 지원 Ubuntu 버전:

- 20.04 (LTS)
- 22.04 (LTS)
- 24.04 (LTS)

| 패키지 파일명 기준 | 사용 가능 ubuntu 버전 |
| ------------------ | ----------------------| 
| `..._ubuntu20.04_amd.deb` | Ubuntu 20.04 전용 |
| `..._ubuntu22.04_amd.deb` | Ubuntu 22.04, 24.04 |

압축 해제 후 아래 명령어로 설치합니다:

```bash
sudo apt install ./mobilint_ctrl_{version}_<Ubuntu_version>_amd64.deb
```	

### aries_flash_firmware 설치

#### Windows

`mobilint_ctrl` 배포 파일 내부 하위 폴더에 `aries_flash_firmware.exe`로 포함되어 있으며 CLI 환경에서 실행 가능합니다.

```bash
mobilint_ctrl_{version}_windows.zip
 └── aries_flash_firmware.exe
```

파워쉘 혹은 cmd 환경에서 아래와 같이 실행할 수 있습니다.

```powershell
aries_flash_firmware.exe status
```

#### Linux

Flash firmware 패키지는 `mobilint_ctrl` 배포 파일 내부 하위 폴더에 `aries_flash_firmware_<Build_version>_ubuntu_<amd64/arm64>.deb` 파일로 포함되어 있습니다.

공식 지원 Ubuntu 버전:

- Ubuntu 20.04 (LTS)
- Ubuntu 22.04 (LTS)
- Ubuntu 24.04 (LTS)

아래 명령어로 설치할 수 있습니다.

```bash
sudo apt install ./aries_flash_firmware_<Build_version>_ubuntu_<amd64/arm64>.deb
```	
