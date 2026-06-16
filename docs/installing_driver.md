# 드라이버 설치

드라이버는 호스트 시스템과 설치된 NPU 장치 간의 통신을 활성화하기 위해 필요합니다. 현재 **Linux/Ubuntu**와 **Windows** 시스템에서 사용 가능합니다.

```{attention}
- 런타임 환경 설정을 진행하기 전에 반드시 드라이버 설치를 완료해야 합니다.
- 호환되는 런타임 및 펌웨어 버전을 [이곳](compatibility.md)에서 확인 하세요.
```

## 목차

- Linux

    - [방법 1: Ubuntu에서 APT를 통한 설치](#ubuntu-apt)

        - [(선택) Secure Boot 상태에서 드라이버 설치](#secure-boot)

    - [방법 2: 수동 다운로드 (기타 리눅스 배포판 대응)](#manual-download)

        - [(선택) 커널 내 드라이버 통합](#kernel-integration)

- Windows

    - [방법 1: 윈도우 업데이트를 통한 설치](#via-windows-update)

    - [방법 2: 윈도우 드라이버 수동 설치](#via-manual)

## Linux

(ubuntu-apt)=
### 방법 1: Ubuntu에서 APT를 통한 설치

모빌린트는 **APT 패키지 관리자**를 통해 설치할 수 있는 Ubuntu용 드라이버 패키지를 제공합니다. 다른 Linux 배포판을 사용하는 경우, 소스 코드를 직접 빌드하여 드라이버 모듈을 설치할 수 있습니다.

#### 요구사항

설치를 시작하기 전에, 다음 사항을 확인해주세요

- Ubuntu 20.04 이상(x86-64 혹은 arm64 기반 아키텍처)의 시스템을 사용해야 합니다.
- sudo 혹은 root 권한이 필요합니다.
- 아래의 빌드 관련 도구들이 설치되어 있어야 합니다:

    ```bash
    sudo apt install linux-headers-$(uname -r) build-essential
    ```

#### 설치 절차

1. 아래 명령을 수행하여 모빌린트 APT 레포지토리를 시스템에 추가합니다.

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

2. 드라이버를 아래 명령을 통해 설치합니다.

    ```bash
    sudo apt install mobilint-aries-driver
    ```

    ```{warning}
    만약, 개발 환경의 PC가 Secure Boot이 활성화된 경우에는 드라이버가 바로 설치되지 않습니다. Secure Boot을 비활성화한 후 드라이버를 설치하거나, [Secure Boot 상태에서 드라이버 설치](#secure-boot)를 참조 바랍니다.
    ```
    
    Secure Boot 활성화 여부는 아래의 명령어로 확인이 가능합니다.  
    
    ```bash
    $ mokutil --sb-state
    SecureBoot enabled
    ```

3. 드라이버 설치가 완료되었다면, 시스템을 재부팅합니다.

#### 설치 확인

재부팅 후, 다음 명령어를 통해 디바이스 노드가 성공적으로 생성되었는지 확인합니다:

```bash
ls -al /dev/aries*
```

NPU가 성공적으로 인식되었다면, `/dev/aries0`과 같은 이름으로 디바이스 노드가 표시됩니다.

![Driver check](/res/image/driver_check.png "드라이버 확인")

디바이스 노드가 보이지 않는 경우, PCI 버스에서 NPU가 인식되는지 확인해주세요:

```bash
sudo lspci -vd 209f:
```

만약 아래 이미지와 유사한 출력이 나타나지 않는다면, 하드웨어 연결 상태(전원 공급, PCIe 슬롯 장착 상태 등)을 점검하거나 다른 슬롯에 장착을 시도해주세요.

![Device check](/res/image/lspci.png "장치 확인")

제시된 방법으로 드라이버를 설치할 수 없는 경우, 아래에 설명된 대체 설치 방법을 따르거나 [tech-support@mobilint.com](mailto:tech-support@mobilint.com) 으로 기술 지원을 요청해주세요.

#### Secure Boot 상태에서 드라이버 설치

Secure Boot이 활성화된 환경에서 드라이버를 설치하기 위해서는 시스템에 드라이버의 MOK(Machine Owner Key) 등록이 필요합니다.

```{warning}
MOK 등록은 원격 작업 환경에서는 할 수 없으며, 디스플레이와 키보드가 연결된 시스템에 직접 접근해서 수행해야합니다.
```

등록 절차는 다음과 같습니다.

우선, 설치 명령어를 입력하면 아래의 안내 메시지가 나옵니다.

```bash
$ sudo apt install mobilint-aries-driver
* READ ME! *****************************************************************
*
* Installation Instruction for Secure Boot Mode
*
* You are now installing aries driver on secure boot mode. MOK (Machine Owner Key) should be enrolled to the system first for succesful driver installation. If you haven't enrolled MOK yet, make sure that you are not accessing the system remotely (via ssh) as you need to enroll the MOK on system booting. Follow the instructions below.
*
* 1. You will be prompted to set password for MOK key enrollment. Set the password (which should be 8 to 16 characters) and remember it for step 2.
* 2. Reboot to enroll MOK to the system. Blue screen will pop up automatically for MOK enrollment. Enroll the key using the password you set in step 1. If you failed to enroll MOK for any reason, move on to step 3.
* 3. Reinstall the driver: \"sudo apt install mobilint-aries-driver\".
* 4. If the installation was successful, check whether the driver is loaded: \"lsmod | grep aries\". If not, follow the instruction from step 1 again.
*
*********************************
```

안내 메시지 뒤에 나오는 안내에 따라 MOK의 패스워드를 입력해 주시기 바랍니다. 이 때, 패스워드는 8~16자리의 문자로 설정해야합니다.

```bash
[Instruction] Set password for MOK enrollment
input password: 
input password again: 
```

MOK 패스워드를 입력하면 설치 실패 에러와 함께 시스템 재부팅 및 드라이버 재설치를 안내 메시지가 나오는 것을 확인하면 시스템을 재부팅해야합니다.

```bash
[Instruction] Enroll MOK on reboot. Then reinstall the driver.
dpkg: error processing package mobilint-aries-driver (--install):
 installed mobilint-aries-driver package post-installation script subprocess returned error exit status 1
```

재부팅을 하면 아래와 같은 "**Shim UEFI Key Management**"라고 적힌 파란색 화면이 나타납니다. 해당 화면이 보이면 아무 키나 입력하면 MOK 설정 화면으로 넘어갑니다.

![MOK_1](/res/image/mok_1.jpg "MOK_1")

"**Perform MOK Management**" 화면이 나타나면, "**Enroll MOK**"를 선택합니다.

![MOK_2](/res/image/mok_2.jpg "MOK_2")

"**Enroll the key(s)?**" 화면이 나타나면, "**Yes**"를 선택합니다.

![MOK_3](/res/image/mok_3.jpg "MOK_3")

"**Enroll MOK**" 화면이 나타나면, "**Continue**"를 선택합니다.

![MOK_4](/res/image/mok_4.jpg "MOK_4")

패스워드 입력창이 나타나면, 드라이버 설치과정에서 설정했던 MOK 패스워드를 입력합니다.

![MOK_5](/res/image/mok_5.jpg "MOK_5")

MOK 등록이 완료되면, "**Reboot**"를 선택하여 시스템을 재부팅합니다.

![MOK_6](/res/image/mok_6.jpg "MOK_6")

MOK 등록을 마치고 재부팅된 이후에, 다시 드라이버를 설치하면 정상적으로 설치되는 것을 확인할 수 있습니다.

```bash
sudo apt install mobilint-aries-driver
```

```{note}
만약, MOK 등록과정에서 MOK 패스워드를 잘못 입력하였을 경우, 드라이버 설치 명령어를 입력하면 새 MOK 패스워드 설정을 요청할 것이고, 새 패스워드를 설정한 이후 재부팅하여 다시 MOK 등록 과정을 수행하면 됩니다.
```

(manual-download)=
### 방법 2: 수동 다운로드 (기타 리눅스 배포판)

커널 모듈을 빌드하려면 커널 헤더와 컴파일러(gcc 등)가 필요하며, 구체적인 요구사항은 사용하는 리눅스 배포판과 환경에 따라 달라질 수 있습니다.

다음 안내는 APT를 사용해 Ubuntu에 필수 패키지를 설치하는 절차를 예시로 보여줍니다. 사용 중인 시스템 환경에 맞게 적절히 적용해주세요.

#### 요구사항

- 모빌린트 [공식 다운로드 센터](https://dl.mobilint.com)에서 드라이버를 다운로드 합니다.

- 드라이버 모듈을 빌드하기 전에 다음 패키지들이 설치되어 있는지 확인합니다.

    ```bash
    sudo apt install linux-headers-$(uname -r) build-essential
    ```

#### 설치 절차

[다운로드 센터](https://dl.mobilint.com)를 통해 다운로드할 수 있는 드라이버 패키지는 커널 모듈을 직접 빌드해 사용할 수 있는 형태로 제공됩니다. 로드 가능한 모듈로 사용하시려면 `make` 명령으로 간단히 빌드하여 커널 모듈을 삽입하거나 제거할 수 있습니다.

```bash
## 커널 모듈 빌드
make

## 커널 모듈 삽입
sudo insmod aries.ko
```

- 커널 모듈 제거

    아래 명령을 통해 삽입된 드라이버 커널 모듈을 제거할 수 있습니다.

    ```bash
    sudo rmmod aries
    ```

(kernel-integration)=
### 선택 사항: 커널 내 통합

드라이버를 커스텀 Linux 커널의 내장 모듈로 포함하려면, 사용중인 배포판의 커널 빌드 및 패키징 절차를 참조해주세요.

제공된 절차로 드라이버를 설치할 수 없는 경우, [tech-support@mobilint.com](mailto:tech-support@mobilint.com)으로 기술 지원을 요청해주세요.

## Windows

모빌린트는 64비트 윈도우 10/11 (빌드 19041 이상)용 WHQL 인증된 **Universal KMDF 드라이버**를 제공합니다.

- 요구사항
	- 윈도우 10 또는 11 (64비트), (Windows 10.0.19041 빌드 이상)
		- 최소 지원 버전을 충족하기 위해 윈도우 업데이트가 필요할 수 있습니다.

먼저 PC가 전원이 꺼진 상태에서 PCIe 슬롯에 연결하고, 전원을 켜서 `장치 관리자`에 정상적으로 장치가 인식이 되었는지 확인합니다.

![장치 관리자](/res/image/windows_driver_uninstalled_ko.png "장치 관리자")

```{note}
- 드라이버가 설치되기 전에는 장치가 "기타 장치"의 "PCI 단순 통신 컨트롤러"로 표시됩니다.
- 드라이버 설치가 정상적으로 이루어지기 위해서, 메인보드와 CPU의 드라이버 설치가 먼저 진행되어야 합니다.
```

장치의 존재가 확인되어야, 아래 2가지 방법 중 하나로 드라이버를 설치를 진행할 수 있습니다.

(via-windows-update)=
### 방법 1: 윈도우 업데이트를 통한 설치

윈도우 업데이트를 통해 드라이버 설치하는 것이 가장 쉬운 방법입니다.

1. 설정에서 윈도우즈 업데이트 실행  
    ![윈도우즈 업데이트](/res/image/windows_update_0_ko.png "윈도우즈 업데이트")  
    "업데이트 확인" 버튼을 누른 후, 완료되기까지 기다립니다.

2. "고급 옵션"을 클릭합니다.  
    ![고급 옵션](/res/image/windows_update_1_ko.png "고급 옵션")  

3. "선택적 업데이트"를 클릭합니다.  
    ![선택적 업데이트](/res/image/windows_update_2_ko.png "선택적 업데이트")  

4. "드라이버 업데이트" 목록에서 "MOBILINT, Inc. AI Device Driver Update" 를 선택 후 "다운로드 및 설치"를 클릭합니다.  
    ![다운로드 및 설치](/res/image/windows_update_3_ko.png "다운로드 및 설치")  

만약 목록이 보이지 않는다면, 이미 최신 드라이버가 설치되었거나, 장치가 먼저 인식이 되었는지 확인해야 합니다.

(via-manual)=
### 방법 2: 윈도우 드라이버 수동 설치

윈도우 업데이트를 통해 쉽게 드라이버 설치가 가능하지만, 부득이하게 인터넷이 불가한 상황에서는 직접 수동 설치가 필요합니다. 수동으로 설치시 다음과 같은 절차에 따릅니다.

1. 인터넷이 되는 다른 호스트 PC 에서 [최신 윈도우 드라이버](https://dl.mobilint.com/releases?series-id=3&product=ARIES)를 다운로드 받아 대상 PC에 옮겨 줍니다.

    ![드라이버 다운로드](/res/image/windows_driver_download.png "드라이버 다운로드")

2. 장치 관리자에서 "드라이버 업데이트" 버튼을 클릭해주세요.

    ![드라이버 업데이트](/res/image/windows_control_pannel_update_driver_ko.png "드라이버 업데이트")

3. "내 컴퓨터에서 드라이버 찾아보기"를 클릭하여 진행합니다.

    ![수동 설치](/res/image/manual_installation.png "수동 설치")

4. 드라이버 파일이 포함된 폴더를 찾아 선택합니다.

    ![드라이버 찾기](/res/image/find_driver.png "드라이버 찾기")

5. 설치가 완료되면 아래 이미지와 같이 드라이버가 정상적으로 설치된 것을 확인할 수 있습니다.

    ![드라이버 설치 완료](/res/image/complete_install.png "드라이버 설치 완료")

    ![설치 후 장치관리자](/res/image/fin_device_manager.png "설치 후 장치관리자")