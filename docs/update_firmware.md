# 펌웨어 업데이트

PCIe 장치의 경우, 안정적인 드라이버 실행과 최적의 성능을 위해서 최신의 펌웨어로 업데이트 하는 것이 좋습니다. **Linux/Ubuntu**와 **Windows** 시스템에서 모두 사용 가능합니다.

```{attention}
- 펌웨어 업데이트를 진행하기 전에 반드시 [드라이버 설치](installing_driver.md)를 완료해야 합니다.
- 호환되는 런타임 및 펌웨어 버전을 [이곳](compatibility.md)에서 확인 하세요.
- 리눅스와 윈도우 모두 동일한 과정으로 업데이트 동작을 진행합니다.
```

## 목차

- [펌웨어 툴 다운로드](#firmware-download)
- [펌웨어 업데이트](#firware-update)
    - [방법 1: GUI 툴로 업데이트](#firmware-update-gui)
    - [방법 2: Console 툴로 업데이트](#firmware-update-console)
- [펌웨어 확인](#firware-version-check)

(firmware-download)=
## 펌웨어 툴 다운로드

[다운로드 사이트](https://dl.mobilint.com/releases?series-id=9&product=ARIES)에서 현재 OS에 적합한 최신의 툴을 다운로드하여 설치합니다.
* 윈도우즈 : mobilint_ctrl_v?_windows.zip
* 리눅스(데비안 패키지) : aries_flash_firmware_?.deb, mobilint_ctrl_?.deb

(firware-update)=
## 최신 펌웨어 업데이트

실행환경에 따라 GUI 또는 Console 환경에서 업데이트 방법을 모두 제공합니다. 업데이트 진행 전에 모든 AI 응용 프로그램을 종료해야 합니다.

(firmware-update-gui)=
### 방법 1: GUI 툴로 업데이트

`mobilint_ctrl` 프로그램을 실행하여, `firmware` 탭을 클릭합니다.

![펌웨어 탭](/res/image/mobilint_ctrl_firmware_0.png "펌웨어 탭")

`Check Firmware Update...` 버튼을 클릭하여, 현재 설치된 펌웨어를 확인합니다.

![현재 펌웨어 확인](/res/image/mobilint_ctrl_firmware_1.png "현재 펌웨어 확인")

`Update Firmware...` 버튼을 클릭하여, 최신 펌웨어 업데이트를 진행합니다.

![펌웨어 업데이트](/res/image/mobilint_ctrl_firmware_2.png "펌웨어 업데이트")

```{attention}
- 이 과정에 약 7~10초가 소요됩니다.
- 업데이트 중 절대 호스트 PC의 전원을 끄지 마세요.
- 최신 펌웨어를 온라인으로 다운 받기 때문에, 인터넷 연결이 필요합니다.
```

만약 이미 최신 펌웨어가 설치되어 있다면 아래와 같이 표시됩니다.

![이미 최신 펌웨어](/res/image/mobilint_ctrl_firmware_3.png "이미 최신 펌웨어")

설치한 펌웨어는 재부팅 후에 적용됩니다.

(firmware-update-console)=
### 방법 2: Console 툴로 업데이트

`aries_flash_firmware update` 명령을 통해 firmware 를 최신으로 업데이트 가능합니다. 이때 인터넷 연결이 필요합니다.

정상적으로 실행된다면 아래와 같이 업데이트 과정이 출력됩니다.

```bash
> aries_flash_firmware update
* PCIe Device Driver connection details.
		> Driver version : 1.8.2 (Revision 2)
		> PCIe card connection information.
				 - Main-System(VID:0x209F,DID:0x0000), Sub-System(VID:0x0402,DID:0x1093), PCIe 4.0 - 8 Lanes
		Device type   : Aries 2 (Firmware v1.2.5 rev#0)
		Product type  : Commercial
		Board type    : Low profile
		DRAM size     : 16 GB

*I: Current firmware.
	- Reading Board [==================================================] Done.
		BSP           : aries-mla100
		Product name  : MLA100 LowProfile
		Version       : v1.2.5 rev.0 (Build date : 2026-03-23 16:14:00 KST)
		Configuration : master(cd87e82d) release
		SHA256 check  : Ok!

*I: Downloading firmware : MLA100 LowProfile v1.2.6
		BSP           : aries-mla100
		Product name  : MLA100 LowProfile
		Version       : v1.2.6 rev.0 (Build date : 2026-05-22 18:28:31 KST)
		Configuration : master(3ac404ba) release
		SHA256 check  : Ok!

*I: Flash new firmware. (Don't turn off the device!!!)
	- Erasing flash [==================================================] Done.
	- Verifying...  [==================================================] Done.
	- Writing flash [==================================================] Done.
	- Verifying...  [==================================================] Done.
*I: Please reboot your system to update device.
*I: Done!
- More detail information : https://docs.mobilint.com
```

```{attention}
- 이 과정에 약 7~10초가 소요됩니다.
- 업데이트 중 절대 호스트 PC의 전원을 끄지 마세요.
- 최신 펌웨어를 온라인으로 다운 받기 때문에, 인터넷 연결이 필요합니다.
```

만약 이미 최신의 펌웨어가 설치되었다면, 아래와 같이 이미 업데이트 되었음을 안내하고, 불필요한 펌웨어 플래싱을 진행하지 않습니다.

```bash
> aries_flash_firmware update
* PCIe Device Driver connection details.
		> Driver version : 1.8.2 (Revision 2)
		> PCIe card connection information.
				 - Main-System(VID:0x209F,DID:0x0000), Sub-System(VID:0x0402,DID:0x1093), PCIe 4.0 - 8 Lanes
		Device type   : Aries 2 (Firmware v1.2.5 rev#0)
		Product type  : Commercial
		Board type    : Low profile
		DRAM size     : 16 GB

*I: Current firmware.
	- Reading Board [==================================================] Done.
		BSP           : aries-mla100
		Product name  : MLA100 LowProfile
		Version       : v1.2.6 rev.0 (Build date : 2026-05-22 18:28:31 KST)
		Configuration : master(3ac404ba) release
		SHA256 check  : Ok!

*I: No need to update, The firmware is already up to date.
- More detail information : https://docs.mobilint.com
```

설치한 펌웨어는 재부팅 후에 적용됩니다.

(firware-version-check)=
## 펌웨어 버전 확인

재부팅 후 `aries_flash_firmware status`를 실행합니다. 아래와 같이 이미 최신의 펌웨어가 설치되었음을 확인 할 수 있습니다.

```bash
> aries_flash_firmware status
* PCIe Device Driver connection details.
        > Driver version : 1.8.2 (Revision 2)
        > PCIe card connection information.
                 - Main-System(VID:0x209F,DID:0x0000), Sub-System(VID:0x0402,DID:0x1093), PCIe 4.0 - 8 Lanes
        Device type   : Aries 2 (Firmware v1.2.6 rev#0)
        Product type  : Commercial
        Board type    : Low profile
        DRAM size     : 16 GB

*I: Current firmware.
    - Reading Board [==================================================] Done.
        BSP           : aries-mla100
        Product name  : MLA100 LowProfile
        Version       : v1.2.6 rev.0 (Build date : 2026-05-22 18:28:31 KST)
        Configuration : master(3ac404ba) release
        SHA256 check  : Ok!

*I: The firmware is already up to date.
```

GUI/CLI 모니터링 툴을 실행하여, 설치된 펌웨어 버전을 확인할 수 있습니다.

#### 1) `mobilint_ctrl` GUI 모니터링 툴
데스크톱 호스트를 위한 향상된 GUI 모드. (펌웨어 업데이트 기능 추가됨.)
![GUI 모니터링 툴](/res/image/mobilint_ctrl.png "mobilint_ctrl")

#### 2) `mobilint_ctrl_cli` CLI 모니터링 툴
GUI 환경 제약된 서버 또는 저사양 호스트를 위한 가벼운 모드.  
콘솔 명령어 : `mobilint_ctrl_cli loop`
![CLI 모니터링 툴](/res/image/mobilint_ctrl_cli.png "mobilint_ctrl_cli")
