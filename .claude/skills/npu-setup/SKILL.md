---
name: npu-setup
description: 신규 서버에서 이 레포(AX_NPU)를 clone한 뒤 Mobilint ARIES NPU 환경을 세팅해 'mobilint-cli status'가 동작하게 만든다. HF 로그인만 하면 SDK(드라이버/런타임 tar, 컴파일러 whl)를 HF private 레포에서 자동으로 download/에 받아(fetch_sdk_from_hf.py) 드라이버 빌드/설치 -> 런타임+CLI 설치 -> 디바이스/상태 점검을 수행한다. (SDK를 download/에 수동 배치해도 됨) "NPU 세팅", "mobilint-cli status 되게", "신규 서버 환경설정" 같은 요청에 사용.
---

# NPU Setup (Mobilint ARIES — mobilint-cli status까지)

## 언제 쓰나

신규 서버에서 AX_NPU 레포를 clone하고, NPU를 쓸 수 있게 환경을 세팅할 때.
목표 종착점은 **`mobilint-cli status`가 정상 출력**되는 것(= 드라이버+디바이스+런타임 OK).

## 대전제

SDK 바이너리는 git에 없다(비공개/대용량). **HF private 레포에서 받는다** — clone 후 준비물은:

1. **HF 로그인** (조직 계정): `huggingface-cli login` 또는 `export HF_TOKEN=...`.
   → SDK(드라이버/런타임/컴파일러)는 setup 스크립트가 **HF `PIA-SPACE-LAB/MXQ_NPU`의 `sdk/aries2_v1.2.0/`에서
   자동으로 `download/`에 받아온다** (`setup/fetch_sdk_from_hf.py`). 수동 배치도 여전히 가능.
2. **NPU 카드 물리 장착** (PCIe 슬롯). 미장착이면 드라이버는 깔려도 `status`에 디바이스가 안 뜬다.
3. **sudo 권한** (드라이버 빌드/설치, modprobe, make install에 필요).

→ 즉 **아무것도 없는 서버라도 "clone → HF 로그인 → 이 skill 실행"** 이면 SDK 다운로드부터 설치·검증까지 자동.
(HF 접근 불가 시엔 SDK를 `download/`에 직접 넣어도 됨: `mobilint-aries2-driver_*.tar.gz`,
`qbruntime_aries2-*_amd64.tar.gz`, `qbcompiler-*+aries2-py3-none-any.whl`.)

## 절차 (에이전트가 수행)

작업 디렉토리는 레포 루트(`AX_NPU/AX_NPU`, `download/`가 보이는 곳)다.
스크립트는 레포 위치를 동적으로 찾으므로 어느 서버에 clone하든 그대로 동작한다.

### 0단계 — HF 로그인 (SDK 자동 다운로드용, 1회)
```bash
huggingface-cli login          # 또는 export HF_TOKEN=hf_...
```

### 1단계 — SDK 확인/자동 fetch (sudo 불필요)
```bash
bash .claude/skills/npu-setup/setup_npu_cli.sh --check
```
`--check`가 `download/`에 SDK 없으면 **HF에서 자동으로 받아온다**(`setup/fetch_sdk_from_hf.py`).
HF 접근이 안 되면 SDK를 `download/`에 수동 배치하도록 안내한다. (직접 받기: `python setup/fetch_sdk_from_hf.py`)

### 2단계 — 전체 세팅 (sudo 필요)
SDK 파일이 준비됐으면:
```bash
sudo bash .claude/skills/npu-setup/setup_npu_cli.sh
```
순서대로 수행: 드라이버 빌드/설치 → `modprobe aries` → 런타임+CLI `make install` → `mobilint-cli status`.

> 에이전트가 sudo를 직접 못 쓰는 권한 모드면, 위 명령을 사용자가 프롬프트에
> `! sudo bash .claude/skills/npu-setup/setup_npu_cli.sh` 로 실행하도록 안내한다.

### 부분 실행 옵션
- `sudo bash ... --runtime` : 드라이버는 이미 있고 런타임/CLI만 설치
- `sudo bash ... --driver`  : 드라이버만 설치
- `bash ... --check`        : 점검만 (sudo 불필요)

## 검증 (성공 기준)

`mobilint-cli status` 출력에 NPU 디바이스/펌웨어 정보가 뜨면 성공.
스크립트 마지막 `[3] 점검`에서 다음이 모두 `[OK]`여야 한다:
- aries 커널 모듈 로드됨
- 디바이스 노드 `/dev/aries0`
- PCI 인식 (vendor 209f)
- mobilint-cli status 출력

## 자주 막히는 곳 (트러블슈팅)

- **status에 디바이스 0개 / `/dev/aries0` 없음** → 카드 미장착이거나 모듈 미로드.
  카드 장착 확인 후 `sudo modprobe aries`. PCI는 `lspci -d 209f:`로 확인.
- **`apt install`로 드라이버가 안 깔린다** → 이 환경은 apt 보류가 쌓인 stale 상태라 의존성 충돌.
  그래서 이 skill은 apt를 쓰지 않고 **tar 소스 직접 빌드**한다(스크립트가 처리).
- **드라이버 빌드 실패(make 없음/헤더 없음)** → `sudo apt install build-essential linux-headers-$(uname -r)`.
  Secure Boot가 켜져 있으면 모듈 로드가 막힐 수 있으니 비활성 권장.
- **커널 업데이트 후 모듈 사라짐** → dkms 미설치 환경이라 자동 재빌드 안 됨.
  `uname -r`이 바뀌면 `sudo bash ... --driver`로 재빌드.
- **`mobilint-cli: command not found`** → 런타임 미설치. `sudo bash ... --runtime`.

## 이 다음 (선택)

`mobilint-cli status`까지 됐으면 NPU 자체는 준비 완료. 모델 추론(Python)까지 하려면:
- `bash setup/setup_conda_host.sh` (conda env + qbruntime + torch/einops/timm/huggingface_hub)
- 그 다음 추론은 두 가지 방식 중 선택 (`tutorial/pe_npu/README.md`):
  - **옵션 B (빠름, 권장)**: 컴파일러 없이 HF에서 미리 컴파일된 자산을 받아 추론.
    `pe_npu.MXQInferenceFull.from_hf()` → `PIA-SPACE-LAB/MXQ_NPU`에서 `pe_full.mxq` 자동 다운로드
    (image→embedding 전부 NPU). NPU + qbruntime + 인터넷만 있으면 됨 (qbcompiler·원본 가중치 불필요).
  - **옵션 A (직접 컴파일)**: calib → `python -m pe_npu.compile --qk16` → 추론. qbcompiler(docker) 필요.
    커스텀 calib/해상도·컴파일 실험용. full NPU cos 0.99.
