---
name: npu-setup
description: 신규 서버에서 이 레포(AX_NPU)를 clone한 뒤 Mobilint ARIES NPU 환경을 세팅해 'mobilint-cli status'가 동작하게 만든다. 세팅 시작 전에 SDK 번들 버전(1.0v / 1.1v)을 사용자에게 반드시 한 번 물어본다. HF 로그인만 하면 SDK(드라이버/런타임 tar, 컴파일러 whl)를 HF private 레포에서 자동으로 download/에 받아(fetch_sdk_from_hf.py) 드라이버 빌드/설치 -> 런타임+CLI 설치 -> 디바이스/상태 점검을 수행한다. (SDK를 download/에 수동 배치해도 됨) "NPU 세팅", "mobilint-cli status 되게", "신규 서버 환경설정" 같은 요청에 사용.
---

# NPU Setup (Mobilint ARIES — mobilint-cli status까지)

## 언제 쓰나

신규 서버에서 AX_NPU 레포를 clone하고, NPU를 쓸 수 있게 환경을 세팅할 때.
목표 종착점은 **`mobilint-cli status`가 정상 출력**되는 것(= 드라이버+디바이스+런타임 OK).

## 대전제

SDK 바이너리는 git에 없다(비공개/대용량). **HF private 레포에서 받는다** — clone 후 준비물은:

1. **HF 로그인** (조직 계정): `huggingface-cli login` 또는 `export HF_TOKEN=...`.
   → SDK(드라이버/런타임/컴파일러)는 setup 스크립트가 **HF `PIA-SPACE-LAB/MXQ_NPU`의
   선택한 버전 폴더(`sdk/v1.0/` 또는 `sdk/v1.1/`)에서 자동으로 받아온다** (`setup/fetch_sdk_from_hf.py`).
   수동 배치도 여전히 가능.
2. **NPU 카드 물리 장착** (PCIe 슬롯). 미장착이면 드라이버는 깔려도 `status`에 디바이스가 안 뜬다.
3. **sudo 권한** (드라이버 빌드/설치, modprobe, make install에 필요).

→ 즉 **아무것도 없는 서버라도 "clone → 토큰 전달 → 이 skill 실행"** 이면 SDK 다운로드부터 설치·검증까지
자동으로 끝난다. 사람이 개입하는 지점은 **① SDK 버전 답하기(질문 1회) ② 토큰/sudo 제공** 둘뿐이다.

자격증명은 환경변수 또는 레포 루트 `.env`(gitignore됨)로 전달한다:
```
HF_TOKEN=hf_...        # SDK·모델 다운로드 (필수)
SUDO_PASS=...          # 드라이버 설치용 (선택 — 없으면 사용자가 직접 sudo 실행)
```
### SDK 번들 버전 선택 (1.0v / 1.1v)

번들 정의는 **`setup/sdk_versions.json`이 단일 기준**이다. 버전마다 파일명 규칙과 컴파일 요건이 다르다.

| 버전 | 구성 | 로컬 경로 | 컴파일 |
|---|---|---|---|
| **1.0** (기본) | 드라이버 1.13 / qbruntime 1.2.0 / qbcompiler 1.1.2 (`aries2` 표기) | `download/` | docker `qbcompiler:1.1-cpu-ubuntu22.04` (GPU 있으면 `1.1-cuda12.8.1-…`) |
| **1.1** | 드라이버 1.14 / qbruntime 1.4.0 / qbcompiler 1.2.0 (`aries` 표기) | `download/sdk/v1.1/` | docker `qbcompiler:1.2-cpu-ubuntu22.04` (GPU 있으면 `1.2-cuda12.8.1-…`) |

> **컴파일은 GPU가 없어도 된다.** 벤더가 버전별로 `-cpu` / `-cuda` 이미지를 쌍으로 배포하고,
> 컴파일 코드가 `torch.cuda.is_available()`로 CPU에 자동 폴백한다(`mblt-sdk-tutorial` 예제 README).
> GPU 없는 호스트는 **cpu 이미지**를 쓸 것 — cuda 이미지는 27.7GB로 훨씬 크다.
> 1.1의 컴파일러 whl에는 mmc가 들어있지만 그 `.so`가 CUDA 빌드 torch에 링크돼 있어
> **호스트 직접 설치는 비권장**이다(CPU torch에서 `libtorch_cuda.so` 없음 — 2026-09-01 실측).
> 이 호스트에 권장되는 이미지는 `python setup/sdk_resolve.py --sdk <버전>` 이 알려준다.

```bash
python setup/sdk_resolve.py --list                                  # 버전 목록·현재 파일 유무
python setup/fetch_sdk_from_hf.py --sdk 1.1                         # HF에서 1.1v 받기
sudo bash .claude/skills/npu-setup/setup_npu_cli.sh all --sdk 1.1   # 1.1v로 세팅
SDK_VERSION=1.1 sudo -E bash .claude/skills/npu-setup/setup_npu_cli.sh   # 환경변수로도 가능
```

`--sdk`를 안 주면 manifest의 기본값(현재 **1.0**)을 쓴다. 버전을 올릴 때는
`sdk_versions.json`의 `default`만 바꾸면 모든 스크립트가 함께 따라간다.

(HF 접근 불가 시엔 SDK를 해당 버전 경로에 직접 넣어도 됨 — 1.0: `mobilint-aries2-driver_*.tar.gz`,
`qbruntime_aries2-*_amd64.tar.gz`, `qbcompiler-*+aries2-py3-none-any.whl`.)

## 절차 (에이전트가 수행)

작업 디렉토리는 레포 루트(`download/`가 보이는 곳)다.
스크립트는 레포 위치를 동적으로 찾으므로 어느 서버에 clone하든 그대로 동작한다.

### ★ 0단계 — 사용자에게 SDK 버전을 먼저 묻는다 (필수, 생략 금지)

세팅을 시작하기 **전에** `AskUserQuestion` 으로 번들 버전을 한 번 확인한다. 이 질문 하나만 받고,
그 뒤 설치·검증은 전부 에이전트가 알아서 끝낸다(추가 질문 없이).

- 질문: "어느 SDK 번들로 세팅할까?"
- 선택지는 `python setup/sdk_resolve.py --list` 출력으로 구성한다. 현재:
  - **1.0v** — 드라이버 1.13 / qbruntime 1.2.0 / qbcompiler 1.1.2. 검증된 기본값.
    컴파일은 docker 이미지(27.7GB) 필요.
  - **1.1v** — 드라이버 1.14 / qbruntime 1.4.0 / qbcompiler 1.2.0. 컴파일은 docker
    `qbcompiler:1.2-cpu-ubuntu22.04`(GPU 없어도 됨). 최신 벤더 SDK.
- 사용자가 "아무거나/모르겠다"라고 하면 manifest 기본값(현재 1.0)으로 진행하고 그 사실을 알린다.

### 1단계 — 원샷 세팅 실행

버전이 정해지면 아래 한 줄로 부트스트랩(python/huggingface_hub) → SDK 다운로드 → 드라이버·런타임
설치 → 점검까지 수행한다. `.env`에 `HF_TOKEN`(필수) / `SUDO_PASS`(선택)가 있으면 자동으로 읽는다.

```bash
bash setup/setup_all.sh --sdk 1.1        # 사용자가 고른 버전으로
bash setup/setup_all.sh --sdk 1.0 --fetch-only   # 다운로드까지만 확인하고 싶을 때
```

- `HF_TOKEN`이 환경변수·`.env`·`huggingface-cli login` 중 어디에도 없으면 다운로드 단계에서
  멈추므로, 사용자에게 토큰을 요청한다.
- sudo 비밀번호가 없고 무권한 모드면 스크립트가 **직접 실행할 명령을 안내하며 종료**한다.
  그때는 사용자에게 프롬프트에서
  `! sudo -E bash .claude/skills/npu-setup/setup_npu_cli.sh all --sdk <버전>` 실행을 요청한다.

### 1-b단계 — 단계별로 하고 싶을 때 (원샷 대신)
```bash
python setup/fetch_sdk_from_hf.py --sdk 1.1                       # SDK만 다운로드
bash .claude/skills/npu-setup/setup_npu_cli.sh --check --sdk 1.1   # 점검만 (sudo 불필요)
sudo bash .claude/skills/npu-setup/setup_npu_cli.sh all --sdk 1.1  # 드라이버+런타임 설치
```
순서대로 수행: 드라이버 빌드/설치 → `modprobe aries` → 런타임+CLI `make install` → `mobilint-cli status`.

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
- **평가(정확도/성능 측정)까지 할 서버라면**: `export HF_TOKEN=hf_... && python -m eval.tta download`
  → `eval/datasets/TTA_인증용/`(이상행동 4종 영상 200 + 라벨 200 + clips 252, 2.2GB). 상세는 `eval/README.md`.
