<!-- 출처: https://docs.mobilint.com/runtime/v1.4/en/installing_utility.html (원본) -->

# Utility Installation

Mobilint provides a set of utilities for few features like checking status of NPU, running benchmark, and viewing MXQ file information. This secction explains how to install these utilities.

## Linux-Specific Utilities

- **mobilint-runtime-gui**: A GUI-based utility that enables users to easily run and test various AI models on the NPU without writing any code.

### Install mobilint-runtime-gui

If you have already added Mobilint's APT repository to your system during {external+aries:doc}`Driver Installation <driver-installation>` or [Runtime Library Installation](installing_runtime_library.md) process, you can install `mobilint-runtime-gui` using following command:

```bash
sudo apt install mobilint-runtime-gui
```

## Windows, Linux Common Utilities

Following utilities are available on both Windows and Linux:

- **mobilint-cli**: Command-line utility for checking NPU status, running benchmarks, and viewing MXQ information.
- **mobilint_ctrl**: GUI utility for checking NPU status and updating firmware.
- **mobilint_ctrl_cli**: CLI utility for checking NPU status.
- **aries_flash_firmware**: CLI utility for updating NPU firmware.

### Install mobilint-cli

#### Option 1: Installation via APT (Linux)

If you have already added Mobilint's APT repository to your system during {external+aries:doc}`Driver Installation <driver-installation>` or [Runtime Library Installation](installing_runtime_library.md) process, you can install `mobilint-cli` using following command:

```bash
sudo apt install mobilint-cli
```

#### Option 2: Installation via DNF (Linux)

If you have already registered Mobilint's DNF repository to your system during [Runtime Library Installation](installing_runtime_library.md#option-2-installing-via-dnf-on-redhat-based-os) process, you can install `mobilint-cli` on RedHat-based OS using following command:

```bash
sudo dnf install mobilint-cli
```

The `mobilint-qb-runtime` package is installed automatically as a dependency if it is not already present.

#### Option 3: Manual Installation from Download Center (Linux)

`mobilint-cli` utility is included in the runtime library (qb Runtime) package provided through official [Mobilint Download Center](https://dl.mobilint.com).

Within the downloaded directory, run the following command to install both runtime library and `mobilint-cli`:

```bash
sudo make install
```

For more details, refer to [Runtime Library Installation Guide](installing_runtime_library.md#option-3-manual-installation-from-download-center).

#### Option 4: Installation via the Installer (Windows)

On Windows, the runtime library installer also installs `mobilint-cli`. Keep the **Install `mobilint-cli` utilities** task selected on the installer's **Select Additional Tasks** screen. If you also keep the task that adds the `bin` folder to `PATH`, you can run it from any directory.

For the full procedure, see [Runtime Library Installation - Windows](installing_runtime_library.md#installing-on-windows).

```{note}
`mobilint-cli` for Windows is available starting from runtime library version **1.4.0**.
```

### Install Mobilint Ctrl

#### Windows

Windows package can be downloaded from official [Mobilint Download Center](https://dl.mobilint.com).

Distribution file: `mobilint_ctrl_{version}_windows.zip`

After downloading `mobilint_ctrl_{version}_windows.zip` file, extract it and double-click the included `mobilint_ctrl.exe` file to run the program.

#### Linux

The Linux package can be downloaded from the Mobilint Download Center.

Supported Ubuntu versions:
- Ubuntu 20.04 (LTS)
- Ubuntu 22.04 (LTS)
- Ubuntu 24.04 (LTS)

| Package Filename          | Supported Ubuntu Versions |
| ------------------------- | ------------------------- | 
| `..._ubuntu20.04_amd.deb` | Ubuntu 20.04 only         |
| `..._ubuntu22.04_amd.deb` | Ubuntu 22.04 and 24.04    |

After extracting the file, install using:

```bash
sudo apt install ./mobilint_ctrl_{version}_<Ubuntu_Version>_amd64.deb
```

### Install aries_flash_firmware

#### Windows

`aries_flash_firmware.exe` is included in the `mobilint_ctrl` Windows distribution package:

```bash
mobilint_ctrl_{version}_windows.zip
 └── aries_flash_firmware.exe
```

It can be executed from a Command Prompt or PowerShell environment:

```powershell
aries_flash_firmware.exe status
```

#### Linux

The firmware flashing package is included inside the `mobilint_ctrl` Linux distribution package: `aries_flash_firmware_<Build_version>_ubuntu_<amd64/arm64>.deb`

Supported Ubuntu versions:

- Ubuntu 20.04 (LTS)
- Ubuntu 22.04 (LTS)
- Ubuntu 24.04 (LTS)

Install using:

```bash
sudo apt install ./aries_flash_firmware_<Build_version>_ubuntu_<amd64/arm64>.deb
```	
