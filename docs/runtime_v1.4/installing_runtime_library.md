<!-- 출처: https://docs.mobilint.com/runtime/v1.4/en/installing_runtime_library.html (원본) -->

# Runtime Library Installation

This section explains how to install the Mobilint NPU **qb Runtime** which is required to run inference on Mobilint NPUs using precompiled or custom models.

```{attention}
- For ARIES devices, complete driver installation before proceeding. Please check ARIES manual {external+aries:doc}`driver installation <driver-installation>` documentation. REGULUS ships with the driver and runtime pre-installed.
- Check compatible driver and firmware versions in the ARIES {external+aries:doc}`compatibility guide <compatibility>`; for runtime–MXQ compatibility, see [Checking Compatibility](compatibility.md).
```

## Table of Contents

- Linux

    - [Option 1: Installing via APT on Debian-based OS](#option-1-installing-via-apt-on-linux-ubuntu)

    - [Option 2: Installing via DNF on RedHat-based OS](#option-2-installing-via-dnf-on-redhat-based-os)

    - [Option 3: Manual Installation from Download Center](#option-3-manual-installation-from-download-center)

        - [Option 3-1: System-Wide Installation](#option-3-1-system-wide-installation)

        - [Option 3-2: Use Without Installation](#option-3-2-use-without-installation)

- Windows

    - [Installing with the Installer](#windows-install)

    - [Uninstalling](#windows-uninstall)

- Common to Windows and Linux

    - [Installing the Python Library with pip](#pip)

## Linux

(option-1-installing-via-apt-on-linux-ubuntu)=
### Option 1: Installing via APT on Debian-based OS

This method installs the runtime system-wide using Debian-based OS's package manager `apt`.

#### Installation Process

1. Add the Mobilint APT repository to your system before proceeding with the installation.

    ```{tip}
    If you’ve already added the Mobilint APT repository during driver installation, you can skip this step.
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

2. Install via apt package manager.

    ```bash
    sudo apt install mobilint-qb-runtime
    ```

3. Verify installation.

    ```bash
    dpkg -L mobilint-qb-runtime
    ```

#### Troubleshooting Anaconda Error

If you are using anaconda virtual environment, you can get error message like below.

```bash
../lib/libstdc++.so.6: version `GLIBCXX_3.4.32' not found
```

This issue is caused by the version confliction of libstdc++ bundled with Anaconda, and the appropriate solution may vary depending on the Anaconda version.
We recommend searching online for the most suitable fix.

Additionally, to prevent such conflicts, it is recommended to use Python’s built-in virtual environment (`venv`), which utilizes the system’s `libstdc++` library.

### Option 2: Installing via DNF on RedHat-based OS

This method installs the runtime system-wide using RedHat-based OS's package manager `dnf`. The packages are built against glibc 2.28 (EL8) and support RHEL, Rocky Linux on both x86_64 and aarch64 architectures.

```{note}
The Mobilint DNF repository provides the NPU driver package (`mobilint-aries-driver`) in addition to `mobilint-qb-runtime` and `mobilint-cli`. To install the NPU driver on RedHat-based OS, refer to the DNF installation method on the {external+aries:doc}`Driver Installation <driver-installation>` page of the ARIES manual.
```

#### Installation Process

1. Add the Mobilint DNF repository to your system before proceeding with the installation.

    ```bash
    # Register the repository
    sudo curl -fsSL https://dl.mobilint.com/dnf/mobilint.repo \
         -o /etc/yum.repos.d/mobilint.repo
    ```

2. Install via dnf package manager.

    ```bash
    # Install the runtime library
    sudo dnf install mobilint-qb-runtime

    # (Optional) Install the Mobilint CLI utility
    sudo dnf install mobilint-cli
    ```

    ````{note}
    On the first installation, `dnf` asks for confirmation before importing Mobilint's GPG signing key. Check that the output below, then please answer `y`.

    ```text
    Importing GPG key 0x________:
     Userid     : "Mobilint_RPM (gpg key for dnf - RSA key.) <infra@mobilint.com>"
     Fingerprint: 
     From       : https://dl.mobilint.com/dnf/gpg.pub
    Is this ok [y/N]: y
    ```
    ````

3. Verify installation.

    ```bash
    rpm -ql mobilint-qb-runtime
    ```

### Option 3: Manual Installation from Download Center

#### Requirements

- Download the appropriate runtime library from Mobilint’s official [Download Center](https://dl.mobilint.com).

#### Installation Process

1. Unzip the downloaded runtime library file. `{RUNTIME_VERSION}` corresponding to your runtime library version.

    ```bash
    tar -xvzf qb-runtime_aries2-v4_v{RUNTIME_VERSION}.tar.gz
    ```

    You should see a directory structure like this:

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

There are two options from here: **System-wide installation**, or **using without installing**. 

#### Option 3-1: System-Wide Installation

Install the runtime library system-wide using the following commands.

```bash
cd qb-runtime_aries2-v4_v{RUNTIME_VERSION}
sudo make install
```

#### Option 3-2: Use Without Installation

Specify the paths to `qb-runtime_aries2-v4_v{RUNTIME_VERSION}/qbruntime/qbruntime/include` and `qb-runtime_aries2-v4_v{RUNTIME_VERSION}/qbruntime/qbruntime/lib` during compilation as shown below:

```bash
g++ -o {output_binary} {source_code} -I{path_to_include} \
    -L{path_to_library} -lqbruntime
```

Then export environment variable `LD_LIBRARY_PATH` to let linker know runtime library(`*.so`) path.

```bash
export LD_LIBRARY_PATH={path_to_library}
```

## Installing on Windows

On Windows, the runtime library is distributed as an installer. The installer copies the runtime library and headers, installs the additional components you select, and registers the environment variables. If you only need the Python library, see [Installing the Python Library with pip](#pip).

```{note}
The Windows installer is available starting from runtime library version **1.4.0**.
```

(windows-install)=
### Installing with the Installer

#### Requirements

- Download the Windows installer `mobilint-qb-runtime-sdk_v{RUNTIME_VERSION}_setup.exe` from Mobilint’s official [Download Center](https://dl.mobilint.com). `{RUNTIME_VERSION}` corresponds to your runtime library version.

#### Installation Process

1. Run the downloaded installer and choose the installation directory. The default is `C:\Users\{user}\AppData\Local\Programs\mobilint-qb-runtime-sdk`; click **Browse** to change it.

    ![Destination folder selection screen](../res/image/windows_installer_install_step1_en.png "Select destination location")

2. Select the additional tasks to perform during installation. All three options below are selected by default; leave them as they are unless you have a reason to change them.

    | Task | Description |
    | --- | --- |
    | Install `mobilint-cli` utilities | Installs the command-line utility for checking NPU status, running benchmarks, and inspecting MXQ files. See [Utility Usage](utility_usage.md) for details. |
    | Set `qbruntime_DIR` environment variable | Registers the path so that CMake `find_package` can locate the runtime library. See [Compiling C++ code - CMake](#compile-cmake). |
    | Add `bin` folder to `PATH` | Makes `mobilint-cli` and `qbruntime.dll` discoverable without specifying a path. |

    ![Additional tasks selection screen](../res/image/windows_installer_install_step2_en.png "Select additional tasks")

3. Review the destination and the additional tasks on the summary screen, then click **Install**.

    ![Installation summary screen](../res/image/windows_installer_install_step3_en.png "Ready to install")

4. The wizard reports the completed installation. Click **Finish** to close the installer.

    ![Setup completion screen](../res/image/windows_installer_install_step4_en.png "Setup complete")

```{note}
Environment variable changes do not apply to terminals that are already open. Open a new terminal after installation.
```

To use the installed library from a **Visual Studio** project, see [Compiling C++ code - Windows](#compile-windows); from a CMake project, see [Compiling C++ code - CMake](#compile-cmake).

(windows-uninstall)=
### Uninstalling

You can uninstall the runtime in two ways.

- **From the Control Panel**: Select `Mobilint qb Runtime SDK` in **Programs and Features**, or in **Apps > Installed apps** in Windows Settings, and remove it.

- **By running the installer again**: Running the downloaded installer again opens the dialog below for handling the existing installation. Choose **Remove** to delete the installed files, or **Reinstall** to overwrite the existing installation.

    ![Dialog offering to remove or reinstall the existing installation](../res/image/windows_installer_remove_en.png "Remove")

## Common to Windows and Linux

(pip)=
### Installing the Python Library with pip

The Python library can be installed with pip on both Windows and Linux.

```bash
pip install mobilint-qb-runtime
```

Wheels are provided for Windows x86-64 and for Linux x86-64/aarch64. 32-bit environments are not supported.

````{note}
The Linux wheel uses the `manylinux_x_y` platform tag, which requires **pip 20.3 or later**. If pip cannot find the package even on a supported Python version (3.8–3.14), check your pip version and upgrade it if needed:

```bash
pip install --upgrade pip
```
````
