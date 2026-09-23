<!-- 출처: https://docs.mobilint.com/compiler/v1.3/en/installation.html (원본) -->

# Installation and Environment Check

This chapter covers the environment needed to run qb Compiler, how to install the Python package, how to choose a target device string, and how to confirm that the compiler can see its commands and default configuration.

## System Requirements

Use a Linux environment for compilation. Mobilint recommends the official Docker images because they keep Python, CUDA, framework, and compiler dependencies aligned.

Recommended baseline:

- Ubuntu 20.04 or later
- Docker
- NVIDIA Container Toolkit when using the CUDA image
- NVIDIA GPU for faster compilation, especially for large vision models and LLMs

A CPU-only Docker image is also supported for environments without NVIDIA GPUs. Compilation may take longer.

## Install qbcompiler

Use a Docker image whose major and minor tag matches the qbcompiler wheel version. For example, use a `1.0-*` Docker image with a `qbcompiler-1.0.*` wheel.

CUDA image example:

```bash
docker pull mobilint/qbcompiler:1.3-cuda12.8.1-ubuntu22.04
docker run -it --gpus all --ipc=host \
  --name {YOUR_CONTAINER_NAME} \
  -v $(pwd):/workspace \
  mobilint/qbcompiler:1.3-cuda12.8.1-ubuntu22.04 /bin/bash
```

If models and datasets live outside the working directory, mount them explicitly:

```bash
docker run -it --gpus all --ipc=host \
  --name {YOUR_CONTAINER_NAME} \
  -v $(pwd):/workspace \
  -v {PATH_TO_MODEL_DIR}:/models \
  -v {PATH_TO_DATASET_DIR}:/datasets \
  mobilint/qbcompiler:1.3-cuda12.8.1-ubuntu22.04 /bin/bash
```

CPU-only image example:

```bash
docker pull mobilint/qbcompiler:1.3-cpu-ubuntu22.04
docker run -it --ipc=host \
  --name {YOUR_CONTAINER_NAME} \
  -v $(pwd):/workspace \
  mobilint/qbcompiler:1.3-cpu-ubuntu22.04 /bin/bash
```

Install the qbcompiler wheel inside the container:

```bash
python -m pip install /path/to/qbcompiler-{VERSION}-py3-none-any.whl
```

Starting with qbcompiler 1.2, wheel filenames do not include an NPU name. For example, the 1.3.0 wheel is named `qbcompiler-1.3.0-py3-none-any.whl`. Wheels before 1.2 may include an NPU name in the local version segment, such as `qbcompiler-1.1.2+aries2-py3-none-any.whl`; install the exact wheel file provided for that release.

## Select the Target Device

Every MXQ is compiled for a target device. Use the exact target device string in CLI options and Python calls:

| Target device string | NPU Chip |
| --- | --- |
| `aries-rb` | `ARIES` |
| `regulus-ra` | `REGULUS` |
| `regulus-rb` | `REGULUS` |
| `regulus-rb-usb` | `REGULUS` |

`regulus-rb-usb` is available from qbcompiler 1.3. It targets a USB-attached REGULUS device, whose NPU I/O boundary is not floating point, so an MXQ built for `regulus-rb` is not interchangeable with one built for `regulus-rb-usb`.

Example CLI option:

```bash
python -m qbcompiler compile --target-device regulus-rb ...
```

Example Python argument:

```python
target_device = "regulus-rb"
```

If you compile for one target device and deploy to a different target device, the MXQ may fail to load or may not run correctly.

## Verify the Installation

After installation, confirm that Python can import the package:

```bash
python - <<'PY'
import qbcompiler
print(qbcompiler.__version__)
PY
```

Then confirm that the CLI is available. This form works on every release:

```bash
python -m qbcompiler --help
python -m qbcompiler compile --help
```

From qbcompiler 1.3 the installation also puts a `qbcompiler` command on PATH,
which runs the same entry point through the same output protocol:

```bash
qbcompiler --help
```

On 1.2 that command does not exist, so a `command not found` there means the
release predates it, not that the installation failed. The rest of this manual
uses the `python -m qbcompiler` form for that reason, except the release notes,
which show the new command as the feature it is. Substitute `qbcompiler` freely
on 1.3.

The form is not the only difference on 1.2. The `compile` and `quantize`
subcommands did not run at all on 1.2.0 — they reached the compile entry point
without a target device and failed internally — so the command examples in this
manual need 1.3 regardless of which spelling you use. `parse`, `presets`,
`dump-config` and `check` are unaffected.

The core CLI commands are:

- `compile`: original model to MXQ in one command
- `parse`: original model to MBLT
- `quantize`: MBLT to MXQ
- `dump-config`: write a default or preset-based config file
- `info`: print an MBLT's provenance record as JSON (qbcompiler 1.3 and later)

## Check Presets and the Default Config

List built-in presets:

```bash
QBCOMPILER_JSONL=1 python -m qbcompiler presets
```

`presets` and `check` report on the JSONL channel, so they print nothing unless it is on. Set `QBCOMPILER_JSONL=1` to see their output; an empty result is not a failed install.

Common presets include:

- `classification`
- `detection`
- `classification_torchvision`
- `yolo_640`
- `yolo_1280`
- `llm`
- `llm_fast`
- `vision_transformer`
- `multimodal`

Dump a config before editing it:

```bash
python -m qbcompiler dump-config --preset classification_torchvision --output compile_config.yaml
```

Use this file as the starting point for repeatable builds. A config file is also the safest way to keep CLI and Python workflows aligned across teams.
