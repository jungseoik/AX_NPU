<!-- 출처: https://docs.mobilint.com/compiler/v1.3/en/compile_configuration.html (원본) -->

# Compile Configuration

[First Compile](first_compile.md) showed the shortest path from a model to an MXQ. This chapter explains how to control that path with compile configuration.

qb Compiler accepts configuration in three ways:

- **Config preset** — a built-in starting point for a model family
- **Config file** — a JSON or YAML file with compile settings
- **Python sub-config objects** — typed config objects passed to the Python API

Use presets for common model families, config files for repeatable builds, and Python sub-config objects for scripted workflows.

## Config Presets

A preset is a named, built-in partial configuration for a common model family. It sets calibration, quantization, preprocessing, and model-family defaults so you do not need to specify every field.

List the presets available in the installed package:

```bash
QBCOMPILER_JSONL=1 python -m qbcompiler presets
```

### Using a preset

From the CLI, pass `--config-preset`:

```bash
python -m qbcompiler compile \
  --model resnet50.onnx \
  --backend onnx \
  --target-device regulus-rb \
  --config-preset classification \
  --calib-data-path ./calib_resnet50 \
  --output resnet50.mxq
```

From Python, pass `config_preset`:

```python
from qbcompiler import mxq_compile

mxq_compile(
    model="resnet50.onnx",
    backend="onnx",
    target_device="regulus-rb",
    config_preset="classification",
    calib_data_path="./calib_resnet50",
    save_path="resnet50.mxq",
)
```

Presets are starting points, not model identifiers. A preset does not remove the need to provide the correct model path, backend, target device, input information, and calibration data. For model-family-specific preset usage, see [Vision](vision.md) and [Transformer](transformer.md). For the full preset list and what each preset sets, see [Reference — Preset List](reference.md#preset-list).

## dump-config: Generate a Config Template

`dump-config` writes a fully expanded `CompileConfig` to a JSON or YAML file. Use it to see every available field and its default value, or to inspect what a preset changes.

### Generate a default template

```bash
python -m qbcompiler dump-config --output compile_config.json
```

### Generate a preset-based template

```bash
python -m qbcompiler dump-config --preset yolo_640 --output yolo_640_config.yaml
```

The output format is chosen by file extension: `.yaml` or `.yml` produces YAML, anything else produces JSON.

### Workflow: dump → edit → compile

1. Generate a template from a preset that matches your model family.
2. Open the file and change only the fields you need to override.
3. Pass the edited file to `compile` or `quantize` with `--compile-config`.

```bash
# 1. Generate
python -m qbcompiler dump-config --preset classification --output my_config.yaml

# 2. Edit my_config.yaml (change calibration, bit settings, etc.)

# 3. Compile with the edited config
python -m qbcompiler compile \
  --model resnet50.onnx \
  --backend onnx \
  --target-device regulus-rb \
  --compile-config my_config.yaml \
  --calib-data-path ./calib_resnet50 \
  --output resnet50.mxq
```

Before using a config in automation, compare the generated template against the default to confirm which values came from the preset and which came from your edits.

## config-save-path: Record the Config a Build Used

`dump-config` shows what a preset or the defaults contain. `--config-save-path`
answers the other question: what did *this* build actually run with, once the
precedence order below had picked between a preset and a config file and the
sub-config objects and explicit arguments had been applied on top.

Available from qbcompiler 1.3 on `quantize` and `compile`:

```bash
python -m qbcompiler compile \
  --model resnet50.onnx \
  --backend onnx \
  --target-device regulus-rb \
  --config-preset classification \
  --calib-data-path ./calib_resnet50 \
  --output resnet50.mxq \
  --config-save-path resnet50_used_config.yaml
```

```python
qbcompiler.mxq_compile(
    model="resnet50.onnx",
    target_device="regulus-rb",
    calib_data_path="./calib_resnet50",
    save_path="resnet50.mxq",
    config_preset="classification",
    config_save_path="resnet50_used_config.yaml",
)
```

The file holds the normalized `CompileConfig` after every layer of the
[Config Resolution Priority](#config-resolution-priority) below has been applied,
with all sub-configurations materialized. Format follows the suffix: `.yaml` or
`.yml` produces YAML, anything else produces JSON. Parent directories are created.

Two properties make it useful in automation:

- It is written **before** compilation starts, so the record survives a run that
  fails or is interrupted — which is exactly when you want to know what the
  settings were.
- It can be fed straight back as `--compile-config` / `compile_config=` to
  reproduce the same compile.

On the `compile` pipeline the file is written by the quantize phase, the phase
that carries the full `CompileConfig`; the parse phase resolves only `device` and
`cpu_offload`. If the destination cannot be written the run fails with
`OUTPUT_WRITE_ERROR` naming that path, rather than an internal error.

## Custom Config File

A compile config file is useful when a build must be repeatable. Keep it in version control next to the model export recipe.

### CLI usage

Pass the file path with `--compile-config`:

```bash
python -m qbcompiler compile \
  --model model.onnx \
  --backend onnx \
  --target-device regulus-rb \
  --compile-config my_config.json \
  --calib-data-path ./calib \
  --output model.mxq
```

### Python usage

Pass the file path as the `compile_config` argument, which also accepts a `CompileConfig` object:

```python
from qbcompiler import mxq_compile

mxq_compile(
    model="resnet50.onnx",
    target_device="regulus-rb",
    calib_data_path="./calib_resnet50",
    save_path="resnet50.mxq",
    backend="onnx",
    compile_config="resnet50_compile_config.json",
)
```

The config file contains compile settings only. Model path, calibration data path, target device, backend, and output path remain normal CLI or Python arguments. JSON and YAML files are interchangeable when they contain the same fields. The file must contain only supported `CompileConfig` fields; do not include metadata keys such as `$preset`, `$description`, or `$skipValidation`.

A preset and a config file are alternatives, not layers. When both are given, the preset is used and the file is ignored — with no warning. Start from `dump-config --preset <name>` and edit the result if you want a preset's values plus your own.

When editing generated config files, `inferenceScheme` controls the runtime core mode prepared into the MXQ. `single`, `multi`, `global4`, and `global8` map to different ARIES core-collaboration modes; see [Reference — inferenceScheme](reference.md#inferencescheme) for the supported values and detailed core-mode links.

For the full `CompileConfig` schema and field descriptions, see [Reference — CompileConfig Schema](reference.md#compileconfig-schema).

## Python Sub-Config Objects

The Python API provides typed sub-config objects for specialized areas. Use them when writing reusable compile scripts:

```python
from qbcompiler import mxq_compile
from qbcompiler.configs import CalibrationConfig, BitConfig

mxq_compile(
    model="resnet50.onnx",
    backend="onnx",
    target_device="regulus-rb",
    calib_data_path="./calib_resnet50",
    save_path="resnet50.mxq",
    config_preset="classification",
    calibration_config=CalibrationConfig(mode=1, output=0),
    bit_config=BitConfig(
        transformer=BitConfig.Transformer(
            weight=BitConfig.Transformer.Weight(
                query=8, key=8, value=8, output=8, head=8, router=8,
                ffn=8,  # or BitConfig.Transformer.Weight.Ffn(up=8, gate=8, down=8)
            ),
        ),
    ),
)
```

Sub-config objects override the corresponding section of the preset or config file. This makes it easy to keep model-family defaults from a preset while adjusting specific quantization or LLM settings in code.

Available sub-config objects include `CalibrationConfig`, `BitConfig`, `LlmConfig`, `HessianQuantConfig`, `LayerBiasCorrectionConfig`, `ModConfig`, `EquivalentTransformationConfig`, `SearchWeightScaleConfig`, `Uint8InputConfig`, `PreprocessingConfig`, `SaveSampleConfig` and `ResourceManagementConfig`, all of which have a matching `*_config` argument. `LayerBiasCorrectionConfig` is new in qbcompiler 1.3. For the role and tuning order of each quantization sub-config, see [Model Quantization — Quantization Configuration](model_quantization.md#quantization-configuration).

## Config Resolution Priority

When the same setting is provided in more than one place, the effective value follows this priority (highest wins):

1. Explicit CLI or Python API arguments
2. Individual sub-config objects
3. `compile_config` file **or** `config_preset` — whichever is given; if both are, the preset wins and the file is not read
4. `CompileConfig` defaults

Use this layering deliberately: pick a starting point with either the preset or the file, and put one-off experiment values on the command line or in the Python call.

When in doubt, add `--config-save-path` to the build and read the file it writes.
