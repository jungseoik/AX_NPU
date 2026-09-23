<!-- 출처: https://docs.mobilint.com/compiler/v1.3/en/basic_compile_flow.html (원본) -->

# Basic Compile Flow

qb Compiler is built around a single pipeline:

```text
Original model -> MBLT -> MXQ
```

`compile` runs the whole pipeline. `parse` and `quantize` expose the two major stages when you need more control.

When an MXQ is generated, the artifact reflects not only the target device but also the runtime core mode selected by `inferenceScheme`. An existing MXQ cannot be run later with an arbitrary core mode; regenerate the MXQ with the required `inferenceScheme` instead. In short, `single` uses independent Local Cores, `multi` is the cluster-level 4-batch mode, and `global4`/`global8` use 4 or 8 Local Cores together for one input. For the supported values and core-mode links, see [Reference — inferenceScheme](reference.md#inferencescheme).

## End-to-End Compile

Use `compile` for normal production builds:

```bash
python -m qbcompiler compile \
  --model model.onnx \
  --backend onnx \
  --calib-data-path calibration/resnet50 \
  --target-device regulus-rb \
  --output model.mxq
```

Python end-to-end compile:

```python
from qbcompiler import mxq_compile

mxq_compile(
    model="model.onnx",
    backend="onnx",
    calib_data_path="calibration/resnet50",
    target_device="regulus-rb",
    save_path="model.mxq",
)
```

The compiler parses the original model, calibrates and quantizes the parsed graph, applies target-specific compilation, and writes the MXQ.

## `parse`: Original Model to MBLT

`parse` reads the source framework format and writes MBLT:

```bash
python -m qbcompiler parse \
  --model model.onnx \
  --backend onnx \
  --target-device regulus-rb \
  --output model.mblt
```

Use this stage when you want to:

- inspect the parsed graph in Netron
- confirm layer support before quantization
- debug input names and shapes
- reuse the same parsed model for multiple target/config experiments

MBLT is not the final deployment artifact. It is the compiler's parsed model graph and weights. See [Model Parsing](model_parsing.md) for details.

## `quantize`: MBLT to MXQ

`quantize` reads an MBLT and writes MXQ:

```bash
python -m qbcompiler quantize \
  --mblt model.mblt \
  --calib-data-path calibration/resnet50 \
  --target-device regulus-rb \
  --output model.mxq
```

In this manual, `quantize` means the complete MBLT-to-MXQ stage. It includes calibration and quantization, but it also includes the target-specific work needed to produce an MXQ package. See [Model Quantization](model_quantization.md) for calibration data preparation and quantization settings.

Use this stage when:

- the MBLT has already been reviewed
- you need to compare presets or config files
- you need to compile the same parsed model for different target device strings
- you are debugging calibration or quantization without reparsing the original model

## `compile` vs. `parse` + `quantize`

Use `end-to-end compile` when the model, calibration data, and target are already known. It is shorter, easier to automate, and keeps the build path simple.

Use `parse` plus `quantize` when model structure matters to the task. This is common for models with unsupported preprocessing/postprocessing, YOLO-style heads (see [Vision Model Input Handling](vision.md)), [LLMs](transformer.md), or any case where you want to inspect CPU head, NPU body, and CPU tail boundaries (see [CPU Offloading](cpu_offloading.md)) before generating MXQ.

## Common Artifact Checklist

For a reproducible compile, keep this set:

- original model and source revision
- calibration sample directory or metadata file
- compile config file or preset name (see [Reference — CompileConfig Schema](reference.md#compileconfig-schema))
- target device string
- generated MBLT, when using staged compilation
- generated MXQ
- validation input and validation output
- extracted body artifact, when CPU head/body/tail boundaries are part of deployment
