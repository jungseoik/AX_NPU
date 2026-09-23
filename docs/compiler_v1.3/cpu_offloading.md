<!-- 출처: https://docs.mobilint.com/compiler/v1.3/en/cpu_offloading.html (원본) -->

# CPU Offloading

## CPU Offloading Overview

CPU Offloading lets qb Compiler keep on the CPU those parts of a model that are unsupported or unsuitable for the NPU, while compiling the rest of the graph for the Mobilint NPU. The output is still an MXQ package, but the runtime must support CPU-offloaded MXQ execution.

Use CPU Offloading when the model has small preprocessing, postprocessing, shape, or indexing operations that are not good NPU candidates. If the unsupported region is large or appears in the middle of the performance-critical body, it is usually better to change the original model, export a cleaner graph, or compile only the NPU body subgraph.

Enable CPU Offloading explicitly in the compile configuration or Python API:

```python
from qbcompiler import mxq_compile

mxq_compile(
    model="model.onnx",
    backend="onnx",
    target_device="regulus-rb",
    calib_data_path="./calib",
    save_path="./model_cpu_offload.mxq",
    cpu_offload=True,
)
```

The same setting can be specified as `cpuOffload: true` in a compile config file, or as `--cpu-offload` if the installed CLI supports that flag.

Confirm that the runtime version on the device supports MXQ files compiled with CPU Offloading. CPU Offloading is not a replacement for arbitrary Python, PyTorch, TensorFlow, or custom application execution — it only covers the graph regions that the CPU offload path supports.

## Unsupported Operator Handling

During `parse`, qb Compiler converts the original model to MBLT and marks which layers are supported by the selected target device. During MXQ generation, CPU Offloading can split the graph into three logical regions:

- `CPU head`: operators before the NPU-executable region.
- `NPU body`: the contiguous body subgraph compiled for the NPU.
- `CPU tail`: operators after the NPU body, often task-specific postprocessing.

This is most useful for models such as YOLO, where the convolutional backbone and neck are NPU-friendly but the final decoding or filtering logic may contain unsupported operators. Netron can be used to inspect the MBLT and confirm the supported layer region before compiling.

The presence of unsupported operators is not automatically harmless. CPU Offloading is intended for graph boundary regions and simple CPU-executable operators. If an unsupported operator splits the NPU body into many small islands, CPU/NPU transfer overhead can dominate latency, and the compiler may fail to produce a practical MXQ.

## Calibration Shape Differences

Without CPU Offloading, vision models are commonly calibrated using the preprocessed tensor that enters the NPU body. For many image models this is an `HWC` tensor after resize, crop, color conversion, scaling, and normalization.

With CPU Offloading enabled, calibration data should match the original model input shape because the CPU head remains part of the compiled graph. If the original exported model expects `NCHW`, provide calibration tensors in `NCHW`. If the original model includes preprocessing and expects raw-like `NHWC` or `uint8` input, calibrate with that same shape and dtype.

The rule is:

- Compile the NPU body only: calibration data matches the NPU body input.
- Compile with CPU Offloading: calibration data matches the original model input.

Preprocessing still matters. The numerical distribution used for calibration should match the real inference input produced by the same application preprocessing path. When accuracy differs from the original model, check calibration shape, preprocessing order, channel order, normalization, and output interpretation before tuning quantization.
