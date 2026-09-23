<!-- 출처: https://docs.mobilint.com/compiler/v1.3/en/model_parsing.html (원본) -->

# Model Parsing

MBLT is Mobilint's internal parsed IR. It contains the model graph and weights after qb Compiler has read the original model and converted it into Mobilint's internal representation. In the normal pipeline, qb Compiler uses MBLT as the intermediate artifact between the original model and the final MXQ package:

```text
Original model -> MBLT -> MXQ
```

Most users compile directly from an original model to MXQ. Creating an MBLT explicitly is useful when you want to inspect the parsed graph, check which layers can run on the NPU, or separate parsing problems from MXQ generation problems.

## How to Create MBLT

Use the `mblt_compile` Python API to convert an original model into MBLT.

```python
from qbcompiler import mblt_compile

mblt_compile(
    model="resnet50.onnx",
    backend="onnx",
    target_device="regulus-rb",
    mblt_save_path="resnet50.mblt",
)
```

The `backend` value selects the original model framework. Supported backend names are `onnx`, `tf`, `tflite`, `torchscript`, and `torch`. HuggingFace transformers models are handled through the `torch` path with the required model, tokenizer, feed inputs, and LLM configuration supplied from Python or a compile config.

The `target_device` value should match the target Mobilint device string used for the final MXQ. Common values are `aries-rb`, `regulus-ra`, `regulus-rb`, and `regulus-rb-usb`.

If you use CPU offloading, create the MBLT with CPU offload enabled so the graph is split according to the same CPU/NPU boundary that the final compile flow will use.

The same operation is available through the `parse` CLI. Note that the CLI path is only recommended when the backend is `onnx`.

```bash
python -m qbcompiler parse \
  --model resnet50.onnx \
  --backend onnx \
  --target-device regulus-rb \
  --output resnet50.mblt
```

With CPU offloading enabled:

```bash
python -m qbcompiler parse \
  --model yolov8n.onnx \
  --backend onnx \
  --target-device regulus-rb \
  --cpu-offload \
  --config-preset yolo_640 \
  --output yolov8n.mblt
```

## How to Create MXQ from MBLT

Use the `mxq_compile` Python API with an MBLT path as `model` to convert an existing MBLT to MXQ.

```python
from qbcompiler import mxq_compile

mxq_compile(
    model="resnet50.mblt",
    target_device="regulus-rb",
    calib_data_path="./calibration/resnet50",
    config_preset="classification",
    save_path="resnet50.mxq",
)
```

Use the same compile config or preset family that you would use for end-to-end compile. This stage is not only calibration and quantization; it also performs the MXQ generation work required to produce the executable package for the target.

The same operation is available through the `quantize` CLI.

```bash
python -m qbcompiler quantize \
  --mblt resnet50.mblt \
  --calib-data-path ./calibration/resnet50 \
  --config-preset classification \
  --target-device regulus-rb \
  --output resnet50.mxq
```

For advanced options, generate or edit a JSON/YAML config and pass it with `--compile-config`. `quantize` has no `--cpu-offload` flag, so a `.mblt` parsed with CPU offloading needs `cpuOffload: true` set in that file — no preset sets it, and a multi-subgraph `.mblt` is refused without it.

```bash
python -m qbcompiler dump-config \
  --preset yolo_640 \
  --output yolo_640_config.yaml

# The .mblt above was parsed with --cpu-offload, so set cpuOffload: true in
# yolo_640_config.yaml before this step — no preset sets it.

python -m qbcompiler quantize \
  --mblt yolov8n.mblt \
  --calib-data-path ./calibration/yolo \
  --compile-config yolo_640_config.yaml \
  --target-device regulus-rb \
  --output yolov8n.mxq
```

## Opening MBLT in Netron

Netron is the recommended visual tool for checking MBLT graph structure and NPU support status. Use it to inspect layer order, tensor flow, input/output shapes, and CPU/NPU boundaries before spending time on calibration and MXQ generation.

Mobilint provides two access paths:

- Online Netron: [http://netron.mobilint.com](http://netron.mobilint.com)
- Windows/Linux downloads: [http://dl.mobilint.com](http://dl.mobilint.com)

To inspect an MBLT:

1. Create the `.mblt` with `parse` or `mblt_compile`.
2. Open Netron.
3. Load the `.mblt` file.
4. Follow the graph from model inputs to model outputs.

If a model contains customer data or proprietary weights, use the downloaded Netron application in the appropriate secure environment instead of uploading the file to the online service.

## Check Model Layers and NPU Support

In Mobilint's Netron view, layer color indicates NPU support:

| Color | Meaning |
| --- | --- |
| Blue | Supported layer. The layer can be assigned to the NPU. |
| Black | Unsupported layer. The layer must run on CPU or be handled outside the NPU body. |

Use the color view as a first-pass support check. Then verify the graph boundary:

- A fully supported model should show the main compute graph as blue from input to output.
- Unsupported preprocessing near the input usually appears as a CPU head.
- Supported middle layers usually form the NPU body.
- Unsupported postprocessing near the output usually appears as a CPU tail.

This view is especially useful for models that include resize, decode, NMS, custom tensor reshaping, or task-specific postprocessing inside the exported graph.

## CPU Head, NPU Body, and CPU Tail

With CPU offloading, qb Compiler may split one parsed model into multiple subgraphs:

| Subgraph | Meaning |
| --- | --- |
| CPU head | Operators before the NPU body that are not assigned to the NPU. This often contains preprocessing or shape preparation. |
| NPU body | The largest NPU-executable body subgraph. This is the part compiled into the core NPU executable path. |
| CPU tail | Operators after the NPU body that are not assigned to the NPU. This often contains decode, filtering, or postprocessing. |

When checking an MBLT in Netron, confirm that the NPU body contains the numerically heavy part of the network. A small CPU head or CPU tail can be acceptable, but a small NPU body usually means the model structure, export options, or compile config should be reviewed.

## Check Truncated Regions in YOLO and Similar Models

YOLO-style models often include detection-head decode or postprocessing operators that are not part of the best NPU execution region. In this case, the expected layout is commonly:

```text
CPU head, if any -> NPU body -> CPU tail
```

For YOLO and similar detection models, check the following before creating the final MXQ:

- The backbone and main detection compute path are blue and included in the NPU body.
- Unsupported black layers are limited to preprocessing, decode, NMS, or other postprocessing regions that you expect to run on CPU.
- The NPU body output tensors are the tensors your runtime or application postprocessing code expects.

If the NPU body ends too early, starts too late, or excludes major convolution/attention blocks, re-check the original model export and compile config. For YOLO models, export options can change whether decode and postprocessing are embedded in the graph, which directly changes the CPU tail and the body subgraph interface.
