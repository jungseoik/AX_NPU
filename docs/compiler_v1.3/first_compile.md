<!-- 출처: https://docs.mobilint.com/compiler/v1.3/en/first_compile.html (원본) -->

# First Compile

This chapter shows the shortest path from an original model to an NPU binary (MXQ).

```text
Original model -> MBLT -> MXQ
```

The examples use a ResNet-50 ONNX model, but the same flow applies to other model families.

## Compile with CLI

### Quickest first run

Specify the model, the target device, and how to calibrate. A compile with neither calibration data nor `--use-random-calib` stops with *No calibration dataset is provided*, so the quickest run asks for random calibration:

```bash
python -m qbcompiler compile \
  --model /workspace/resnet50.onnx \
  --backend onnx \
  --target-device regulus-rb \
  --use-random-calib \
  --output /workspace/resnet50.mxq
```

On success, the MXQ file is created.

```bash
ls -lh /workspace/resnet50.mxq
```

### Add calibration data for better accuracy

The result above was quantized against random data rather than real inputs, so accuracy may be significantly degraded. For production builds, provide representative calibration data with `--calib-data-path`. If you haven't prepared calibration data yet, see [Prepare Calibration Data](#prepare-calibration-data).

```bash
python -m qbcompiler compile \
  --model /workspace/resnet50.onnx \
  --backend onnx \
  --calib-data-path /workspace/calibration/resnet50 \
  --target-device regulus-rb \
  --output /workspace/resnet50.mxq
```

The CLI mirrors the main pipeline stages. You can also run individual stages directly:

```bash
# Original model -> MBLT (parse only)
python -m qbcompiler parse --model model.onnx --backend onnx \
  --target-device regulus-rb --output model.mblt

# MBLT -> MXQ (quantize only)
python -m qbcompiler quantize --mblt model.mblt --calib-data-path calib \
  --target-device regulus-rb --output model.mxq
```

Use `python -m qbcompiler <command> --help` to see the exact options available in the installed version.

## Compile with Python

The same operation is available from Python.

```python
from qbcompiler import mxq_compile

mxq_compile(
    model="/workspace/resnet50.onnx",
    backend="onnx",
    calib_data_path="/workspace/calibration/resnet50",
    target_device="regulus-rb",
    save_path="/workspace/resnet50.mxq",
)
```

For PyTorch models, provide a `feed_dict` whose keys match the model's `forward` parameter names:

```python
import torch
import torchvision
from qbcompiler import mxq_compile

model = torchvision.models.resnet50(pretrained=True).eval().cpu()
feed_dict = {"x": torch.randn(1, 3, 224, 224)}

mxq_compile(
    model=model,
    backend="torch",
    feed_dict=feed_dict,
    calib_data_path="/workspace/calibration/resnet50",
    target_device="regulus-rb",
    save_path="/workspace/resnet50.mxq",
)
```

The tensor shapes in `feed_dict` must be valid for the original model. For HuggingFace/transformers LLMs, use the `torch` backend and the LLM-specific settings described in the [Transformer](transformer.md) chapter.

Key arguments at a glance:

| Purpose | Python argument |
| --- | --- |
| Original model | `model` |
| Input framework | `backend` |
| Calibration samples | `calib_data_path` |
| Target NPU | `target_device` |
| Output MXQ path | `save_path` |
| PyTorch example inputs | `feed_dict` |

The same compile configuration concepts are available from CLI and Python. Use a config file when the build has many options or must be reproduced exactly.

qb Compiler provides built-in presets for common model families such as `classification`, `yolo_640`, `llm`, and `vision_transformer`. List available presets with `QBCOMPILER_JSONL=1 python -m qbcompiler presets` — `presets` reports on the JSONL channel and prints nothing without it. See [Vision](vision.md) and [Transformer](transformer.md) for model-family specific compilation guides.

## Prepare Calibration Data

Calibration data should match the images or tensors the compiled model will see during real inference. For image models, you can either pass a directory of raw image files directly to `calib_data_path` and define the model preprocessing with `PreprocessingConfig`, or use prepared NumPy tensors.

### Use Raw Images with a Preprocessing Pipeline

For the standard TorchVision ResNet-50 recipe, keep representative JPEG or PNG images in a directory such as `/workspace/calibration/imagenet-1k-selected`. Do not convert them to `.npy` files. Pass that directory to `calib_data_path` and configure the preprocessing pipeline in the Python compile call:

```python
from qbcompiler import (
    CalibrationConfig,
    PreprocessingConfig,
    Uint8InputConfig,
    mxq_compile,
)

preprocess_pipeline = [
    {"op": "resize", "size": 256, "mode": "bilinear"},
    {"op": "centerCrop", "height": 224, "width": 224},
    {
        "op": "normalize",
        "scaleToUint8": True,  # [0, 255] -> [0.0, 1.0]
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "fuseIntoFirstLayer": True,
    },
]

preprocessing_config = PreprocessingConfig(
    apply=True,
    auto_convert_format=True,
    pipeline=preprocess_pipeline,
    input_configs={},
)

calibration_config = CalibrationConfig(
    method=1,
    output=0,
    mode=1,
    max_percentile={"percentile": 0.9999, "topk_ratio": 0.01},
)

mxq_compile(
    model="/workspace/resnet50.onnx",
    backend="onnx",
    calib_data_path="/workspace/calibration/imagenet-1k-selected",
    save_path="/workspace/resnet50.mxq",
    target_device="regulus-rb",
    device="cpu",
    inference_scheme="single",
    image_channels=3,
    preprocessing_config=preprocessing_config,
    uint8_input_config=Uint8InputConfig(apply=True, inputs=[]),
    calibration_config=calibration_config,
)
```

`image_channels=3` converts grayscale calibration images to RGB when necessary. `auto_convert_format=True` handles the input format conversion. The pipeline uses bilinear resize, center crop, and ImageNet-style scaling and normalization. `fuseIntoFirstLayer=True` and `Uint8InputConfig` let the MXQ model accept uint8 image input while fusing normalization into the first layer.

`inference_scheme="single"` compiles the MXQ for Single core mode. Other values such as `multi`, `global4`, and `global8` select different ARIES runtime core modes; they are rejected outright on a single-core REGULUS target like the one used here. See [Reference — inferenceScheme](reference.md#inferencescheme) before changing this value.

Use this method only when the pipeline matches the model's training and inference recipe. If the model uses a different image size, color order, resize rule, crop rule, scaling, or normalization, update the pipeline accordingly.

### Use Prepared NumPy Tensors

qb Compiler can also create preprocessed NumPy calibration samples with either a YAML preprocessing description or a Python preprocessing function.

YAML-based preprocessing example:

```python
from qbcompiler.calibration import make_calib

make_calib(
    args_pre="/workspace/resnet50.yaml",
    data_dir="/workspace/calibration/cali_1000",
    save_dir="/workspace/calibration",
    save_name="resnet50",
    max_size=100,
)
```

Example preprocessing YAML:

```yaml
Datatype: Image
GetImage:
    to_float32: false
    channel_order: RGB

Pre-Order: [ResizeTorch, CenterCrop, Normalize, SetOrder]
Pre-processing:
    ResizeTorch:
        size: 256
        interpolation: bilinear
    CenterCrop:
        size: [224, 224]
    Normalize:
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
        to_float_div255: true
    SetOrder:
        shape: HWC
```

Manual preprocessing example:

```python
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torchvision.transforms import InterpolationMode
from qbcompiler.calibration import make_calib_man

def preprocess_resnet50(img_path: str):
    img = Image.open(img_path)
    out = F.pil_to_tensor(img)
    out = F.resize(out, size=256, interpolation=InterpolationMode.BILINEAR)
    out = F.center_crop(out, output_size=(224, 224))
    out = out.to(torch.float, copy=False) / 255.0
    out = F.normalize(out, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return np.transpose(out.numpy(), axes=[1, 2, 0])

make_calib_man(
    pre_ftn=preprocess_resnet50,
    data_dir="/workspace/calibration/cali_1000",
    save_dir="/workspace/calibration",
    save_name="resnet50",
    max_size=100,
)
```

Both examples create:

- `/workspace/calibration/resnet50`: a directory of preprocessed `.npy` samples
- `/workspace/calibration/resnet50.txt`: a metadata file listing the samples

Either path can be used as calibration input.

## Verify the Compiled Model

An MXQ file that builds without errors does not guarantee correct inference. Run the compiled model on the NPU and compare the results against the original floating-point model.

### Run inference with qbruntime

Install the Mobilint runtime library (`pip install mobilint-qb-runtime`) and run the MXQ on the target device. See the {external+runtime:doc}`Runtime Installation Guide <en/installing_runtime_library>` for setup details.

```python
import qbruntime
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F, InterpolationMode

acc = qbruntime.Accelerator(0)
mc = qbruntime.ModelConfig()
mxq_model = qbruntime.Model("/workspace/resnet50.mxq", mc)
mxq_model.launch(acc)

img = Image.open("test.jpg").convert("RGB")
out = F.pil_to_tensor(img)
out = F.resize(out, size=256, interpolation=InterpolationMode.BILINEAR)
out = F.center_crop(out, output_size=(224, 224))
out = out.to(torch.float) / 255.0
out = F.normalize(out, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
image = np.transpose(out.numpy(), axes=[1, 2, 0])  # HWC float32

output = mxq_model.infer(image)
```

The preprocessing must match how the model was calibrated. The example above matches the CLI compile at the top of this chapter, whose MXQ takes float32 input. If you built the MXQ with `Uint8InputConfig` and `fuseIntoFirstLayer=True` as in [Use Raw Images with a Preprocessing Pipeline](#use-raw-images-with-a-preprocessing-pipeline), pass the cropped RGB `uint8` array instead — do not divide by `255` and do not normalize, because the model now does both. Note that every compile on this page writes the same `/workspace/resnet50.mxq`. For more uint8 input workflows, see the [SDK Tutorial — Image Classification](https://github.com/mobilint/mblt-sdk-tutorial/tree/master/compilation/image_classification).

For complete runtime examples including model loading, postprocessing, and top-k output, see the [SDK Tutorial — Runtime](https://github.com/mobilint/mblt-sdk-tutorial/tree/master/runtime/python/image_classification).

### Check the output

Run a few representative inputs and confirm the output is reasonable:

- **Classification**: top-5 predictions should include the expected class
- **Detection**: bounding boxes should appear at sensible positions and sizes
- **LLM**: generated text should be coherent and follow the prompt

If the output looks wrong — misclassified images, empty detections, garbled text — revisit the calibration data quality, try a different preset, or adjust quantization options. See [Model Quantization](model_quantization.md) for tuning guidance.

## Inspect Compile Outputs

For repeatable builds, keep these files together:

- original model or model revision
- calibration data directory or `.txt` metadata file
- compile config or preset name
- target device string
- generated `.mxq`
