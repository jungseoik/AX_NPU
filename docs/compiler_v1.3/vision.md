<!-- 출처: https://docs.mobilint.com/compiler/v1.3/en/vision.html (원본) -->

# Vision

Vision model accuracy depends on the exact input format used at compile time. The model may have been trained with normalized floating-point tensors, but the deployed application may naturally produce raw RGB `uint8` images. qb Compiler can support both styles when the compile configuration, calibration data, and inference path use the same input format.

## Image Input Format

Start by identifying what the exported source model expects:

- color order: RGB or BGR
- dtype: usually `uint8`, `float32`, or framework-specific tensor dtype
- numeric range: `0..255`, `0..1`, or normalized values
- layout: HWC, NHWC, or NCHW
- spatial size: fixed, dynamic, or preset-specific such as `640 x 640`
- batch dimension: present or omitted in per-sample calibration files

Do not infer these values from the model family alone. A ResNet, YOLO, or ViT export can differ depending on the export script and whether preprocessing was included in the graph.

## HWC, NHWC, and NCHW

`HWC` means one image without an explicit batch dimension: height, width, channels. `NHWC` adds batch first. `NCHW` is the common PyTorch channel-first layout: batch, channels, height, width.

Older image calibration examples often save preprocessed samples as HWC NumPy arrays when compiling without CPU offloading. CPU-offloaded or framework-visible flows may instead require the original model input shape, such as NCHW. Use the shape expected by the compiler.

When accuracy is unexpectedly poor, verify layout first. A tensor with the right total element count but wrong channel order or axis order can compile successfully and still produce incorrect results.

## `uint8` Input

`uint8` input means the compiled model receives image pixels as integers in the `0..255` range. This is useful when the runtime application captures or decodes images and wants to avoid doing floating-point preprocessing before calling the NPU.

If the compiled model accepts `uint8`, calibration samples must also use `uint8` input in the same format. This holds in particular when the compiler configuration fuses the conversion and normalization steps. Do not calibrate with normalized `float32` samples and then run inference with raw `uint8` samples through the same MXQ input.

## Raw RGB Input

Raw RGB input means the application supplies decoded RGB pixels before model-specific normalization. A typical raw RGB image has shape `(H, W, 3)` and dtype `uint8`.

Raw RGB is not the same as the tensor used to train many models. For example, TorchVision ImageNet models are commonly trained with RGB images that are converted to floating point, divided by `255`, and normalized with ImageNet mean and standard deviation. If the runtime input is raw RGB, the compile configuration must account for that preprocessing so the NPU sees the same effective values as the trained model.

## Normalization Fuse

Normalization fuse moves simple preprocessing operations such as scale, mean subtraction, and standard deviation division into the compiled graph or compile-time input handling. This lets the application pass raw or lightly processed image data while the compiled model still receives normalized values internally.

Use normalization fuse only when the fused operation exactly matches the training and inference preprocessing:

- same color order
- same scaling factor, such as division by `255`
- same per-channel mean
- same per-channel standard deviation
- same channel order for those mean and standard deviation values

If normalization is fused, calibration must use the input format sent to the compiled model. For example, with raw RGB `uint8` input plus fused ImageNet normalization, calibration samples should represent raw RGB `uint8` images after resize/crop but before normalization.

## Preprocessing Pipeline

A vision preprocessing pipeline should be written as an ordered contract. For classification, the contract may be:

```text
decode image -> RGB -> resize -> center crop -> uint8 or float conversion -> normalize -> layout conversion
```

For YOLO-family detection, the contract may be:

```text
decode image -> RGB -> aspect-ratio resize -> letterbox padding -> scale -> layout conversion
```

Every step changes the numeric distribution seen during calibration. Resize interpolation, padding value, crop policy, and color conversion all matter.

## Calibration and Inference Must Match

The calibration data generator and inference preprocessing should share code or share a single written specification. If they diverge, the compiler calibrates one distribution and the deployed MXQ receives another.

Common mismatches include:

- calibration uses RGB but inference uses BGR
- calibration divides by `255` but inference sends `0..255`
- calibration normalizes but inference relies on a fuse that was not enabled
- calibration uses HWC but inference sends NCHW
- calibration uses center crop but inference uses resize without crop
- `yolo_640` calibration is reused for a `yolo_1280` export

When debugging, save one runtime input tensor and compare it numerically with one calibration tensor produced from the same image.

## Classification Preset

Use `classification` when the model is a conventional classification network but does not exactly follow the TorchVision recipe. Use `classification_torchvision` when the model was exported from TorchVision weights and uses the standard ImageNet preprocessing pattern: RGB image decode, resize, crop, scaling by 255, and channel-wise normalization. Keep the compile config, calibration data, and runtime preprocessing aligned with the exported model.

Image classification models are usually the simplest vision models to compile. ResNet, EfficientNet, MobileNet, and similar CNN models often have a single image input and a single logits output.

### CLI Examples

Generic classification preset:

```bash
python -m qbcompiler compile \
  --model resnet50.onnx \
  --backend onnx \
  --target-device regulus-rb \
  --config-preset classification \
  --calib-data-path ./calib_resnet50 \
  --output resnet50.mxq
```

For TorchVision ImageNet weights, use `classification_torchvision` instead of the generic `classification` preset:

```bash
python -m qbcompiler compile \
  --model resnet50.onnx \
  --backend onnx \
  --target-device regulus-rb \
  --config-preset classification_torchvision \
  --calib-data-path ./calib_resnet50 \
  --output resnet50.mxq
```

(classification-preset-preprocessing-pipeline)=

### Preprocessing Pipeline

`classification` configures the classification calibration defaults only. `classification_torchvision` extends it and enables a three-channel uint8 image input plus the following preprocessing pipeline:

| Step | Preset operation | Values |
| --- | --- | --- |
| 1 | Resize | shortest side to `256`, bilinear interpolation, aspect ratio preserved |
| 2 | Center crop | `224 x 224` |
| 3 | Scale | Convert uint8 pixel values from `[0, 255]` to floating-point values in `[0.0, 1.0]` |
| 4 | Normalize | Mean `[0.485, 0.456, 0.406]`; standard deviation `[0.229, 0.224, 0.225]` |

The preset also enables `autoConvertFormat`, sets `scaleToUint8=true` for normalization, and sets `fuseIntoFirstLayer=true`. Prepare calibration images and runtime inputs for this same RGB, resize, crop, and normalization contract. Use the generic `classification` preset, or a custom config, when the model uses a different image size, interpolation, color order, scaling rule, or normalization values.

For this recipe, apply resize and center crop before calibration and runtime inference. The fused operation is normalization, so calibration samples should be cropped RGB uint8 images before normalization.

### Checklist

- Export the model with fixed input shape, commonly `1x3x224x224`.
- Generate calibration tensors from images representative of the deployment data.
- Match calibration preprocessing with inference preprocessing.
- Verify whether the model expects `NCHW`, `NHWC`, or `HWC` at the compiled input.
- After MXQ generation, compare top-k predictions with the original model.

## Detection and YOLO Presets

Use `detection` for general detection models that do not match a YOLO image size preset. Detection exports often differ in whether decode, NMS, and other postprocessing are inside the graph, so inspect the MBLT before assuming the runtime input and output interface. Use `yolo_640` or `yolo_1280` when the exported model input size and preprocessing pipeline match that square input size. The preset name must match the exported model shape. Do not compile a 1280-exported model with `yolo_640` or a 640-exported model with `yolo_1280`.

YOLO models usually compile as an NPU body plus application-side or CPU-offloaded postprocessing. The convolutional feature extractor is the main NPU workload, while decode, NMS, slicing, concatenation, or output formatting can contain unsupported or low-value operators.

### CLI Example

```bash
python -m qbcompiler compile \
  --model yolov8n.onnx \
  --backend onnx \
  --target-device regulus-rb \
  --config-preset yolo_640 \
  --calib-data-path ./calib_yolo_640 \
  --output yolov8n_body.mxq
```

For YOLO models with a `1280 x 1280` input size, use `yolo_1280`.

### Preset Details

Both YOLO presets extend `detection`, which sets `calibration.mode=1` and `calibration.output=1`. They enable three-channel uint8 input and `autoConvertFormat`, then apply the following letterbox operation:

| Preset | Target size | Padding value |
| --- | --- | --- |
| `yolo_640` | `640 x 640` | `114` |
| `yolo_1280` | `1280 x 1280` | `114` |

The preset, exported ONNX or framework model shape, letterbox preprocessing, calibration tensors, and runtime input preparation must all agree on the same image size. Calibration should reproduce the training/inference letterbox path: color conversion, aspect-ratio-preserving resize, padding value, scaling, and layout conversion. If you compile only the NPU body, calibration data should match the body input. If you enable CPU Offloading and keep preprocessing or postprocessing in the compiled graph, calibration data should match the original model input.

### Netron Inspection

Use Netron before final compilation:

- Open the parsed MBLT.
- Confirm the main YOLO body is blue, meaning NPU-supported.
- Identify CPU head, NPU body, and CPU tail regions.

## Vision Transformer

Vision Transformer models are sensitive to shape and layout because patch embedding and attention blocks introduce reshape, transpose, gather, and positional embedding patterns. Patch embedding and normalization behavior can make calibration more sensitive to preprocessing mismatch than in conventional CNNs. Use the `vision_transformer` preset as the starting point.

(vision-transformer-cli-example)=

### CLI Example

```bash
python -m qbcompiler compile \
  --model vit.onnx \
  --backend onnx \
  --target-device regulus-rb \
  --config-preset vision_transformer \
  --calib-data-path ./calib_vit \
  --output vit.mxq
```

(vision-transformer-preset-details)=

### Preset Details

The `vision_transformer` preset sets `calibration.method=1` and `calibration.mode=0`. It also sets transformer activation precision to 16 bits for both `output` and `ffn`. This preset does not define image preprocessing; provide the resize, crop, scaling, color conversion, and normalization required by the exported model in the calibration and runtime paths.

For quantization configuration options including layer-level overrides, see [Model Quantization — Quantization Configuration](model_quantization.md#quantization-configuration). For how to use presets, config files, and `dump-config`, see [Compile Configuration](compile_configuration.md). For transformer-architecture specific quantization (attention projections, FFN), see also [Transformer](transformer.md).

(vision-transformer-checklist)=

### Checklist

- Export with fixed image size and fixed sequence length.
- Keep batch size fixed unless the model and target flow explicitly support another value.
- Confirm patch embedding is represented by supported operators.
- Check position embedding interpolation. Dynamic interpolation may need to be moved outside the compiled graph.
- Use representative calibration images with the same resize, crop, scaling, and normalization used at inference.

### Parse Failure Guidance

If parse fails around patchify, reshape, or attention blocks, inspect the MBLT in Netron and decide whether to re-export the model, simplify dynamic shape logic, or compile an extracted body.

## Multimodal

Use `multimodal` when image input is only one part of the model input contract; in that case, image calibration must stay aligned with the text or other modality inputs in the same sample.
