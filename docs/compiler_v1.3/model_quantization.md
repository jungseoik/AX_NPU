<!-- 출처: https://docs.mobilint.com/compiler/v1.3/en/model_quantization.html (원본) -->

# Model Quantization

Calibration data is representative input data used during the `MBLT -> MXQ` stage. qb Compiler uses it to observe activation ranges and choose quantization parameters for the compiled model.

Good calibration data is not just the right shape. It must represent the same distribution, dtype, layout, scaling, and preprocessing as the input that the compiler expects at inference time.

## Quantization Overview

### Quantization Flow

Quantization converts the numeric ranges in an MBLT model into the representation used by an MXQ artifact. In qb Compiler this is part of the `MBLT -> MXQ` stage. It reduces memory traffic and enables efficient NPU execution, but it also changes numeric behavior.

The normal workflow is:

```text
source model -> MBLT -> calibration (compute statistics) -> quantization -> MXQ
```

During calibration, the compiler observes activation ranges from calibration samples, computes statistics from them, and uses those statistics to choose scale values for activations and weights. Quantization then applies those scales and emits an MXQ package tuned for the selected target device. Some configuration changes affect only numeric quality, while others affect the generated graph or NPU execution characteristics.

Use this order when tuning:

1. Confirm the calibration data was generated using the actual inference preprocessing.
2. Validate the default preset.
3. Adjust global quantization settings such as calibration and bit behavior.
4. Add layer-level overrides only for layers proven to be sensitive.
5. Use search or optimization configs when simple changes are not enough.

The sections below cover detailed quantization techniques and configuration controls.

### Why Calibration is Required

Most source models are trained in floating point. MXQ artifacts are optimized for Mobilint NPUs and usually use quantized tensors internally. Calibration estimates numeric ranges for activations so the compiler can map floating-point behavior into quantized execution with minimal accuracy loss.

If calibration samples are too different from real inference inputs, the observed ranges will be wrong. The compiled model can then saturate values, lose small signals, or assign too much precision to ranges that never appear in production.

## Calibration Data

### Calibration Data Format

Calibration data can be stored as NumPy arrays, PyTorch tensors, or image files in a directory, or as a text/JSON metadata file that lists sample paths. Each sample must match the model input that qb Compiler calibrates:

- the same number of inputs
- the same input names or order expected by the compile path
- the shape expected by the compiler
- the same dtype and numeric range
- the same layout, such as HWC, NHWC, or NCHW

For a single image input, a calibration sample is often either a source image for compiler-side preprocessing or one preprocessed image tensor. For multi-input models, one calibration sample must contain the full set of inputs for the same inference example.

### Presets and Calibration Data

A preset can add preprocessing before the original model input. Prepare calibration samples for the input format defined by the compile configuration, not for the original floating-point model input.

| Preset | Compiler-side preprocessing | Calibration data to provide |
| --- | --- | --- |
| `classification` | None | Prepared tensor matching the exported model input. |
| `classification_torchvision` | Three-channel uint8 input; resize shortest side to `256` with bilinear interpolation, preserving aspect ratio; center crop `224 x 224`; scale `[0, 255]` to `[0.0, 1.0]`; normalize with mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`. | Three-channel uint8 images after resize and center crop to `224 x 224`, but before normalization. `fuseIntoFirstLayer` fuses normalization; resize and crop must be applied outside the MXQ model. |
| `detection` | None | Prepared tensor matching the exported model input. |
| `yolo_640` | Three-channel uint8 input and letterbox to `640 x 640` with padding value `114`. | Raw three-channel uint8 images before letterboxing. Do not letterbox them a second time. |
| `yolo_1280` | Three-channel uint8 input and letterbox to `1280 x 1280` with padding value `114`. | Raw three-channel uint8 images before letterboxing. Do not letterbox them a second time. |
| `vision_transformer` | None | Prepared tensor matching the exported model input and its model-specific preprocessing. |
| `llm` / `llm_fast` | No tokenizer or embedding preprocessing. | Tokens, embeddings, masks, positions, and cache tensors required by the compile flow. Match the configured sequence and cache lengths. |
| `multimodal` | No image or text preprocessing. | Every prepared vision and language input required by the exported model. |

Do not use a preset when the model's training/export preprocessing differs from the preset pipeline. Use a custom configuration and prepare calibration data in the input format that the configuration expects.

### Image Model Calibration

For a model without compiler-side preprocessing, image calibration starts from representative raw images, then applies the same resize, crop, color conversion, scaling, layout conversion, and normalization used by inference before storing the tensor.

For example, external TorchVision-style ImageNet preprocessing usually looks like:

1. decode the image as RGB
2. resize with the training recipe interpolation
3. center crop to the model input size
4. scale `uint8` pixels to floating point, often by dividing by `255`
5. normalize with the training mean and standard deviation
6. store the tensor in the layout expected by qb Compiler

For YOLO-family models without compiler-side preprocessing, the external pipeline usually includes RGB conversion, aspect-ratio-preserving resize, letterbox padding, scaling, and a fixed square input such as `640 x 640` or `1280 x 1280`. qb Compiler provides the `yolo_640` and `yolo_1280` [config presets](vision.md#detection-and-yolo-presets) for these input sizes. If the exported model input is `640 x 640`, use the `yolo_640` preset. If the exported model input is `1280 x 1280`, use the `yolo_1280` preset. With these presets, the compiler performs letterboxing, so provide the raw three-channel uint8 images described above instead of pre-letterboxed tensors. Mismatched preset and calibration input sizes produce incorrect quantization results.

### Calibration Data Shapes

The calibration tensor shape must correspond to the compiler-visible model input.

Common examples:

| Model family | Typical sample shape | Notes |
| --- | --- | --- |
| `classification_torchvision` | `(224, 224, 3)` | Three-channel uint8 image after resize/crop and before fused normalization. |
| Classification without compiler-side preprocessing | `(224, 224, 3)` | Preprocessed RGB image in HWC form when that is the compiler-visible input. |
| ONNX or framework input that remains channel-first | `(1, 3, 224, 224)` or `(3, 224, 224)` | Match the model input or compile API expectation. |
| `yolo_640` | `(H, W, 3)` | Raw three-channel uint8 image before preset letterboxing to `640 x 640`. |
| `yolo_1280` | `(H, W, 3)` | Raw three-channel uint8 image before preset letterboxing to `1280 x 1280`. |
| Multi-input model | one tensor per input | Preserve names, order, shape, dtype, and sample alignment. |
| LLM | token, embedding, mask, position, or cache-related tensors required by the compile flow | Match the HuggingFace/transformers preprocessing and configured sequence/cache lengths. If the compile flow requires position encoding tensors such as RoPE or ALiBi as separate inputs, include them in the calibration data. |

Some flows calibrate the NPU body after CPU head extraction, while CPU-offloaded flows may calibrate the original model-visible input. When CPU offloading is involved, confirm whether the calibration shape belongs to the original model or to the extracted NPU body.

### Multi-input Calibration

For a model with multiple inputs, do not store each input stream independently unless the compile tool explicitly asks for that layout. Calibration sample `0` for every input must come from the same real example.

For example, a multimodal model may take an image tensor, token IDs, and an attention mask. The image must correspond to the same prompt/text data in that sample. Randomly pairing image and text inputs can produce activation ranges that do not occur during real inference.

### LLM Calibration

LLM calibration should use text representative of the deployment workload and the same tokenizer, sequence length policy, attention mask behavior, position handling, and KV-cache configuration used for inference.

Some compile flows run the embedding layer on the CPU and calibrate only the NPU body. In that case the calibration input is the post-embedding tensor, not raw token IDs. Generate those embeddings with the same model weights and dtype assumptions used by the compile script. Keep batch size, sequence length, and cache length consistent with the selected [`llm` or `llm_fast`](transformer.md#llm-and-llm_fast-presets) preset.

### Random Calibration

Random calibration data can be useful for smoke tests, parser checks, or confirming that a compile path runs. It is not a substitute for representative calibration data when accuracy matters.

Use random calibration only when the goal is build validation. For release artifacts, use samples from the expected inference distribution.

### Matching Inference Preprocessing

Calibration preprocessing must match inference preprocessing because calibration decides the numeric ranges that the MXQ artifact will use. If inference supplies raw RGB `uint8` images but calibration used normalized `float32`, or if calibration used BGR while inference uses RGB, the calibrated ranges describe a different model input.

This is the most common source of quantization accuracy loss. Keep the preprocessing implementation shared between calibration generation and runtime inference whenever possible. If normalization is fused into the compiled model, provide calibration data after the required spatial preprocessing but before normalization; do not apply normalization a second time.

### Calibration Data Preparation Notes

Use enough samples to represent the input distribution expected in the deployment scenario, including brightness, object scale, backgrounds, and prompt lengths for LLMs. Avoid corrupt files, mixed color formats, and samples with unexpected channel counts.

Record the preset name, target device, source model export command, calibration script version, and preprocessing parameters with the generated calibration data. Those details are required to reproduce an MXQ artifact.

## Quantization Configuration

For how to supply configuration via presets, config files, or Python sub-config objects, see [Compile Configuration](compile_configuration.md). For the full `CompileConfig` schema, see [Reference — CompileConfig Schema](reference.md#compileconfig-schema).

### CalibrationConfig

`CalibrationConfig` controls how qb Compiler collects activation statistics from calibration samples.

Use it when the model output is sensitive to activation ranges, when calibration samples are limited, or when the calibration distribution does not match deployment data. This is usually the first config to inspect for vision models, detection models, multimodal encoders, and LLM prompt calibration.

It changes how ranges are estimated from sample tensors. For example, a configuration may choose a more conservative range, ignore rare outliers, or use a percentile-style estimate instead of the absolute maximum. Conservative ranges reduce clipping risk but can waste quantization levels. Aggressive outlier handling often improves average accuracy but can hurt rare inputs that legitimately produce large activations.

Inference speed impact is negligible, but accuracy impact can be high. The generated MXQ still runs with the same NPU operators, so runtime speed does not change much, but better ranges can reduce numerical error without changing runtime precision. Compile time may increase if calibration uses more samples or additional statistics.

Practical guidance:

- Use representative calibration samples from the same preprocessing pipeline used at inference time.
- Increase calibration sample diversity before changing advanced quantization options.
- Avoid random calibration for accuracy evaluation; use it only to smoke-test compilation.
- For multi-input models, calibrate all inputs together with realistic correlations.

qbcompiler 1.3 replaces the lookup-table clustering settings with two explicit
sections. `calibration.clusteringMethods`, `calibration.clusteringMethodsList` and the
`LutClusteringMethod` enumeration are gone; use these instead:

- `calibration.groupLut` — `irlsIter` (3) iterations of the IRLS scale solve,
  `droEps` (0, disabled) for how conservatively to clip the worst points, and
  `coverFloor` (0.85) as the cluster scale lower bound.
- `calibration.optimizeLut.optimizationLevel` — `0` for a fast surrogate-only
  search, `1` (the default) to refine against the true objective.

`calibration.maxSampleSizeForQuantScheme` (16) caps how many calibration samples
each quant-scheme stage uses, and `calibration.layerOverrides.method` sets the
calibration method for named layers.

### BitConfig

`BitConfig` controls the quantization bit-width policy used for tensors and weights.

Use it when you need to trade accuracy for smaller/faster execution, or when a model family is known to require higher precision in part of the graph. Lower bit-widths reduce memory bandwidth and can improve throughput, but they reduce numeric resolution. Higher bit-widths usually improve accuracy at the cost of memory and may reduce performance depending on the target device and operator.

It changes the precision assigned to activations, weights, or selected layer outputs. A global bit policy is simple and predictable, but a mixed policy can preserve accuracy by keeping sensitive layers at higher precision while allowing the rest of the graph to use a faster/lower precision path.

Accuracy impact is often high for the first layer, last layer, normalization-adjacent layers, attention projections, and detection heads. Performance impact depends on the operators affected; changing many large convolution or matrix multiplication layers has a larger runtime effect than changing a small output head.

qbcompiler 1.3 splits the transformer settings more finely.
`bit.transformer.activation` and `bit.transformer.weight` gained `router`, which
sets the bit-width of the MoE router gate, and their single `ffn` value became a `{up, gate, down}`
sub-object so the SwiGLU gate and the two projections can differ:

```python
BitConfig.Transformer.Weight(
    query=8, key=8, value=8, output=8, head=8, router=8,
    ffn=BitConfig.Transformer.Weight.Ffn(up=8, gate=8, down=4),
)
```

Passing a plain integer for `ffn` still sets all three, so existing configuration
files and code that wrote `ffn=8` keep working. Code that *read* `.ffn` as an
integer does not.

`bit.layerOverrides` also gained `weight8Bits`, the counterpart to the existing
`weight16Bits`: it forces named layers down to 8-bit weights when the global policy
is wider.

### HessianQuantConfig

`HessianQuantConfig` enables optimization-oriented quantization. It is useful when the default scale selection is valid but validation shows a remaining accuracy gap.

Use it after calibration data has been checked and after simple global settings have been tried. It is most useful for large matrix-heavy models, transformer blocks, and layers where weight quantization error dominates activation range error.

It changes how quantization parameters are optimized, usually by using sample activations or layer reconstruction criteria to reduce the difference between floating-point and quantized behavior. This can improve accuracy without increasing runtime precision.

Accuracy impact can be significant on sensitive layers. Runtime performance usually stays close to the selected bit policy because the optimization is done at compile time. Compile time and temporary memory usage may increase, so it is best applied selectively when possible.

From qbcompiler 1.3, two settings control where the accumulated Hessian lives and
how large it is. `accumulationDevice` does not change the result, only its cost; `hessianDtype` changes both, because the stored accumulator is rounded after every batch:

- `hessianDtype` (`fp32`, the default, or `bf16`). `bf16` halves the Hessian's
  memory footprint — host RAM when it lives on the CPU, VRAM when it lives on the
  GPU. Compute is unaffected: the per-batch matmul and accumulation stay float32
  and the solve upcasts the accumulator, so only the stored one is bfloat16.
  This matters most for MoE models, which carry a Hessian per expert FFN.
- `accumulationDevice` (`auto`, the default, or `cpu` / `gpu`). `gpu` accumulates
  on the calibration device instead of copying the whole d x d accumulator off it
  once per layer per batch, which the vendor benchmark measures at 3-5x faster,
  and parks the Hessian on the
  host as soon as the last batch is in, so peak VRAM for the rest of the compile
  does not rise. `cpu` bounds VRAM during calibration itself, for models whose
  summed Hessian does not fit alongside the activations. `auto` follows
  `resourceManagement.useGPUOnlyForCalibration`, the setting that already states
  whether this compile is VRAM-bound — inversely, since that setting means *keep
  the GPU for calibration only*: `true`, its default, resolves `auto` to `cpu`,
  and `false` resolves it to `gpu`. A CPU compile always accumulates on the host.

Also from 1.3, `attributes.blockSize` defaults to 256 rather than 128, the
sequential solve is replaced by a parallel fixed-point iteration, and a layer whose
Hessian is entirely zero is skipped with a message instead of aborting the compile.

### LayerBiasCorrectionConfig

`LayerBiasCorrectionConfig` corrects the systematic per-channel error that
quantization introduces into activations. Available from qbcompiler 1.3.

Use it when validation shows a consistent shift rather than added noise — mean
output drift, a detection head whose scores are uniformly low, a classifier whose
logits are offset. It is cheap enough to try before reaching for `ModConfig`, and
unlike MOD it needs no labels and no training loop.

It measures the difference between float and quantized activations across the
calibration set, then folds a damped correction into the integer convolution
biases. Corrections are recomputed between iterations so that changes upstream are
accounted for. Nothing about the runtime graph or its precision changes; only bias
values differ.

```python
qbcompiler.mxq_compile(
    model="model.onnx",
    target_device="aries-rb",
    calib_data_path="calib",
    save_path="model.mxq",
    layer_bias_correction=True,
)
```

`layer_bias_correction_config=` tunes it. The tunables live on `Attributes`, not
on the config itself, and the config rejects unknown top-level fields:

```python
from qbcompiler.configs import LayerBiasCorrectionConfig

cfg = LayerBiasCorrectionConfig(
    apply=True,
    attributes=LayerBiasCorrectionConfig.Attributes(
        num_samples=256,
        iterations=5,
        correction_rate=0.05,
    ),
)
```

`num_samples` (256) caps the calibration samples used per iteration, `iterations`
(5) sets how many damped rounds to run, and `correction_rate` (0.05) is the
fraction of the measured error applied per round. Raise `iterations` before
raising `correction_rate`: large single steps can overshoot.

Accuracy impact is moderate but reliable on models with a systematic bias.
Inference speed is unchanged. Compile time grows by roughly the cost of
`iterations` extra calibration passes. Disk use grows too: the pass needs the
float weights, so a `weightMemory.method` of `DeleteFloat` is promoted to
`SaveFloat` and the weights are written out rather than dropped. `method` is an
index into `weightMemory.methodList`, not one of those names: `0` is
`DeleteFloat`, `1` `SaveFloat`, `2` `MoveFloat`, `3` `KeepFloat`, `4` `KeepAll`.

### ModConfig

`ModConfig` enables MOD (Minimize Output Difference) optimization, a training/gradient-based method that directly minimizes the difference between the quantized outputs and the floating-point outputs. Use it as a last step when a validation accuracy gap remains even after both default scale selection and reconstruction-based correction such as `HessianQuantConfig`.

It requires representative calibration/training data and is best used after first trying simple global settings and HessianQuant. It is especially useful for models that are particularly sensitive to quantization, for detection and transformer families whose output quality depends heavily on the error of a few layers, and for layers that activation-range correction alone cannot recover.

This setting makes the quantization parameters (activation scale, zero-point, weight scale)—and, if needed, the weights and biases—learnable, then trains them with an optimizer configured with an epoch count, a learning-rate schedule, and a loss function (based on output difference, combined with reconstruction/task loss when needed). Because it uses the floating-point model as a teacher and steers the quantized outputs toward it, accuracy can be recovered without raising runtime precision.

Accuracy improvement can be largest in the hardest cases. Because training is done at compile time, runtime performance stays close to the selected bit policy. However, because it runs an actual training loop, compile time, GPU memory, and data-preparation effort can increase substantially compared to other methods, so it is best to narrow the scope to specific layers and raise training-related settings only as much as needed.

### EquivalentTransformationConfig

`EquivalentTransformationConfig` controls transformations that preserve the mathematical function of the model while making quantization easier.

Use it when a model has scale imbalance across adjacent layers, such as convolution/batch-normalization patterns, linear layers followed by normalization-sensitive operations, or transformer projections with uneven channel magnitudes. These transformations are especially useful when the floating-point model is accurate but quantization error is concentrated in a few layers.

It changes how equivalent computations are distributed across the graph. For example, scale factors may be moved between adjacent operations so that weight and activation ranges become easier to quantize. The floating-point function is intended to remain equivalent, but the quantized approximation can improve because the ranges are better conditioned.

Accuracy impact is often positive for models with outlier channels or imbalanced weights. Inference speed is barely affected, though graph changes can affect fusion and scheduling. Compile time may increase modestly.

### SearchWeightScaleConfig

`SearchWeightScaleConfig` searches for better weight scale values than the default heuristic.

Use it when validation points to weight quantization as the main source of error, especially in convolution, fully connected, or attention projection layers. It is a good next step after confirming calibration quality and before raising precision broadly across the model.

It changes the weight scale selection. Instead of accepting a default range estimate, the compiler evaluates candidate scales and chooses values that reduce quantization error according to the configured objective.

Accuracy impact can be high on weight-sensitive layers. Inference speed is barely affected because the resulting MXQ still uses the same quantized execution path. Compile time can increase noticeably, so restrict search scope with layer-level overrides when only a small part of the model is problematic.

### SaveSampleConfig

`SaveSampleConfig` saves calibration samples or intermediate sample information for later analysis or reuse.

Use it when debugging accuracy degradation, creating a reproducible quantization investigation, or sharing a minimal calibration case with another engineer. It is also useful before running heavier optimization/search flows so that each experiment uses the same observed sample set.

It changes the artifacts emitted during calibration rather than the model execution itself. The saved data can help compare floating-point and quantized tensors, inspect outlier samples, or reproduce scale generation.

Accuracy and runtime performance are not directly changed. Disk usage and build time can increase, especially for large inputs or many saved intermediate tensors.

### RuntimeOptions

`RuntimeOptions` records runtime metadata used by the generated package. It is not normally the first place to tune quantization behavior.

Use the calibration, bit, optimization, and layer override settings for numeric quality. Change runtime metadata only when a documented workflow or runtime integration requires it, and keep it aligned with the qb Compiler/runtime version used for deployment.

### Layer-Level Overrides

Layer-level overrides let you apply quantization settings to specific layers instead of changing the entire model.

Use overrides when a validation diff, tensor comparison, or model knowledge identifies a small number of sensitive layers. They are preferable to global changes when only the input stem, output head, attention projection, normalization-adjacent layer, or detection head needs special handling.

Common override uses:

- Keep selected layers at a higher bit-width, or force selected layers down to 8-bit weights with `bit.layerOverrides.weight8Bits`.
- Use per-channel weight quantization for layers with uneven channel ranges.
- Disable a graph modification or search pass for a layer that becomes unstable.
- Apply weight-scale search only to layers with high reconstruction error.
- Preserve output-layer behavior when post-processing is sensitive to small numeric differences.

Overrides improve accuracy with less performance cost than broad global changes. The tradeoff is maintenance: layer names can change when the source model is exported differently, so keep overrides close to the model version and verify them after re-export.

## Accuracy Tuning

### Accuracy Degradation Handling

Treat quantization accuracy loss as a data and localization problem before changing many knobs.

Recommended sequence:

1. Compare preprocessing: calibration input normalization, layout, dtype, resize, padding, tokenization, and sequence lengths must match real inference.
2. Validate with a representative metric, not only a few sample outputs.
3. Check whether the loss is global or concentrated in specific classes, boxes, tokens, or layers.
4. Increase or rebalance calibration samples if the distribution is weak.
5. Tune `CalibrationConfig` for activation clipping or outliers.
6. Tune `BitConfig` or add layer-level overrides for sensitive layers.
7. Enable `EquivalentTransformationConfig`, `SearchWeightScaleConfig`, or `HessianQuantConfig` when the remaining error is localized and reproducible.
8. Use `SaveSampleConfig` to fix the sample set for repeatable experiments.

Typical symptoms and first actions:

| Symptom | Likely cause | First action |
| --- | --- | --- |
| Accuracy is poor on almost every sample | preprocessing or calibration mismatch | Rebuild calibration data from the inference pipeline |
| Rare inputs fail badly | activation outliers clipped | Try a more conservative calibration range or add those samples |
| One head or class regresses | sensitive output layer | Add a layer override for the head |
| Transformer quality drops after several blocks | accumulated projection/attention error | Try weight-scale search or HessianQuant on projection layers |
| Rebuilds produce different quality | unstable calibration/search inputs | Save and reload validated scales or samples |

### Recommended Configuration Examples

Start from a preset whenever possible, then change only the quantization sub-config that matches the observed issue.

#### Default Production Build

Use this when validation passes with the preset.

```python
from qbcompiler import CompileConfig

config = CompileConfig.from_preset("classification")
# Provide calibration data through the normal compile or quantize command.
```

Impact: best maintainability and shortest tuning time. Accuracy and performance follow the tested preset defaults.

#### More Robust Activation Calibration

Use this when outputs are generally close but a subset of samples shows clipping-like errors.

```python
from qbcompiler import CompileConfig, CalibrationConfig

config = CompileConfig.from_preset("detection")
config.calibration = CalibrationConfig(
    # Choose the range/statistics policy supported by your qb Compiler version.
    # Prefer representative samples before increasing quantization complexity.
)
```

Impact: can recover accuracy without changing runtime precision. Compile time may increase if more samples or statistics are used.

#### Protect a Sensitive Output Head

Use this when most of the model is accurate but the final logits, boxes, masks, or token scores are sensitive.

```python
from qbcompiler import BitConfig, CompileConfig

config = CompileConfig.from_preset("yolo_640")
# Per-layer overrides belong to the sub-config they act on, not to
# CompileConfig: config.bit.layer_overrides raises precision for named
# layers, config.calibration.layer_overrides changes their calibration
# method. See the BitConfig and CalibrationConfig sections above for what
# each one accepts.
```

Impact: often recovers task metrics with less performance cost than raising precision globally. Verify layer names after model export.

#### Weight-Sensitive Transformer or Large Linear Model

Use this when calibration is correct but quality loss is concentrated in matrix-heavy blocks.

```python
from qbcompiler import CompileConfig, HessianQuantConfig, SearchWeightScaleConfig

config = CompileConfig.from_preset("llm")
config.search_weight_scale = SearchWeightScaleConfig(apply=True)
config.hessian_quant = HessianQuantConfig(apply=True)
```

Impact: can improve accuracy without changing runtime bit-width. Compile time and memory usage can increase, so narrow the scope if only a few layers are problematic.

#### Reproducible Quantization Experiment

Use this when comparing compiler versions, presets, or model exports.

```python
from qbcompiler import CompileConfig, SaveSampleConfig

config = CompileConfig.from_preset("vision_transformer")
config.save_sample = SaveSampleConfig(apply=True)
```

Impact: fixes the sample set for repeatable experiments and makes validation easier to reproduce. Disk usage and build time can increase for large inputs or many saved intermediate tensors.
