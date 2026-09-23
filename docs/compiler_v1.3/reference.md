<!-- 출처: https://docs.mobilint.com/compiler/v1.3/en/reference.html (원본) -->

# Reference

This chapter is a reference for the current qbcompiler command line, Python API, configuration schema, supported inputs, and related terms. The source of truth is the current qbcompiler implementation.

## CLI Reference

Run the CLI as:

```bash
python -m qbcompiler <command> [options]
```

The current command set is `compile`, `parse`, `quantize`, `dump-config`, `info`, `check`, `presets`, `validate`, and `extract-body`. The table below covers the five this manual builds its examples on; `presets` is described under [Compile Configuration](compile_configuration.md).

| Command | Purpose | Main options |
| --- | --- | --- |
| `compile` | End-to-end model to `.mxq`: parse to temporary `.mblt`, then quantize. | `--model`, `--output`, `--backend`, `--device`, `--target-device`, `--cpu-offload` / `--no-cpu-offload`, `--config-preset`, `--compile-config`, `--calib-data-path`, `--use-random-calib` / `--no-use-random-calib`, `--config-save-path` |
| `parse` | Convert an input model to `.mblt` intermediate IR. | `--model`, `--output`, `--backend`, `--device`, `--target-device`, `--cpu-offload` / `--no-cpu-offload`, `--config-preset`, `--compile-config` |
| `quantize` | Convert a runnable `.mblt` file to `.mxq`. | `--mblt`, `--output`, `--target-device`, `--calib-data-path`, `--use-random-calib` / `--no-use-random-calib`, `--config-preset`, `--compile-config`, `--device`, `--config-save-path` |
| `dump-config` | Write a default or preset-derived `CompileConfig` template. | `--output`, `--preset` |
| `info` | Print the provenance recorded in a `.mblt`. | `--mblt` |

`--backend` defaults to `onnx`. `--calib-data-path` is repeatable. `--compile-config` accepts JSON, YAML, or YML and is the extension point for settings that do not have dedicated CLI flags. `dump-config` chooses JSON or YAML from the output file suffix.

For subprocess integrations, the CLI has an opt-in JSONL protocol. Set `QBCOMPILER_JSONL=1`, `true`, `yes`, or `on` to emit structured `status`, `progress`, `result`, `log`, and `error` messages on stdout. Without that environment variable, normal logs are written to stderr.

## Python API Reference

Public functions are available from `qbcompiler`.

| API | Purpose |
| --- | --- |
| `mxq_compile(...)` | Main Python entry point. Compiles a model or runnable `.mblt` to `.mxq`, delegating to one of the two functions below based on the input. |
| `mxq_compile_from_source(...)` | Parses a source model (ONNX / PyTorch / TensorFlow / TF Lite / TorchScript), then compiles it to `.mxq`. Takes the same arguments as `mxq_compile`. |
| `mxq_compile_from_mblt(...)` | Compiles an existing `.mblt` to `.mxq`. Its first parameter is `mblt`, not `model`. The graph is already parsed, so the parser-only arguments (`save_subgraph_type`, `output_subgraph_path`, `feed_dict`, `dynamic_axes`, `in_dformats`, `yolo_decode_include`, `exclude_first_subgraph`, `model_part`, `model_part_options`) are absent. |
| `mxq_compile_with_callback(...)` | `mxq_compile` with a progress callback. |
| `mblt_compile_with_callback(...)` | `mblt_compile` with a progress callback. Takes its output path before its target device. |
| `read_provenance(mblt_path)` | Returns the provenance record of a `.mblt`. `None` when the file carries no record — written before provenance existed, or with recording off. Both container formats carry one otherwise. |
| `mblt_compile(...)` | Exports a model to `.mblt` without producing `.mxq`. |
| `list_presets()` | Returns preset metadata dictionaries. |
| `dump_default_config(output_path, preset=None)` | Writes a `CompileConfig` template to JSON/YAML. |
| `get_body_subgraph(mblt_path, output_path)` | Writes a `.mblt` containing the largest NPU-runnable subgraph, together with any subgraphs it calls. A subgraph that is only ever called is not chosen as the body. |

The main `mxq_compile` parameters are:

```python
from qbcompiler import mxq_compile

mxq_compile(
    model="model.onnx",
    target_device="aries-rb",
    calib_data_path="calib",
    save_path="model.mxq",
    backend="onnx",
    device="gpu",
    config_preset="classification",
    compile_config=None,
)
```

Important parameters include `model`, `target_device`, `calib_data_path`, `save_path`, `backend`, `feed_dict`, `dynamic_axes`, `in_dformats`, `yolo_decode_include`, `exclude_first_subgraph`, `device`, `inference_scheme`, `use_random_calib`, `cpu_offload`, `layer_bias_correction`, `buffer_mode`, `input_shape_dict`, `force_npu_input_reposition`, `force_npu_output_reposition`, `image_channels`, `split_blocks`, `split_parts`, `config_preset`, `compile_config`, `config_save_path`, `model_part`, `model_part_options`, and sub-config objects such as `calibration_config`, `bit_config`, `llm_config`, `hessian_quant_config`, `layer_bias_correction_config`, `mod_config`, `equivalent_transformation_config`, `search_weight_scale_config`, `uint8_input_config`, `preprocessing_config`, and `save_sample_config`.

`config_save_path`, `model_part` and `model_part_options` are new in qbcompiler 1.3:

- `config_save_path` writes the fully resolved `CompileConfig` — the normalized configuration after every layer of the precedence order below has been applied — before compilation starts. A `.yaml` or `.yml` suffix writes YAML, any other suffix writes JSON, and parent directories are created. Because it is written up front, the record survives a compile that fails after it starts, and the file can be passed straight back as `compile_config=` to reproduce the compile. It is written by the quantize phase, so a `compile` that fails while parsing produces nothing. The CLI exposes it as `--config-save-path` on `quantize` and `compile`.
- `model_part` names which part of a multi-part torch model to parse (`"vision"`, `"language"`, `"encoder"`, and so on), and `model_part_options` passes extra arguments that part needs, such as `{"mel_frames": 100}`. A model declaring exactly one part resolves it from `None`; a model declaring several requires a name. List them with `available_parts(model)` from `qbcompiler.model_dict.parser.patcher.parts`. These replace the removed `hf_config` argument, which now raises `ValueError`.

Configuration resolution priority (highest wins): explicit arguments → sub-config objects → `compile_config` file *or* `config_preset`, whichever is given → defaults. Passing both uses the preset and ignores the file. For details and usage examples, see [Compile Configuration — Config Resolution Priority](compile_configuration.md#config-resolution-priority).

## CompileConfig Schema

`CompileConfig` is a Pydantic model with `extra="forbid"` and aliases enabled. Use alias names in JSON/YAML files. A full template can be generated with:

```bash
python -m qbcompiler dump-config --output compile_config.yaml
python -m qbcompiler dump-config --preset yolo_640 --output yolo_640.yaml
```

Top-level fields:

| Alias | Python field | Default | Meaning |
| --- | --- | --- | --- |
| `modelPaths` | `model_paths` | `[]` | Model path list. Usually supplied by API/CLI instead. |
| `calibDataPaths` | `calib_data_path` | `[]` | Calibration dataset path list. |
| `savePaths` | `save_paths` | `["./tmp.mxq"]` | MXQ output path list. |
| `useRandomCalib` | `use_random_calib` | `false` | Generate random calibration data. |
| `inferenceScheme` | `inference_scheme` | `single` | Compile-time NPU core-assignment scheme used by the generated MXQ. See [inferenceScheme](#inferencescheme). |
| `cpuOffload` | `cpu_offload` | `false` | Enable CPU offloading for unsupported groups. |
| `forceNpuInputReposition` | `force_npu_input_reposition` | `false` | Force input reposition operations onto NPU. |
| `forceNpuOutputReposition` | `force_npu_output_reposition` | `false` | Force output reposition operations onto NPU. |
| `bufferMode` | `buffer_mode` | `1` | Buffer serialization mode. |
| `inputShapeDict` | `input_shape_dict` | `{}` | Multi-shape compile specification for supported models. |
| `device` | `device` | `gpu` | Compilation compute device: `gpu` or `cpu`. |
| `dtype` | `dtype` | `float` | Computation dtype metadata. |
| `debug` | `debug` | `false` | Enable debug mode. |
| `trace` | `trace` | `false` | Enable trace mode. |
| `imageChannels` | `image_channels` | `0` | Number of image channels; `0` means auto-detect. |
| `configVersion` | `config_version` | `1.0.0` | Config schema version. |
| `splitBlocks` | `split_blocks` | `[]` | LLM multi-MXQ split points by block index. |
| `splitParts` | `split_parts` | `0` | Split LLM transformer blocks into N parts. |

Nested sections:

| Alias | Type | Purpose |
| --- | --- | --- |
| `uint8Input` | `Uint8InputConfig` | Treat model inputs as uint8, optionally by input name. |
| `preprocessing` | `PreprocessingConfig` | Input preprocessing pipeline such as resize, letterbox, crop, normalize, and format conversion. |
| `resourceManagement` | `ResourceManagementConfig` | Weight dtype, GPU calibration policy, and weight-memory method. |
| `calibration` | `CalibrationConfig` | Quantization method, output quantization, calibration mode, clipping/statistics, and layer overrides. |
| `bit` | `BitConfig` | Quantization precision settings. |
| `hessianQuant` | `HessianQuantConfig` | HessianQuant optimization settings, including `hessianDtype` and `accumulationDevice`. |
| `layerBiasCorrection` | `LayerBiasCorrectionConfig` | Calibration-derived per-channel bias correction folded into integer convolution biases. |
| `mod` | `ModConfig` | MOD training/optimization settings. |
| `llm` | `LlmConfig` | LLM sequence/cache/runtime/debug settings. If `runtime.batchSize` is greater than `1`, `inferenceScheme` must be `single`, `global4` or `global8` — `multi` and `all` are rejected — and the Batch LLM runtime API uses `BatchParam` metadata; see {external+runtime:doc}`qb Runtime Release Notes <en/release_note>`. |
| `moe` | `MoeConfig` | Sparse MoE calibration expert selection. |
| `equivalentTransformation` | `EquivalentTransformationConfig` | SmoothQuant-like, rotation, and FFN equivalent transformations. |
| `searchWeightScale` | `SearchWeightScaleConfig` | Weight-scale search. |
| `loadScale` | `LoadScaleConfig` | Load externally supplied scale entries. |
| `runtimeOptions` | `RuntimeOptions` | Internal runtime metadata. |
| `saveSample` | `SaveSampleConfig` | Save sample data for inspection/debugging. |

`CompileConfig.from_file()` accepts `.json`, `.yaml`, and `.yml`. For compatibility, it also flattens grouped keys such as `quantization.calibration`, `quantization.bit`, `advancedQuantization.hessianQuant`, `advancedQuantization.mod`, `advancedQuantization.EquivalentTransformation`, `advancedQuantization.searchWeightScale`, `advancedQuantization.loadScale`, and `advancedQuantization.layerBiasCorrection`, which 1.3 adds. That grouped spelling is also the one the tool writes — `dump-config` and `--config-save-path` both emit it. Flattening overwrites rather than merges, so a top-level `calibration` block added beside a `quantization.calibration` one is discarded with no message: do not mix the two spellings in one file.

### inferenceScheme

`inferenceScheme` selects how NPU core work is assigned when qb Runtime uses the MXQ for inference. Because the compiled MXQ is prepared for the selected core-assignment scheme, an MXQ built only for one mode cannot later be switched arbitrarily to another mode at runtime. One built with `all` carries every mode the model and target support, but the caller must name one: qb Runtime's default `CoreMode::Auto` accepts only an MXQ with exactly one mode, so an `all` build needs an explicit `setSingleCoreMode()` or `setGlobal4CoreMode()` — see {external+aries:doc}`ARIES ModelConfig Configuration <modelconfig>`.

Available choices are `single`, `multi`, `global4`, `global8`, and `all`. `single` uses independent Local Cores, `multi` is the cluster-level 4-batch mode, and `global4`/`global8` use 4 or 8 Local Cores together for one input. `all` prepares the MXQ for every core mode supported by the selected model and target. Support can vary by model, target, and compiler version. REGULUS targets have a single NPU core, so MXQ files compiled for REGULUS can use only `single` mode. From qbcompiler 1.3 this is enforced rather than left to fail later: `multi`, `global4` and `global8` are rejected outright on a single-core target, and `all` narrows to `single` there. The check runs before quantization regardless of whether the scheme or the target device was set first. `global` is a legacy/compatibility value; use `global4` or `global8` for new settings.

If `llm.attributes.runtime.batchSize` is greater than `1`, LLM/KV-cache transformer compilation accepts `single`, `global4` and `global8`, and rejects `multi` and `all` rather than narrowing them — each segment compiles under exactly one scenario, and guessing which one was meant would build for a core set nobody asked for. Vision batch inference and LLM batch inference do not use the same core-mode rule.

For what each mode means, see {external+aries:doc}`ARIES Core Mode <core-mode>`. For runtime core/cluster selection, see {external+aries:doc}`ARIES ModelConfig Configuration <modelconfig>`.

## RuntimeOptions

`RuntimeOptions` currently contains only:

| Field | Default | Meaning |
| --- | --- | --- |
| `version` | `0.0.0` | Compiler/runtime version metadata. |

This section is runtime metadata rather than a normal user tuning surface. Prefer the compile, calibration, preprocessing, LLM, and optimization sections for user-controlled behavior.

## Preset List

| Preset | Extends | Behavior |
| --- | --- | --- |
| `classification` | none | Image classification defaults: `calibration.mode=1`, `calibration.output=0`. |
| `detection` | none | Object detection defaults: `calibration.mode=1`, `calibration.output=1`. |
| `classification_torchvision` | `classification` | Enables uint8 input and standard Torchvision preprocessing: resize shortest side to 256, center crop 224x224, normalize ImageNet mean/std, `imageChannels=3`. |
| `yolo_640` | `detection` | Enables uint8 input and 640x640 letterbox preprocessing with pad value 114, `imageChannels=3`. |
| `yolo_1280` | `detection` | Enables uint8 input and 1280x1280 letterbox preprocessing with pad value 114, `imageChannels=3`. |
| `llm` | none | Enables LLM config with `maxSequenceLength=4096`, `maxCacheLength=4096`, `calibration.mode=0`, `calibration.output=0`, full-sequence-length LLM calibration, and the QK, UD, VO, SpinR1, SpinR2 and OptimizeFFN equivalent transformations. |
| `llm_fast` | `llm` | Turns off those six equivalent transformations and full-sequence-length calibration, trading accuracy for compile time. |
| `vision_transformer` | none | Transformer-oriented calibration/bit settings. |
| `multimodal` | none | Enables LLM handling and uses `calibration.method=3`. |

## Supported Frameworks

The public backend strings are:

| Backend | Input |
| --- | --- |
| `onnx` | ONNX model path. This is the primary path and is used by `validate_model`. |
| `tf` | TensorFlow SavedModel or supported TensorFlow/Keras model path. |
| `tflite` | TensorFlow Lite model path. |
| `torchscript` | TorchScript module/model. |
| `torch` | PyTorch model or HuggingFace/transformers model path/object flows. HuggingFace LLMs use this backend path, not a separate `hf` backend. |

## Supported Target Device

Use these target device strings in CLI and Python API calls:

| target device | Meaning |
| --- | --- |
| `regulus-ra` | REGULUS RA target. |
| `aries-rb` | ARIES RB target. |
| `regulus-rb` | REGULUS RB target. |
| `regulus-rb-usb` | REGULUS RB target connected over USB. From qbcompiler 1.3. |

The validator accepts the public strings above. Internal enum names use underscores, but user-facing values use hyphens.

## Supported Operators

[Mobilint IR Operations List](supported.md#mobilint-ir-operations-list) lists the operations Mobilint hardware runs natively. It is not the whole answer: an operation outside that list may still compile through a graph-level transformation, and whether any given one runs natively also depends on the model, backend, shapes and target device, which the parser and target-device allocation logic settle per compile.

Available reference paths:

- For non-ONNX backends, parse/compile the model and inspect unsupported groups with `.mblt` and CPU offloading workflows.
- If a model contains unsupported groups, `cpuOffload` / `cpu_offload` can partition those groups for CPU execution where the runtime flow supports it.

## Glossary

| Term | Meaning |
| --- | --- |
| `MXQ` | Mobilint executable package produced by quantization/compilation and run on Mobilint NPUs. |
| `MBLT` | Mobilint intermediate model format used between parsing and MXQ generation. |
| `parse` | Convert a source model into `.mblt`. |
| `quantize` | Convert a runnable `.mblt` into `.mxq` using calibration/config settings. |
| `compile` | End-to-end parse plus quantize flow. |
| `backend` | Source model framework identifier such as `onnx`, `torch`, or `tflite`. |
| `target_device` | Mobilint NPU target device string such as `aries-rb`. |
| NPU Chip | Mobilint NPU product name such as ARIES or REGULUS. |
| target device | Compile target such as `aries-rb`, `regulus-ra`, `regulus-rb`, or `regulus-rb-usb`; used as the target device string. |
| `calibration data` | Representative input tensors used to derive quantization scales/statistics. |
| `CPU offloading` | Partitioning unsupported graph groups for CPU execution while supported body subgraphs run on NPU. |
| `body subgraph` | The largest NPU-runnable supported subgraph extracted from a partitioned `.mblt`. |
| `preset` | Built-in partial `CompileConfig` for common model families. |

## License

See [License](license.md) for the open-source license notices included with this manual.
