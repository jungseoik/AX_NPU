<!-- 출처: https://docs.mobilint.com/compiler/v1.3/en/CHANGELOG.html (원본) -->

# Changelog

## [v1.3.0] - 2026-09-10

### Added

- Layer bias correction, a calibration-only accuracy pass that measures the systematic per-channel error quantization introduces and folds it into the biases of the layers carrying an integer one. Enable it with `layer_bias_correction=True`. No labels and no training loop, but it needs the float weights kept: a `weightMemory.method` of `DeleteFloat` is silently promoted to `SaveFloat`, so they are written to disk rather than dropped. The other methods are left as set. The field is an index into `weightMemory.methodList`, not one of those names: `0` is `DeleteFloat` and `1` is `SaveFloat`.
- Hessian-based weight quantization can hold its Hessian in half precision (`hessian_dtype="bf16"`), and `accumulation_device` chooses where it is accumulated, trading memory against speed. It defaults to `auto`, which follows `useGPUOnlyForCalibration` inversely: that setting's own default resolves `auto` to the CPU, the slower of the two, so a compile that enables Hessian-based quantization and says nothing about the device takes that path.
- A model made of several parts, such as a vision-language model, can be compiled one named part at a time with `model_part` and `model_part_options`.
- The parse stage is seeded, so the same inputs produce the same parsed graph. Each one also records how it was produced: the source revisions, the resolved parser settings, the parse inputs, and a hash of the source model. Read it with `read_provenance()` or the new `info` subcommand, and set `QBCOMPILER_PROVENANCE=0` to record nothing. The compiled MXQ records the same source revisions and the configuration it was compiled with. Neither is an on-disk format change.
- `config_save_path`, and `--config-save-path` on the command line, writes a compile's fully resolved configuration before it starts, so the settings a build actually used survive a run that fails and can be fed straight back to reproduce it.
- `mxq_compile_from_source` and `mxq_compile_from_mblt` expose the two halves of the compile entry point, for callers that stage the steps. The combined entry point is unchanged.
- A separate bit-width for the mixture-of-experts router gate, and a per-layer override that forces 8-bit weights.
- `calibration.maxSampleSizeForQuantScheme` is reachable from `CompileConfig`. The cap and its default of 16 are not new — the quantizer has enforced them since v1.2.0 — but nothing exposed them to the Python layer.
- Two parser settings on `ParserConfig`, each also readable from the matching `MBLT_HL_PARSER_` environment variable. `force_np_supported_types` casts torch weights to a numpy dtype, so bfloat16 becomes float32; it is on by default and was unconditional before, so what is new is being able to turn it off and keep the original dtype, skipping that copy on a model that does not fit. `transform_skip_on_error` reports a transform rule that raises and carries on, instead of failing the parse.
- `regulus-rb-usb`, a target device for USB-attached REGULUS. Its NPU I/O boundary is not floating point, so its executables are not interchangeable with `regulus-rb` ones.
- Qwen3-VL compiles for batched language-model inference.
- Installing the package puts a `qbcompiler` command on PATH; the previous `python -m qbcompiler` form still works and runs the same thing.

### Fixed

- **[Breaking]** The `llm` and `llm_fast` presets had been swapped since v1.1.0. v1.3.0 corrects them: `llm` optimizes for accuracy, `llm_fast` for compile time. A pipeline pinned to either changes behaviour — swap the name to keep the old one.
- Code generation is reproducible: recompiling one model with the same settings produces the same binary. Three causes are fixed — two sets of pointers iterated in heap order, and parallel bundle emission drawing ids from one shared counter, so the order threads reached it decided which id each bundle got.
- **[Breaking]** Calibration inputs given as a name map are matched by shape, and by name among inputs that share one. Two same-shape inputs fell back to positional order, silently calibrating against the wrong one; a name in such a group that does not match the model is now an error.
- Two code-generation faults on the first-generation REGULUS target that silently produced wrong results: two bitwise-ALU lowerings, and a write to a register that generation does not have. An executable built for it with an earlier release is worth rebuilding.
- **[Breaking]** An operation that needs the multi-function unit is correctly rejected on `regulus-ra`, which does not have that unit. It had been classified as supported and exported as an empty layer, so a model using one built on 1.2 and produced wrong results. A model that compiled before may now fail to build.
- **[Breaking]** The preprocessing resize operation applies its `alignCorners` and `antialias` settings instead of overwriting them with fixed values. Those values are still the defaults, so only a pipeline that sets either one changes.
- The preprocessing colour conversion between RGB and BGR works. The channel swap used a negative-step slice, which the tensor library rejects, so it raised instead of running.
- For a split model, the advertised input and output order follows the source model's declared order. It had been the order the sub-networks were walked in, while samples are numbered in the declared order, so a caller zipping buffers positionally put every one on the wrong input. An executable built for a split model before this release is worth rebuilding.
- `weightMemory.method` had no effect. The value was parsed but never reached the resource manager, so float weights were always deleted after quantization whatever was asked for. The field is an index into `weightMemory.methodList` rather than one of its names: `0` `DeleteFloat`, `1` `SaveFloat`, `2` `MoveFloat`, `3` `KeepFloat`, `4` `KeepAll`.
- Quantization accuracy and stability, across asymmetric zero-points, histogram calibration weighting, the 16-bit weight range, EfficientViT.
- The `compile` and `quantize` subcommands run. Both reached the compile entry point without a target device, which is a required argument, so every invocation failed internally. `quantize` now takes a target device of its own, and `compile`'s drives both phases rather than the parse phase alone.
- Command-line argument errors no longer exit silently: every failure prints a coded message, or emits a structured event instead where that channel is on. A missing `--target-device` lists the available ones instead of failing internally, and an unsupported `--backend` is reported as an argument error rather than an internal one.

### Changed

- **[Breaking]** Transformer feed-forward bit-width is set per projection rather than as one value. A single number still applies to all three, so configuration files keep working; code that read the field as a number does not.
- **[Breaking]** A configuration file written by 1.2 may name a key this release removed. An unknown key has always been refused, so such a file fails to load — it is the key set that changed, not the loader. Regenerate it with `dump-config` and carry across the values that were set.
- **[Breaking]** The exception, progress and logging modules moved under a `reporting` package. Importing the exception classes from the package root is unchanged.
- **[Breaking]** `inference_scheme` is validated against the target device's core count before quantization rather than failing later: multi-core schemes are rejected on single-core devices, and `all` narrows to single-core there.
- **[Breaking]** A batch-LLM compile honours the requested `inferenceScheme` instead of forcing `single`. `global4` and `global8` now build what they name, and `multi` and `all` are rejected with an error rather than quietly becoming `single`. A pipeline that passed either of those and relied on the silent fallback now fails; one pinned to `global4` gets a different MXQ than it did on 1.2.
- Hessian-based quantization uses a larger default block size and a parallel solve instead of a sequential one, and skips a layer whose Hessian is empty instead of aborting the compile.
- The body-subgraph export writes the body together with every subgraph it calls, rather than the body alone, and a subgraph that is only ever called is no longer chosen as the body.
- The ONNX parser handles `GroupNormalization`, which it did not before. A model containing one parses and partitions differently with no change to the call.
- The multi-function unit is served by a lookup table embedded in the library.
- Binary code generation, the stage after quantization, is up to roughly 3x faster. Sparse liveness storage applies to every compile; building each compilation unit in a forked process applies to the Global inference schemes, which single-core devices do not accept.
- **[Breaking]** `torch` models are parsed by the same pipeline as `onnx`; previously only `onnx` took that path. The resulting graph, its operation coverage, the split between NPU and host, and the numerics can all differ with no change to the call. `tf`, `tflite` and `torchscript` still use the legacy parser.
- **[Breaking]** An unrecognised backend name is rejected at the API boundary, with the accepted names listed. Several names the parser registry accepted before — including the one for Hugging Face models — are no longer backend names; load the model yourself and pass it as a torch model.
- **[Breaking]** The parser package holding the new pipeline took the name the legacy one used, and the legacy pipeline moved beside it under a `_legacy` suffix. Code importing the old name binds to a different module rather than failing.
- The parser transforms subgraphs in parallel, one forked worker per subgraph, up to `ParserConfig.num_work` — settable there or with `MBLT_HL_PARSER_NUM_WORK`, and by default the CPUs the process is allowed to use, less two. Worker count is not memory-aware: peak host memory scales with it, so models with unusually heavy subgraphs need a lower `num_work`.
- **[Breaking]** The `classification_torchvision` preset resizes by shortest side rather than to an exact height and width. For a non-square calibration image that is a different tensor, so a pipeline pinned to this preset changes numerics with no error. Two further changes reach a configuration that already resized by shortest side: the scaled long side truncates where it used to round, and a centre crop splits an odd gap to even rather than always rounding down. Either can move the crop by a pixel.
- Quantized activations are stored with real integer dtypes rather than float, lowering host memory use on large models.

### Removed

- **[Breaking]** The `_V2` suffixed entry points. Dropping the suffix is enough for the two `mxq_` ones. For the two `mblt_` ones it is not: `mblt_compile` and `mblt_compile_with_callback` take the output path before the target device, where `mblt_compile_V2` and `mblt_compile_with_callback_V2` took the target device first, so a positional caller swaps two strings silently.
- **[Breaking]** `hf_config`, which now raises `ValueError` naming its replacement. Load the Hugging Face model yourself and name the part to compile with `model_part`.
- The lookup-table clustering settings, replaced by explicit grouping and optimization sections. They existed only in the Python layer and no compiler component read them, so a compile that set them already behaved as the new defaults do.
- The bundled multi-function wrapper library.

## [v1.2.0] - 2026-06-26

### API
- Unified qbcompiler support for multiple hardware targets, including REGULUS and ARIES, into a single compiler.

### CLI
- Introduced the `python -m qbcompiler` CLI, with `parse` and `quantize` as the subcommand names for producing an MBLT and an MXQ, alongside the end-to-end `compile`. Advertised by `check` as capabilities.

## [v1.1.0] - 2026-03-31

### API
- The configuration of the mxq_compile function has been structurally refactored.
- For ONNX model parsing, graph optimization is now handled by the newly refactored parser based on mblt-graph.
- Improved quantization accuracy for non-linear activation functions.

## [v1.0.2] - 2026-02-12

### API
- Added support for all in inference_scheme.
- Fixed an issue where the optimize option was not being applied correctly.


## [v1.0.0] - 2026-01-30

### API
- Introduced support for Qwen3 LLMs.
- Expanded compatibility to include the YOLO26 series.
- Removed the deprecated singlecore compile and startdramoffset options.
- Added support for custom masks and dynamic RoPE.
- Added NPU-accelerated execution for parts of the pre-processing pipeline, along with new input process configuration.
  - Added support for uint8 inputs.
  - Added NPU-based normalization.
- Added support for Torch tensors and image files as calibration data.
- Removed redundant quantization configuration options and unified Percentile, MSE, and KL into a histogram-based observer.
- Added support for dynamic core and memory allocation for all models.


## [v0.12.0.0] - 2026-01-02

### API

- Refactored compilation and quantization configuration.
- Optimized CPU memory usage during compilation.
- Added support for Transformers v4.57.1 (aligned with the 0.12 Mobilint Docker release).
- Added support for the MiniCPM model.

## [v0.11.0.0] - 2025-09-10

### API

- Added support for Torch parser
- Added support for Yolo12l and Yolo12x
- Expanded support range for ViT models
- Fixed minor bugs for GRU/RNN/LSTM

## [v0.10.0.0] - 2025-07-24

### API

- Added Inference_scheme global4/global8 modes
- Added support for Yolov10 series, Yolo11 series, Yolo12n, Yolo12s, and Yolo12m
- Added LLM config options
- Added support for GRU/RNN/LSTM

## [v0.9.0.5] - 2025-06-25

### API

- Changed YOLO model decoding to be linked with the model zoo
- Added layer config options

## [v0.9.0.4] - 2025-05-22

### API

- Added support for direct compilation of HuggingFace LLM models

## [v0.9.0.3] - 2025-04-02

### API

- Added support for saving mblt using mmap
- Added support for diverse visualization types

## [v0.9.0.2] - 2025-01-16

### API

- Added API for saving model architecture
- Added support for Visualization tool
- Added support for Run min output difference QAT mode

## [v0.9.0.1] - 2024-12-13

### API

- Added API for global core (beta)
- Added API for YOLO post (beta)
- Added support for diverse hardware

## [v0.9.0.0] - 2024-11-27

### API

- Updated the high-level parsing processes

### Docker

- ONNX: 1.13.0 -> 1.16.2
- TensorFlow: 2.9.0 -> 2.17.0
- Torch: 1.13.0 -> 2.4.1

## [v0.8.5] - 2024-06-20

### API

- Added support for FastPercentile quantization method

## [v0.8.4] - 2024-05-20

### API

- Connected TF backend to ONNX backend by TF2ONNX
- Enabled compilation of models with custom input shape
- Supported more operations

### Docker

- ONNX: 1.12.0 -> 1.13.0

## [v0.8.3] - 2024-03-07

### API

- Added support for TF Lite backend

## [v0.8.2] - 2024-02-23

## [v0.8.1] - 2023-12-08

## [v0.8.0] - 2023-11-02

### API

- Deprecated TVM backend

## [v0.7.12] - 2023-09-12

## [v0.7.11] - 2023-08-31

### API

- Added support for TorchScript backend

## [v0.7.10] - 2023-08-11

## [v0.7.9] - 2023-08-11

## [v0.7.8] - 2023-08-08

## [v0.7] - 2023-03-23

- Added multi-channel quantization
- Supported more operations

### API

- Improved calibration dataset processing
- Added support for CPU offloading (beta version)

## [v0.6] - 2022-08-10

- Made minor updates

## [v0.5] - 2022-07-01

### Docker

- Switched from Conda to Virtualenv
- Python: 3.7.7 -> 3.8.10
- Torch: 1.8.1 -> 1.10.1
- TensorFlow: 1.15.0 -> 2.3.0
- ONNX:1.6.0 -> 1.11.0

### Parser

- Refactored code

### API

- Enabled saving sample inference results (inputs and outputs)

## [v0.4] - 2022-02-23

### Optimizer

- Made minor updates in fusing reshape

## [v0.3] - 2022-02-05

### Parser

- Identified preprocess and postprocess of the model
- Excluded preprocess and postprocess if they were unsupported by the NPU

### API

- Added integer inference simulation in the Python API

## [v0.2] - 2021-12-01

- First release
