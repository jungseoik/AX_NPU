<!-- 출처: https://docs.mobilint.com/compiler/v1.3/en/release_note.html (원본) -->

# Release Notes

## v1.3.0

**Release date:** September 10, 2026<br>
**Type:** Minor

Compiling one part of a model at a time, reproducible `.mblt` files with a provenance record, a faster and leaner HessianQuant, and a `qbcompiler` command on PATH.

### Added

- **[Highlight]** **`model_part` / `model_part_options` — compile one part of a model at a time** — a multi-part model such as a VLM is compiled one part at a time rather than as a single graph. `model_part` names the part (`"vision"`, `"language"`, `"encoder"`, …); `model_part_options` passes extra arguments the part needs. A model that declares exactly one part resolves it from `None`; a model that declares several requires a name.

  ```python
  # Which parts does this model declare?
  from qbcompiler.model_dict.parser.patcher.parts import available_parts
  print(available_parts(model))

  qbcompiler.mblt_compile(
      model=model,
      mblt_save_path="audio.mblt",
      target_device="aries-rb",
      backend="torch",
      model_part="audio",
      model_part_options={"mel_frames": 100},
  )
  ```

  This replaces the removed `hf_config` argument, which loaded a Hugging Face model on the caller's behalf. Load the model yourself and pass it as `model`.

- **[Highlight]** **Reproducible `.mblt` files, and a provenance record in every one** — two changes that together make a parse repeatable.

  The parse stage is now seeded. `ParserConfig.random_seed` had been dead configuration — both backend parsers stored it and never read it, and it was left unset — which let two runs of the same command write different weight bytes: a traced torch model whose `forward` calls `torch.randn` bakes that value into the graph as a constant, and the ONNX loader fabricates dummy inputs from `numpy.random` when no `feed_dict` is given. `random_seed` now defaults to `12345` and is applied. Loading a model seeds `random`, `numpy.random` and `torch` for the duration of the load and restores the caller's RNG state afterwards. The legacy parser, still the path for `tf`, `tflite` and `torchscript`, seeds with a different constant.

  Every `.mblt` now also records how it was produced: the qbcompiler and mblt-graph commits, the resolved `ParserConfig`, the `parse()` inputs (`dynamic_axes` plus a fingerprint of `feed_dict`), the write options that change the payload, and the source model's SHA-256 — or its Hugging Face hub revision for a torch model. This is not an on-disk format change: the record goes into the manifest section for the current format and into the model dictionary for the legacy one, so older readers are unaffected.

  ```bash
  qbcompiler info --mblt model.mblt
  ```

  ```python
  print(qbcompiler.read_provenance("model.mblt"))
  ```

  `read_provenance()` gives `None`, and `qbcompiler info` prints `null`, only when the file carries no record — one written before provenance existed, or with recording off. The legacy container carries a record too; it travels inside the serialized model dictionary rather than a manifest section. A field that cannot be determined is recorded as an error string rather than failing the parse. Set `QBCOMPILER_PROVENANCE=0` to record nothing, which also skips hashing the source model. An MXQ built from a current-format `.mblt` embeds that manifest verbatim, next to the target device and the resolved quantization config. A legacy-format `.mblt` has no manifest section, so its MXQ records why instead of the record itself.

- **[Highlight]** **A faster, leaner HessianQuant** — two new knobs decide where the accumulated Hessian lives and how large it is.

  ```python
  from qbcompiler.configs import HessianQuantConfig

  qbcompiler.mxq_compile(
      model="model.onnx",
      target_device="aries-rb",
      calib_data_path="calib",
      save_path="model.mxq",
      hessian_quant_config=HessianQuantConfig(
          apply=True,
          hessian_dtype="bf16",
          accumulation_device="gpu",
      ),
  )
  ```

  `hessian_dtype="bf16"` halves the accumulated Hessian's memory footprint — host RAM when it lives on the CPU, VRAM when it lives on the GPU. Compute is unaffected: the per-batch matmul and accumulation still run in float32 and the solve upcasts the accumulator, so only the stored one is bfloat16. This matters most for MoE models, which carry a Hessian per expert FFN.

  `accumulation_device="gpu"` accumulates on the calibration device instead of copying the whole *d*×*d* accumulator off it once per layer per batch, which measures 3–5× faster on the vendor's benchmark; the Hessian is parked on the host as soon as the last batch is in, so peak VRAM for the rest of the compile does not rise. `"cpu"` bounds VRAM during calibration itself and is the slower of the two. **The default resolves to `"cpu"`**, so a compile that enables HessianQuant and leaves the device alone takes the slower path; pass `accumulation_device="gpu"` if the calibration GPU has the headroom. The default `"auto"` follows `resourceManagement.useGPUOnlyForCalibration`, the setting that already states whether a compile is VRAM-bound — and follows it inversely, since that setting means *keep the GPU for calibration only*: `true`, its default, resolves `"auto"` to `"cpu"`, and `false` resolves it to `"gpu"`. A CPU compile always accumulates on the host.

  HessianQuant also replaced its sequential solve with a parallel one, and a layer whose Hessian is entirely zero is now skipped with a message instead of aborting the compile.

- **Layer bias correction** — a new calibration-only accuracy pass. It measures the systematic per-channel error between the float and quantized activations and folds damped corrections into the biases of the layers that carry an integer one, leaving inserted layers and the model's own outputs alone, recomputing them between iterations so that upstream changes are accounted for. It uses no labels and no Minimum Output Difference optimization, so it costs a few extra calibration passes rather than a training loop. It does cost disk: enabling it promotes a `weightMemory.method` of `DeleteFloat` to `SaveFloat`, so the float weights are written out instead of dropped. The other methods are left as set. `method` is an index into `weightMemory.methodList`, not one of those names: `0` is `DeleteFloat`, `1` `SaveFloat`, `2` `MoveFloat`, `3` `KeepFloat`, `4` `KeepAll`.

  ```python
  qbcompiler.mxq_compile(
      model="model.onnx",
      target_device="aries-rb",
      calib_data_path="calib",
      save_path="model.mxq",
      layer_bias_correction=True,
  )
  ```

  `layer_bias_correction_config=` tunes it. The tunables live on `Attributes`, not on the config itself:

  ```python
  from qbcompiler.configs import LayerBiasCorrectionConfig

  cfg = LayerBiasCorrectionConfig(
      apply=True,
      attributes=LayerBiasCorrectionConfig.Attributes(
          num_samples=256,      # calibration samples per iteration
          iterations=5,         # damped correction rounds
          correction_rate=0.05, # fraction of the measured error per round
      ),
  )
  ```

- **A `qbcompiler` command on PATH** — installing qbcompiler now provides a console script, so the CLI no longer has to be spelled `python -m qbcompiler`.

  ```bash
  qbcompiler compile --model model.onnx --target-device aries-rb \
      --calib-data-path calib --output model.mxq
  ```

  Both forms run the same entry point through the same output protocol, and usage and argument-error text name whichever form was invoked. Subcommands are unchanged. See [Installation and Environment Check](installation.md).

- **`config_save_path` — record the configuration a compile actually used** — writes the fully resolved `CompileConfig` before compilation starts: the normalized configuration after every layer of the precedence order has been applied, with all sub-configurations materialized. Because it is written up front, the record survives a compile that fails or is interrupted, and the file can be fed straight back as `compile_config=` to reproduce the run.

  ```bash
  qbcompiler compile --model model.onnx --target-device aries-rb \
      --calib-data-path calib --output model.mxq \
      --config-save-path used_config.yaml
  ```

  ```python
  qbcompiler.mxq_compile(
      model="model.onnx",
      target_device="aries-rb",
      calib_data_path="calib",
      save_path="model.mxq",
      config_save_path="used_config.yaml",
  )
  ```

  A `.yaml` or `.yml` suffix writes YAML; any other suffix writes JSON. Parent directories are created. On the `compile` pipeline the file is written by the quantize phase, the phase that carries the full configuration. An unwritable destination is reported as `OUTPUT_WRITE_ERROR` with the offending path, not as an internal error. See [Compile Configuration](compile_configuration.md).

- **`mxq_compile_from_source()` and `mxq_compile_from_mblt()`** — the two things `mxq_compile()` does, available directly. `mxq_compile()` keeps its signature and behavior and routes between them, so existing callers need no change. The `.mblt` entry point omits the parser-only arguments that cannot apply to an already-parsed graph (`save_subgraph_type`, `output_subgraph_path`, `feed_dict`, `dynamic_axes`, `in_dformats`, `yolo_decode_include`, `exclude_first_subgraph`, `model_part`, `model_part_options`). The signature omits them; it does not reject them, since it ends in `**kwargs` and forwards what it is given. `mxq_compile()` does reject the last two with a `.mblt` input. Its first parameter is `mblt`, not `model`.

- **`regulus-rb-usb` target device** — a fourth compile target, alongside `aries-rb`, `regulus-ra` and `regulus-rb`. Its NPU I/O boundary is not floating point, so an MXQ built for `regulus-rb` is not interchangeable with one built for `regulus-rb-usb`. See [Installation and Environment Check](installation.md).

- **[Breaking]** **Finer bit-width control for transformers** — `bit.transformer.activation` and `bit.transformer.weight` gained `router`, for the MoE router gate, and replaced the single `ffn` bit-width with a `{up, gate, down}` sub-object so the SwiGLU gate and the two projections can differ. Passing a plain integer for `ffn` still sets all three, so existing configuration files keep working — but Python that reads the field back now gets an object where it used to get a number. `bit.layerOverrides` gained `weight8Bits`, the counterpart to the existing `weight16Bits`.

- **The cap on calibration samples per quantization-scheme stage is settable** — `calibration.maxSampleSizeForQuantScheme`. The cap and its default of 16 are not new; reaching them from `CompileConfig` is. See [Model Quantization](model_quantization.md#calibrationconfig).

- **Two more parser settings** — both read from `ParserConfig` or the matching `MBLT_HL_PARSER_` environment variable. `force_np_supported_types` (on by default) casts torch weights to a numpy dtype, so bfloat16 becomes float32; it was unconditional before, so what is new is being able to turn it off, keep the original dtype and skip that copy on a model that does not fit. `transform_skip_on_error` reports a transform rule that raises and carries on instead of failing the parse.

- **Qwen3-VL for batch LLM** — Qwen3-VL compiles for batched LLM inference. Batch LLM is requested by setting `llm.attributes.runtime.batchSize` above 1 with `llm.apply` enabled — the `llm` presets turn `llm.apply` on but leave `batchSize` at 1, so the batch size is yours to set either way — and applies to single-bundle models only; a multi-bundle model is rejected with a message. The runtime side of this feature is qb Runtime v1.2.0's `BatchParam`.

### Revised

- **[Breaking]** **The `llm` and `llm_fast` presets had been swapped since v1.1.0. v1.3.0 corrects them.** Through v1.1.x and v1.2.0, `config_preset="llm"` applied none of the accuracy transformations and `llm_fast` applied all of them — the opposite of what the names say. As of v1.3.0, `llm` enables the equivalent transformations (QK, UD, VO, SpinR1, SpinR2, OptimizeFFN) and full-sequence-length calibration, and `llm_fast` disables them for a faster compile. A pipeline pinned to either preset will see its compile time and accuracy change. To keep v1.2.0 behavior, swap the preset name.
- **[Breaking]** **`inference_scheme` is now validated against the target device.** `multi`, `global4` and `global8` each bundle two or more NPU cores, so on a single-core device — every REGULUS variant — they are rejected outright rather than failing later. `all`, which asks for whatever the device supports, narrows to `single` there instead of failing. The check runs before quantization regardless of whether the scheme or the target device was set first, so an impossible combination fails in seconds rather than after a full calibration.
- `HessianQuant`'s `attributes.blockSize` default changes from 128 to 256.
- **[Breaking]** The `classification_torchvision` preset now resizes with `size` rather than `height` / `width`. `size` scales the shortest side and preserves aspect ratio, following torchvision `Resize(<int>)`; `height` / `width` resize to an exact shape. For a non-square calibration image those are different tensors, so a pipeline pinned to this preset changes numerics with no error. Both spellings have been accepted since v1.2.0 and `size` takes precedence when both are given. Two further changes reach a configuration that already spelled `size`: the shortest-side computation truncates where it used to round, and `centerCrop` splits an odd gap to even instead of always rounding down. Either can move the crop by a pixel.
- The `compile` and `quantize` subcommands run. On 1.2.0 both reached the compile entry point without a target device, which is a required argument, so every invocation failed internally — silently by default, and as `INTERNAL_ERROR` only on the JSONL channel where that was switched on. `quantize` now takes `--target-device` of its own, and `compile --target-device` drives both phases rather than the MBLT phase alone.
- **[Breaking]** The preprocessing `resize` operation applies `alignCorners` and `antialias` instead of ignoring them. Both were parsed and then overwritten with fixed values; those values are still the defaults, so only a pipeline that sets either one changes.
- The preprocessing `colorConvert` operation converts between RGB and BGR. The channel swap used a negative-step slice, which the tensor library rejects, so the conversion raised instead of running.
- `resourceManagement.weightMemory.method` is now actually applied to the resource manager; it had been accepted and ignored.
- **[Breaking]** A configuration file written by 1.2 may name a key this release removed. An unknown key has always been refused, so such a file fails to load — it is the key set that changed, not the loader. Regenerate it with `dump-config` and carry across the values that were set.
- Missing or invalid CLI arguments no longer exit silently. Every error path now prints an `ERROR [CODE]: message` line to stderr, or, when the JSONL channel is enabled, emits a structured `error` event with code `ARGUMENT_ERROR` there instead — one or the other, not both. Omitting `--target-device` lists the available devices instead of raising an internal `TypeError`, and an unsupported `--backend` is reported as an argument error rather than an internal one.
- **[Breaking]** A batch-LLM compile honours the requested `inferenceScheme` instead of forcing `single`. `global4` and `global8` now build what they name, and `multi` and `all` are rejected with an error rather than quietly becoming `single`. A pipeline that passed either of those and relied on the silent fallback now fails; one pinned to `global4` gets a different MXQ than it did on 1.2.
- The ONNX parser handles `GroupNormalization`, which it did not at v1.2.0. An ONNX model containing one parses and partitions differently with no change to the call.
- **[Breaking]** An operation that needs the multi-function unit is correctly rejected on `regulus-ra`, which does not have that unit. It had been classified as supported and exported as an empty layer, so a model using one built on 1.2 and produced wrong results. A model that compiled before may now fail to build.
- **[Breaking]** `backend="torch"` is parsed by the same pipeline as `onnx`. In v1.2.0 only `onnx` took that path. The resulting graph, its operation coverage, the NPU/CPU split and the numerics can all differ, with no change to the call. `tf`, `tflite` and `torchscript` still use the legacy parser.
- **[Breaking]** An unrecognised `backend` is rejected at the API boundary with the accepted names listed. `hf`, `nemo`, `wenet`, `outetts` and `mellotts` were reachable backend names in v1.2.0 and are not accepted now; `hf` in particular is replaced by loading the model yourself and passing `backend="torch"`.
- **[Breaking]** Calibration inputs given as a JSON name map are matched by shape, and by name among inputs that share one. Two same-shape inputs used to fall back to positional order, silently calibrating against the wrong one; a name in such a group that does not match the model is now an error.
- For a split model, the advertised input and output order follows the source model's declared order. It had been collected by walking the sub-networks, which yields graph-appearance order, while the quantizer numbers its samples in the declared order — so a caller zipping buffers positionally put every one on the wrong input. A split-model MXQ built before 1.3 is worth rebuilding.
- The selection that keeps only the largest supported subgraph keeps every subgraph it calls along with it, rather than the body alone, and the target of a call is no longer eligible as the body. That is what `get_body_subgraph()` and the `extract-body` subcommand write, and the parse path uses the same selection — so for a model whose body calls subgraphs, the resulting `.mblt` differs and a script post-processing it sees a different file.
- Two `regulus-ra` code-generation faults that silently produced wrong results are fixed: two bitwise-ALU lowerings, and a write to a register that does not exist on that generation. An MXQ built for `regulus-ra` with an earlier release is worth rebuilding.
- **[Breaking]** `qbcompiler.model_dict` now holds the new parser pipeline. The legacy pipeline moved to `qbcompiler.model_dict_legacy`, and `qbcompiler.model_dict_new` is gone. Code importing from `qbcompiler.model_dict` binds to a different module rather than failing, so the change does not announce itself.
- A model with more than one subgraph has them transformed in parallel, one forked worker per subgraph, up to `ParserConfig.num_work` — settable there or with `MBLT_HL_PARSER_NUM_WORK`, and by default the CPUs the process is allowed to use, less two. A single-subgraph model is unaffected. The worker count is not memory-aware: peak host memory scales with it, so a model with unusually heavy subgraphs needs a lower `num_work`.
- Binary code generation, the stage after quantization, is up to roughly 3× faster. Sparse liveness storage applies to every compile; forking each compilation unit into its own process applies to the Global inference schemes (`global4`, `global8`), which single-core devices do not accept. Three sources of non-determinism in the same stage were fixed alongside it.
- Quantization accuracy and stability: EfficientViT accuracy improved; asymmetric zero-points are now carried correctly through matmul accumulator bias, the *x*² scale-factor search, `InputConstant` integer codes and average-pooling depthwise conversion; 16-bit weights use the BINT16 range `[-32768, 32639]` rather than the full INT16 range; sample weighting in histogram-mode calibration was corrected; and a model input that feeds a padded convolution on `aries-rb` or `regulus-ra` is kept symmetric where making it asymmetric would cost it the hardware input reshape.
- The Multi-Function Unit is served by a bit-exact fp16 lookup table embedded in the library, replacing the `libmfu_wrapper.so` that used to ship beside it.
- Quantized activations are stored with real integer dtypes rather than float, lowering host memory use on large models.
- **[Breaking]** **`qbcompiler.exceptions`, `qbcompiler.progress` and `qbcompiler.logging`** moved to `qbcompiler.reporting.exceptions`, `qbcompiler.reporting.progress` and `qbcompiler.reporting.logging`. No top-level shim remains, so the old paths raise `ModuleNotFoundError`. Importing the exception classes from the package root — `from qbcompiler import QBCompilerError` — is unchanged and remains the supported form.

### Removed

- **[Breaking]** **The `_V2` function aliases.** `mxq_compile_V2`, `mblt_compile_V2`, `mxq_compile_with_callback_V2` and `mblt_compile_with_callback_V2` are gone. Dropping the suffix is enough for the two `mxq_` functions. Both `mblt_` ones also changed argument order: `mblt_compile` and `mblt_compile_with_callback` take the output path before the target device, where `mblt_compile_V2` and `mblt_compile_with_callback_V2` took the target device first. A positional caller swaps two strings with no error; keyword callers are unaffected.

  ```python
  # before (v1.2.0)
  qbcompiler.mxq_compile_V2(model=..., target_device=..., ...)

  # after (v1.3.0)
  qbcompiler.mxq_compile(model=..., target_device=..., ...)
  ```
- **[Breaking]** **`hf_config`.** Passing it now raises a `ValueError` naming the replacement. Load the Hugging Face model yourself, pass it as `model` with `backend="torch"`, and select the piece to compile with `model_part` / `model_part_options`.
- **`CalibrationConfig.clustering_methods`**, `clusteringMethods`, `clusteringMethodsList` and the `LutClusteringMethod` enumeration. They existed only in the Python layer — no compiler component read them at v1.2.0 — so a compile that set them already behaved as the new defaults do. The lookup-table search is configured by `calibration.groupLut` (`irlsIter`, `droEps`, `coverFloor`) and `calibration.optimizeLut.optimizationLevel` — `0` for a surrogate-only fast search, `1` to refine against the true objective.
