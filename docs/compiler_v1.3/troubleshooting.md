<!-- 출처: https://docs.mobilint.com/compiler/v1.3/en/troubleshooting.html (원본) -->

# Troubleshooting

Most qb Compiler failures can be narrowed down by checking which stage failed:

```text
Original model -> parse -> MBLT -> quantize -> MXQ
```

Run `parse` on its own to find out whether the original model converts to MBLT at all. Use Netron on the MBLT when operator support or graph boundaries are unclear. Use `quantize` when calibration, quantization, or MXQ generation is the likely issue.

## Installation Issues

Symptoms include `qbcompiler: command not found`, Python import failures, or shared library errors when the CLI starts.

```bash
python -m pip show qbcompiler
python -c "import qbcompiler; print(qbcompiler.__file__)"
python -m qbcompiler --help
```

Use the Python environment where qb Compiler is installed. If multiple virtual environments, Conda environments, or containers are present, confirm that `python`, `pip`, and `qbcompiler` resolve to the same environment. For shared library errors, check the installation guide and confirm that `LD_LIBRARY_PATH` or the container image includes the required runtime libraries.

## Target Device Errors

Valid target device strings are:

- `regulus-ra`
- `aries-rb`
- `regulus-rb`
- `regulus-rb-usb` (from qbcompiler 1.3)

Use the exact target device string in CLI options and compile config. Do not use NPU Chip names such as ARIES or REGULUS as the value. The MXQ must be compiled for the same target device expected by the runtime deployment.

## Model Parse Failures

Parse failures usually mean the original model cannot be loaded, input shape cannot be inferred, or the exported graph contains unsupported framework constructs.

Checks:

- Confirm `backend` matches the original model: `onnx`, `tf`, `tflite`, `torchscript`, or `torch`.
- Re-export the model with fixed input shapes.
- Put PyTorch models in eval mode before export.
- Remove training-only nodes, random branches, debug outputs, and Python-side control flow.
- For ONNX models, a file cleaned up with an ONNX optimization library such as `onnxsim` parses more reliably.
- For HuggingFace models, confirm that the model loader, tokenizer, and `trust_remote_code` settings match the source repository.
- For LLM and HuggingFace models, set up your export environment to match either the `transformers` version used in the qb Compiler Docker image or the `transformers` version specified in the model parsing script.

When possible, run `parse` separately and open the resulting MBLT in Netron. If `parse` does not create MBLT, the issue is in model loading, framework conversion, or graph normalization before NPU compilation.

## Unsupported Layer

Netron shows supported layers in blue and unsupported layers in black. If compile reports an unsupported operator or the NPU body is smaller than expected, first decide whether the unsupported layer belongs in the compiled NPU body.

Unsupported preprocessing or postprocessing near the graph boundary may be handled with CPU Offloading or application-side code. Unsupported layers inside the main compute body usually require model re-export, graph simplification, or an equivalent supported operator pattern.

Useful actions:

- Locate the unsupported layer in Netron.
- Check whether it is in the `CPU head`, `NPU body`, or `CPU tail`.
- Replace custom operators with standard ONNX or framework operators before export.
- Freeze dynamic shape logic into constants when deployment shape is fixed.

## Calibration Data Errors

Calibration errors usually appear as file loading failures, input count or tensor name mismatches, or shape/dtype mismatches during quantization.

Checks:

- Each calibration sample must match the compiled model input contract.
- Multi-input models need one tensor per input for every sample.
- File naming and directory layout must be consistent with the calibration loader.
- Dtype should match the expected input path, such as `float32` for normalized tensors or `uint8` for raw input flows.
- For CPU Offloading, calibration shape should match the original model input. For NPU-body-only compilation, it should match the NPU body input.
- Preprocessing must match runtime inference: channel order, layout, scaling, normalization, resize, crop, letterbox, padding, tokenization, and masks.

Use a small calibration set first to validate format, then increase sample count.

## Quantization Accuracy Drops

Quantization is the likely issue when floating-point MBLT output is acceptable but MXQ output is not, specific layers show large error after quantization, or accuracy is sensitive to calibration sample choice.

Actions:

- Confirm parsing, preprocessing, and calibration shape before tuning quantization.
- Increase calibration sample count and make samples more representative.
- Compare a single preprocessed tensor from the source pipeline and the calibration generator.
- Try the task preset before manual quantization changes.
- Use per-channel weight quantization for convolution-heavy models when available.
- Keep sensitive output layers or task heads at a safer quantization setting if supported.
- Validate intermediate outputs to isolate the first layer with large error.

For YOLO, verify letterbox size, padding value, and scale restoration. For LLMs, verify tokenizer, embedding weight, sequence truncation, and attention mask generation before changing quantization settings.

For a systematic approach to diagnosing and recovering quantization accuracy, including sub-config tuning order and recommended configuration recipes, see [Model Quantization — Accuracy Degradation Handling](model_quantization.md#accuracy-degradation-handling).

## MXQ Generation Failures

MXQ generation failures usually happen after calibration starts, when memory is exhausted, or when `quantize` finishes without writing MXQ.

Checks:

- Confirm the MBLT was generated for the same target device.
- Ensure calibration data is readable and has enough valid samples.
- Reduce batch size, sequence length, cache length, or input resolution to isolate memory pressure.
- For LLMs, try a smaller `max_sequence_length` and `max_cache_length` first.
- Confirm the output path is writable and has enough disk space.
- Avoid compiling unrelated postprocessing into the NPU body when it creates unsupported or large graph regions.

If `compile` fails, split the flow into `parse` and `quantize` to identify the failing stage.

## LLM Sequence, Cache, and Batch Errors

Common symptoms are attention shape mismatch, KV cache tensor mismatch, runtime prompt length exceeding compiled limits, or batch size mismatch.

Checks:

- `max_sequence_length` must be at least the prompt sequence length used by the compiled graph.
- `max_cache_length` must match the intended KV cache capacity.
- Batch size must be fixed consistently across calibration, compile settings, and runtime.
- Calibration embedded tensors should use the same hidden size and dtype as the original model embedding.
- Dynamic RoPE, dynamic mask, `splitBlocks`, and `splitParts` settings should match the model architecture and memory target.

For diagnosis, compile with a short sequence length and cache length first. Once that path works, increase limits gradually and watch compiler memory usage.

## MBLT Does Not Open in Netron

Use Mobilint Netron rather than an arbitrary upstream Netron build:

- Online: `http://netron.mobilint.com`
- Windows/Linux download: `http://dl.mobilint.com`

Confirm that the file is an MBLT produced by the current compiler and that it is not a partially written file from a failed `parse`. If the graph looks simplified, that can be normal: MBLT is a parsed IR, so graph normalization and operator fusion may change the visual structure compared with the source model.
