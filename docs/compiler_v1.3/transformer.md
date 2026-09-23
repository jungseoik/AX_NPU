<!-- 출처: https://docs.mobilint.com/compiler/v1.3/en/transformer.html (원본) -->

# Transformer

This chapter describes LLM compilation for HuggingFace `transformers` models through the PyTorch path, using `backend="torch"`.

The LLM flow follows the same qb Compiler pipeline used by other models:

```text
HuggingFace transformers model
  -> MBLT
  -> MXQ
```

For LLMs, the important difference is that the compiled model is treated as a transformer decoder body with runtime state. The runtime keeps the KV cache while the NPU executes the repeated transformer blocks.

For Vision Transformer (ViT) models, see [Vision — Vision Transformer](vision.md#vision-transformer).

## LLM Compile Overview

A typical HuggingFace LLM compile has four parts:

1. Prepare the model in HuggingFace `transformers` format.
2. Prepare calibration data that matches the tensor input used by the compiled decoder body.
3. Load the model with `transformers` yourself and pass the instance with `backend="torch"`.
4. Enable LLM settings with `LlmConfig`, either directly or through the `llm` / `llm_fast` preset.

Use GPU compilation for practical LLM workloads. Transformer decoder models are large, and calibration plus quantization can require substantial memory.

## Supported Model Structure: Transformer + KV Cache

qb Compiler targets causal decoder-style transformer models with KV cache. The expected structure is:

- token embedding and tokenizer are handled outside the compiled NPU body;
- the compiled body receives embedded token tensors or the model input form selected by the frontend;
- each transformer block contains attention, RoPE or another positional encoding path, MLP/FFN, residual connections, and normalization;
- attention uses past key/value state so generation can append new tokens without recomputing the full prefix;
- the output head can be quantized separately from the internal transformer blocks.

The model should be loadable by HuggingFace `transformers`, usually with `AutoModelForCausalLM`. Architectures such as LLaMA, Qwen, Gemma, and similar decoder-only families are the main target. Models with unusual custom attention, sliding-window behavior, MoE routing, or nonstandard cache layout may require model-specific handling and should be validated before relying on generated MXQ output.

## HuggingFace LLM Compilation

qb Compiler does not load the model on your behalf. Load it with `transformers`, build an example input, and pass the live instance:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import qbcompiler

model_id = "meta-llama/Llama-3.2-1B-Instruct"

model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)
example = tokenizer("Hello", return_tensors="pt")

qbcompiler.mxq_compile(
    model=model,
    backend="torch",
    target_device="aries-rb",
    feed_dict={"input_ids": example["input_ids"]},
    calib_data_path="calib",
    save_path="model.mxq",
    weight_dtype="bfloat16",
)
```

Three things this path requires:

- `model` must be the loaded instance. `backend="torch"` parses a live `torch.nn.Module`; a model id or a directory path is not loaded for you.
- `feed_dict` is required. The torch path traces the model, and tracing without an example input fails with *feed_dict must be specified*.
- Leave `model_part` out for a plain causal LM. Parts exist for models that declare more than one, such as a vision-language model's `vision` and `language` towers. A model that declares none is parsed as loaded, one that declares exactly one resolves it from `None`, and naming a part the model does not declare raises `KeyError`. List what a model declares with `available_parts(model)` from `qbcompiler.model_dict.parser.patcher.parts`.

Set `weight_dtype` to match the original model weights, for example `bfloat16` or `float16`. The value is usually visible in the model's `config.json` or model card. For gated HuggingFace models, authenticate with `huggingface-cli login` before compiling.

A `transformers` model compiled with `backend="torch"` emits a `UserWarning` suggesting `backend="hf"` when no part was resolved for it and its class has no registered root patcher. The suggestion is stale — qb Compiler 1.3 no longer accepts `hf` as a backend name — but the warning means nothing is applying model-specific handling, so it is worth acting on rather than ignoring.

Up to qb Compiler 1.2 this section was spelled with `hf_config`, which loaded the HuggingFace model for you. qb Compiler 1.3 removed it; passing it now raises `ValueError`. See [Release Notes](release_note.md).

## LlmConfig

`LlmConfig` is the LLM-specific sub-config. It enables LLM graph handling and records sequence, cache, calibration, and runtime metadata.

```python
from qbcompiler.configs import LlmConfig

llm_config = LlmConfig(
    apply=True,
    attributes=LlmConfig.Attributes(
        maxSequenceLength=4096,
        maxCacheLength=4096,
        maxDataLength=4096,
        maxCoreDataLength=128,
        calibration=LlmConfig.Attributes.Calibration(
            useFullSeqLength=True,
        ),
        runtime=LlmConfig.Attributes.Runtime(
            batchSize=1,
            dynamicRope=False,
            dynamicMask=False,
        ),
    ),
)
```

The same fields can be written in JSON/YAML compile config files with their schema names:

```json
{
  "llm": {
    "apply": true,
    "attributes": {
      "maxSequenceLength": 4096,
      "maxCacheLength": 4096,
      "maxDataLength": 4096,
      "maxCoreDataLength": 128,
      "calibration": {
        "useFullSeqLength": true
      },
      "runtime": {
        "batchSize": 1,
        "dynamicRope": false,
        "dynamicMask": false
      }
    }
  }
}
```

Some compatibility examples use helper functions such as `get_llm_config(...)`. The schema-backed form above is preferred when writing reusable config files because it maps directly to `CompileConfig`.

## Sequence Length

For single-batch LLM (`runtime.batchSize=1`), `maxSequenceLength` is the maximum input length that the resulting MXQ can receive and process in a single `infer` call. It affects attention shapes, RoPE/mask generation, calibration shape, and runtime memory requirements.

Choose the smallest value that covers the deployment requirement. Increasing it from 4K to 8K or beyond can sharply increase runtime memory requirements.

`maxSequenceLength` may be equal to `maxCacheLength`, but it can also be smaller. The KV cache can be accumulated across multiple `infer` calls, so the input processed by one call does not always need to cover the full cache length.

Chunked prefill is useful when a large prompt would otherwise be passed in one `infer` call. Large single-call inputs can hit the qb Runtime timeout, which is derived from the measured operator latency, with a 10-second fallback. From qb Runtime 1.3.1 it can be overridden with `QBRUNTIME_NPU_TIMEOUT_MS`. Applications may split prefill into chunks such as 128, 256, or 512 tokens and call `infer` multiple times while accumulating the KV cache.

On NPUs, input/output buffer memory also scales with `maxSequenceLength`. REGULUS deployments may be more sensitive to this memory burden, so set `maxSequenceLength` appropriately small. Treat this as a separate sizing concern from the runtime-timeout reason for chunked prefill.

For current LLM workflows, set calibration samples to the same shape policy used at compile time. If `useFullSeqLength=True`, calibration uses full-length sequences and better represents the worst-case activation range.

## Cache Length

`maxCacheLength` is the maximum KV cache length recorded for runtime generation. It must be greater than or equal to the length values used by the deployment: `maxSequenceLength` for single-batch LLM (`runtime.batchSize=1`), and both `maxDataLength` and `maxCoreDataLength` for Batch LLM (`runtime.batchSize>1`).

When `maxCacheLength` is smaller than the intended prompt plus generated context, runtime generation will hit the cache limit. When it is much larger than required, the runtime memory budget can become unnecessarily large.

## Batch Size

The LLM runtime batch size is configured in `llm.attributes.runtime.batchSize`. Most edge deployments use `batchSize=1` for interactive generation. When `batchSize` is greater than `1`, compile-time `inferenceScheme` must be `single`, `global4` or `global8`.

This is different from vision-model batch inference that may use `multi` mode. For LLM and other KV-cache transformer models, the NPU distributes batched transformer work by splitting operations into one-core units. Each segment compiles under exactly one scenario, so `multi` and `all` are rejected for `LlmConfig` batch compilation — with an error naming the schemes that work, not a silent fallback. `global4` and `global8` are accepted, and were fixed to `single` before 1.3. For what the core modes mean in general, see {external+aries:doc}`ARIES Core Mode <core-mode>`.

The qb Runtime infer API also changes for Batch LLM. Instead of passing independent vision-style batch tensors, runtime code passes batch metadata such as `BatchParam` with the concatenated LLM input. See the Batch LLM entry in {external+runtime:doc}`qb Runtime Release Notes <en/release_note>` for the current API shape.

When using a batch size greater than `1`, keep these points in mind:

- `runtime.batchSize`;
- `maxDataLength` and `maxCoreDataLength`;
- runtime input packing and `BatchParam` metadata;
- target device memory budget, because KV cache memory is reserved for the configured batch size.

For Batch LLM, `maxSequenceLength` is not used as the single-call input limit. `maxDataLength` and `maxCoreDataLength` are used only when Batch LLM is enabled, and `maxDataLength` plays the role analogous to single-batch `maxSequenceLength`.

`maxDataLength` is the total data input length limit for one `infer` call. `maxCoreDataLength` is the per-KV-cache-session, per-single-core maximum token length that one `infer` call can process. In Batch LLM, each KV cache session is handled by one single core.

For example, if `maxCoreDataLength=128` and `maxDataLength=1024`, one `infer` call can process up to 1024 total tokens, while each session is limited to 128 tokens. This can be 128 tokens * 8 sessions or 64 tokens * 16 sessions, as long as the number of sessions does not exceed `runtime.batchSize`.

## KV Cache

The KV cache stores the key and value tensors produced by previous tokens. During autoregressive generation, the runtime appends the new token's K/V tensors to the cache and attention reads both the cached prefix and the current token.

This avoids recomputing the whole prompt for every generated token. It also means the compiled model and runtime metadata must agree on cache length, batch size, head layout, and positional encoding behavior.

If a model uses sliding-window attention or another bounded-cache mechanism, verify that the HuggingFace model path and generated MBLT express that cache policy correctly before producing a final MXQ.

## `llm` and `llm_fast` Presets

The built-in `llm` preset enables the LLM config and sets the default sequence/cache length to 4096:

| Setting | Value |
| --- | --- |
| `llm.attributes.maxSequenceLength` | `4096` |
| `llm.attributes.maxCacheLength` | `4096` |
| `calibration.mode` | `0` |
| `calibration.output` | `0` |
| `llm.attributes.calibration.useFullSeqLength` | `true` |
| `equivalentTransformation.{QK,UD,VO,SpinR1,SpinR2,OptimizeFFN}.apply` | `true` |

```python
mxq_compile(
    model=model,
    backend="torch",
    target_device="aries-rb",
    config_preset="llm",
    feed_dict={"input_ids": example["input_ids"]},
    calib_data_path=calib_path,
    save_path="model.mxq",
)
```

The `llm_fast` preset extends `llm` and skips accuracy-oriented optimization in exchange for faster compilation. It turns off the equivalent-transform passes that `llm` enables for transformer blocks — QK, UD, VO, SpinR1, SpinR2 and OptimizeFFN — and turns off full-sequence-length calibration. Neither preset enables HessianQuant; set `hessian_quant_config` explicitly if you want it.

Use `llm` as the conservative starting point. Use `llm_fast` when compile time matters more, then validate model quality on representative prompts.

## `multimodal` Preset

The `multimodal` preset is the configuration reference point for models with more than one input family, such as image tensors plus token tensors. It sets `calibration.method=3` and enables `llm.apply=true`.

The preset does not prescribe image or text preprocessing, input sizes, sequence lengths, or cache lengths. Define those values to match the exported vision and language submodels. For calibration, prepare data for every compiled input; for image-plus-text models, this may mean image tensors for the vision path and token, embedding, mask, or cache tensors for the language path.

Use the model-specific Python script rather than generic CLI examples because the graph usually combines multiple input contracts. The `multimodal` preset provides a starting configuration, but it does not replace model-specific export, preprocessing, and calibration code.

## LLM Quantization Settings

For detailed quantization sub-config descriptions and recommended configuration recipes, see [Model Quantization — Quantization Configuration](model_quantization.md#quantization-configuration). For how to use presets, config files, and `dump-config`, see [Compile Configuration](compile_configuration.md).

LLM quantization usually needs different precision choices for different transformer projections. The transformer weight fields under `BitConfig` let you control query, key, value, output, FFN, and head weights independently:

```python
from qbcompiler.configs import BitConfig, CalibrationConfig, SearchWeightScaleConfig

calibration_config = CalibrationConfig(mode=0, output=0)

bit_config = BitConfig(
    transformer=BitConfig.Transformer(
        weight=BitConfig.Transformer.Weight(
            query=4,
            key=4,
            value=8,
            output=4,
            ffn=4,
            head=8,
        ),
    ),
)

search_weight_scale_config = SearchWeightScaleConfig(
    apply=True,
    transformer=SearchWeightScaleConfig.Transformer(
        query=True,
        key=True,
        value=True,
        out=True,
        ffn=True,
    ),
)
```

A practical progression is:

- start with 8-bit transformer weights to establish a quality baseline;
- move query/key/output/FFN to 4-bit for size and bandwidth reduction;
- keep value and head at 8-bit if perplexity or generation quality drops;
- enable search weight scale when using aggressive low-bit settings;
- validate with task-level prompts, not only compile success.

## Dynamic RoPE and Dynamic Mask

`dynamicRope` and `dynamicMask` are runtime options inside `LlmConfig.Attributes.Runtime`.

`dynamicRope=True` makes RoPE position data dynamic at runtime instead of fully fixed at compile time. Use it when the deployment needs variable decode positions that cannot be represented well by a fixed compile-time position path.

`dynamicMask=True` makes the attention mask a runtime input. Use it when prompt lengths, padding patterns, or causal/padding mask combinations vary at runtime.

Dynamic settings can make one compiled package cover more runtime cases, but they also add runtime input requirements and can reduce optimization opportunities. Keep them disabled unless the deployment needs that flexibility.

## `splitBlocks` and `splitParts`

Large LLMs may be split into multiple MXQ files. This can help with very large transformer stacks, packaging constraints, or runtime scheduling.

`splitBlocks` gives explicit split points by transformer block index:

```json
{
  "splitBlocks": [4, 8, 12]
}
```

`splitParts` evenly divides transformer blocks into the requested number of parts:

```json
{
  "splitParts": 4
}
```

Use only one split strategy for a compile job. Prefer `splitParts` when blocks are uniform and you only need a simple even split. Prefer `splitBlocks` when the model has nonuniform blocks or when target memory measurements show that specific boundaries work better.

## End-to-End Example

The following example compiles a HuggingFace LLaMA-style model through `backend="torch"` using an explicit `LlmConfig`.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

from qbcompiler import mxq_compile
from qbcompiler.configs import (
    BitConfig,
    CalibrationConfig,
    LlmConfig,
    ResourceManagementConfig,
    SearchWeightScaleConfig,
)

model_id = "meta-llama/Llama-3.2-1B-Instruct"
calib_path = "/workspace/Llama-3.2-1B-Instruct-Wikipedia-en"
save_path = "./Llama-3.2-1B-Instruct.mxq"

model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)
example = tokenizer("Hello", return_tensors="pt")

resource_management_config = ResourceManagementConfig(
    weight_dtype="bfloat16",
    use_gpu_only_for_calibration=True,
)

llm_config = LlmConfig(
    apply=True,
    attributes=LlmConfig.Attributes(
        maxSequenceLength=4096,
        maxCacheLength=4096,
        calibration=LlmConfig.Attributes.Calibration(
            useFullSeqLength=True,
        ),
        runtime=LlmConfig.Attributes.Runtime(
            batchSize=1,
            dynamicRope=False,
            dynamicMask=False,
        ),
    ),
)

calibration_config = CalibrationConfig(mode=0, output=0)

bit_config = BitConfig(
    transformer=BitConfig.Transformer(
        weight=BitConfig.Transformer.Weight(
            query=8,
            key=8,
            value=8,
            output=8,
            ffn=8,
            head=8,
        ),
    ),
)

search_weight_scale_config = SearchWeightScaleConfig(
    apply=False,
    transformer=SearchWeightScaleConfig.Transformer(
        query=True,
        key=True,
        value=True,
        out=True,
        ffn=True,
    ),
)

mxq_compile(
    model=model,
    backend="torch",
    target_device="aries-rb",
    feed_dict={"input_ids": example["input_ids"]},
    calib_data_path=calib_path,
    save_path=save_path,
    device="gpu",
    resource_management_config=resource_management_config,
    calibration_config=calibration_config,
    bit_config=bit_config,
    search_weight_scale_config=search_weight_scale_config,
    llm_config=llm_config,
)
```

For larger models, add `splitParts` or `splitBlocks` through a `CompileConfig` or a JSON/YAML config file, then validate each produced MXQ with the runtime flow used by the application.

## LLM Calibration Data

For many LLMs, calibration uses embedded tensors rather than token IDs because embedding can be handled outside the NPU graph. The typical workflow is:

1. Prepare text representative of the deployment prompts.
2. Tokenize the text with the model tokenizer.
3. Pass the tokenized input through the model embedding layer.
4. Save the resulting tensors as the calibration dataset.

The calibration tensor shapes must match the compiled decoder body input. If the model frontend selects a specific input form, the calibration tensors must follow the same contract.

If compilation fails, reduce sequence and cache length first to confirm that the model path is valid, then scale back up within memory limits.

## Multimodal (VLM) Compilation

HuggingFace multimodal (VLM) models are compiled with dedicated Python scripts through `mblt_compile` and `mxq_compile` rather than generic CLI examples. Start from the tutorial code or reference implementation for each model, then adapt the script and configuration to the exported model contract. See the reference implementations at https://github.com/mobilint.

The [`multimodal` preset](#multimodal-preset) is the configuration starting point for these scripts.

Practical guidance:

- Compile and validate the vision encoder alone when possible.
- Compile and validate the LLM path alone when possible.
- Combine the graph only after each subpath has a clear shape and preprocessing contract.
- Move tokenizer logic, image decoding, prompt formatting, and other Python-level preprocessing outside the compiled graph.
- Prepare calibration data for every compiled input. Sample names, shapes, and ordering must match the original model inputs.
- Use CPU Offloading only for bounded graph regions that the deployment runtime supports.
