<!-- 출처: https://docs.mobilint.com/runtime/v1.4/en/release_note.html (원본) -->

# Release Notes

## v1.4.0

**Release date:** August 21, 2026
**Type:** Minor

The mbltml NPU management library, a `mobilint-cli top` monitor, one runtime build for every NPU device, and beta support for `NPUData` and mixed core modes.

### Added

- **[Highlight]** **mbltml — NPU management library** — a new library that observes the state of every Mobilint NPU in the system: driver and firmware versions, temperature, clocks, power rails, memory usage, utilization, per-core activity, and the processes currently holding a device. It is distributed as `mobilint-ml` and versioned in lockstep with qb Runtime, so mbltml v1.4 pairs with qb Runtime v1.4.

  ```bash
  pip install mobilint-ml         # PyPI
  sudo apt install mobilint-ml    # Debian, Ubuntu
  sudo dnf install mobilint-ml    # RHEL, Rocky Linux
  ```

  The module is `mbltml`, not `mobilint-ml`: use `import mbltml` from Python and `#include <mbltml/mbltml.h>` from C or C++. Every query takes a target device type and a device number. Device numbers are assigned per device type, so an `aries-rb` and a `regulus-ra` can both be device #0.

  ```cpp
  // C API example (the header is usable from both C and C++)
  #include <mbltml/mbltml.h>
  #include <stdio.h>

  int main() {
      if (mbltmlInit() != MBLTML_SUCCESS) {
          return 1;
      }

      unsigned int count = 0;
      mbltmlGetTargetDeviceCount(MBLTML_TARGET_DEVICE_ARIES_RB, &count);

      for (int dev_no = 0; dev_no < (int)count; ++dev_no) {
          int temperature = 0;
          double utilization = 0.0;
          mbltmlGetTemperature(MBLTML_TARGET_DEVICE_ARIES_RB, dev_no, &temperature);
          mbltmlGetTotalUtilization(MBLTML_TARGET_DEVICE_ARIES_RB, dev_no, &utilization);
          printf("aries-rb #%d: %d C, utilization %.2f\n", dev_no, temperature, utilization);
      }

      mbltmlShutdown();
      return 0;
  }
  ```

  ```python
  # Python example
  import mbltml

  mbltml.mbltmlInit()
  try:
      for dev_no in range(mbltml.mbltmlGetTargetDeviceCount(mbltml.MBLTML_TARGET_DEVICE_ARIES_RB)):
          temperature = mbltml.mbltmlGetTemperature(mbltml.MBLTML_TARGET_DEVICE_ARIES_RB, dev_no)
          utilization = mbltml.mbltmlGetTotalUtilization(mbltml.MBLTML_TARGET_DEVICE_ARIES_RB, dev_no)
          print(f"aries-rb #{dev_no}: {temperature} C, utilization {utilization:.2f}")
  finally:
      mbltml.mbltmlShutdown()
  ```

  - [mbltml C API reference](../doxygen/html_en_mbltml/group__MbltmlCAPI)
  - [mbltml Python API reference](../doxygen/html_en_mbltml/group__MbltmlPyAPI)

- **[Highlight]** **One runtime build for every NPU device** — a single qb Runtime artifact now supports every target device. {doxylink}`Accelerator <mobilint::Accelerator>` accepts a target device name — `"aries-rb"`, `"regulus-ra"`, `"regulus-rb"`, `"regulus-ra-usb"`, or `"regulus-rb-usb"` — alongside the device number, and names are case-insensitive.

  The aliases `"auto"`, `"aries"`, `"regulus"`, and `"regulus-usb"` resolve to a concrete target device, but only when exactly one matching kind of device is attached. An ambiguous alias fails instead of guessing, so systems holding more than one kind of NPU must name the device explicitly.

  ```cpp
  // C++ example
  mobilint::StatusCode sc;

  // What is attached: each target device name, and its device numbers.
  for (const auto& name : mobilint::getAvailableDevices()) {
      for (int dev_no : mobilint::getAvailableDeviceNumbers(name)) {
          printf("%s #%d\n", name.c_str(), dev_no);
      }
  }

  auto acc = mobilint::Accelerator::create("aries-rb", 0, sc);
  if (!sc) {
      fprintf(stderr, "Error code %d\n", int(sc));
      exit(1);
  }
  printf("opened %s\n", acc->getDeviceName().c_str());
  ```

  ```python
  # Python example
  for name in qbruntime.get_available_devices():
      print(name, qbruntime.get_available_device_numbers(name))

  acc = qbruntime.Accelerator("aries-rb", 0)
  print(acc.get_device_name())
  ```

  Three helpers come with it: `getAvailableDevices()` lists the detected target device names, `getAvailableDeviceNumbers(device_name)` narrows the device numbers to one name or alias, and `Accelerator::getDeviceName()` reports which target device an alias resolved to.

- **`mobilint-cli top`** — continuously monitor ARIES NPU status and resource usage from the terminal. The command runs the `mobilint-ctrl-cli` monitor that already shipped with the utility package, so there is nothing extra to install or launch. Currently, it is not available in REGULUS (SoC) builds of `mobilint-cli`.

  ```bash
  mobilint-cli top
  ```

  See [Utility Usage](utility_usage.md) for the full `mobilint-cli` command list.

- **(Beta) `NPUData`** — {doxylink}`NPUData <mobilint::NPUData>` is a handle to a single model input or output tensor that owns its storage and moves between host (CPU) and NPU memory on request. It makes the residency of inference data explicit, which matters when several NPUs — or several models on one NPU — pass tensors to each other.

  Acquire one from a launched model with `acquireInputNPUData()` / `acquireOutputNPUData()` (`acquire_input_npu_data()` / `acquire_output_npu_data()` in Python), fill it while it is on the host, then `launch()` it onto an accelerator and infer. Call `cpu()` to bring a result back before you read it.

  ```cpp
  // C++ example
  mobilint::StatusCode sc;

  mobilint::NPUData in = model->acquireInputNPUData({224, 224, 3}, 0, false, sc);
  float* host = in.data<float>(sc);
  std::copy(image.begin(), image.end(), host);

  in.launch(*acc);  // upload once; the tensor now lives in NPU memory

  std::vector<mobilint::NPUData> inputs = {in};
  std::vector<mobilint::NPUData> outputs = model->infer(inputs, sc);

  outputs[0].cpu();  // bring the result back to the host
  const float* result = outputs[0].data<float>(sc);
  ```

  ```python
  # Python example
  npu_in = model.acquire_input_npu_data([224, 224, 3], idx=0, upload=False)
  npu_in[...] = image      # writable as a numpy view while it is on the CPU
  npu_in.launch(acc)       # upload once; the tensor now lives in NPU memory

  outputs = model.infer_npu_data([npu_in])
  print(outputs[0].dev_no, outputs[0].hardware_name)

  outputs[0].cpu()         # bring the result back to the host
  result = outputs[0][...]
  ```

  Every input and output of one call must share the same residency — all on the host, or all on the NPU. When all of them are already on the NPU, inference performs no reposition and no host copy. An NPU-resident tensor can also go to any model that expects the same shape and element type on the same accelerator, so chained models exchange activations without a host round-trip. This is the pattern the API was built for, in Mixture-of-Experts style graphs.

  `NPUData` targets advanced use rather than typical inference, and applies only to single-NPU-op (non-CPU-offload), relocatable (MXQv7+) models.

- **(Beta) Bundles of different core modes in one MXQ** — {doxylink}`setAutoCoreMode() <mobilint::ModelConfig::setAutoCoreMode()>` now resolves the core mode for each bundle separately instead of once for the whole model. The rule is per bundle: every bundle must carry exactly one core mode, and bundles may carry different ones from each other. Such an MXQ previously could not use `CoreMode::Auto`; it now runs as a single model. The feature was developed for BatchLLM optimization, and `ModelConfig`'s default constructor already selects `CoreMode::Auto`, so no code change is needed to opt in.

  ```cpp
  // C++ example
  mobilint::ModelConfig cfg;  // CoreMode::Auto by default
  cfg.setAutoCoreMode();
  auto model = mobilint::Model::create(MXQ_FILE_PATH, cfg, sc);
  ```

  ```python
  # Python example
  cfg = qbruntime.ModelConfig()
  cfg.set_auto_core_mode()
  model = qbruntime.Model(MXQ_FILE_PATH, cfg)
  ```

  A single bundle compiled for several core modes at once — for example with the compiler's `scheme="all"` — is still ambiguous, so `CoreMode::Auto` cannot resolve it. Model creation fails with a message naming that bundle; set the core mode explicitly for such an MXQ.

### Revised

- Fixed a static initialization order fiasco (SIOF) crash in the Windows static-library distribution.

### Removed

- `ModelConfig::early_latencies` and `ModelConfig::finish_latencies` — deprecated fields that had no effect.

## v1.3.2

**Release date:** July 16, 2026
**Type:** Patch

### Revised

- **Resolved v1.3.1 known issue** — Fixed the failure that could occur when running large models on Windows.

## v1.3.1

**Release date:** July 9, 2026
**Type:** Minor

16-bit integer support, SIMD level selection, a configurable NPU timeout, DNF (RPM) packages for RedHat-based OS, and reliability and performance improvements.

### Added

- **16-bit integer support** — Runtime now supports 16-bit integers as an internal data type. Model input and output data types are not affected.
- **DNF (RPM) packages for RedHat-based OS** — `mobilint-qb-runtime` and `mobilint-cli` can now be installed from the Mobilint DNF repository on RHEL, Rocky Linux (x86_64, aarch64). See [Runtime Library Installation](installing_runtime_library.md).
- **SIMD level selection** — Scale and transpose operations now support AVX-512. By default, qb Runtime selects the fastest SIMD level the system supports; set the `QBRUNTIME_SIMD_LEVEL` environment variable (`auto`, `avx512`, `avx2`, or `sse2`) to override it.
- **Configurable NPU timeout** — Set the `QBRUNTIME_NPU_TIMEOUT_MS` environment variable to control how long qb Runtime waits for the NPU before reporting a timeout.

### Revised

- `inferSpeedrun` no longer crashes when used with models that accept variable-length input.
- Fixed issues affecting the `inferAsync` API.
- `Model::dispose()` no longer waits 3 seconds when `Model::releaseBuffer()` was not called.
- Improved inference and data-transfer performance on Linux.

### Known Issues

- **Large models on Windows** — Some large models, including 7B LLMs, may fail to run on Windows. A fix is in progress and planned for v1.3.2.

## v1.2.0

**Release date:** April 2, 2026
**Type:** Minor

Adds Batch LLM support.

### Added

- **BatchParam** — a new struct {doxylink}`BatchParam <mobilint::BatchParam>` for Batch LLM inference. It holds the per-batch information needed during inference:
    - `sequence_length` : the sequence length for each batch.
    - `cache_size` : the cache size each batch will use.
    - `cache_id` : the cache identifier for each batch. All inputs in the same context must share one cache ID, and the value must be within the model's maximum batch count.

  To run Batch LLM, concatenate multiple inputs into a single input — along the `seq_len` dimension when the shape is `(1, seq_len, hidden_dim)` — then pass a `BatchParam` for each input:

  ```python
  import qbruntime
  import numpy as np

  ## Check the maximum batch count supported by the model.
  print(model.get_cache_infos()[0].num_batches)

  ## Concatenate inputs along the 2nd dimension (axis=1).
  batch_input = np.concatenate([input0, input1], axis=1)

  ## qbruntime.BatchParam(sequence_length, cache_size, cache_id)
  batch_params = [
      qbruntime.BatchParam(10, 0, 0),
      qbruntime.BatchParam(80, 0, 1),
  ]
  res = model.infer([batch_input], params=batch_params)

  batch_params2 = [
      qbruntime.BatchParam(1, 10, 0),
      qbruntime.BatchParam(1, 80, 1),
  ]
  res = model.infer(res, params=batch_params2)
  ```

### Known Issues

- Running LLM models on ARM (aarch64) systems may fail with a "Bus Error". Present since v1.1.0; a driver patch is planned.

## v1.1.0

**Release date:** March 23, 2026
**Type:** Minor

Automatic core-mode selection, data-type query APIs, and performance optimizations.

### Added

- **`CoreMode::Auto`** — the runtime auto-selects the available core mode from the MXQ. Set `CoreMode::Auto` in your `ModelConfig` (the default constructor already uses it), so non-default modes such as `Multi`, `Global4`, and `Global8` no longer need manual construction. See {doxylink}`setAutoCoreMode() <mobilint::ModelConfig::setAutoCoreMode()>`.
- `getModelInputDataType()` / `getModelOutputDataType()` — query a model's input and output data types at runtime.
- `getAvailableDeviceNumbers()` — retrieve the list of available NPU device numbers.

```{note}
If the MXQ was compiled with a flag like `scheme="all"` that produces multiple core modes, you must still select the core mode manually.
```

### Revised

- REGULUS now uses the dynamic-allocation approach introduced in v1.0.0, for a consistent usage pattern.
- Improved data-transfer performance to NPU devices on Windows.
- Optimized internal type conversion.
- Fixed a compile error caused by `std::filesystem` on GCC versions below 9.
- Fixed an intermittent deadlock in certain models.
- **[Breaking]** The supported REGULUS driver revision changes from REV0 to REV1.

### Known Issues

- Running LLM models on ARM (aarch64) systems may fail with a "Bus Error". A driver patch is planned.

```{seealso}
For the complete changelog, see the [Changelog](CHANGELOG.md) page.
```

## v1.0.0

**Release date:** January 31, 2026
**Type:** Major

A major release focused on scalability, consistency, and a structural refactor for future expansion. To upgrade, follow the [Migration Guide](migration_guide.md).

### Added

- **uint8 inference** — uint8 quantized models can be compiled with qb Compiler and executed by qb Runtime, reducing CPU overhead during preprocessing for models with uint8 inputs.
- **Activation slots** — `setActivationSlots(int num)` (C++) and `set_activation_slots(num)` (Python) tune pipelining between NPU inference and data transfer. More slots use more NPU memory but improve throughput in multithreaded workloads.

```{note}
For models that use cache (e.g., LLMs), the activation slot count is currently limited to 1.
```

### Revised

- **[Highlight]** Model-count limit removed — models compiled with the latest qb Compiler (MXQv7) load and run concurrently within available DRAM, regardless of compile-time core mode. This helps multi-model services, mixed core-mode execution, and large models such as LLMs, with no code changes.
- **[Breaking]** SDK qb naming unified — runtime library `maccel` → **qb Runtime**, compiler `qubee` → **qb Compiler**. Packages, headers, and module names changed accordingly.

### Removed

- Legacy packages (`mobilint-npu-runtime`, `aries-driver`) are no longer maintained. See the [Migration Guide](migration_guide.md).
