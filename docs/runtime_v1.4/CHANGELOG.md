<!-- 출처: https://docs.mobilint.com/runtime/v1.4/en/CHANGELOG.html (원본) -->

# Changelog

## [v1.4.0] - 2026-08-21

### Added

- Validate NPU-only model I/O order against the NPU op layer (!295)
- Support multiple REGULUS-USB devices on Windows (!304)
- CMakeLists.txt : Add MSVC /MT flag for static build (!302)
- `QBRUTNIME_CLANG_TIDY` compile option added (!306)
- Support beta-version of NPUData class (!309)
- `CoreMode::Auto` now detects the CoreMode of each bundle independently, so bundles of
  different CoreModes can run on the same model (!311)

### Fixed

- Remove deprecated `latencies` related APIs (!285)
- Fix static initialization order crash in Windows (!296)
- Validate SequenceLength without input axes at `Model::create` (!293)
- Export `statusCodeToString` from the shared library (!301)
- Fix max sequence length (!312)

### Changed

- Migrate HW selection to runtime (!276)
- `Model` class no longer creates a temporary directory (!291)
- `regulus-rb` MXQ no longer runs on a `regulus-rb-usb` device, and vice versa (!308)

### Removed

- Remove windows release resources (!302)

## [v1.3.2] - 2026-07-10

### Fixed

- Fix wrong I/O base address mapping in `initPinnedInferBundlePrepManager` by filtering out DDR_IMEM entries from the relocation table (!288)

## [v1.3.1] - 2026-07-09

### Added

- Support avx512 for scale and transpose operations (!268)
- Add `QBRUNTIME_NPU_TIMEOUT_MS` environment variable (!274)
- Support cmake Config/Target for find_package() (!284)

### Fixed

- Change timer module from `system_clock` to `steady_clock` to prevent timeout malfunction (!273)
- Fix segfault when running `inferSpeedrun` of model using variable-length input (!277, !283)
- Fix wrong initRmem skip for multi-shape model (!278)
- Fix failure of releasing activation index (!278)
- Fix bugs of RNN/LSTM models on Regulus SoC (!282)
- GCC8 build failure resolved (!281)

### Removed

- Remove legacy Aries(v1) driver/product and NOC interleaving support (!264)
- Support Asymmetric Quantization for Int8 NPU DataType (!265)
- Remove `QBRUNTIME_USE_SSE2_FOR_SIMD` option from root CMakeLists.txt (!268)

## [v1.3.0] - 2026-06-12

### Added

- Support custom 16bit integer type, BInt16 (!248)
- Support PinnedMemory for Regulus SoC (!263)
- Support Initstart - Rmem attachment & Rmem broadcasting (!261)

### Fixed

- Support regulus-usb in getAvailableDevices() on Linux (!258)
- Fix inferAsync API (!266)

### Changed

- Optimized DMA process on Linux (!247)
- Don't wait 3 seconds on Model::dispose() without Model::releaseBuffer() (!249)
- Reduced data copy during inference on Regulus NPU (!255)
- Refactor `infer` to resolve reposition-dma bottleneck (!266)

## [v1.2.0] - 2026-04-02

### Added

- Support regulus-usb-driver on Linux (!227)
- Add revision verification for regulus-usb-driver on Linux (!238)
- Support regulus-usb-driver on Windows (!239)
- Add resnet50_static.sln for Windows deployment (!241)
- Support Batch-LLM model (!240)

### Fixed

- Fix `getOutputScales` to get last bundle's scale, not first (!233)
- Fix a bug of dumpCacheMemory & loadCacheMemory for primitive LLM Models (!235)

## [v1.1.0] - 2026-03-23

### Added

- type: Supports `CoreMode::Auto` (!208)
- model: Adds `getModelInputDataType` & `getModelOutputDataType` (!220)
- type: Add `getDeviceCount()` (!218)
- Support relocation based dynamic allocation on Regulus (!216)

### Fixed

- model_impl: Fix a bug `CoreMode::Auto` to select only a single core/cluster on MXQv7 (!215)
- model_impl: Fix a compile error driven by `std::filesystem` on GCC < 9 (!214)
- npu_task_scheduler: Fix deadlock in activation index allocation for models containing InNPU (!213)
- relocation_helper: Using move semantic instead of copy (!222)

### Changed

- npu_watchdog: Optimize `initRmem` to copy all RMEMs at once (!217)
- reposition: Optimize reposition by MemoryPool (!224)
- simd_x86_64_scale: Optimize to save SIMD output by `_mm256_stream_*` (!224)
- ndarray: Allocate with 32-Byte alignment to optimize reposition (!224)
- Update supported revision number of regulus-driver from REV0 to REV1 (!216)
- Rename `LogLevel` enum from UPPER_CASE to PascalCase (!232)

## [v1.0.0] - 2026-01-29

### Added

- model.py: Support infer APIs that can specify HWC or CHW (!187)
- Support relocation based dynamic allocation and 1-core-multi-model runtime (!188)
- CMakeLists.txt : Add MSVC multi-core build flag (!196)

### Fixed

- CMakeLists.txt : Fix SIMD not applied on Windows (!196)
- Fix a bug of CacheInfo update failure after launch (!199)
- Fix issue related to InNPU IOType input (!200)

### Changed

- Rename library from maccel to qbruntime (!193, !196)

### Removed

- model: Remove resetCacheMemory (!199)
- Remove model claim/unclaim (!188)

## [v0.31.0] - 2026-01-16

### Added

- Sync op_desc.cc with unsupported.cc in quantizer (!174)
- env_var: Support `MACCEL_VISIBLE_DEVICES` (!180)
- Add `--export-sample-dir` option to testinfer (!181)
- Add driver revision number matching to Regulus (!189)
- Support UINT8 Models (!185)

### Fixed

### Changed

- Modify the build to get the version from the Git tag (!165)
- Windows: Update `build_release.bat` and `build_wheel.bat` (!169)
- aries: Allow MMIO-based write to NPU SRAM (!172)
- Supports revised regulus-npu-driver (!178)

### Removed

- Remove documents from maccel repository (!141)
- Remove `build_wheel.sh` and `build_wheel.bat` (!179)
- Revert "Support multi-batch inference of CPU-Offload models" (!191)

## [v0.30.1] - 2025-10-20

### Added

- CI: build & ctest check added (!131)
- Check MXQ core consistency (!150)
- Add @note comments for `inferAsync` family (!154)
- Show supported cores in error message when ModelConfig mismatches MXQ file (!153)

### Fixed

- Fix issue reading files over 2GB on Windows (!148)
- npu_watchdog: Fix inference error in activation buffer model with Global-Core (!152)

### Changed

- Supports Windows driver rev#1 : S/G DMA, core claim/unclaim (!166)

### Removed

- Remove the Trace API from `acc` (!164)

## [v0.30.0] - 2025-08-29

### Added

- Supports int8 asynchronous inference (!120)
- Add getModelSummary function (!125)
- Send model's memory usage to aries2-driver at Model::launch (!145)
- Add Python comments for doxygen generation (!144)
- Support BiRNN, BiLSTM (!143)

### Fixed

- Fix compile error by `std::packaged_task` in Windows (!129)
- Fix out of index error of async API unittests (!129)
- Windows: Restrict each NPU core to load only one model at a time (!130)
- Fix Already-launched `Model` error handling (!140)
- Fix a bug where the variable-length model only accepted a fixed shape (!143)

### Changed

- Update README.md (!127)
- Change Python API error message (!147)
- Add numpy dependency to install maccel whl file (!138)

## [v0.29.0] - 2025-07-25

### Added

- npu_task_scheduler: Add `NPUTaskScheduler` (!101)
- type: Add `CacheInfo` (!119)
- Support multi input shape model (!118)

### Fixed

- Fix testinfer.cc help message (!111)
- model_impl: Fix the logic calculating 2nd activation space when a model has multi-bundle (!114)
- acc_impl: Block to upload `IMEM_INIT`/`IMEM_INITSTART` at `Model::launch` for Regulus (!117)
- Fix compiler error caused by designated initializer in Ubuntu18.04 (!121)

### Changed

- Enable additional compiler warnings and fix triggered warnings (!108)
- model: arguments of `dumpCacheMemory` and `loadCacheMemory` change from file path to directory path (!119)
- model: arguments of `dumpCacheMemory` and `loadCacheMemory` change from a buffer to buffer vector (!119)

## [v0.28.0] - 2025-06-25

### Added

- Support model using framebuffer logic (!109)
- Support model using command queue inference (!109)

### Fixed

- Fix bugs where models with global-core failed to run when EXPERIMENTAL_READ_ALL_OUTPUTS_AT_ONCE=ON (!107)

### Changed

- model_impl: Use CacheRearrange IMEM in `Model::resetCacheMemory` (!100)

## [v0.27.0] - 2025-06-17

### Added

- Support Cache Rearrange (Tail-Move, Tail-Filter) by Supplementary Infer (!84)
- Add async API python wrapper (!87)
- Support `cmake --install` (!93)
- Add header installation for `cmake --install` (!94)
- Add INSTALL_MACCEL cmake option (!99)
- Add a post-build event to `resnet50.sln` to copy a DLL file (!98)

### Fixed

- Fix bugs related to `SequenceLength` (!95)

### Changed

- Use prettified argv[0] in testinfer help message (!91)

### Removed

- Remove exception handling for dev_no in sleep device (!90)

## [v0.26.0] - 2025-05-20

### Added

- Windows: Support building with custom OpenBLAS path (!80)
- Implement Async API for C++ (!42)
- Support multi-batch inference of CPU-Offload models (!78)
- Support MXQv5 and models with multiple sequence lengths (!86)

### Fixed

- Resolve build error on Ubuntu 18.04 caused by `_mm256_set_m128i` (!73)
- Fix markdown syntax errors and content errors for doxygen documentation (!77)
- Fix CMake error of `add_dependencies` during cross-compilation (!83)
- Fixed error in infer<float*> and infer<int8_t*> about reversed input shapes (!85)

### Changed

- Improve the Python wheel build process - remove unnecessary logs and ensure rebuilding whl every time (!71)
- Refactor unittests for the Python API (!72)
- Update CMAKE_CXX_STANDARD from 14 to 17 (!75)
- reposition_test: Refactor all HWC<->CHW transpose code into unified functions (!79)
- CMakeLists.txt: Change the default value of MACCEL_CPU_OFFLOAD to bonfire (!80)
- Improve the execution time of the `Model::create` (!81)
- Rename Ticket to Future for consistency with common async terminology (!82)

### Removed

- Remove `lib_info.py` and `resnet50_accuracy.py` (!72)

## [v0.25.0] - 2025-04-03

### Added

- Support FP16 & FP32 NPU Data Type (!38)
- testinfer: Support a help message to describe arguments and how to use (!41)
- Support `Global4` and `Global8` core modes (!40)
- pymaccel: Add `set_global4_core_mode`, `set_global8_core_mode`, and `set_multi_core_mode` to python API (!51)
- Support LLM Model (!48)
- pymaccel: Add `max_height`, `max_width`, `max_channel`, and `max_cache_size` in `BufferInfo` (!58)
- Support `Multi` core mode (!56)
- Add Python unittests to verify the Python API (!65)
- Apply doxygen to maccel (!66)

### Fixed

- Windows: Fix updating temperature to new union sturcture (!43)
- Windows: Release `pMem` after updating memory consumption (!43)
- tensor_utils: Fix `moveToNDArray` to execute copy if libtorch is used for CPU offload (!45)
- Windows: Distinguish the target architecture in `.whl` file between MinGW Python and native Windows Python (!44)
- Windows: Fix build errors in the English version of Windows by removing Korean comments from `build_wheel.bat` and `build_release.bat` (!49)
- interleaving: Fix a bug where the return value of `interleave` was calculated larger than intended (!46)
- acc_impl: Fix a bug where IMEM is overwritten by IMEM_CACHE_REARRANGE in `Model::launch` (!50)
- Fix `NDArrayData` to have thead-safe refcount by using `std::shared_ptr` (!52)
- simd_x86_64_transpose: Add missing `defined(__GNUC__) &&` to prevent compilation in MSVC (!55)
- npu_op_desc_test: Fix the failure of `NPUOpDescTest.getRmemAddresses` in Aries1 (!57)
- pymaccel: Fix `get_latency_set_policy` being overlapped by `get_latency_consumed` (!58)
- testinfer: Fix `set_global8_core_mode` being overlapped by `set_global4_core_mode` in Python (!58)
- Fix the logic for calculating DDR memory usage to avoid referencing `hwdep` and to support multi-core mode (!62)
- Windows: Fix to pass the correct Aries1 DDR usage to Windows monitoring tools (!63)
- Fix Python APIs - add `__repr__`, `is None`, `reposition_outputs`, `np.ascontiguousarray`, and more (!67)
- test_model: Fix `test_checkInferConsistency` to correct comparison (!67)

### Changed

- Windows: power monitor change to power/voltage/current (!47)
- driver: Remove the 4KB margin in allocHostMemory due to an interleaving bug (!46)
- pymaccel: Change the notation of some methods in `ModelConfig` from camelCase to snake_case (!51)
- Windows: Update Windows codes to match Windows driver v1.6 (!61)
- ndarray: Attach `noexcept` to move constructor/assignment of `NDArrayData` (!64)
- Make `ModelConfig` API more user-friendly - Add `setSingleCoreMode` (!59)
- Revise the Python API wrapper of maccel (!65)
- Improve docs - Markdown formatting, Python `ModelConfig` example in advanced_usage, and more (!68)

### Removed

- model_impl: Remove redundant trace events in  `inferBufferOutput` and `inferSpeedrun` (!37)
- Remove `packed` logic (!43)

## [v0.24.0] - 2024-02-12

### Added

- Windows: Tracks memory usage of Aries and sends it to the Windows monitoring tool (!39)

### Fixed

- Modify the naming of .whl files to properly reflect the target platform, such as Windows or Linux (!36)

## [v0.23.0] - 2024-01-24

### Added

- type.h: Add option in ModelConfig to force a single NPU bundle execution (!29)
- Support FP16, FP32 Output NPU Buffer (!34)

### Fixed

- type.h: Support Global core mode (!28)

## [v0.22.0] - 2024-01-23

### Added

- Support Aries2 on maccel(!26)
- Windows: Add aries performance monitoring (!30)

### Fixed

- Fix build error on Windows by using findPythonInterp in MSVC (!31)

### Changed

- Windows: Send 8->1->0 in postInfer on Windows (!32)

## [v0.21.0] - 2025-01-17

### Added

- init.cc: Add initializer to set default LogLevel according to MACCEL_LOG_LEVEL (!9)
- model: Add `shape` parameter to `std::vector<T*>` infer api (!17)
- Support RNN/LSTM models with fixed sequence length inputs (!17)
- testinfer: Add `seq-sizes` argument for variable length inputs (!17)
- CMakeLists.txt: Automatically generate libmaccel.so* symbolic links based on the `SOVERSION` and `VERSION` extracted from the Git tag (!19)
- Windows: Support Aries1/2 on Windows (!24)

### Fixed

- reposition: fix logic of need_repos when shape is kept but reposition occurs (!11)
- Modify the naming of .so and .whl files to appropriately reflect the target architecture (!22)
- Fix whl file name error by adding `linux_` prefix (!25)

### Changed

- Support multi-model on single NPU core for Regulus (!6)
- acc: Revise Accelerator::getCoreList to retrieve all available cores from driver (!10)
- regulus: Fix regulus timeout unit from nsec to mesc (!15)
- CMakeLists.txt: Add a cmake flag `MACCEL_GLIBCXX_DEBUG` to apply compile option `D_GLIBCXX_DEBUG` (!8)
- npu_watchdog: dumpDDRBin to dump DDR for each bundle & sequence_index (!17)
- CMakeLists.txt: Remove Warning from FindPython (!23)

## [v0.20.0] - 2024-11-15

### Added

- op_desc.cc: Add support for additional CPU offload operations (!3)
- type: Add `CoreAllocationPolicy` to support automatic NPU core allocation (!10)

### Fixed

- CMakeLists.txt: Consider CMAKE_BUILD_TYPE as Release, when it is not specified (!7)

### Changed

- op_desc.cc: Update implementations for some of CPU offload operations (!3)
- CMakeLists.txt: CMAKE_BUILD_TYPE=Release limits LogLevel to INFO (!7)

## [v0.19.0] - 2024-09-12

### Added

- simd_x86_64_scale: Implement SIMD(AVX2, SSE2)-based scale functions in x86-64 (#432, #434)
- simd_x86_64_transpose: Implement SIMD(AVX2, SSE2)-based transpose functions in x86-64 (#434)
- transpose: Implement transpose functions for inferCHW (#434)
- simd_aarch64_scale: Implement SIMD(NEON)-based scale functions in ARM64 (#435)
- simd_aarch64_transpose: Implement SIMD(NEON)-based transpose functions in ARM64 (#435)
- Add definitions of platform which represents host's architecture & SIMD (#441)
- aries_win: Implement `postInfer` for Windows (#446)
- Support multi-card in Windows (#448, #450)
- Support Regulus NPU (#449)
- model_impl : Implement `runNPUModelTest` for use in `inferSpeedrun` and `inferOutputDiff` of _Thread-Benchmark_ (#462)
- Support MXQv3 (!2)

### Fixed

- reposition: Fix calcReposIndicesDefaultBase when `original_size != reshaped_size` (#437)
- reposition_test: Fix vector out of bound in scale_list (#438)
- Update SIMD Option for MSVC (#441)
- Fix some unittests to support MSVC (#446)
- win_ddk: Fix pure function for MSVC (#453)

### Changed

- reposition: Use SIMD for float repositions (#432, #434, #435)
- PCIeDriver: Exclude RiscV area in Windows heap allocator (#443)
- reposition: Use SIMD-based transpose for CHW int8_t data (#442)
- npu_watchdog: Unify `postInfer` in both Windows and Linux (#446)
- reposition: MXQ compiled by default use efficient reposition (#457)
- reposition: repositionFloat & repositionInt are integrated (#460)

### Removed

- reposition: Remove OpenMP (#447)
- reposition: Remove default reposition (#457)

## [v0.18.0] - 2024-04-30

### Changed

- reposition: Revise `need_repos` condition to compare size of original and buffer (#431)

## [v0.17.0] - 2024-04-12

### Added

- testinfer: Add `batch-size`, `num-cores` options (#414)
- pymaccel: Add `set_log_level`, `start_tracing_events`, `stop_tracing_events` (#418)
- pymaccel: Implement `maccel.load()` API in Python (#422)
- model: Implement `acquireInputBuffers`, `acquireOutputputBuffers`, `releaseBuffers` (#428)
- model: Implement new infer API return multi-batch output by reference (#428)
- model: Implement new `repositionOutputs` which use `vector<vector<float>>& output` (#428)
- model: Implement `inferBufferToFloat` (#428)

### Fixed

- npu_watchdog: Fix default time duration overflow (#425, #426)
- resnet50_test_cc: Fix wrong comparison & scale and add diff test (#428)
- reposition: Fix fillgap logic (#429)

### Changed

- aries: Determine to use interrupt mode by `ARIES_IOC_GET_SIGNAL_TYPE` ioctl rather than driver version (#417)
- model_impl: Skip benchmark for re-launched model (#421)
- npu_watchdog: Change Default timeout from 1s to 10s (#424)
- reposition: Use `scale.scale` when scale is uniform (#420)

## [v0.16.0] - 2024-02-20

### Added

- build_wheel: MSVC provides python wheel (#409)

### Fixed

- Fix pymaccel build error from latest setuptools version (#412)
- reposition: Fix ch-wise scale bug in inferCHW (#415)
- Fix input shape check for batch in Python (#416)

### Changed

- aries: Enhance robustness of pread/pwrite usage by apply loop (#413)

### Removed

- resnet50_msvc: Remove `.TestDrive`, `PCIeDriverSystem.dll` (#409)
- VERSION: Remove `VERSION` (#412)

## [v0.15.0] - 2024-01-03

### Added

- testinfer: Implement inferBuffer infer-api
- acc_impl: Add lock_guard for AcceleratorImpl
- pymaccel: Implement constructor that create and launch at the same time in Python (#391)
- model_impl, npu_watchdog: Implement IMEM_INIT, IMEM_INITSTART

### Fixed

- memory_pool: Fix not to wait forever when model.dispose() called without releaseBuffer
- reposition: Apply __pragma for MSVC (#393)
- aries_win: Addtional allocate 4KB host memory to fix interleaving bug (#396)
- Fix default log level bug in release build
- model_impl: Fix segfault in user_outputs by rollback of resize removal (#401)

### Changed

- testinfer: Modularize doMain by implementing processInputs, processOutputs, printSummary
- model_desc: Revise model shape to exclude batch-size (#392)
- model_desc: checkIfInputShapeMatchAndEnsureBatchDim -> doesInputShapeMatch (#392)
- op_desc: Update CPU offload code (#386)
- tensor_utils: Remove release when move tensor to NDArray
- Change PACKAGE_FILENAME as `maccel_${VENDOR}_${PRODUCT}_${VERSION}`

## [v0.14.0] - 2023-11-03

### Added

- pypymaccel (#369)
- acc: Introduce startTracingEvents(), while deprecating Acc::startTracingEvents()
- model: Implement infer APIs using NDArray

### Fixed

- reposition: Calulcate CHW indices for efficient type
- `acc_impl`: Respect `OMP_NUM_THREADS` env var (#378)
- driver: Check {read,write}MemoryBuffer's return value (#370)

### Changed

- omp: Limit number of threads for `OMP parallel block` when other `OMP` blocks are running (#383)
- 윈도우에서 아래 파일들의 추가 종속성을 없앰 .TestDrive SystemDDK.dll SystemHAL.dll PCIeDriverSystem.dll
- Convert `maccel.h` to all-in-one-style header (#366)
- type: Include all cluster/cores by default

## [v0.13.0] - 2023-09-13

### Added

- Add copyright notice (#351)
- npu_watchdog: Dump DDR.bin for debugging (#348)

### Fixed

- reposition: Fill 0 when reposition is ill-fitted
- reposition: Fix filling gap bug for channel 1

### Changed

- cmake: Add 'd' postfix for windows dll debug build
- log: Change default log level
- model: Implement inferHeightBatch using inferBatch (#349)

## [v0.12.0] - 2023-09-01

### Added

- aries: Introduce new env var, which prevents {un,}claiming cores (#338)
- Interrupt 지원 (Linux) (#324)
- windows_interrupt (#315)
- INFO, WARNING, ERROR 로그가 출력됩니다.
  추가된 `setLogLevel()` API로 로그 레벨을 조절할 수 있습니다. (OFF도 가능)
- Implement int8->int8 infer API in output-as-parameter fashion

### Fixed

- reposition: Fill gap when input channel is 1
- reposition: Don't use efficient reposition if it's CHW format
- Fix Visual Studio build error (#322)
- reposition: Make output correct for AVX2
- memory_pool: When reset, free all pre-allocated memory

### Changed

- reposition: Implement `efficient` logic for int input/output (#329)
- Limit OMP num threads at runtime
- Implement RepositionType::EfficientRuntimeInterleave

### Removed

- model: Remove some infer APIs

## [v0.11.1] - 2023-08-25

- aries: Introduce new env var, which prevents {un,}claiming cores (#339)

## [v0.11.0] - 2023-06-21

### Added

- Implement experimental `inferHeightBatch()` API
- model: Implement new infer API which takes output as parameter

### Fixed

- Fix inferSpeedrun segfault
- reposition: When CHW, use `default` method

### Changed

- reposition: Implement scale using NEON
- Refactor reposition functions

## [v0.10.0] - 2023-06-14

### Added

- Implement batch input inference API (#290)

### Fixed

- Fix `EXPERIMENTAL_READ_OUTPUTS_AT_ONCE` compile error
- Fix inferCHW bug
- Mitigate sleep() problem on windows (#283)

### Changed

- Do not repos when original channel is the same as pe num (#294)
- Allocate input/output buffers at once (#293)
- Implement `RepositionType::Efficient` (#291)
- pymaccel check input shape (#292)
- pybind11 infer no convert (#289)
- Implement `MemoryPool` and apply to `ModelImpl` (#288)
- Move `NPUTaskQueue`-releated members to `AcceleratorImpl` (#286)
- reposition: Clean up omp directives
- Remove model runner

## [v0.9.1] - 2023-05-23

### Changed

- Disable interleaving

## [v0.9.0] - 2023-05-19

### Added

- Implement manual packed, multi input inference
- Claim cores when model launches (#277)

### Changed

- List up more detailed items for profiler (#278)
- kibum/factor-out-aries-reset (#276)
- Try to apply bonfire for CPU offload (#273)
- root일때만 whl 빌드 (#274)

## [v0.8.0] - 2023-04-18

### Fixed

- Use input's memory format for output's memory format (#270)

### Changed

- Support width-wise reshape with 1 or 2 channels, just like 3 channels (#272)
- Build a wheel file while `make`ing
- Allow lowercase for `DRIVER_TYPE`

## [0.7.0] - 2023-03-23

### Added

- Initial work on CPU-offloading.

### Removed

- Temporarily removed non-float infer APIs

## [0.6.0] - 2023-03-23

### Added

- add windows driver interface. (tested.) (#263)
- Implement GlobalMode/MultiMode (#262)
- core 관련 함수 추가 (#251)
- add modelconfig func (#230)
- Reset 구현 (#229)
- Implement inferSpeedrun (#232)
- model: Add infer functions using `int8_t` (#182)
- version 기능 추가 (#173)
- `model_impl`: Implement ChannelFirst (#166)

### Changed

- Plus 0.5 then floor to convert float -> int
- `npu_watchdog`: Set reset timeout ratio to 10 (#241)
- Enable log for default cmake build
- pymaccel infer gil release 추가 (#214)
- profile: Reimplement profiler (#210)
- Aries 이후 하드웨어를 지원하는 패치 (#185)
- `model_manager`: Always wait until `NPU_FINISH == 1`

### Fixed

- Fix generate export header process
- acc 먼저 파괴시 model dispose 호출 (#252)
- 동작중인 Core에 launch 방지 (#211)
- 모델 업로드 전 DRAM 영역 초기화 로직 추가 (#206)
