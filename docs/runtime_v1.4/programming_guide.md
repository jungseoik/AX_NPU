<!-- 출처: https://docs.mobilint.com/runtime/v1.4/en/programming_guide.html (원본) -->

# Programming Guide

This document will help you further understand Mobilint's qb Runtime and utility (`mobilint-cli`).

This section includes explanations for the core components in the library as well as the typical inference process.

```{warning}
To use ARIES-powered devices, you must first have the driver and runtime library installed in your system. See {external+aries:doc}`Driver Installation <driver-installation>` in the ARIES manual and [Runtime Library](installing_runtime_library.md) installation.
```

## Mobilint Runtime Library Features

Mobilint's runtime library provides a flexible and robust API for integrating Mobilint NPUs into your C++ or Python applications. Below are key features and supported behaviors:

- C++: The C++ library supports defensive programming through {doxylink}`StatusCode <mobilint::StatusCode>`.

- Python: Each step handles success or failure internally, immediately raising an error and terminating execution if an issue occurs.

- Currently supports `UINT8`, `INT8` and `float32` input data types for inference.

- `qb Runtime` supports high-dimensional data inference using {doxylink}`NDArray <mobilint::NDArray>` type. Input data using NDArray can be structured as shown below.

	```cpp
	mobilint::StatusCode sc;
	mobilint::NDArray<float> input({224, 224, 3}, sc);
	```
	
	```{tip}
	For detailed initialization methods of `NDArray`, please refer to API references.
	```

## Inference Process

To utilize runtime library, an **MXQ** file is required. An **MXQ (Mobilint ExeCUtable)** file refers to the optimized model file format compiled on {external+compiler:doc}`qb Compiler <index>`, Mobilint’s official compiler.

Running inference on Mobilint's NPU using the `qb Runtime` involves four essential steps:

1. Load the NPU device. [(Step 1: `Accelerator`)](#step-1-accelerator)
2. Load the compiled model (MXQ file). [(Step 2: `Model`)](#step-2-model)
3. Upload model to the NPU device. [(Step 3: Pass `Model` information to the `Accelerator`)](#step-3-pass-model-information-to-the-accelerator)
4. Run inference using user input. [(Step 4: Run inference using passed `Model` information and input data)](#step-4-run-inference-using-passed-model-information-and-input-data)

In Mobilint's `qb Runtime`, abstracted objects `Accelerator` and `Model` are used throughout the steps.

### Step 1: `Accelerator`

The {doxylink}`Accelerator <mobilint::Accelerator>` object represents the NPU device to be used. It abstracts a single device identified by the number appended to the device name (e.g., `/dev/aries0` -> `Accelerator0`, `/dev/aries1` -> `Accelerator1`). If no specific number is provided in `Accelerator` initialization, NPU device `0` is used by default.

```cpp
// C++ example
mobilint::StatusCode sc;
auto acc = mobilint::Accelerator::create(0, sc);
if (!sc) {
	fprintf(stderr, "Error code %d\n", int(sc));
	exit(1);
}
```
```python
# Python example
acc = qbruntime.Accelerator(0)
```

### Step 2: `Model`
The {doxylink}`Model <mobilint::Model>` object represents a model contained in an MXQ file. Upon creation, it immediately reads the MXQ file and stores the relevant information. It then uses the `Accelerator` object to run inference using this model.

```cpp
// C++ example
auto model = mobilint::Model::create(MXQ_FILE_PATH, sc);
if (!sc) {
	fprintf(stderr, "Error code %d\n", int(sc));
	exit(1);
}
```
```python
# Python example
model = qbruntime.Model(MXQ_FILE_PATH)
```

### Step 3: Pass `Model` information to the `Accelerator`.
Pass the information with the {doxylink}`launch() <mobilint::Model::launch(Accelerator &	acc)>` method of `Model` object.

### Step 4: Run inference using passed `Model` information and input data
Get the input data and run inference using the `mobilint::Model::infer()` method of `Model` object.

```cpp
// C++ example
sc = model->launch(*acc);
if (!sc) {
	fprintf(stderr, "Error code %d\n", int(sc));
	exit(1);
}

auto result = model->infer({INPUT}, sc);
if (!sc) {
	fprintf(stderr, "Error code %d\n", int(sc));
	exit(1);
}
```
```python
# Python example
model.launch(acc)
result = model.infer([INPUT])
```

### Inference Scenario

```cpp
// C++ example
#include "qbruntime/qbruntime.h"

const char* MXQ_PATH = "path/to/mxq.mxq"

int main() {
	mobilint::StatusCode sc;
	auto acc = mobilint::Accelerator::create(sc);        // Step 1
	if (!sc) exit(1);
	
	auto model = mobilint::Model::create(MXQ_PATH, sc);  // Step 2
	if (!sc) exit(1);
	
	sc = model->launch(*acc);                            // Step 3
	if (!sc) exit(1);
	
	// Some preprocessing for input data
	
	auto result = model->infer(preprocessed_input, sc);  // Step 4
	if (!sc) exit(1);
}
```
```python
## Python example
import qbruntime

MXQ_PATH = "path/to/mxq.mxq"

## Step 1
acc = qbruntime.Accelerator()
## Step 2
model = qbruntime.Model(MXQ_PATH)
## Step 3
model.launch(acc)

## Some preprocessing for input data

## Step 4
result = model.infer(preprocessed_input)
```

## Compile C++ source code

### Linux/Ubuntu

The compilation process will vary depending on how the runtime library was installed.

- If the library was installed system-wide (installation via apt, dnf, or `sudo make install`), compile with:

	```bash
	g++ -o {outfile_name} {source_code} -lqbruntime
	```

- If you're using the runtime library without installation (using downloaded file from the Download Center):

	```bash
	g++ -o {outfile_name} -I{path/to/include} -L{path/to/library} {source_code} -lqbruntime
	```

(compile-windows)=
### Windows

On Windows, the runtime library is installed with the installer. See [Runtime Library Installation - Windows](installing_runtime_library.md#installing-on-windows) for the installation steps. During compilation, specify the `include` and library (`lib`) paths under the installation directory as shown below.

1. Open the Visual Studio project you are working on.

2. Click the button below to modify the project settings.

	![Project Setting](../res/image/project_setting.png "project setting")

3. As shown in the two images below, add the `include` and `lib` folders in the C/C++ setting and Linker setting, respectively.

	![Include](../res/image/project_setting_include_dir.png "Include")

	![Library](../res/image/project_setting_library_dir.png "Library")

4. Add the following settings according to the build mode. Use `qbruntime.lib` in Release mode and `qbruntimed.lib` in Debug mode.

	![Build Mode](../res/image/project_setting_build_mode.png "Build mode")

(compile-cmake)=
### CMake

If you installed the runtime through the `apt` or `dnf` package, or with the Windows installer, the CMake package configuration files are installed as well, so you can use the library with `find_package`.

```cmake
find_package(qbruntime REQUIRED)

add_executable(my_app main.cc)
target_link_libraries(my_app PRIVATE qbruntime::qbruntime)
```

The `qbruntime::qbruntime` target carries both the header path and the library, so you do not need to set include paths or link options yourself.

````{note}
If you cleared the **Set `qbruntime_DIR` environment variable** task in the Windows installer, point CMake at the package directory when you configure the project:

```powershell
cmake -B build -Dqbruntime_DIR="C:\Users\{user}\AppData\Local\Programs\mobilint-qb-runtime-sdk\lib\cmake\qbruntime"
```
````
