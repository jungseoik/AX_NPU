<!-- 출처: https://docs.mobilint.com/runtime/v1.4/en/introduction.html (원본) -->

# Build AI inference on Mobilint NPUs

qb Runtime gives developers, engineers, and researchers the building blocks to run AI inference on Mobilint NPUs

- [Get Started](getting_started.md)
- [API Reference](../doxygen/html_en/group__CPPAPI)

```{tip}
**qb Runtime v1.4.0** released — the mbltml NPU management library, `mobilint-cli top`, and one runtime build for every NPU device. See the [release notes](release_note.md).
```

```{important}
Driver installation, firmware update, and NPU core configuration guides for ARIES devices have moved to the {external+aries:doc}`ARIES manual <overview>`. This manual covers the qb Runtime and its utilities.
```

## What is qb Runtime?

**qb Runtime** is the inference engine of SDK qb, Mobilint's software development kit for its Neural Processing Unit (NPU). It loads models compiled into Mobilint's executable format (**MXQ**) by the {external+compiler:doc}`qb Compiler <index>` and runs them on the NPU using internal optimized logic.

![Runtime Execution](../res/image/SDK_Runtime_Execution.png)

- **C++ and Python interfaces**: Integrate NPU inference into your application with the runtime library's C++ and Python APIs.
- **Linux and Windows support**: Deploy on either operating system for easier system integration.

## Mobilint Model Zoo

Mobilint provides a growing **Model Zoo** of open-source models precompiled for Mobilint NPUs. These **`.mxq`** (Mobilint eXeCUtable) files are optimized for Mobilint's runtime and can be deployed immediately on supported NPU hardware. New models are added regularly to support evolving AI workloads and customer requests.

- [490+ pre-compiled models](model_zoo.md): Computer vision, natural language processing, and multimodal AI models ready to run as `.mxq`.

## Get Started

Go from installing the product to maximizing its performance with advanced functions.

- [Installation and first inference](getting_started.md)


```{tip}
Have questions or need custom deployment support?
Contact us at [https://discuss.mobilint.com](https://discuss.mobilint.com).
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Overview
Getting Started <getting_started>
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Installation
Installing Runtime Library <installing_runtime_library>
compatibility
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Tutorials
(Basic) Running Pre-Compiled ResNet50 <tutorial_resnet50>
Programming Guide <programming_guide>
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Utilities
Installing Utility <installing_utility>
utility_usage
Mobilint Model Zoo <model_zoo>
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Deep Dive
advanced_usage
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Deployment
Mobilint Device Plugin <kubernetes_device_plugin>
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Supported Hardware
REGULUS System-on-Module (SoM) <regulus-som>
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Reference
C++ API <../doxygen/html_en/group__CPPAPI>
Python API <../doxygen/html_en/group__PythonAPI>
Classes <../doxygen/html_en/annotated_classes>
Files <../doxygen/html_en/files>
Deprecated List <../doxygen/html_en/deprecated>
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: mbltml
C API <../doxygen/html_en_mbltml/group__MbltmlCAPI>
Python API <../doxygen/html_en_mbltml/group__MbltmlPyAPI>
Classes <../doxygen/html_en_mbltml/annotated>
Files <../doxygen/html_en_mbltml/files>
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Release Notes
Release Notes <release_note>
Changelog <CHANGELOG>
```

