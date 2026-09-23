<!-- 출처: https://docs.mobilint.com/runtime/v1.4/en/getting_started.html (원본) -->

# Getting Started

Here, you can find a comprehensive list of pages to get you started with your product, from installing it to maximizing its performance with advanced functions.

## Installation

```{important}
- The installation steps below apply to **ARIES-based devices only**. Driver, firmware, and hardware setup for ARIES products live in the {external+aries:doc}`ARIES manual <overview>`.
- **REGULUS** devices ship with the essential SDK stack (driver, runtime library) and utilities pre-installed. Proceed directly to the **Tutorials** section below.
```

Install the SDK software required to use an ARIES NPU by following these documents in order.

1. {external+aries:doc}`Driver Installation (ARIES manual) <driver-installation>`: The kernel driver that exposes the NPU to your operating system.
2. [Installing Runtime Library](installing_runtime_library.md): The C++ and Python runtime for loading models and running inference.
3. [Checking Compatibility](compatibility.md): Verify that your runtime and MXQ model versions match.

```{note}
You need a [Download Center](https://dl.mobilint.com) account to access and download some files and modules. Please contact contact@mobilint.com for more information.
```

### Optional Setup

These steps are not required for a standard setup — perform them only when needed.

- {external+aries:doc}`Firmware Update (ARIES manual) <firmware-update>`: Needed only when the compatibility check reports a device firmware older than the version the SDK requires.

```{note}
Compiling your own models into MXQ is covered in the separate {external+compiler:doc}`qb Compiler documentation <index>` and is not part of this manual. To run models right away, use the pre-compiled models in the [Model Zoo](model_zoo.md).
```

## Tutorials

After installing all essential SDK components required for NPU operation, refer to the following tutorials to learn how to successfully run AI inference on a Mobilint NPU.

- [Running Pre-Compiled ResNet50](tutorial_resnet50.md): Run your first inference with a ready-made model.
- [NPU Programming Guide](programming_guide.md): The core runtime API and the inference workflow.

## Utilities

Explore the utility tools provided by Mobilint. These help you monitor the NPU's status or run tests. If you want to use an open-source pre-trained model off the shelf, visit our Model Zoo to check the availability of a pre-compiled model.

- [Installing Utility](installing_utility.md): Install Mobilint's command-line tools.
- [Utility Usage](utility_usage.md): Monitor NPU status and run diagnostics.
- [Model Zoo](model_zoo.md): 490+ pre-compiled open-source models.

## Advanced

Once you get the hang of the system, follow the Advanced Usage sections to further optimize your NPU usage or refer to the advanced tutorials.

- [Advanced Usage](advanced_usage.md): Optimize throughput, batching, and memory.
- {external+aries:doc}`Utilizing Multi-Core Modes (ARIES manual) <core-mode>`: Distribute inference across NPU cores.

## Deployment

To deploy NPU workloads on a Kubernetes cluster, install the device plugin so that Pods can request NPUs as a schedulable resource.

- [Mobilint Device Plugin](kubernetes_device_plugin.md): Schedule NPUs as resources on a Kubernetes cluster.
