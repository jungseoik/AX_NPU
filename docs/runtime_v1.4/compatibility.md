<!-- 출처: https://docs.mobilint.com/runtime/v1.4/en/compatibility.html (원본) -->

# Checking Compatibility

This document provides compatibility information for the software products offered by Mobilint. When configuring or updating the system, you must refer to the table below to ensure that compatible versions are used. Using incompatible combinations may result in unexpected behavior or system errors, and normal operation cannot be guaranteed.

<style>
table.table {
  table-layout: fixed;
  width: 100%;
}
</style>

```{note}
For **ARIES** devices, see the {external+aries:doc}`ARIES manual <compatibility>` for driver, firmware, and OS compatibility. This page covers **runtime–MXQ** compatibility for qb Runtime.
```

## MXQ Compatibility Matrix

To run a deep learning model optimized for Mobilint's hardware (`MXQ` file extension), you must use a runtime that supports the specific `MXQ` format version on which the model is based.

This section provides a compatibility matrix of `MXQ` files for each runtime:

| Runtime Ver.    | MXQ Ver.      |
| --------------- | ------------- |
| 0.28.0 ~ 0.28.0 | MXQv1 ~ MXQv5 |
| 0.29.0 ~ 0.30.1 | MXQv1 ~ MXQv6 |
| 1.0.0  ~ latest | MXQv1 ~ MXQv7 |

## How to check the version

### Runtime Library

If you downloaded the runtime library from the [official Download Center](http://dl.mobilint.com), the version information is typically available at the time of download.

Alternatively, you can print out version via runtime provided method :

```cpp
// C++ Example
std::cout << mobilint::getQbRuntimeVersion() << "\n";
```

```python
# Python Example
print(qbruntime.__version__)
```

### MXQ

- You can check MXQ version using `mobilint-cli` utility tool.

- Run following command:

  ```bash
  mobilint-cli mxqtool show {mxq_path}
  ```

  Example output:

  ```bash
  $ mobilint-cli mxqtool show resnet50_IMAGENET1K_V1.mxq
  Format Version:           0x60000
  Compiler Version:         0.10.0.0
  Hardware Version:         Aries2
  ...
  ```

- The most significant digit of "Format Version" value indicates MXQ version. For example, "0x60000" corresponds to MXQv6.
