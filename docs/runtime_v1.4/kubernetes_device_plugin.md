<!-- 출처: https://docs.mobilint.com/runtime/v1.4/en/kubernetes_device_plugin.html (원본) -->

# Mobilint Device Plugin for Kubernetes

The Mobilint Device Plugin integrates ARIES NPUs with Kubernetes so that you can request an NPU the same way you request a CPU or GPU.

## Overview

The Mobilint Device Plugin implements the Kubernetes Device Plugin API to register ARIES NPUs as Kubernetes resources.

After installation, the kubelet publishes the node's ARIES devices as the `mobilint.com/npu` resource, and you request that resource from a Pod.

When a Pod is allocated an NPU, the device plugin uses the Container Device Interface (CDI) to inject the selected device into the container.

## Prerequisites

- The **ARIES driver** must be installed on every NPU node. See {external+aries:doc}`Driver Installation <driver-installation>` in the ARIES manual. Verify on each node with:

    ```bash
    lsmod | grep aries
    ls /dev/aries*
    ```

- The container runtime must support **CDI (Container Device Interface)**.

    | Runtime | Version |
    | --- | --- |
    | containerd | 1.7+ |
    | CRI-O | 1.23+ |

## Installation

### 1. Label the NPU nodes

The device plugin is deployed only to nodes that carry the `mobilint.com/npu.present=true` label. Label each NPU node:

```bash
kubectl label node <NODE_NAME> mobilint.com/npu.present=true --overwrite
```

List node names with `kubectl get nodes`. You can automate this step with Node Feature Discovery (see NFD Integration below).

### 2. Install the device plugin

Install with Helm:

```bash
helm install mobilint-device-plugin \
  oci://ghcr.io/mobilint/charts/mobilint-device-plugin \
  -n kube-system
```

If you are not using Helm, apply the DaemonSet manifest directly:

```bash
kubectl apply -f https://raw.githubusercontent.com/mobilint/mobilint-device-plugin/main/deploy/daemonset.yaml
```

### OpenShift

On OpenShift, the plugin's privileged container and hostPath volumes are blocked by SCC.  
Set `openShift.enabled=true` to grant the required `privileged` SCC. Install into a dedicated namespace (not `kube-system`).

```bash
oc new-project mobilint-device-plugin

helm install mobilint-device-plugin \
  oci://ghcr.io/mobilint/charts/mobilint-device-plugin \
  -n mobilint-device-plugin --set openShift.enabled=true
```

## Verifying the Installation

Check that the device plugin Pod is running:

```bash
kubectl -n kube-system get pods \
  -l app.kubernetes.io/name=mobilint-device-plugin
```

The device plugin Pod should be `READY 1/1` on every NPU node.

Check that the node advertises the NPU resource:

```bash
kubectl get node <NODE_NAME> \
  -o jsonpath='{.status.allocatable.mobilint\.com/npu}'
```

It prints the number of NPUs on the node (for example, `4`).

## Using It in a Workload

Request an NPU by setting `mobilint.com/npu` under the Pod's `resources.limits`:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: npu-example
spec:
  containers:
    - name: app
      image: ubuntu:latest
      command: ["sleep", "infinity"]
      resources:
        limits:
          mobilint.com/npu: 1
```

Save the manifest as `npu-example.yaml`, apply it, and confirm the NPU device is visible inside the container:

```bash
kubectl apply -f npu-example.yaml
kubectl exec -it npu-example -- ls -l /dev/aries*
```

## Monitoring

The device plugin exposes metrics and status endpoints on port `:9400` on each node.

| Endpoint | Description |
| --- | --- |
| `GET /metrics` | Per-device NPU telemetry in Prometheus text format |
| `GET /process` | Details of the processes currently using the NPU (JSON) |
| `GET /readyz` | Readiness probe (returns 200 once kubelet registration completes) |

The endpoints are always served from the Pod. To have Prometheus scrape them, enable the scrape configuration below.

If you use [Prometheus Operator](https://github.com/prometheus-operator/prometheus-operator), enable the Service and ServiceMonitor when installing with Helm:

```bash
helm install mobilint-device-plugin \
  oci://ghcr.io/mobilint/charts/mobilint-device-plugin \
  -n kube-system \
  --set metrics.service.enabled=true \
  --set metrics.serviceMonitor.enabled=true
```

### Metrics

| Metric | Type | Unit | Description |
| --- | --- | --- | --- |
| `mobilint_npu_health` | gauge | 0/1 | Whether the monitor sample was read successfully |
| `mobilint_npu_info` | gauge | — | Static info (model, driver/firmware version, PCIe, etc.) exposed as labels; value is always 1 |
| `mobilint_npu_temperature_celsius` | gauge | °C | Die temperature |
| `mobilint_npu_clock_npu_hz` | gauge | Hz | NPU core clock |
| `mobilint_npu_clock_noc_hz` | gauge | Hz | NoC (interconnect) clock |
| `mobilint_npu_power_watts` | gauge | W | Total power |
| `mobilint_npu_current_amperes` | gauge | A | Total current |
| `mobilint_npu_voltage_volts` | gauge | V | Total voltage |
| `mobilint_npu_fan_duty` | gauge | % | Cooling fan duty |
| `mobilint_npu_fd_count` | gauge | — | Number of open file descriptors on the device |
| `mobilint_npu_memory_total_bytes` | gauge | bytes | Total NPU memory |
| `mobilint_npu_memory_used_bytes` | gauge | bytes | Used NPU memory |
| `mobilint_npu_utilization_ratio` | gauge | 0–1 | Overall NPU utilization |
| `mobilint_npu_process_count` | gauge | — | Number of processes using the NPU |
| `mobilint_npu_core_utilization_ratio` | gauge | 0–1 | Per-core utilization (`cluster` and `core` labels) |

### Per-process Details (`/process`)

`mobilint_npu_process_count` provides only the process count. Details for individual processes, such as memory and utilization, are available from the `/process` JSON endpoint.

```bash
kubectl -n kube-system port-forward <device-plugin-pod> 9400:9400
curl localhost:9400/process
```

```json
[
  {
    "device": "aries0",
    "processes": [
      { "pid": 420300, "memory_used_bytes": 3890802880, "utilization": 0.712 }
    ]
  }
]
```

## NFD Integration

[Node Feature Discovery](https://github.com/kubernetes-sigs/node-feature-discovery) (NFD) can apply the `mobilint.com/npu.present=true` label automatically through the NodeFeatureRule that Mobilint provides.

First, install NFD:

```bash
helm repo add nfd https://kubernetes-sigs.github.io/node-feature-discovery/charts
helm install nfd nfd/node-feature-discovery \
  -n node-feature-discovery --create-namespace \
  --set master.extraLabelNs={mobilint.com}
```

Then enable NFD integration when installing the device plugin:

```bash
helm install mobilint-device-plugin \
  oci://ghcr.io/mobilint/charts/mobilint-device-plugin \
  -n kube-system \
  --set nodeFeatureDiscovery.enabled=true
```

## Uninstalling

If you installed with Helm:

```bash
helm uninstall mobilint-device-plugin -n kube-system
```

If you installed from the manifest:

```bash
kubectl delete -f https://raw.githubusercontent.com/mobilint/mobilint-device-plugin/main/deploy/daemonset.yaml
```

Remove the node label as well:

```bash
kubectl label node <NODE_NAME> mobilint.com/npu.present-
```
