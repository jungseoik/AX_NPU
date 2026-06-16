# Mobilint Device Plugin for Kubernetes

Mobilint Device Plugin은 ARIES NPU를 Kubernetes와 통합하여, 사용자가 CPU나 GPU와 동일한 방식으로 NPU를 요청할 수 있게 합니다.

## 개요

Mobilint Device Plugin은 Kubernetes Device Plugin API를 구현하여 ARIES NPU를 Kubernetes 리소스로 등록합니다.

설치 후 kubelet은 노드의 ARIES 디바이스를 `mobilint.com/npu` 리소스로 게시하며, 사용자는 Pod에서 이 리소스를 요청할 수 있습니다.

Pod에 NPU가 할당되면 Device Plugin은 CDI(Container Device Interface)를 사용하여 선택된 디바이스를 컨테이너에 주입합니다.

## 사전 요구사항

- 모든 NPU 노드에 **ARIES 드라이버**가 설치되어 있어야 합니다. [드라이버 설치](installing_driver.md)를 참조하세요. 각 노드에서 다음으로 확인할 수 있습니다.

    ```bash
    lsmod | grep aries
    ls /dev/aries*
    ```

- 컨테이너 런타임이 **CDI(Container Device Interface)**를 지원해야 합니다.

    | 런타임 | 버전 |
    | --- | --- |
    | containerd | 1.7+ |
    | CRI-O | 1.23+ |

## 설치

### 1. 노드 라벨 지정

Device Plugin은 `mobilint.com/npu.present=true` 라벨이 있는 노드에만 배포됩니다. 각 NPU 노드에 라벨을 지정합니다.

```bash
kubectl label node <NODE_NAME> mobilint.com/npu.present=true --overwrite
```

노드 이름은 `kubectl get nodes`로 확인할 수 있습니다. Node Feature Discovery를 사용하면 이 과정을 자동화할 수 있습니다(아래 NFD 연동 참조).

### 2. Device Plugin 설치

Helm으로 설치합니다.

```bash
helm install mobilint-device-plugin \
  oci://ghcr.io/mobilint/charts/mobilint-device-plugin \
  -n kube-system
```

Helm을 사용하지 않으려면 DaemonSet 매니페스트를 직접 적용합니다.

```bash
kubectl apply -f https://raw.githubusercontent.com/mobilint/mobilint-device-plugin/main/deploy/daemonset.yaml
```

## 설치 검증

Device Plugin Pod이 실행 중인지 확인합니다.

```bash
kubectl -n kube-system get pods \
  -l app.kubernetes.io/name=mobilint-device-plugin
```

모든 NPU 노드에서 Device Plugin Pod이 `READY 1/1` 상태여야 합니다.

노드가 NPU 리소스를 노출하는지 확인합니다.

```bash
kubectl get node <NODE_NAME> \
  -o jsonpath='{.status.allocatable.mobilint\.com/npu}'
```

노드에 연결된 NPU 개수(예: `4`)가 출력되면 정상입니다.

## 워크로드에서 사용

Pod의 `resources.limits`에 `mobilint.com/npu`를 지정하여 NPU를 요청합니다.

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

위 매니페스트를 `npu-example.yaml`로 저장하여 적용한 뒤, 컨테이너 내부에서 NPU 디바이스가 보이는지 확인합니다.

```bash
kubectl apply -f npu-example.yaml
kubectl exec -it npu-example -- ls -l /dev/aries*
```

## 모니터링
Device Plugin은 각 노드에서 `:9400` 포트로 메트릭과 상태 엔드포인트를 제공합니다.

| 엔드포인트 | 설명 |
| --- | --- |
| `GET /metrics` | Prometheus 텍스트 포맷의 디바이스별 NPU 텔레메트리 |
| `GET /process` | 현재 NPU를 사용 중인 프로세스 상세(JSON) |
| `GET /readyz` | readiness probe (kubelet 등록 완료 시 200) |

엔드포인트는 Pod에서 항상 제공됩니다. Prometheus가 수집하도록 하려면 아래 스크레이프 설정을 활성화하세요.

[Prometheus Operator](https://github.com/prometheus-operator/prometheus-operator)를 사용하는 경우, Helm 설치 시 Service와 ServiceMonitor를 활성화합니다.

```bash
helm install mobilint-device-plugin \
  oci://ghcr.io/mobilint/charts/mobilint-device-plugin \
  -n kube-system \
  --set metrics.service.enabled=true \
  --set metrics.serviceMonitor.enabled=true
```

### 메트릭

| 메트릭 | 타입 | 단위 | 설명 |
| --- | --- | --- | --- |
| `mobilint_npu_health` | gauge | 0/1 | 모니터 샘플 읽기 성공 여부 |
| `mobilint_npu_info` | gauge | — | 정적 정보(모델·드라이버/펌웨어 버전·PCIe 등)를 라벨로 노출, 값은 항상 1 |
| `mobilint_npu_temperature_celsius` | gauge | °C | 다이 온도 |
| `mobilint_npu_clock_npu_hz` | gauge | Hz | NPU 코어 클럭 |
| `mobilint_npu_clock_noc_hz` | gauge | Hz | NoC(인터커넥트) 클럭 |
| `mobilint_npu_power_watts` | gauge | W | 총 전력 |
| `mobilint_npu_current_amperes` | gauge | A | 총 전류 |
| `mobilint_npu_voltage_volts` | gauge | V | 총 전압 |
| `mobilint_npu_fan_duty` | gauge | % | 쿨링 팬 duty |
| `mobilint_npu_fd_count` | gauge | — | 디바이스에 열린 fd 수 |
| `mobilint_npu_memory_total_bytes` | gauge | bytes | 총 NPU 메모리 |
| `mobilint_npu_memory_used_bytes` | gauge | bytes | 사용 중 NPU 메모리 |
| `mobilint_npu_utilization_ratio` | gauge | 0–1 | 전체 NPU 사용률 |
| `mobilint_npu_process_count` | gauge | — | NPU를 사용 중인 프로세스 수 |
| `mobilint_npu_core_utilization_ratio` | gauge | 0–1 | 코어별 사용률(`cluster`·`core` 라벨) |

### 프로세스별 상세 (`/process`)

`mobilint_npu_process_count`는 프로세스 개수만 제공합니다. 개별 프로세스의 메모리·사용률 같은 상세 정보는 `/process` JSON 엔드포인트로 제공됩니다.

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


## NFD 연동

[Node Feature Discovery](https://github.com/kubernetes-sigs/node-feature-discovery)(NFD)를 사용하면 Mobilint가 제공하는 NodeFeatureRule을 통해 NPU가 있는 노드에 `mobilint.com/npu.present=true` 라벨을 자동으로 부여할 수 있습니다.

먼저 NFD를 설치합니다.

```bash
helm repo add nfd https://kubernetes-sigs.github.io/node-feature-discovery/charts
helm install nfd nfd/node-feature-discovery \
  -n node-feature-discovery --create-namespace \
  --set master.extraLabelNs={mobilint.com}
```

이후 Device Plugin 설치 시 NFD 연동을 활성화합니다.

```bash
helm install mobilint-device-plugin \
  oci://ghcr.io/mobilint/charts/mobilint-device-plugin \
  -n kube-system \
  --set nodeFeatureDiscovery.enabled=true
```

## 제거

Helm으로 설치한 경우:

```bash
helm uninstall mobilint-device-plugin -n kube-system
```

매니페스트로 설치한 경우:

```bash
kubectl delete -f https://raw.githubusercontent.com/mobilint/mobilint-device-plugin/main/deploy/daemonset.yaml
```

노드 라벨도 제거합니다.

```bash
kubectl label node <NODE_NAME> mobilint.com/npu.present-
```
