<!-- 출처: https://docs.mobilint.com/runtime/v1.4/en/advanced_usage.html (원본) -->

# Advanced Usage

This section covers some of the advanced features of the runtime library that allow developers to optimize performance and customize resource allocation on the NPU.

- [Multi-threading](#important-multi-threading)
- [Tracing NPU Usage](#tracing-npu-usage)

These features are not required for basic operation, but can significantly enhance
throughput and flexibility in multi-stream, multi-model, or other various environments where fine-grained control or scalability is needed.

```{tip}
ARIES contains eight (8) independent NPU processing cores, and REGULUS has one (1) independent NPU core.
```

## (Important) Multi-threading

```{note}
The following instruction is applicable to ARIES-powered products only.
```

By default, inference using the runtime library (via `mobilint::Model::infer()` method) operates
in a blocking or synchronous I/O manner. This approach has been chosen as the
default because it is the simplest structurally and the easiest to understand when
working with an unfamiliar runtime library.

However, blocking I/O has the following limitation:

- When inference is performed using a single thread, the CPU sends a task to NPU and waits for result each time. As a result, CPU cannot immediately sends next inputs, so multiple NPU cores(or clusters) cannot be utilized simultaneously. In other words, CPU must wait until core/cluster finishes its task before sending next input, which prevents taking advantage of multiple cores or clusters.

To address this issue, you can either:
- Implement **multi-threaded** program directly, or
- Use `inferAsync()` method provided by Mobilint runtime.

For users familiar with **multi-threaded** programming, performance can be improved by implementing a custom multi-threaded program, as shown in simple example below.

```cpp
void work(Model* model) {
    StatusCode sc;
    NDArray<float> input(model->getModelInputShape()[0], sc);
    std::vector<NDArray<float>> output;
    for (int i = 0; i < 10; i++) {
        sc = model->infer({input}, output);
    }
}

int main() {
    StatusCode sc;
    auto acc = Accelerator::create(sc);
    auto model = Model::create(MXQ_PATH, sc);
    model->launch(*acc);

    std::vector<std::thread> threads;
    for (int i = 0; i < NUM_THREAD; i++) {
        threads.emplace_back(work, model.get());
    }

    for (int i = 0; i < NUM_THREAD; i++) {
        threads[i].join();
    }

    return 0;
}
```

Alternatively, if implementing a multi-threaded program is challenging, you can use `inferAsync()` method provided by Mobilint runtime library, as show in example below.

```{caution}
The `inferAsync()` method currently has the following limitations:

- RNN/LSTM/LLM models are not yet supported.
- Models requiring CPU offloading are not yet supported.
- Only single-batch inference supported.

For much details, please refer to API Reference under the `Model` class, section "Asynchronous Inference".
```

```python
## Python example
import qbruntime
from collections import deque

mc = qbruntime.ModelConfig()
mc.set_async_pipeline_enabled(True)

acc = qbruntime.Accelerator()

model = qbruntime.Model(MXQ_PATH, mc)
model.launch(acc)

## Method 1: Simple usage example
future_results = []
results = []

for i in range(NUM_INFERENCE):
    rand_input = np.random.rand(224, 224, 3).astype(np.float32)
    future_result = model.infer_async(rand_input)
    future_results.append(future_result)

for future in future_results:
    res = future.get()
    results.append(res)

## Method 2: Focus on real-time data processing
future_results = deque()
for i in range(NUM_INFERENCE):
    rand_input = np.random.rand(224, 224, 3).astype(np.float32)
    future_result = model.infer_async(rand_input)
    future_results.append(future_result)
    
    while future_results and (future_results[0].wait_for(0) or i == NUM_INFERENCE - 1):
        future = future_results.popleft()
        res = future.get()
        ## RESULT PROCESSING ...
```

```cpp
// C++ example
#include "qbruntime/qbruntime.h"

#include <queue>
#include <vector>

int main() {
    mobilint::StatusCode sc;
    mobilint::ModelConfig mc;

    auto acc = mobilint::Accelerator::create(sc);
    if (!sc) exit(1);

    mc.setAsyncPipelineEnabled(true);

    auto model = mobilint::Model::create(MXQ_PATH, mc, sc);
    if (!sc) exit(1);

    sc = model->launch(*acc);
    if (!sc) exit(1);

    // Note: For simplicity, validating `sc` value are omitted below.
    // Method 1: Simple usage example
    std::vector<mobilint::Future<float>> future_results;
    std::vector<std::vector<mobilint::NDArray<float>>> results;

    for (int i = 0; i < NUM_INFERENCE; i++) {
        auto rnd_inputs = mobilint::NDArray<float>({224, 224, 3}, sc);
        mobilint::Future<float> future = model->inferAsync({rnd_inputs}, sc);
        future_results.push_back(std::move(future));
    }

    for (auto& future_result: future_results) {
        std::vector<mobilint::NDArray<float>> res = future_result.get(sc);
        results.push_back(std::move(res));
    }

    // Method 2: Focus on real-time data processing
    std::queue<mobilint::Future<float>> future_results;
    for (int i = 0; i < NUM_INFERENCE; i++) {
        auto rnd_inputs = mobilint::NDArray<float>({224, 224, 3}, sc);
        mobilint::Future<float> future = model->inferAsync({rnd_inputs}, sc);
        future_results.push(std::move(future));

        while (!future_results.empty() && (future_results.front().waitFor(0) ||
                                            i == NUM_INFERENCE - 1)) {
            auto future_result = std::move(future_results.front());
            future_results.pop();

            std::vector<mobilint::NDArray<float>> res = future_result.get(sc);
            // RESULT PROCESSING ...
        }
    }
}
```

## Tracing NPU Usage

![Example image](../res/image/tracing_log.png "Trace log example")

Runtime library offers a tracing feature. You can start tracing with {doxylink}`startTracingEvents() <mobilint::startTracingEvents(const char *path)>`.

User can specify the trace log's path in `path` argument. The log should result in a .json file format, which can be viewed in [Perfetto UI](https://ui.perfetto.dev/). This trace log would be recorded until {doxylink}`stopTracingEvents() <mobilint::stopTracingEvents()>` is called.

You can enable tracing by using the following methods:

```cpp
// c++ example
#include "qbruntime/qbruntime.h"
mobilint::startTracingEvents("path/to/trace.json");

// Tracing target NPU events

mobilint::stopTracingEvents();
```

```python
## python example
import qbruntime
qbruntime.start_tracing_events("path/to/trace.json")

## Tracing target NPU events

qbruntime.stop_tracing_events()
```
