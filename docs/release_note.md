# 릴리즈 노트

## v1.2.0

**릴리즈 날짜:** 2026년 4월 2일

Batch LLM을 지원하기 위한 업데이트가 이루어졌습니다.

### 주요 변경 사항

#### BatchParam

Batch LLM을 지원하기 위해 새로운 구조체 {doxylink}`BatchParam <mobilint::BatchParam>` 이 도입되었습니다.

`BatchParam` 은 Batch LLM 추론 시 각 batch의 실행에 필요한 정보를 담는 구조체입니다.

**구조체 필드**

- `sequence_length` : 각 batch의 sequence_length 길이를 의미합니다.
- `cache_size` : 각 batch가 사용할 cache 크기입니다.
- `cache_id` : 각 batch가 이용할 cache의 identifier 입니다.
    - 각 batch 입력 context들은 동일한 cache id를 사용하여 추론을 진행해야 합니다.
    - 이 `cache_id` 값은 모델이 지원하는 최대 배치 개수 내에서 지정되어야 합니다.

**사용 방법**

Batch LLM을 사용하기 위해서는 여러 입력을 하나의 입력으로 결합한 뒤, 각 입력에 대한 `BatchParam`을 함께 전달해야 합니다.

입력의 차원이 (1, seq_len, hidden_dim)인 경우, 배치 입력들은 `seq_len` 차원을 기준으로 이어붙여야 합니다.

    - `seq_len` : 토큰 개수
    - `hidden_dim` : 각 토큰의 embedding 차원

예를 들어, sequence length 10, 80 길이의 두 입력을 하나로 합쳐 진행한다고 할 때, 아래 예시 코드와 같이 사용할 수 있습니다.

```python
import qbruntime
import numpy as np

## 모델이 지원하는 최대 배치 개수를 확인합니다.
print(model.get_cache_infos()[0].num_batches)

## Batch LLM 사용을 위해 입력들을 하나의 입력으로 이어붙입니다.
## 이때, 3차원 기준 2번째 차원으로 이어붙여야 합니다.
batch_input = np.concatenate([input0, input1], axis=1)

## qbruntime.BatchParam(각 입력 길이, 사용 캐시, 캐시 id)
batch_params = [
    qbruntime.BatchParam(10, 0, 0),
    qbruntime.BatchParam(80, 0, 1),
]

res = model.infer([batch_input], params=batch_params)

## 이 res 데이터는 [[첫번째 배치의 output], [두번째 배치의 output], ...]
## 형태로 출력됩니다.

batch_params2 = [
    qbruntime.BatchParam(1, 10, 0),
    qbruntime.BatchParam(1, 80, 1),
]

res = model.infer(res, params=batch_params2)
```

### 알려진 문제

- ARM (aarch64) 환경에서 LLM 모델을 실행할 경우 "Bus Error"와 함께 실행을 실패할 수 있습니다. 해당 문제는 v1.1.0부터 존재하며, 드라이버 패치를 통해 조속히 해결할 예정입니다. 불편을 드려 죄송합니다.

## v1.1.0

**릴리즈 날짜:** 2026년 3월 23일

qb Runtime v1.1.0에서는 자동 코어 모드 선택, 데이터 타입 조회 API, 그리고 성능 최적화가 이루어졌습니다.

### 주요 변경 사항

#### CoreMode::Auto

런타임이 모델의 코어 모드를 자동으로 선택할 수 있게 되었습니다. `ModelConfig`에서 `CoreMode::Auto`를 설정하면 MXQ의 코어 모드를 자동으로 감지하여 적용합니다. 기존에는 `Multi`, `Global4`, `Global8` 등 기본이 아닌 코어 모드를 사용하려면 직접 `ModelConfig`를 생성해야 했지만, 이제 자동으로 가능한 모드가 선택됩니다. 기본 생성자에서도 Auto 모드가 사용되므로 별도의 설정 없이 사용하는 것을 권장합니다.

```{note}
이때, MXQ 컴파일 시 `scheme="all"` 과 같은 플래그로 여러 모드를 가지는 MXQ 를 컴파일했다면 기존처럼 직접 코어 모드를 설정하여 사용해야 합니다.
```

```{seealso}
더 자세한 사항은 {doxylink}`setAutoCoreMode() <mobilint::ModelConfig::setAutoCoreMode()>` 를 확인해주세요.
```

#### 신규 API

- {doxylink}`getModelInputDataType() <mobilint::Model::getModelInputDataType() const>`,  {doxylink}`getModelOutputDataType() <mobilint::Model::getModelOutputDataType() const>` — 모델의 입력 및 출력 데이터 타입을 런타임에서 조회할 수 있어 더 유연한 파이프라인 구성이 가능합니다.
- {doxylink}`getAvailableDeviceNumbers() <mobilint::getAvailableDeviceNumbers()>` — 사용 가능한 NPU 디바이스 번호 목록을 조회합니다.

#### REGULUS 동적 할당

v1.0.0에서 도입된 동적 할당 방식을 REGULUS에도 동일하게 적용하여, 일관된 사용 방식을 확보하였습니다.

#### 성능 개선

- Windows에서 NPU 장치로의 데이터 전송 성능을 개선했습니다.
- 내부적으로 사용되던 타입 변환 과정을 최적화하여 성능을 개선하였습니다.

### 버그 수정

- GCC 9 미만에서 `std::filesystem`으로 인한 컴파일 오류를 해결했습니다.
- 일부 모델에서 간헐적으로 발생하던 데드락을 수정했습니다.

### 호환성 변경

- REGULUS driver의 지원 리비전 번호가 REV0에서 REV1으로 변경되었습니다.

### 알려진 문제

- ARM (aarch64) 환경에서 LLM 모델을 실행할 경우 Bus Error와 함께 실패할 수 있습니다. 해당 문제는 드라이버 패치를 통해 조속히 해결할 예정입니다. 불편을 드려 죄송합니다.

```{seealso}
전체 변경 사항은 [업데이트 기록](CHANGELOG.md) 페이지를 참조하세요.
```

## v1.0.0 — Major Release

**릴리즈 날짜:** 2026년 1월 31일

![Update_illust](/res/image/qb_release.jpg)

내부 아키텍처와 SDK qb 전반에 걸친 중요한 개선이 포함된 Major Update입니다. 성능 확장성, 일관성, 향후 기능 확장을 고려한 구조 개편에 중점을 두었습니다.

### 주요 변경 사항

#### SDK qb 이름 통일

기존에는 구성 요소별로 서로 다른 명칭을 사용하고 있어, SDK qb 전반의 구조를 처음 접하는 사용자에게 혼란을 줄 수 있었습니다. 이를 개선하기 위해 SDK qb 내 핵심 구성 요소의 명칭을 다음과 같이 통일하였습니다.

- 기존 런타임 라이브러리 maccel → qb Runtime
- 기존 컴파일러 qubee → qb Compiler

이번 명칭 통일을 통해 SDK qb 구성 요소 간의 역할과 관계가 보다 직관적으로 드러나며, 문서화 및 향후 기능 확장 시 일관된 사용자 경험을 제공할 수 있게 되었습니다.

#### 모델 수 제한 해제

기존에는 NPU 코어 수에 따라 동시에 구동할 수 있는 모델 수에 제한이 존재했습니다. 이번 업데이트에서는 해당 구조를 개선하여, 모델 수 제한을 해제하였습니다.

- 최신 qb Compiler로 컴파일된 모델은 컴파일 시 지정된 코어 모드와 무관하게 DRAM 메모리 범위 내에서 더 많은 모델을 동시에 로딩 및 실행할 수 있습니다.

이를 통해 다음과 같은 이점을 얻을 수 있습니다.

- 다수의 모델을 동시에 운용하는 서비스 시나리오에서 유연성 향상
- 다양한 코어 모드의 모델들을 동시에 사용 가능
- LLM 모델등 큰 모델 사용 상에 존재했던 코어 제약 제거

본 변경은 런타임의 내부 로직 변경및 최적화를 통해 이루어졌으며, 사용자 수준에서는 MXQv7 기반으로 컴파일된 모델이라면 기존 추론 코드 변경 없이 바로 적용됩니다.

#### 멀티 쓰레딩 성능 개선

이번 업데이트와 함께, c++ 라이브러리에서는 `.setActivationSlots(int num)`, 파이썬에서는 `.set_activation_slots(num)` 함수를 통해 NPU 추론과 데이터 이동 간의 파이프라이닝 최적화 방식을 더 자유롭게 사용하실 수 있도록 지원합니다.

위 함수는, 하나의 모델에서 사용할 입력 slot 수를 제어할 수 있는 방식을 제공하여 더 많은 slot 사용 시에는 NPU 메모리를 더 많이 차지하지만 npu 연산과의 파이프라이닝이 활성화되어 멀티 쓰레딩 상황에서의 성능면에서 이점을 얻을 수 있습니다.

```{note}
현재 LLM등 Cache를 사용하는 모델에 한해서는 activation slot 수가 1로 제한됩니다.
```

#### uint8 추론 지원

이번 업데이트에서는 uint8 정수형 기반 추론을 공식 지원합니다.

- uint8 양자화 모델을 qb Compiler에서 컴파일 가능
- qb Runtime에서 해당 모델의 추론 실행 지원

이를 통해 uint8 입력을 사용하는 모델들에 대해 전처리 과정에서 cpu 연산을 줄일 수 있습니다.

### 마이그레이션 가이드

이름 통일에 따라 패키지, 헤더, 모듈 이름이 변경되었습니다. 기존 패키지는 더 이상 유지보수되지 않습니다.

#### 설치

##### I. APT 패키지 목록 업데이트

패키지 설치 전에 APT 패키지 인덱스를 최신 상태로 갱신합니다.

``` bash
sudo apt update
```

##### II. 런타임 라이브러리 설치

기존 패키지명 `mobilint-npu-runtime` 에서 `mobilint-qb-runtime` 으로 변경되었습니다.

``` bash
# C++ 라이브러리
sudo apt install mobilint-qb-runtime

# 파이썬 패키지
pip install mobilint-qb-runtime
```

##### III. 드라이버 설치

드라이버 패키지 역시 네이밍 정책 변경에 따라 설치 패키지명이 `aries-driver` 에서 `mobilint-aries-driver`로 변경되었습니다.

``` bash
sudo apt install mobilint-aries-driver
```

#### C++ 라이브러리 변경 사항

- 컴파일 링킹 플래그 변경

    ```bash
    # 기존 컴파일 방식
    g++ -o example example.cpp -lmaccel

    # 변경된 컴파일 방식
    g++ -o example example.cpp -lqbruntime
    ```

- 헤더 파일 경로 변경

    ```cpp
    // 기존 헤더 파일
    # include "maccel/maccel.h"

    // 변경된 헤더 파일
    # include "qbruntime/qbruntime.h"
    ```

#### 파이썬 패키지 변경 사항

- 모듈 이름 변경

    ```python
    # 기존 파이썬 모듈
    import maccel

    # 변경된 파이썬 모듈
    import qbruntime
    ```
