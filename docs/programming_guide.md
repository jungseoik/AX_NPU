# 프로그래밍 가이드

본 문서는 모빌린트에서 제공하는 런타임 `qb Runtime`(이하 런타임)과 유틸리티 도구 (`mobilint-cli`)의 사용 방법을 설명합니다.

런타임의 주요 구성 요소와 일반적인 추론 절차에 대해 안내합니다.

```{warning}
ARIES 기반 장치를 사용하기 위해서는 먼저 드라이버와 런타임 라이브러리가 설치되어 있어야 합니다. 설치 관련 안내는 [드라이버 설치](installing_driver.md), [런타임 라이브러리 설치](installing_runtime_library.md)를 참고하세요.
```

## 모빌린트 런타임의 주요 기능

모빌린트의 런타임은 C++, Python 애플리케이션에 모빌린트 NPU를 통합할 수 있도록 유연하고 안정적인 API를 제공합니다. 다음은 주요 기능과 지원하는 동작들입니다:

- C++ : C++ 라이브러리는 {doxylink}`StatusCode <mobilint::StatusCode>`를 통해서 방어적 프로그래밍을 지원합니다.

- Python : 각 메서드 혹은 기능의 성공 또는 실패 여부를 내부적으로 처리하며, 오류 발생 즉시 예외를 발생시키고 종료합니다.

- 현재 지원되는 입력 데이터 타입은 `UINT8`, `INT8`, `float32` 입니다.

- `qb Runtime`은 {doxylink}`NDArray <mobilint::NDArray>` 클래스를 통해 고차원 데이터의 추론을 지원합니다. 아래와 같이 `NDArray`를 구성할 수 있습니다.

```cpp
mobilint::StatusCode sc;
mobilint::NDArray<float> input({224, 224, 3}, sc);
```

```{tip}
`NDArray`의 초기화 방법에 대한 더 다양한 방법들은 API Reference를 참고해주세요.
```

## 추론 절차

런타임을 사용하기 위해선 **MXQ (Mobilint ExeCUtable)** 파일이 필요합니다. MXQ 파일은 모빌린트 공식 컴파일러인 `qb Compiler` 를 통해 모빌린트 NPU에서 작동할 수 있도록 최적화된 모델 형식입니다.

`qb Runtime`을 사용해서 모빌린트 NPU에서 추론을 수행하려면 보통 다음의 4단계를 거칩니다:

1. NPU 장치를 불러옵니다. [(1단계: `Accelerator`)](#accelerator)
2. 컴파일된 모델(MXQ)를 불러옵니다. [(2단계: `Model`)](#model)
3. 모델을 NPU 장치에 업로드합니다. [(3단계: `Model` 정보를 `Accelerator`에 전달)](#model-accelerator)
4. 사용자 입력을 사용하여 추론을 수행합니다. [(4단계: 전달된 `Model` 정보와 `input` 데이터 사용하여 추론 실행)](#model-input)

`qb Runtime`에서는 NPU 장치와 MXQ 모델을 각각 추상화한 `Accelerator`, `Model`이라는 두 객체를 사용해 추론 과정을 수행합니다.

(accelerator)=
### 1단계: `Accelerator`

{doxylink}`Accelerator <mobilint::Accelerator>` 객체는 사용할 NPU 장치를 나타냅니다. 이는 인식된 장치 이름의 번호를 통해 각 장치를 구별합니다. 예를 들어 `/dev/aries0`은 `Acclerator0`, `/dev/aries1`은 `Accelerator1`에 연결하여 사용할 수 있습니다. `Accelerator` 객체 생성자에 어떠한 장치 숫자도 제공하지 않은 경우, `0`번 디바이스를 기본으로 사용합니다.

```cpp
// C++ 예시
mobilint::StatusCode sc;
auto acc = mobilint::Accelerator::create(0, sc);
if (!sc) {
	fprintf(stderr, "Error code %d\n", int(sc));
	exit(1);
}
```
```python
# Python 예시
acc = qbruntime.Accelerator(0)
```

(model)=
### 2단계: `Model`

{doxylink}`Model <mobilint::Model>` 객체는 MXQ 파일에 포함된 모델을 나타냅니다. 생성 시점에 MXQ 파일을 읽어 필요한 정보들을 내부에 저장합니다. 이후 `Accelerator` 객체를 사용해 NPU에서 모델 추론을 수행합니다.

```cpp
// C++ 예제
auto model = mobilint::Model::create(MXQ_FILE_PATH, sc);
if (!sc) {
	fprintf(stderr, "Error code %d\n", int(sc));
	exit(1);
}
```
```python
# Python 예제
model = qbruntime.Model(MXQ_FILE_PATH)
```

(model-accelerator)=
### 3단계: `Model` 정보를 `Accelerator`에 전달

{doxylink}`launch() <mobilint::Model::launch(Accelerator& acc)>` 메서드를 사용하여 `Model` 객체의 정보를 `Accelerator`에 전달합니다.

### 4단계: 전달된 `Model` 정보와 `input` 데이터 사용하여 추론 실행

입력 데이터를 준비한 뒤, `mobilint::Model::infer()` 메서드를 사용해 추론을 수행합니다.

```cpp
// C++ 예제
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
# Python 예제
model.launch(acc)
result = model.infer([INPUT])
```

(model-input)=
### 추론 시나리오

```cpp
// C++ 예제
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
## Python 예제
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

## C++ 소스 코드 컴파일

### Linux/Ubuntu

런타임 라이브러리의 설치 방식에 따라 컴파일 절차가 다릅니다.

- 시스템 전역에 설치된 경우 (`apt` 또는 `sudo apt install`로 설치):

```bash
g++ -o {outfile_name} {source_code} -lqbruntime
```

- 설치 없이 배포된 라이브러리를 직접 사용하는 경우 (배포 사이트에서 다운로드한 파일 사용):

```bash
g++ -o {outfile_name} -I{path/to/include} -L{path/to/library} {source_code} -lqbruntime
```

(compile-windows)=
### Windows

Windows에서는 런타임 라이브러리는 설치 없이 사용하는 방법으로 지원됩니다. 컴파일 시 `include` 경로와 라이브러리 `lib` 경로를 수동으로 지정해야합니다.

1. 작업 중인 Visual Studio 프로젝트를 엽니다.

2. 아래 버튼을 클릭하여 프로젝트 설정을 수정합니다.

	![Project Setting](/res/image/project_setting.png "project setting")

3. 아래 이미지처럼 C/C++ 설정에 `include` 폴더를, Linker 설정에 `lib` 폴더를 각각 추가합니다.

	![Include](/res/image/project_setting_include_dir.png "Include")

	![Library](/res/image/project_setting_library_dir.png "Library")

4. 빌드 모드에 따라 아래 설정을 적용합니다.
	- Release 모드 : `qbruntime.lib` 사용
	- Debug 모드 : `qbruntimed.lib` 사용

	![Build Mode](/res/image/project_setting_build_mode.png "Build mode")
