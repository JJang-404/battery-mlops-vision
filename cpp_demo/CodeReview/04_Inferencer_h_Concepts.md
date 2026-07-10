# Inferencer.h 개념 정리

## 질문 1. `Inferencer`는 무슨 클래스인가?

`Inferencer`는 ONNX 모델을 로드하고, 전처리된 입력 데이터를 모델에 넣어서 추론 결과를 받아오는 클래스다.

이 프로젝트의 전체 흐름에서 위치를 보면 다음과 같다.

```text
원본 이미지
    |
    | Preprocessor
    v
NCHW float vector
    |
    | Inferencer
    v
모델 출력 logits
    |
    | Postprocessor
    v
결함 mask / overlay / 판정 결과
```

즉 `Preprocessor`가 이미지를 모델 입력 형태로 바꾸고, `Inferencer`는 그 입력을 실제 ONNX 모델에 넣어 실행한다.

기존 C# 프로젝트의 `Inferencer.cs`와 같은 역할이다.

```cpp
class Inferencer {
public:
    explicit Inferencer(const string& onnxPath, bool useGPU = true);
    bool usingGPU() const { return gpu_; }

private:
    Ort::Env env_;
    Ort::Session session_{nullptr};
    Ort::AllocatorWithDefaultOptions allocator_;
    std::string inputName_, outputName_;
    bool gpu_ = false;

    static constexpr int64_t INPUT_SHAPE[4] = {1, 3, 513, 513};
};
```

각 멤버의 큰 역할은 다음과 같다.

```text
env_        ONNX Runtime 전체 실행 환경
session_    ONNX 모델을 로드한 실행 세션
allocator_  ONNX Runtime이 문자열/메타데이터를 만들 때 쓰는 메모리 할당자
inputName_  모델 입력 이름
outputName_ 모델 출력 이름
gpu_        GPU 사용 성공 여부
INPUT_SHAPE 모델 입력 tensor shape
```

## 질문 2. `explicit`는 무엇인가?

`explicit`는 생성자 앞에 붙여서 "암시적 변환을 막는" 키워드다.

현재 생성자는 아래 형태다.

```cpp
explicit Inferencer(const string& onnxPath, bool useGPU = true);
```

이 생성자는 문자열 하나만 넣어도 객체를 만들 수 있다.

```cpp
Inferencer inferencer("models/battery_deeplab_v1.onnx");
```

문제는 `explicit`가 없으면 C++이 어떤 상황에서 문자열을 자동으로 `Inferencer` 객체로 바꾸려고 할 수 있다는 점이다.

예를 들어 이런 함수가 있다고 하자.

```cpp
void runModel(Inferencer inferencer);
```

`explicit`가 없으면 아래 같은 호출이 실수로 허용될 수 있다.

```cpp
runModel("models/battery_deeplab_v1.onnx");
```

C++이 문자열을 보고 "아, `Inferencer` 생성자로 바꿀 수 있네" 하고 자동 변환할 수 있기 때문이다.

`explicit`를 붙이면 이런 자동 변환을 막고, 개발자가 의도적으로 객체를 만들게 한다.

```cpp
runModel(Inferencer("models/battery_deeplab_v1.onnx"));
```

한 줄로 말하면:

```text
explicit = 생성자를 이용한 원치 않는 자동 변환을 막는 안전장치
```

생성자 인자가 1개이거나, 기본값 때문에 사실상 1개처럼 호출 가능한 생성자에는 `explicit`를 붙이는 습관이 좋다.

## 질문 3. `Ort`는 무엇인가?

`Ort`는 ONNX Runtime C++ API가 제공하는 namespace다.

```cpp
#include <onnxruntime_cxx_api.h>
```

이 헤더를 include하면 `Ort::Env`, `Ort::Session`, `Ort::Value` 같은 C++ 클래스들을 사용할 수 있다.

여기서 `Ort`는 보통 ONNX Runtime의 줄임 표현이다.

```text
ONNX Runtime -> ORT
```

정확히는 C++ 코드에서 아래처럼 사용한다.

```cpp
Ort::Env
Ort::Session
Ort::AllocatorWithDefaultOptions
Ort::Value
Ort::MemoryInfo
Ort::RunOptions
Ort::Exception
```

주의할 점은 `ORT::`가 아니라 `Ort::`다. C++ namespace 이름은 대소문자를 구분한다.

## 자주 보이는 `Ort::` 문장 해석

### `Ort::Env env_;`

ONNX Runtime 실행 환경 객체다.

보통 프로그램 또는 모델 실행의 기반이 되는 환경을 만든다고 보면 된다. 로그 설정, 런타임 초기화 같은 역할을 한다.

가이드의 `.cpp`에서는 생성자 초기화 목록에서 이렇게 초기화한다.

```cpp
Inferencer::Inferencer(const std::string& onnxPath, bool useGpu)
    : env_(ORT_LOGGING_LEVEL_WARNING, "battery_demo")
{
}
```

뜻은:

```text
warning 이상 로그만 출력하고,
환경 이름은 battery_demo로 둔다.
```

### `Ort::Session session_{nullptr};`

ONNX 모델을 로드한 실행 세션이다.

모델 파일을 열고 나면 실제 추론은 이 `session_`을 통해 수행한다.

```cpp
session_ = Ort::Session(env_, onnxPath.c_str(), opts);
```

뜻은:

```text
env_ 환경에서
onnxPath 경로의 모델을
opts 설정으로 로드해서
session_에 저장한다.
```

`{nullptr}`로 시작하는 이유는 멤버 변수 선언 시점에는 아직 모델 경로와 옵션이 없기 때문이다. 생성자 본문에서 실제 세션을 만들어 대입한다.

### `Ort::AllocatorWithDefaultOptions allocator_;`

ONNX Runtime이 문자열이나 메타데이터를 반환할 때 사용할 기본 메모리 할당자다.

예를 들어 모델의 input 이름을 가져올 때 사용한다.

```cpp
auto inName = session_.GetInputNameAllocated(0, allocator_);
```

### `Ort::SessionOptions opts;`

세션을 만들 때 적용할 옵션이다.

예를 들어 그래프 최적화, CUDA/CoreML 같은 실행 provider 설정을 여기에 넣는다.

```cpp
Ort::SessionOptions opts;
opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
```

### `Ort::Value inputTensor`

ONNX Runtime에 실제로 넘길 tensor 객체다.

`std::vector<float>` 자체를 바로 모델에 던지는 것이 아니라, ONNX Runtime이 이해하는 `Ort::Value` tensor로 감싸서 전달한다.

```cpp
Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
    memInfo,
    const_cast<float*>(inputData.data()),
    inputData.size(),
    INPUT_SHAPE,
    4
);
```

뜻은:

```text
inputData 벡터의 메모리를 사용해서
shape가 [1, 3, 513, 513]인 float tensor를 만든다.
```

### `Ort::Exception`

ONNX Runtime에서 발생하는 C++ 예외 타입이다.

예를 들어 GPU provider 등록 실패, 모델 로드 실패, tensor shape 불일치 같은 상황에서 발생할 수 있다.

```cpp
catch (const Ort::Exception& e) {
    std::cout << e.what() << "\n";
}
```

## 질문 4. `INPUT_SHAPE`는 왜 `static constexpr`로 했는가?

먼저 용어를 바로 잡으면, `static constexpr`는 예외처리가 아니다.

```cpp
static constexpr int64_t INPUT_SHAPE[4] = {1, 3, 513, 513};
```

이 코드는 "컴파일 시점에 정해지는 클래스 공통 상수"를 만드는 코드다.

각 단어의 의미는 다음과 같다.

```text
static     객체마다 따로 갖지 않고 클래스가 공유하는 값
constexpr 컴파일 시점에 확정 가능한 상수
int64_t   64비트 정수 타입
[4]       원소 4개짜리 배열
```

왜 필요한가?

ONNX Runtime은 tensor를 만들 때 shape 정보를 필요로 한다.

```cpp
INPUT_SHAPE = {1, 3, 513, 513}
```

각 값의 의미는 다음과 같다.

```text
1   batch size
3   channel
513 height
513 width
```

`static`으로 둔 이유:

```text
모든 Inferencer 객체가 같은 모델 입력 shape를 사용하기 때문
객체마다 INPUT_SHAPE 배열을 따로 가질 필요가 없음
```

`constexpr`로 둔 이유:

```text
실행 중 바뀌면 안 되는 값이고,
컴파일 시점에 이미 알 수 있는 고정값이기 때문
```

즉 이 코드는 예외처리가 아니라, "이 모델은 항상 이 입력 크기를 쓴다"는 약속을 코드에 박아둔 것이다.

다만 모델을 바꿔서 입력 크기가 바뀌면 이 값도 같이 바꿔야 한다.

## 질문 5. `<stdexcept>`는 `.cpp`와 `.h` 아무 곳에서나 사용해도 되는가?

기술적으로는 `.cpp`와 `.h` 둘 다에서 사용할 수 있다.

하지만 기준이 있다.

```text
헤더 파일(.h) 안에서 std::runtime_error 같은 타입을 직접 사용하면 .h에 include
구현 파일(.cpp) 안에서만 throw하면 .cpp에 include
```

예를 들어 `.cpp`에서만 아래처럼 쓴다면:

```cpp
#include <stdexcept>

void foo() {
    throw std::runtime_error("error");
}
```

이 경우에는 `.cpp`에만 `<stdexcept>`를 include하면 된다.

반대로 헤더 안에서 직접 사용한다면 헤더에 include해야 한다.

```cpp
// SomeHeader.h
#pragma once
#include <stdexcept>

class SomeClass {
public:
    void check() {
        throw std::runtime_error("error");
    }
};
```

또는 함수 선언에 예외 타입이 직접 등장하는 경우도 헤더에 필요하다.

```cpp
std::runtime_error makeError();
```

이 프로젝트 기준으로는 보통 다음이 좋다.

```text
Preprocessor.h     선언만 둔다
Preprocessor.cpp   이미지 로드 실패 throw가 있으므로 <stdexcept> include

Inferencer.h       선언과 멤버 변수만 둔다
Inferencer.cpp     모델 로드, GPU 등록, 추론 실패 처리에서 필요하면 <stdexcept> 또는 Ort::Exception 처리
```

즉 아무 곳에나 넣어도 컴파일은 될 수 있지만, 필요한 곳에만 include하는 것이 바람직하다.

## 정리

`Inferencer`는 ONNX 모델 실행 담당 클래스다.

`explicit`는 생성자의 의도치 않은 자동 변환을 막는다.

`Ort::`는 ONNX Runtime C++ API의 namespace다.

`INPUT_SHAPE`의 `static constexpr`는 예외처리가 아니라 모델 입력 크기를 나타내는 컴파일 타임 상수다.

`<stdexcept>`는 `.cpp`와 `.h` 모두에서 쓸 수 있지만, 실제로 그 타입을 사용하는 파일에 include하는 것이 원칙이다.
