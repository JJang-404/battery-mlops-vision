# Inferencer.cpp - run 함수와 GetTensorMutableData 정리

## 질문 1. `run()` 함수는 무엇을 하는가?

`Inferencer::run()`은 이미 전처리가 끝난 입력 데이터를 ONNX 모델에 넣고, 모델 출력값을 `vector<float>`로 돌려주는 함수다.

현재 코드:

```cpp
vector<float> Inferencer::run(const vector<float>& inputData) {
    auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memInfo,
        const_cast<float*>(inputData.data()), inputData.size(),
        INPUT_SHAPE, 4
    );

    const char* inNames[] = {inputName_.c_str()};
    const char* outNames[] = {outputName_.c_str()};

    auto outputs = session_.Run(Ort::RunOptions{nullptr},
                                inNames, &inputTensor, 1, outNames, 1
    );

    float* dataPtr = outputs[0].GetTensorMutableData<float>();
    size_t count = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
    return vector<float>(dataPtr, dataPtr + count);
}
```

전체 흐름은 다음과 같다.

```text
inputData
    |
    | 1. ONNX Runtime tensor로 감싸기
    v
inputTensor
    |
    | 2. session_.Run()으로 모델 실행
    v
outputs
    |
    | 3. 출력 tensor 내부 float 데이터 주소 얻기
    v
dataPtr
    |
    | 4. vector<float>로 복사해서 반환
    v
return vector<float>
```

## `inputData`는 어디서 오는가?

`inputData`는 `Preprocessor`가 만든 NCHW 형식의 float 배열이다.

```text
원본 이미지
    |
    | Preprocessor::LoadAndPreprocess()
    | 또는 Preprocessor::matToNCHW()
    v
vector<float> inputData
```

이 프로젝트 기준으로 `inputData`의 논리적 shape는 다음과 같다.

```text
1 x 3 x 513 x 513
```

실제 C++ 타입은 1차원 배열인 `vector<float>`다.

```text
개수 = 1 * 3 * 513 * 513
```

## 1단계. `MemoryInfo` 만들기

```cpp
auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
```

이 코드는 ONNX Runtime에게 "이 tensor 데이터는 CPU 메모리에 있다"고 알려준다.

현재 `inputData`는 `std::vector<float>`이고, 일반 CPU 메모리에 저장되어 있다. 그래서 `CreateCpu()`를 사용한다.

```text
OrtArenaAllocator  ONNX Runtime의 CPU arena allocator 사용
OrtMemTypeDefault  기본 메모리 타입
```

처음에는 이렇게 이해하면 충분하다.

```text
inputData는 CPU 메모리에 있는 데이터다.
이 정보를 ONNX Runtime tensor 생성 시 같이 넘겨준다.
```

## 2단계. `std::vector<float>`를 `Ort::Value` tensor로 감싸기

```cpp
Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
    memInfo,
    const_cast<float*>(inputData.data()), inputData.size(),
    INPUT_SHAPE, 4
);
```

ONNX Runtime은 `std::vector<float>` 자체를 바로 모델 입력으로 받지 않는다.

모델에 넣으려면 ONNX Runtime이 이해하는 tensor 객체인 `Ort::Value`로 만들어야 한다.

각 인자의 의미는 다음과 같다.

```text
memInfo                           inputData가 어떤 메모리에 있는지
const_cast<float*>(inputData.data()) inputData의 실제 데이터 시작 주소
inputData.size()                  float 원소 개수
INPUT_SHAPE                       tensor shape = {1, 3, 513, 513}
4                                 shape 차원 개수
```

여기서 중요한 점은 `CreateTensor`가 `inputData`의 데이터를 새로 복사하는 방식이 아니라, 기존 `inputData`의 메모리를 tensor처럼 바라보게 만드는 방식이라는 점이다.

그래서 `inputTensor`가 살아있는 동안 `inputData`도 같이 살아 있어야 한다. 현재 함수에서는 `inputData`가 함수 인자로 들어와 있고, `session_.Run()`이 끝날 때까지 유효하므로 괜찮다.

## 왜 `const_cast`를 쓰는가?

`inputData`는 함수 인자에서 `const vector<float>&`다.

```cpp
vector<float> Inferencer::run(const vector<float>& inputData)
```

즉 이 함수는 입력 데이터를 수정하지 않겠다는 의미다.

그런데 ONNX Runtime의 `CreateTensor<float>()` API는 내부적으로 수정하지 않더라도 `float*` 포인터를 요구한다.

그래서 아래처럼 `const`를 제거해서 넘긴다.

```cpp
const_cast<float*>(inputData.data())
```

이 코드는 "API 모양 때문에 const를 제거해서 넘기지만, 우리가 의도적으로 inputData를 수정하려는 것은 아니다"라고 이해하면 된다.

## 3단계. 입력 이름과 출력 이름 준비

```cpp
const char* inNames[] = {inputName_.c_str()};
const char* outNames[] = {outputName_.c_str()};
```

ONNX Runtime은 모델을 실행할 때 "어떤 입력 이름에 어떤 tensor를 넣을지"를 알아야 한다.

예를 들어 모델 입력 이름이 `"input"`이라면:

```text
input 이라는 입력 슬롯에 inputTensor를 넣는다
```

출력도 마찬가지다.

```text
outputName_에 해당하는 출력 tensor를 받아온다
```

`inputName_`과 `outputName_`은 생성자에서 모델 metadata를 조회해서 저장한 값이다.

## 4단계. `session_.Run()`으로 실제 추론 실행

```cpp
auto outputs = session_.Run(Ort::RunOptions{nullptr},
                            inNames, &inputTensor, 1,
                            outNames, 1
);
```

이 줄이 실제 모델 실행이다.

각 인자의 의미는 다음과 같다.

```text
Ort::RunOptions{nullptr}  기본 실행 옵션
inNames                  입력 이름 배열
&inputTensor             입력 tensor 주소
1                        입력 tensor 개수
outNames                 받고 싶은 출력 이름 배열
1                        출력 tensor 개수
```

실행 결과는 `outputs`에 담긴다.

```cpp
auto outputs = ...
```

`outputs`는 보통 `std::vector<Ort::Value>` 형태로 생각하면 된다.

현재 모델은 출력이 하나라고 가정하므로 `outputs[0]`을 사용한다.

## 질문 2. `GetTensorMutableData<float>()`는 무엇인가?

문제의 코드:

```cpp
float* dataPtr = outputs[0].GetTensorMutableData<float>();
```

이 코드는 `outputs[0]` tensor 내부에 들어있는 실제 float 데이터의 시작 주소를 가져온다.

즉 `dataPtr`은 모델 출력 tensor의 첫 번째 float 값을 가리키는 포인터다.

```text
outputs[0]
    |
    | GetTensorMutableData<float>()
    v
float* dataPtr
```

`dataPtr`을 배열처럼 보면 다음처럼 접근할 수 있다.

```cpp
float first = dataPtr[0];
float second = dataPtr[1];
```

## MutableData에서 `Mutable`은 무슨 뜻인가?

`Mutable`은 "수정 가능한"이라는 뜻이다.

```cpp
GetTensorMutableData<float>()
```

이 함수는 `float*`를 반환한다. `float*`는 값을 읽을 수도 있고, 마음만 먹으면 수정할 수도 있다.

```cpp
dataPtr[0] = 0.0f; // 문법상 가능
```

하지만 이 코드에서는 출력값을 수정하려는 목적이 아니다. 단지 ONNX Runtime tensor 안에 있는 데이터를 읽어서 `vector<float>`로 복사하려는 것이다.

```cpp
return vector<float>(dataPtr, dataPtr + count);
```

즉 이름은 `MutableData`지만, 여기서는 "출력 tensor 내부 데이터 주소를 얻기 위한 함수"로 사용하고 있다.

## 왜 중요한가?

중요하다.

`session_.Run()`의 결과는 `Ort::Value`라는 ONNX Runtime 객체 안에 들어 있다. 그런데 후처리 코드는 보통 C++ 표준 컨테이너인 `vector<float>`로 다루는 것이 편하다.

그래서 아래 과정이 필요하다.

```text
Ort::Value 출력 tensor
    |
    | GetTensorMutableData<float>()
    v
float* dataPtr
    |
    | vector<float>(dataPtr, dataPtr + count)
    v
std::vector<float>
```

이 줄이 없으면 모델 출력값을 일반 C++ 코드에서 쉽게 꺼내 쓰기 어렵다.

## `count`는 왜 구하는가?

```cpp
size_t count = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
```

`dataPtr`은 시작 주소만 알려준다.

하지만 포인터만으로는 데이터가 몇 개 있는지 알 수 없다.

그래서 출력 tensor의 원소 개수를 따로 구한다.

```text
dataPtr = 시작 주소
count   = 원소 개수
```

이 둘을 합쳐서 `vector<float>`를 만든다.

```cpp
return vector<float>(dataPtr, dataPtr + count);
```

여기서 `dataPtr + count`는 마지막 원소 다음 주소를 의미한다.

C++의 iterator 범위는 보통 아래 형식이다.

```text
[begin, end)
```

즉 시작은 포함하고, 끝은 포함하지 않는다.

## 왜 바로 `dataPtr`을 반환하지 않고 `vector<float>`로 복사하는가?

`dataPtr`은 `outputs[0]` 내부 메모리를 가리키는 포인터다.

그런데 `outputs`는 `run()` 함수 안의 지역 변수다.

```cpp
auto outputs = session_.Run(...);
```

함수가 끝나면 `outputs`는 사라진다. 그러면 `dataPtr`이 가리키던 메모리도 더 이상 안전하게 사용할 수 없다.

그래서 함수가 끝나기 전에 데이터를 `vector<float>`로 복사해서 반환한다.

```cpp
return vector<float>(dataPtr, dataPtr + count);
```

이렇게 하면 반환된 vector는 자기 메모리를 따로 가지므로 함수 밖에서도 안전하게 사용할 수 있다.

## 한 줄씩 다시 보기

```cpp
auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
```

입력 데이터가 CPU 메모리에 있다는 정보를 만든다.

```cpp
Ort::Value inputTensor = Ort::Value::CreateTensor<float>(...);
```

`vector<float>` 입력 데이터를 ONNX Runtime tensor로 감싼다.

```cpp
const char* inNames[] = {inputName_.c_str()};
const char* outNames[] = {outputName_.c_str()};
```

모델의 입력 이름과 출력 이름을 준비한다.

```cpp
auto outputs = session_.Run(...);
```

ONNX 모델을 실제로 실행한다.

```cpp
float* dataPtr = outputs[0].GetTensorMutableData<float>();
```

첫 번째 출력 tensor 안의 실제 float 데이터 시작 주소를 얻는다.

```cpp
size_t count = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
```

출력 tensor의 float 개수를 구한다.

```cpp
return vector<float>(dataPtr, dataPtr + count);
```

출력 데이터를 안전하게 `vector<float>`로 복사해서 반환한다.

## 주의할 점

`dataPtr`은 `outputs[0]`이 살아 있는 동안만 안전하다.

아래처럼 포인터만 밖으로 반환하면 위험하다.

```cpp
float* Inferencer::badRun(...) {
    auto outputs = session_.Run(...);
    return outputs[0].GetTensorMutableData<float>(); // 위험
}
```

함수가 끝나면 `outputs`가 사라져서 포인터가 무효가 될 수 있다.

현재 코드처럼 `vector<float>`로 복사해서 반환하는 방식은 안전하다.

## 한 줄 요약

`run()`은 ONNX 모델을 실제로 실행하는 함수이고, `GetTensorMutableData<float>()`는 모델 출력 tensor 내부의 float 데이터 시작 주소를 얻는 중요한 함수다. 다만 그 포인터는 임시로만 쓰고, 현재 코드처럼 `vector<float>`로 복사해서 반환하는 것이 안전하다.
