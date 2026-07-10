# Preprocessor.cpp - `#include <stdexcept>`

## 한 줄 요약

`#include <stdexcept>`는 C++ 표준 예외 클래스들을 사용하기 위해 포함하는 헤더이다.

예를 들어 잘못된 입력값, 실행 중 오류, 범위 초과 같은 상황을 `throw`로 알려줄 때 사용한다.

## 자주 쓰는 예외 클래스

```cpp
#include <stdexcept>

throw std::runtime_error("실행 중 오류가 발생했습니다.");
throw std::invalid_argument("입력값이 올바르지 않습니다.");
throw std::out_of_range("인덱스가 범위를 벗어났습니다.");
```

대표적으로 아래 예외들이 있다.

- `std::runtime_error`: 프로그램 실행 중 발생한 일반적인 오류
- `std::invalid_argument`: 함수 인자가 잘못되었을 때
- `std::out_of_range`: 배열, 벡터 등의 범위를 벗어난 접근
- `std::logic_error`: 코드 논리상 잘못된 상태

## Preprocessor에서 왜 필요할 수 있나?

전처리 과정에서는 이미지 크기, 채널 수, 입력 텐서 크기 등이 예상과 다르면 이후 추론 결과가 잘못될 수 있다.

이럴 때 조용히 넘어가지 않고 예외를 던져서 문제를 바로 알릴 수 있다.

```cpp
if (image.empty()) {
    throw std::invalid_argument("input image is empty");
}
```

## 기억할 점

`<stdexcept>`는 "오류 상황을 명확하게 표현하기 위한 표준 예외 도구"라고 생각하면 된다.

`throw std::...` 형태의 코드를 만나면, 해당 지점에서 오류를 발생시키고 호출한 쪽으로 문제를 전달한다는 뜻이다.

---

## 추가 질문 1. `planeSize`는 무엇인가?

가이드 코드에는 보통 아래처럼 나온다.

```cpp
const int planeSize = INPUT_SIZE * INPUT_SIZE;
```

여기서 `plane`은 "한 채널짜리 2D 이미지 한 장"이라고 이해하면 된다.

예를 들어 모델 입력 크기가 `513 x 513`이면:

```text
planeSize = 513 * 513 = 263,169
```

즉 `planeSize`는 한 채널이 가지고 있는 픽셀 개수다.

3채널 이미지라면 채널별로 plane이 3개 있다.

```text
B 채널 plane: 513 x 513 = 263,169개
G 채널 plane: 513 x 513 = 263,169개
R 채널 plane: 513 x 513 = 263,169개
```

그래서 전체 float 개수는 다음과 같다.

```text
3 * planeSize
= 3 * 263,169
= 789,507
```

이 값이 ONNX 모델에 들어가는 이미지 1장의 전체 입력 데이터 개수다.

## 추가 질문 2. `vector<float> out(3 * planeSize);`에서 `out`은 무엇인가?

```cpp
std::vector<float> out(3 * planeSize);
```

`out`은 전처리가 끝난 결과를 담는 float 배열이다.

이 배열은 `cv::Mat` 이미지를 ONNX 모델이 받을 수 있는 형태로 바꾼 최종 결과물이다.

흐름은 다음과 같다.

```text
cv::Mat 이미지
    |
    | resize
    | 정규화
    | HWC -> NCHW 변환
    v
std::vector<float> out
```

`out`이라는 이름은 output의 줄임말로 보면 된다. 여기서는 "전처리 함수의 출력값"이라는 의미다.

예를 들어 `matToNCHW()` 함수는 대략 이런 역할을 한다.

```cpp
std::vector<float> Preprocessor::matToNCHW(const cv::Mat& bgr) {
    const int planeSize = INPUT_SIZE * INPUT_SIZE;
    std::vector<float> out(3 * planeSize);

    // bgr 이미지 픽셀을 읽어서 out에 NCHW 순서로 저장

    return out;
}
```

즉 `out`은 단순히 임시 변수처럼 보이지만, 실제로는 추론 모델에 넘겨줄 입력 tensor 데이터다.

## 왜 `3 * planeSize`인가?

이미지는 3채널이기 때문이다.

```text
planeSize = 한 채널의 픽셀 개수
3 * planeSize = 3채널 전체 픽셀 개수
```

`INPUT_SIZE = 513`이면:

```text
planeSize = 513 * 513
out 크기  = 3 * 513 * 513
```

ONNX Runtime에 batch size 1로 넣으면 논리적 shape는 다음과 같다.

```text
1 x 3 x 513 x 513
```

하지만 C++의 `std::vector<float>`는 실제 메모리에서 1차원 배열이다.

그래서 4차원 tensor를 1차원 배열에 아래 순서로 펼쳐서 저장한다.

```text
채널 0 전체 -> 채널 1 전체 -> 채널 2 전체
```

## `out`에 값을 넣는 index 계산

보통 아래처럼 저장한다.

```cpp
out[c * planeSize + y * INPUT_SIZE + x] = value;
```

각 항목의 의미는 다음과 같다.

```text
c * planeSize       c번째 채널의 시작 위치
y * INPUT_SIZE      y번째 행의 시작 위치
x                   x번째 열 위치
```

예를 들어 `INPUT_SIZE = 513`, `c = 2`, `y = 10`, `x = 20`이면:

```text
index = 2 * 513 * 513 + 10 * 513 + 20
```

이렇게 계산하면 1차원 `vector<float>` 안에서도 "2번 채널의 10행 20열" 위치를 정확히 찾아갈 수 있다.

## 이름을 `out`으로 써도 괜찮은가?

짧은 함수 안에서는 `out`도 괜찮다.

다만 공부하거나 협업하는 코드라면 아래처럼 더 구체적인 이름도 좋다.

```cpp
std::vector<float> nchw;
std::vector<float> inputTensor;
std::vector<float> preprocessed;
```

이 프로젝트에서는 `matToNCHW()`의 반환값이라는 점을 생각하면 `nchw` 또는 `inputTensor`가 의미를 더 잘 드러낸다.

## 한 줄 요약

`planeSize`는 한 채널 이미지의 픽셀 개수이고, `out`은 OpenCV 이미지를 모델 입력용 NCHW float 배열로 변환한 최종 결과다.
