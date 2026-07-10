# Preprocessor.h: NCHW와 cv::Mat 입력 경로 정리

## 질문 1. NCHW가 무엇인가?

`NCHW`는 딥러닝 모델에 이미지를 넣을 때 사용하는 4차원 tensor 배열 순서다.

```text
N = Number of images, batch size
C = Channel
H = Height
W = Width
```

현재 프로젝트의 모델 입력 shape는 문서와 C# 코드 기준으로 아래와 같다.

```text
1 x 3 x 513 x 513
```

의미는 다음과 같다.

```text
N = 1      한 번에 이미지 1장 처리
C = 3      BGR 또는 RGB 3채널
H = 513    모델 입력 높이
W = 513    모델 입력 너비
```

즉 `NCHW`는 이미지 한 장을 이런 순서로 펼친다는 뜻이다.

```text
첫 번째 채널 전체
두 번째 채널 전체
세 번째 채널 전체
```

예를 들어 `513x513x3` 이미지를 `float[]` 또는 `std::vector<float>`로 만들면 크기는 다음과 같다.

```text
1 * 3 * 513 * 513 = 789,507개 float
```

## OpenCV의 기본 이미지 구조는 HWC에 가깝다

OpenCV의 `cv::Mat` 이미지는 보통 픽셀 단위로 저장된다.

```text
H x W x C
```

예를 들어 어떤 좌표 `(y, x)`에 접근하면 그 위치의 3채널 값이 함께 나온다.

```cpp
cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);
```

OpenCV 기본 색상 순서는 RGB가 아니라 BGR이다.

```text
pixel[0] = B
pixel[1] = G
pixel[2] = R
```

반면 ONNX 모델 입력은 보통 `NCHW` 형태를 기대한다.

```text
N x C x H x W
```

그래서 전처리에서 하는 핵심 작업 중 하나가 `HWC -> NCHW` 변환이다.

## HWC와 NCHW의 차이

OpenCV `cv::Mat` 관점:

```text
(y=0, x=0): B,G,R
(y=0, x=1): B,G,R
(y=0, x=2): B,G,R
...
```

NCHW tensor 관점:

```text
B 채널 전체: 513 x 513
G 채널 전체: 513 x 513
R 채널 전체: 513 x 513
```

기존 C# `Preprocessor.cs`도 같은 방식으로 저장한다.

```csharp
for (int c = 0; c < 3; c++)
{
    for (int h = 0; h < 513; h++)
    {
        for (int w = 0; w < 513; w++)
        {   
            var pixel = resized.At<Vec3b>(h, w);
            floatBuffer[index++] = pixel[c] / 255.0f;
        }
    }
}
```

이 코드는 채널을 먼저 돌고, 그 다음 높이와 너비를 돈다. 그래서 결과 배열은 `CHW` 순서가 된다. batch size `N=1`까지 포함해서 보면 `NCHW`다.

## C++에서 index 계산은 어떻게 하나?

`std::vector<float> out(3 * INPUT_SIZE * INPUT_SIZE);`를 만들고, 특정 채널 `c`, 좌표 `(y, x)` 값을 넣을 때 보통 아래처럼 계산한다.

```cpp
const int planeSize = INPUT_SIZE * INPUT_SIZE;
out[c * planeSize + y * INPUT_SIZE + x] = value;
```

의미는 다음과 같다.

```text
c * planeSize       해당 채널의 시작 위치로 이동
y * INPUT_SIZE      해당 행으로 이동
x                   해당 열로 이동
```

예를 들어 `c=1`, `y=10`, `x=20`, `INPUT_SIZE=513`이면:

```text
index = 1 * 513 * 513 + 10 * 513 + 20
```

이렇게 하면 G 채널 영역 안의 `(10, 20)` 위치에 값을 저장한다.

## 질문 2. 이미 로드된 이미지 Mat으로 호출 가능하다는 뜻은?

이 말은 "이미지 파일 경로를 다시 가져온다"는 뜻이 아니다.

`cv::Mat`은 OpenCV가 사용하는 이미지 객체다. 이미지가 이미 메모리에 올라와 있으면, 그 이미지 데이터를 바로 전처리 함수에 넘길 수 있다는 뜻이다.

파일 기반 흐름:

```text
이미지 경로 문자열
    |
    | cv::imread(path)
    v
cv::Mat
    |
    | matToNCHW(mat)
    v
NCHW float vector
```

카메라 기반 흐름:

```text
카메라 캡처
    |
    | camera.read(frame)
    v
cv::Mat frame
    |
    | matToNCHW(frame)
    v
NCHW float vector
```

즉 카메라로 이미지를 받은 경우에는 이미지 경로가 없다. 프레임이 이미 `cv::Mat`으로 메모리에 존재한다.

그래서 아래 두 함수를 분리하는 것이 적절하다.

```cpp
static std::vector<float> LoadAndPreprocess(const std::string& imagePath);
static std::vector<float> matToNCHW(const cv::Mat& bgr);
```

`LoadAndPreprocess()`는 테스트 이미지처럼 파일 경로가 있을 때 쓰고, 내부에서 `cv::imread()`로 `cv::Mat`을 만든 뒤 `matToNCHW()`를 호출한다.

`matToNCHW()`는 이미 `cv::Mat`이 있을 때 바로 쓴다.

## 실제 비전 프로그램에서도 이렇게 쓰는가?

맞다. 실제 비전 프로그램에서는 `cv::Mat` 또는 카메라 SDK가 제공하는 이미지 버퍼를 바로 처리하는 방식이 일반적이다.

산업용 비전 프로그램의 일반적인 흐름은 다음과 같다.

```text
카메라 Trigger
    |
이미지 Grab
    |
cv::Mat 또는 SDK Image Buffer
    |
전처리 resize / normalize / NCHW 변환
    |
ONNX Runtime 추론
    |
후처리 mask / bbox / defect 판정
    |
PLC, Stage, UI, 저장 시스템으로 결과 전달
```

운영 중인 비전 검사에서는 매 프레임을 디스크에 저장했다가 다시 읽는 방식이 비효율적이다.

```text
카메라 -> 파일 저장 -> 파일 다시 읽기 -> 추론
```

이 방식은 느리고, SSD I/O가 늘고, 실시간성이 떨어진다. 그래서 실제 검사 루프에서는 보통 아래처럼 처리한다.

```text
카메라 -> 메모리 프레임 -> 바로 추론
```

다만 검사 결과 기록, 불량 이미지 저장, 디버깅 목적이라면 원본 이미지나 오버레이 이미지를 파일로 저장할 수 있다. 저장은 "추론 입력을 만들기 위한 필수 과정"이 아니라 "로그와 추적성 확보를 위한 선택 과정"에 가깝다.

## test_images와 카메라 입력의 관계

`test_images/`는 개발과 검증을 위한 오프라인 입력이다.

장점은 다음과 같다.

- 카메라가 없어도 전처리, 추론, 후처리를 테스트할 수 있다.
- 같은 이미지로 반복 실행하므로 결과 비교가 쉽다.
- C# 구현과 C++ 구현의 출력 차이를 확인하기 좋다.

하지만 실제 장비에서는 보통 `test_images/` 대신 카메라에서 받은 프레임을 사용한다.

그래서 `Preprocessor`는 두 경로를 모두 지원하는 구조가 좋다.

```text
오프라인 테스트:
image path -> LoadAndPreprocess()

실제 장비:
camera frame -> matToNCHW()
```

## Preprocessor.h 관점에서 추천 구조

```cpp
#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

class Preprocessor {
public:
    static constexpr int INPUT_SIZE = 513;

    // 파일 경로 기반: test_images 같은 오프라인 테스트에 사용
    static std::vector<float> LoadAndPreprocess(const std::string& imagePath);

    // 메모리 이미지 기반: 카메라 캡처 프레임 또는 이미 로드된 이미지에 사용
    static std::vector<float> matToNCHW(const cv::Mat& bgr);
};
```

## 한 줄 요약

`NCHW`는 ONNX 모델이 이미지를 받는 배열 순서이고, `cv::Mat` 입력 함수는 카메라에서 받은 이미지를 파일 경로 없이 메모리에서 바로 전처리하기 위한 실제 비전 프로그램용 구조다.
