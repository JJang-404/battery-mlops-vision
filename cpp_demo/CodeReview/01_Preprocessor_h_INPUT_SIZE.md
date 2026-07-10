# Preprocessor.h INPUT_SIZE 정리

## 검토 항목

`Preprocessor.h`에서 아래처럼 선언한 값이 실제 `test_images/` 안의 이미지 크기를 따라가야 하는지 확인했다.

```cpp
class Preprocessor {
public:
    static constexpr int INPUT_SIZE = 224;
};
```

## 결론

`INPUT_SIZE`는 `test_images/`의 원본 이미지 크기가 아니라, ONNX 모델이 요구하는 입력 tensor의 이미지 크기다.

현재 프로젝트 문서 기준으로는 `224`가 아니라 `513`을 쓰는 것이 맞다.

```cpp
static constexpr int INPUT_SIZE = 513;
```

`test_images/` 안의 PNG 파일들은 현재 모두 `1920x1080` 원본 이미지다. 이 이미지를 그대로 모델에 넣는 것이 아니라, `Preprocessor`에서 모델 입력 크기인 `513x513`으로 resize한 뒤 NCHW float 배열로 바꿔서 ONNX Runtime에 전달한다.

## 기준은 무엇인가?

기준은 다음 순서로 확인한다.

1. 모델의 입력 shape
2. 모델 학습 당시 사용한 입력 크기
3. 기존 C# 또는 Python 전처리 코드의 resize 크기
4. 프로젝트 문서와 `Inferencer`의 입력 tensor shape

이 중 가장 강한 기준은 1번, 즉 ONNX 모델의 실제 입력 shape다.

## 이 프로젝트에서 확인된 근거

`CPP_DEMO_GUIDE.md`의 `Preprocessor` 설계 코드에는 다음처럼 되어 있다.

```cpp
static constexpr int INPUT_SIZE = 513;
```

그리고 `Inferencer` 설계 코드에도 입력 shape가 다음처럼 고정되어 있다.

```cpp
static constexpr int64_t INPUT_SHAPE[4] = {1, 3, 513, 513};
```

즉 전처리 결과는 아래 형태가 되어야 한다.

```text
1 x 3 x 513 x 513
```

의미는 다음과 같다.

```text
1   = batch size
3   = RGB channel
513 = height
513 = width
```

## test_images 크기는 왜 기준이 아닌가?

현재 `test_images/`의 이미지 크기는 다음과 같다.

```text
RGB_cell_cylindrical_0761_241.png : 1920 x 1080
RGB_cell_cylindrical_0923_183.png : 1920 x 1080
RGB_cell_cylindrical_1154_160.png : 1920 x 1080
```

이 크기는 카메라 또는 샘플 이미지의 원본 해상도다. 모델은 보통 고정된 tensor shape를 받기 때문에, 원본 이미지는 전처리 단계에서 모델 입력 크기로 변환된다.

흐름은 다음과 같다.

```text
test_images 원본 이미지
    1920 x 1080
        |
        | cv::resize
        v
모델 입력 이미지
    513 x 513
        |
        | RGB 변환, 정규화, HWC -> NCHW
        v
ONNX 입력 tensor
    1 x 3 x 513 x 513
```

## 224는 언제 쓰는 값인가?

`224x224`는 ResNet, MobileNet 같은 이미지 분류 모델에서 자주 쓰는 입력 크기다.

하지만 이 프로젝트의 모델은 `battery_deeplab_v1.onnx`이고, 문서상 DeepLab 기반 segmentation 모델로 정리되어 있다. Segmentation 모델은 픽셀 단위 mask를 예측하므로 입력 크기와 출력 mask 크기가 함께 중요하다.

따라서 이 프로젝트에서 `224`를 넣으면 다음 문제가 생길 수 있다.

- `Inferencer`의 `{1, 3, 513, 513}` 입력 shape와 전처리 결과 크기가 맞지 않는다.
- 모델이 학습된 입력 크기와 달라져 정확도가 떨어질 수 있다.
- 후처리 단계에서 출력 mask 크기 가정이 깨질 수 있다.

## Preprocessor.h에서 권장 선언

현재 프로젝트 구현과 맞추면 `Preprocessor.h`는 아래 형태가 적절하다.

```cpp
#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

class Preprocessor {
public:
    static constexpr int INPUT_SIZE = 513;

    static std::vector<float> LoadAndPreprocess(const std::string& imagePath);
    static std::vector<float> matToNCHW(const cv::Mat& bgr);
};
```

## 실제 모델 기준을 확인하는 방법

가장 정확한 방법은 ONNX 모델의 input metadata를 출력하는 것이다.

Python에 `onnx` 패키지가 있으면 다음처럼 확인할 수 있다.

```bash
python3 -c "import onnx; m=onnx.load('models/battery_deeplab_v1.onnx', load_external_data=False); [print(i.name, [d.dim_value if d.HasField('dim_value') else d.dim_param for d in i.type.tensor_type.shape.dim]) for i in m.graph.input]"
```

ONNX Runtime C++로도 `session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape()`를 출력해서 확인할 수 있다.

현재 정리는 프로젝트의 `Preprocessor`, `Inferencer` 구현값과 `test_images` 실제 크기를 근거로 작성했다. 최종 확인은 ONNX 모델 input metadata 또는 ONNX Runtime 세션의 입력 shape 출력값으로 검증한다.

## 한 줄 요약

`INPUT_SIZE`는 테스트 이미지 파일의 실제 해상도가 아니라 모델 입력 크기다. 이 프로젝트에서는 `224`가 아니라 `513`을 기준으로 잡는 것이 맞다.
