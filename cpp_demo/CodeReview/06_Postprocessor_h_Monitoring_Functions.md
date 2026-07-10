# Postprocessor.h 자동 생성 함수와 운영 모니터링 관점

## 검토 항목

`Postprocessor.h`에 추가된 아래 함수들이 컴퓨터 비전 모니터링 관점에서 필요한 역할을 하는지 검토했다.

```cpp
static cv::Mat getSegmentationMask(const vector<float>& logits);
static cv::Mat getColorMap();
static cv::Mat applyColorMapToMask(const cv::Mat& mask, const cv::Mat& colorMap);
static void drawContours(cv::Mat& overlay, const cv::Mat& mask, const cv::Scalar& color, int thickness);
```

또한 WPF 화면에 결과를 표시할 때 적합한 구조인지 함께 검토했다.

## 결론

이 함수들은 모델 정확도 자체에는 필수 기능이 아니다.

하지만 비전 검사 결과를 사람이 확인하고, 운영 화면에서 결함 위치와 종류를 빠르게 판단하기 위한 모니터링 기능으로는 유용하다.

특히 WPF 같은 UI에서 아래 정보를 보여줄 계획이라면 활용도가 높다.

```text
원본 이미지
결함 mask
색상 overlay
결함 contour
클래스별 면적 또는 pixel count
```

다만 현재 `CPP_DEMO_GUIDE.md`의 원래 설계는 외부 호출 함수로 `buildOverlay()` 하나만 두고, 세부 로직은 내부에 감추는 구조다.

따라서 이 함수들은 `public`으로 노출하기보다는 지금처럼 `private` helper로 두는 것이 적절하다.

## 전체 후처리 흐름

모델 출력은 바로 사람이 보기 좋은 이미지가 아니다.

`Inferencer::run()`이 반환하는 값은 보통 아래 형태의 logits 배열이다.

```text
1 x 3 x 513 x 513
```

이 값은 각 픽셀이 각 클래스일 점수를 담은 float 배열이다.

후처리는 이 점수 배열을 사람이 볼 수 있는 결과로 바꾼다.

```text
logits
    |
    | getSegmentationMask
    v
class index mask
    |
    | applyColorMapToMask
    v
color mask
    |
    | overlay / drawContours
    v
운영자가 볼 수 있는 검사 결과 이미지
```

## `getSegmentationMask`

역할:

```cpp
static cv::Mat getSegmentationMask(const vector<float>& logits);
```

`getSegmentationMask()`는 모델 출력 logits에서 픽셀별 클래스를 고르는 함수다.

예를 들어 클래스가 3개라면 각 픽셀마다 아래 점수들이 있다.

```text
class 0: background 점수
class 1: pollution 점수
class 2: damaged 점수
```

각 픽셀에서 가장 점수가 높은 클래스를 선택한다.

```text
mask[y][x] = argmax(class score)
```

결과 mask는 보통 `CV_8UC1` 형태가 된다.

```text
0 = background
1 = pollution
2 = damaged
```

운영 관점:

이 함수는 후처리의 핵심이다. overlay를 만들든, 불량 면적을 계산하든, contour를 그리든 먼저 class mask가 필요하다.

그래서 `getSegmentationMask()`는 모니터링뿐 아니라 검사 결과 계산에도 중요한 함수다.

## `getColorMap`

역할:

```cpp
static cv::Mat getColorMap();
```

`getColorMap()`은 클래스 번호를 색상으로 바꾸기 위한 색상표를 만드는 helper로 볼 수 있다.

예를 들면 현재 프로젝트의 클래스 색상은 BGR 기준으로 다음과 같이 잡혀 있다.

```text
0: background = 검정 또는 투명 처리
1: Pollution  = 노랑
2: Damaged    = 빨강
```

운영 관점:

색상은 모델 결과를 사람이 빠르게 이해하게 만드는 UI 요소다.

예를 들어 WPF 화면에서 결함 종류를 아래처럼 구분할 수 있다.

```text
노랑 = 오염
빨강 = 손상
```

단, 색상표 자체는 모델 판단에 영향을 주지 않는다. 순수하게 시각화와 UX를 위한 기능이다.

## `applyColorMapToMask`

역할:

```cpp
static cv::Mat applyColorMapToMask(const cv::Mat& mask, const cv::Mat& colorMap);
```

`applyColorMapToMask()`는 class index mask를 사람이 볼 수 있는 컬러 이미지로 바꾸는 함수다.

입력 mask는 숫자 이미지다.

```text
0 0 0 1 1 0
0 2 2 2 0 0
```

이 숫자를 색상으로 바꾸면 운영자가 보기 쉬워진다.

```text
0 -> 투명 또는 검정
1 -> 노랑
2 -> 빨강
```

운영 관점:

WPF 화면에 segmentation 결과를 보여줄 때 유용하다.

예를 들어 다음 화면 구성이 가능하다.

```text
왼쪽: 원본 이미지
오른쪽: 컬러 mask 또는 overlay 이미지
```

또는 원본 이미지 위에 반투명 컬러 mask를 합성해서 결함 영역을 바로 확인할 수 있다.

## `drawContours`

역할:

```cpp
static void drawContours(cv::Mat& overlay,
                         const cv::Mat& mask,
                         const cv::Scalar& color,
                         int thickness);
```

`drawContours()`는 결함 영역의 외곽선을 그리는 함수다.

색상 fill만 있으면 결함 영역이 넓게 칠해져서 경계가 흐릿하게 보일 수 있다.

contour를 그리면 운영자가 결함의 정확한 위치와 크기를 더 빠르게 볼 수 있다.

```text
fill     = 결함 영역 전체를 반투명하게 표시
contour  = 결함 영역의 경계를 선명하게 표시
```

운영 관점:

실제 비전 검사 모니터링에서는 contour가 꽤 유용하다.

특히 아래 상황에서 유용하다.

```text
결함 위치를 작업자가 빠르게 확인해야 할 때
불량 영역 경계를 보고 싶을 때
원본 이미지가 복잡해서 색상 fill만으로는 잘 안 보일 때
검사 결과 이미지를 리포트 또는 로그로 저장할 때
```

단, contour 계산은 `findContours` 같은 연산을 추가로 수행하므로 고속 실시간 처리에서는 옵션으로 제어할 수 있게 설계하는 것이 바람직하다.

## WPF 화면에 보여줄 때 적합한가?

적합하다.

WPF는 C++의 `cv::Mat`을 직접 표시하기보다는 보통 아래 중 하나로 결과를 받는다.

```text
1. C++에서 overlay 이미지를 파일로 저장하고 WPF가 로드
2. C++에서 이미지 buffer를 넘기고 WPF BitmapSource로 변환
3. C#에서 OpenCvSharp Mat 또는 byte[] 형태로 받아 표시
4. C++ DLL / CLI / interop으로 결과 이미지를 전달
```

이때 WPF가 최종적으로 보여주기 좋은 결과는 보통 `overlay` 이미지다.

```text
원본 이미지 + 반투명 결함 색상 + contour + legend
```

따라서 `buildOverlay()`가 최종 표시 이미지를 만들고, 내부 helper 함수들이 mask/color/contour를 처리하는 구조는 WPF 연동에도 적합하다.

다만 WPF에서 더 많은 정보를 표시해야 한다면 아래 결과들을 분리해서 받을 수 있도록 API를 확장할 수 있다.

```text
overlay image
raw mask
class별 pixel count
class별 defect ratio
contour points 또는 bounding boxes
```

초기 단계에서는 `buildOverlay()` 하나로 시작하고, UI 요구가 명확해지면 추가 API를 늘리는 방향이 적절하다.

## 현재 설계에서 적절한 점

현재 `Postprocessor.h`는 외부에 `buildOverlay()`만 공개하고 있다.

```cpp
static cv::Mat buildOverlay(const vector<float>& logits,
                            const cv::Mat& originalBgr,
                            float fillAlpha = 0.25f,
                            int contourThickness = 3);
```

이 구조는 외부 API를 단순하게 유지한다는 점에서 적절하다.

외부 코드는 복잡한 내부 절차를 몰라도 된다.

```cpp
cv::Mat overlay = Postprocessor::buildOverlay(logits, originalBgr);
```

이 한 줄로 화면 표시용 결과를 얻을 수 있다.

`getSegmentationMask`, `getColorMap`, `applyColorMapToMask`, `drawContours`는 내부 helper로 감추면 된다.

## 현재 설계에서 주의할 점

`CPP_DEMO_GUIDE.md`의 원래 설계에는 아래 private 함수만 있었다.

```cpp
static void drawLegend(cv::Mat& img);
static const cv::Scalar CLASS_COLORS[NUM_CLASSES];
```

현재 `Postprocessor.h`에는 자동 생성된 helper가 더 많이 들어가 있다.

```cpp
static cv::Mat getSegmentationMask(const vector<float>& logits);
static cv::Mat getColorMap();
static cv::Mat applyColorMapToMask(const cv::Mat& mask, const cv::Mat& colorMap);
static void drawContours(cv::Mat& overlay, const cv::Mat& mask, const cv::Scalar& color, int thickness);
```

함수 분리는 내부 구현을 읽기 쉽게 만드는 데 도움이 된다.

다만 아래 기준은 유지하는 것이 바람직하다.

```text
외부 호출용 public API는 buildOverlay 중심
세부 처리 함수는 private helper
성능 옵션은 buildOverlay 인자로 제어
WPF에서 필요한 데이터가 늘어나면 public API 추가
```

## SW 운영 관점에서 필요한가?

운영 시스템에서는 필요한 기능에 가깝다.

운영 시스템에서는 모델 결과를 숫자로만 보는 것이 아니라, 사람이 확인 가능한 형태로 보여주는 기능이 중요하다.

예를 들어 생산 현장에서는 다음 질문에 빠르게 답해야 한다.

```text
어디가 불량인가?
어떤 종류의 불량인가?
불량 영역이 얼마나 큰가?
판정 근거 이미지를 저장할 수 있는가?
작업자가 UI에서 바로 이해할 수 있는가?
```

이 질문들에 답하려면 mask, color overlay, contour가 도움이 된다.

특히 contour와 color overlay는 "모델이 왜 불량이라고 판단했는지"를 운영자가 직관적으로 확인하는 데 유용하다.

## 추천 구조

현재 단계에서는 아래 구조를 추천한다.

```cpp
class Postprocessor {
public:
    static cv::Mat buildOverlay(const vector<float>& logits,
                                const cv::Mat& originalBgr,
                                float fillAlpha = 0.25f,
                                int contourThickness = 3);

private:
    static void drawLegend(cv::Mat& img);
    static const cv::Scalar CLASS_COLORS[NUM_CLASSES];

    static cv::Mat getSegmentationMask(const vector<float>& logits);
    static cv::Mat applyColorMapToMask(const cv::Mat& mask);
    static void drawContours(cv::Mat& overlay,
                             const cv::Mat& mask,
                             const cv::Scalar& color,
                             int thickness);
};
```

`getColorMap()`은 꼭 별도 함수일 필요는 없다.

이미 `CLASS_COLORS`가 있으므로 클래스 색상표는 `CLASS_COLORS` 하나로 관리하는 편이 더 단순할 수 있다.

색상 개수가 많아지거나 OpenCV의 LUT 방식으로 처리하고 싶다면 `getColorMap()`을 유지해도 된다.

## 한 줄 요약

`getSegmentationMask`, `getColorMap`, `applyColorMapToMask`, `drawContours`는 모델 추론 자체에는 필수는 아니지만, 비전 검사 결과를 운영자가 이해하고 WPF 화면에 표시하기 위한 모니터링 기능으로 적합하다. 다만 외부에는 `buildOverlay()` 중심으로 공개하고, 이 함수들은 `private` helper로 유지하는 것이 바람직하다.
