# Postprocessor.cpp 모니터링 함수 구현 정리

## 이번 구현의 목적

`Postprocessor.h`에 선언되어 있던 모니터링용 helper 함수들을 `Postprocessor.cpp`에 실제 구현했다.

목표는 `buildOverlay()` 하나를 외부 대표 함수로 유지하면서, 내부 작업을 아래 helper로 나누는 것이다.

```cpp
getSegmentationMask()
getColorMap()
applyColorMapToMask()
drawContours()
drawLegend()
```

## 전체 흐름

```text
logits
    |
    | getSegmentationMask
    v
mask513
    |
    | resize
    v
maskFull
    |
    | getColorMap + applyColorMapToMask
    v
colorMask
    |
    | addWeighted + classMask copy
    v
overlay
    |
    | drawContours + drawLegend
    v
최종 모니터링 이미지
```

## 구현한 함수 역할

### `getSegmentationMask`

모델 출력 logits에서 픽셀별 가장 높은 class를 골라 `CV_8UC1` mask를 만든다.

```text
0 = background
1 = Pollution
2 = Damaged
```

이 함수가 후처리의 기준 mask를 만든다.

### `getColorMap`

`CLASS_COLORS`를 OpenCV `cv::Mat` 형태의 색상표로 변환한다.

현재 클래스 색상은 BGR 기준이다.

```text
0 = 검정
1 = 노랑
2 = 빨강
```

### `applyColorMapToMask`

숫자 class mask를 컬러 mask로 바꾼다.

```text
mask 값 1 -> 노랑 픽셀
mask 값 2 -> 빨강 픽셀
```

background class인 0은 검정으로 둔다.

### `drawContours`

class별 binary mask에서 contour를 찾고, overlay 이미지 위에 외곽선을 그린다.

운영 화면에서는 fill보다 contour가 결함 경계를 더 선명하게 보여준다.

### `drawLegend`

화면 오른쪽 아래에 클래스 색상 설명 박스를 그린다.

`m`이라는 축약 변수는 `margin`으로 바꿨고, `ry`는 `rowY`로 바꿨다.

## 코드에서 같이 정리한 부분

`drawLegnd` 오타를 `drawLegend`로 수정했다.

`buildOverlay()`가 `mask513`이 아니라 최종 `overlay`를 반환하도록 수정했다.

```cpp
return overlay;
```

`printf`는 `cout`으로 바꿨다.

```cpp
cout << fixed << setprecision(2)
     << "[mask] " << name << ": " << px << " px ("
     << 100.0 * px / totalPixels << "%)\n";
```

`cv::compare(maskFull, c, classMask, cv::CMP_EQ)`를 사용해 class별 binary mask를 명시적으로 만들었다.

## 왜 `buildOverlay()` 하나만 public으로 두는가?

외부 코드는 아래 한 줄만 알면 된다.

```cpp
cv::Mat overlay = Postprocessor::buildOverlay(logits, originalBgr);
```

mask 생성, color map 적용, contour 그리기, legend 추가는 모두 내부 구현 세부사항이다.

그래서 helper 함수들은 `private`로 두는 것이 적절하다.

## WPF 연동 관점

WPF에서는 최종적으로 사람이 볼 수 있는 overlay 이미지가 가장 먼저 필요하다.

현재 구조는 C++ 쪽에서 `cv::Mat overlay`를 만들고, 필요 시 아래 방식 중 하나로 WPF에 전달할 수 있다.

```text
파일 저장 후 WPF 로드
이미지 buffer 전달
C# OpenCvSharp Mat 변환
C++/CLI 또는 DLL interop
```

WPF에서 원본, mask, overlay, 통계값을 분리해서 표시해야 한다면 public API를 추가할 수 있다.

## 한 줄 요약

이번 구현은 segmentation 결과를 운영자가 보기 좋은 overlay 이미지로 만들기 위해 mask 생성, 색상 적용, contour, legend를 helper 함수로 분리한 구조다.
