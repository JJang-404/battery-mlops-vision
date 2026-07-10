# Postprocessor.cpp 질문 정리

## 1. `mask513`은 무엇인가?

`mask513`은 모델 출력 크기인 `513 x 513` 기준의 segmentation mask다.

```cpp
cv::Mat mask513(SIZE, SIZE, CV_8UC1);
```

의미는 다음과 같다.

```text
SIZE x SIZE = 513 x 513
CV_8UC1     = unsigned char 1채널 이미지
```

각 픽셀에는 색상이 아니라 class 번호가 들어간다.

```text
0 = background
1 = Pollution
2 = Damaged
```

즉 `mask513`은 "모델이 각 픽셀을 어떤 클래스로 봤는지"를 저장하는 숫자 이미지다.

## 2. for문의 계산 방식
핵심 for문은 픽셀을 하나씩 돌면서 class 점수를 비교한다.

```cpp
for (int y=0; y<SIZE; y++) {
    for (int x=0; x<SIZE; x++) {
        int bestClass = 0;
        float bestVal = logits[y * SIZE + x];

        for (int c=1; c<NUM_CLASSES; c++) {
            float val = logits[c * planeSize + y * SIZE + x];
            if (val > bestVal) {
                bestVal = val;
                bestClass = c;
            }
        }

        mask513.at<uchar>(y, x) = static_cast<uchar>(bestClass);
    }
}
```

`y`와 `x`는 이미지 좌표다.

```text
y = row, 세로 위치
x = column, 가로 위치
```

`c`는 class 번호다.

```text
c = 0 background
c = 1 Pollution
c = 2 Damaged
```

`logits`는 NCHW 순서로 펼쳐진 1차원 배열이다.

```text
class 0 전체 plane
class 1 전체 plane
class 2 전체 plane
```

그래서 특정 class `c`의 `(y, x)` 위치 점수는 이렇게 찾는다.

```cpp
logits[c * planeSize + y * SIZE + x]
```

## 3. 3중 for문에서 `c`는 왜 1부터 시작하나?

처음에 class 0을 기준값으로 잡기 때문이다.

```cpp
int bestClass = 0;
float bestVal = logits[y * SIZE + x];
```

이 줄은 class 0의 점수를 먼저 `bestVal`로 넣는다.

그러면 for문에서는 나머지 class만 비교하면 된다.

```cpp
for (int c=1; c<NUM_CLASSES; c++)
```

즉 class 0은 이미 비교 기준으로 들어갔으므로, class 1부터 비교한다.

## 4. `mask513.at<uchar>(y, x) = bestClass;`는 무엇인가?

`cv::Mat::at<T>(y, x)`는 OpenCV Mat의 특정 픽셀에 접근하는 함수다.

```cpp
mask513.at<uchar>(y, x)
```

의미는 다음과 같다.

```text
mask513 이미지에서
y행 x열 위치의
uchar 타입 값
```

`uchar`는 unsigned char이고, 0~255 사이의 정수 하나를 저장한다.

`mask513`은 `CV_8UC1`이므로 각 픽셀이 `uchar` 1개다.

따라서 아래 코드는 해당 위치에 class 번호를 저장한다.

```cpp
mask513.at<uchar>(y, x) = static_cast<uchar>(bestClass);
```

## 5. `cv::compare(...)`와 `(maskFull == c)`는 같은가?

거의 같은 목적이다.

```cpp
cv::Mat classMask;
cv::compare(maskFull, c, classMask, cv::CMP_EQ);
```

이 코드는 `maskFull`의 값이 `c`와 같은 위치를 255로 만들고, 아닌 위치를 0으로 만든다.

```cpp
cv::Mat classMask = (maskFull == c);
```

이 표현도 OpenCV에서 같은 목적의 binary mask를 만든다.

결과는 보통 다음과 같다.

```text
같음     -> 255
다름     -> 0
```

둘 다 사용할 수 있지만, `cv::compare()`가 더 명시적이다.

공부하거나 협업하는 코드에서는 `cv::compare(maskFull, c, classMask, cv::CMP_EQ)`가 의도를 더 분명하게 보여준다.

## 6. contour에서 왜 `vector<vector<cv::Point>>`를 쓰나?

하나의 contour는 여러 점으로 이루어진 선이다.

```text
contour 1개 = vector<cv::Point>
```

그런데 한 이미지 안에는 결함 덩어리가 여러 개 있을 수 있다.

```text
contour 여러 개 = vector<vector<cv::Point>>
```

예를 들어 오염 영역이 3군데라면:

```text
contours[0] = 첫 번째 오염 영역 외곽선 점들
contours[1] = 두 번째 오염 영역 외곽선 점들
contours[2] = 세 번째 오염 영역 외곽선 점들
```

그래서 OpenCV의 `findContours()`는 여러 contour를 담기 위해 `vector<vector<cv::Point>>`를 사용한다.

## 7. `printf`를 `cout`으로 바꿀 수 있나?

바꿀 수 있다.

기존 코드:

```cpp
printf("[mask] %s: %d px (%.2f%%)\n", name.c_str(), px, 100.0*px/total);
```

`cout` 버전:

```cpp
cout << fixed << setprecision(2)
     << "[mask] " << name << ": " << px << " px ("
     << 100.0 * px / totalPixels << "%)\n";
```

둘 다 가능하다.

단순 출력이면 `printf`도 문제 없다. C++ 스타일로 맞추고 싶으면 `cout`을 쓰면 된다.

이번 구현에서는 `cout`으로 바꿨다.

## 8. `drawLegend` 함수에서 `m` 변수의 풀네임은?

`m`은 `margin`으로 보는 것이 가장 자연스럽다.

```cpp
const int margin = 20;
```

legend 박스를 이미지 오른쪽 아래에서 얼마나 떨어뜨릴지 정하는 여백 값이다.

```text
margin = 가장자리 여백
```

## 9. `ry`의 풀네임은 무엇인가?

`ry`는 `rowY` 또는 `rectangleY`로 풀어 쓸 수 있다.

현재 맥락에서는 `rowY`가 가장 자연스럽다.

```cpp
int rowY = y + 12 + (c - 1) * 33;
```

의미는 legend 안에서 각 class 항목이 그려질 y좌표다.

```text
rowY = legend row의 y 위치
```

## 10. `+12`, `*33`, `+10`, `+38` 같은 숫자는 직접 화면을 보면서 정하나?

맞다. 이런 숫자는 UI 배치용 magic number에 가깝다.

정확한 값은 보통 모니터링 화면에서 직접 보면서 조정한다.

예를 들어:

```text
20  = legend 박스와 이미지 가장자리 사이 여백
12  = legend 내부 위쪽 여백
33  = 클래스 항목 사이 세로 간격
10  = 색상 사각형의 왼쪽 여백
18  = 색상 사각형 크기
38  = 텍스트 시작 x 위치
15  = 텍스트 baseline 보정
```

처음에는 적당한 기본값으로 시작하고, 실제 화면에서 다음을 보며 조정한다.

```text
글자가 잘리는지
색상 박스와 텍스트 간격이 자연스러운지
이미지 크기가 달라도 legend가 화면 안에 들어오는지
결함 overlay를 가리지 않는지
작업자가 멀리서 봐도 읽을 수 있는지
```

운영 UI에서는 이런 숫자를 상수 이름으로 분리하는 것이 유지보수에 적합하다.

```cpp
const int LEGEND_MARGIN = 20;
const int LEGEND_ROW_GAP = 33;
const int COLOR_BOX_SIZE = 18;
```

## 한 줄 요약

`mask513`은 모델 출력 크기의 class 번호 이미지이고, for문은 각 픽셀마다 가장 점수가 높은 class를 고르는 과정이다. contour는 여러 결함 덩어리를 다루기 위해 `vector<vector<cv::Point>>`를 쓰며, legend의 숫자 배치는 실제 모니터링 화면을 보면서 조정하는 UI 값이다.
