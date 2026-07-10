# C++ 다중 이미지 처리와 WPF 모니터링 구현

## 구현 목적

한 장만 검사하던 C++ 데모를 폴더 배치 검사로 확장하고, Windows 운영자가 검사 상태와
결과를 한 화면에서 확인할 수 있는 WPF 모니터링 앱을 추가했다.

## C++ `main.cpp`

### 입력 규칙

```text
인자 없음  -> test_images 폴더 전체
파일 인자  -> 해당 이미지 한 장
폴더 인자  -> 해당 폴더의 모든 지원 이미지
```

`findImages()`가 PNG, JPG, JPEG, BMP만 골라 파일명순으로 정렬한다. 모델은 검사 시작 시
한 번만 생성하고 모든 이미지에서 같은 `Inferencer` 세션을 재사용한다. 모델을 매 이미지마다
다시 로드하지 않으므로 배치 처리에 적합하다.

각 이미지 처리 순서:

```text
이미지 로드
-> RGB 변환·513x513 resize·ImageNet normalize
-> ONNX Runtime 추론
-> 픽셀별 argmax와 결함 overlay 생성
-> monitoring_output 폴더 저장
-> 원본과 결과를 좌우로 표시
```

개별 이미지에서 오류가 발생해도 전체 프로그램을 즉시 종료하지 않고 다음 이미지 검사를
계속한다. 마지막에는 성공 및 실패 건수를 터미널에 출력한다.

## 전처리 보정

`Preprocessor.cpp`의 `MEAN`, `STD`가 선언만 되고 사용되지 않던 문제를 수정했다.

```cpp
float value = pixel[channel] / 255.0f;
value = (value - MEAN[channel]) / STD[channel];
```

이는 RGB 각 채널을 0~1 범위로 바꾼 후 ImageNet 평균과 표준편차로 정규화하는 과정이다.
학습 및 기존 C# 전처리 조건과 입력 분포를 맞추기 위해 필요하다.

## WPF 프로젝트 구조

```text
csharp_demo/BatteryMonitor/
├── BatteryMonitor.csproj
├── App.xaml
├── App.xaml.cs
├── MainWindow.xaml
├── MainWindow.xaml.cs
└── InspectionRecord.cs
```

WPF는 Windows 전용이므로 macOS에서는 C++ 데모를 사용하고, Windows 또는 Windows 가상
머신에서 `BatteryMonitor`를 실행한다.

기존 C# 프로젝트의 `Inferencer.cs`, `Postprocessor.cs`는 링크 방식으로 재사용한다.
모델과 테스트 이미지도 빌드 출력 폴더로 자동 복사된다.

## 화면 구성

기존 검사 운영 화면에서 사용하던 큰 제목, 상단 명령 버튼, 상태 표시, 데이터 테이블 구조를 참고했다.
외부 공개가 제한되는 내부 코드나 전용 리소스는 포함하지 않고, 회색톤의 검사 모니터링 화면으로 구성했다.

- `START`: 선택 폴더의 이미지 전체 검사
- `STOP`: 현재 배치 중지 요청
- `FOLDER`: 검사 이미지 폴더 선택
- `EXIT`: 프로그램 종료
- `ORIGINAL IMAGE`: 현재 원본 이미지
- `INSPECTION OVERLAY`: 결함 색상 및 외곽선 결과
- `CURRENT INSPECTION`: 파일명, OK/NG, 클래스 픽셀 수, 처리 시간
- `BATCH STATUS`: 전체·처리·OK·NG 건수와 진행률
- `INSPECTION HISTORY`: 처리된 검사 결과 목록
- `PREVIOUS/NEXT`: 이전 또는 다음 결과 확인
- `SYSTEM LOG`: 검사 시작, 완료, 오류 로그

현재 판정 규칙은 Pollution과 Damaged 예측 픽셀이 모두 0이면 `OK`, 하나라도 있으면 `NG`다.
실제 운영에서는 미세 노이즈를 제거하기 위해 클래스별 최소 픽셀 수 또는 면적 기준을 추가하는
것이 바람직하다.

## Windows 빌드와 실행

Windows PowerShell:

```powershell
cd C:\path\to\battery-mlops-vision\csharp_demo\BatteryMonitor
dotnet restore
dotnet build -c Release
dotnet run -c Release
```

또는 Visual Studio 2022에서 `BatteryMonitor.csproj`를 열고 시작 프로젝트로 실행한다.

GPU 패키지는 CUDA 등록을 먼저 시도하며 실패하면 기존 `Inferencer` 코드에 따라 CPU로
폴백한다. CUDA를 사용하려면 Windows에 호환되는 NVIDIA 드라이버, CUDA 및 cuDNN 런타임이
필요하다.

## 운영상 주의점

- WPF는 macOS에서 네이티브 실행되지 않는다.
- 테스트 이미지와 모델은 프로젝트 파일의 복사 설정으로 출력 폴더에 배치된다.
- 검사 결과 overlay는 실행 폴더의 `monitoring_output`에 저장된다.
- 현재 STOP은 진행 중인 ONNX `Run()`을 강제로 끊지 않고 한 장이 끝난 뒤 중지한다.
- 실제 생산 판정에는 픽셀 임계값, 최소 contour 면적, 모델 confidence 기준이 필요하다.
