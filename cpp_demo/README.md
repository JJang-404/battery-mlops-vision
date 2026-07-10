# Battery 검사 모니터링 실행 안내

모니터링 화면은 **WPF**로 작성되어 있으며 Windows에서 실행합니다. C++ 추론 예제는 `cpp_demo`, WPF 화면은 `csharp_demo/BatteryMonitor`에 있습니다.

## 1. 모델 위치

다음 두 파일을 `cpp_demo/models`에 둡니다.

```text
cpp_demo/models/
├─ battery_deeplab_v2.onnx
└─ battery_deeplab_v2.onnx.data
```

외부 데이터 방식의 모델은 두 파일이 반드시 같은 폴더에 있어야 합니다. WPF 프로젝트를 빌드하면 두 파일이 실행 폴더의 `models`로 자동 복사됩니다.

## 2. WPF 모니터링 실행

저장소 루트(`Battery_Project`)에서 PowerShell로 실행합니다.

```powershell
dotnet run --project .\csharp_demo\BatteryMonitor\BatteryMonitor.csproj -c Release
```

처음 실행할 때는 NuGet 패키지 복원 때문에 시간이 걸릴 수 있습니다.

## 3. 상단 버튼 기능

### START

선택된 입력 폴더의 이미지를 파일명 순서로 검사합니다.

- 지원 확장자: `.png`, `.jpg`, `.jpeg`, `.bmp`
- 선택 폴더 바로 아래 파일만 검사하며 하위 폴더는 검색하지 않습니다.
- 원본, 결함 오버레이, OK/NG, 결함 픽셀 수와 처리 시간을 표시합니다.
- 새 배치를 시작하면 화면의 이전 검사 이력은 초기화됩니다.
- 결과 이미지는 실행 폴더의 `monitoring_output`에 저장됩니다.

기본 입력 폴더는 빌드 시 복사되는 `test_images`입니다.

### STOP

진행 중인 배치 검사를 중단 요청합니다.

- ONNX 추론 한 건이 실행 중이면 해당 추론이 끝난 뒤 중단됩니다.
- 중단 전까지 완료된 결과는 화면과 출력 폴더에 남습니다.

### FOLDER

검사할 **입력 이미지 폴더**를 선택합니다. 결과 폴더를 여는 기능은 아닙니다.

- 선택 직후 상단에 폴더 경로가 표시됩니다.
- `Total`에 지원 이미지 개수가 표시됩니다.
- 첫 번째 원본 이미지는 미리보기로 표시되지만, 오버레이와 검사 이력은 아직 생성되지 않습니다.
- 지원 이미지가 없으면 안내 메시지가 나타납니다.
- 검사 중에는 폴더를 변경할 수 없습니다.

폴더를 선택한 다음 `START`를 눌러야 검사가 시작됩니다.

`PREVIOUS`와 `NEXT`는 폴더의 원본 파일을 탐색하는 버튼이 아니라 **검사가 완료된 이력**을 탐색하는 버튼입니다. 따라서 `START`를 누르고 검사 결과가 생성되기 전에는 비활성화됩니다.

### EXIT

모니터링 프로그램을 종료합니다. 검사 중이면 중단을 요청한 뒤 창을 닫습니다.

## 4. 그 밖의 화면 기능

- `ORIGINAL IMAGE`: 선택된 검사 결과의 원본
- `INSPECTION OVERLAY`: 모델이 찾은 결함을 표시한 결과
- `CURRENT INSPECTION`: 파일명, OK/NG, 결함별 픽셀 수, 처리 시간
- `BATCH STATUS`: 전체·처리·OK·NG 개수와 진행률
- `INSPECTION HISTORY`: 현재 배치에서 완료된 검사 목록
- `PREVIOUS` / `NEXT`: 완료된 이전·다음 결과 확인
- `SYSTEM LOG`: 폴더 변경, 검사 시작·완료·중단 및 오류 기록

현재 판정은 `Pollution` 또는 `Damaged` 픽셀이 하나라도 있으면 `NG`, 둘 다 0이면 `OK`입니다.

## 5. 빌드만 확인하기

```powershell
dotnet build .\csharp_demo\BatteryMonitor\BatteryMonitor.csproj -c Release
```

빌드 결과는 `csharp_demo/BatteryMonitor/bin/Release/net8.0-windows/`에 생성됩니다.
