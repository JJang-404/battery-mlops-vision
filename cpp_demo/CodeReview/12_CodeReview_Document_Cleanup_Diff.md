# CodeReview 문서 정리 비교

## 목적

`CodeReview/` 문서를 Git에 올릴 때 학습 증빙 문서로 읽히도록 표현을 정리했다.
내부 운영 화면 고유명, 오래된 실행 상태 문구, 실제 코드와 맞지 않는 함수명을 점검했다.

## 주요 수정 비교

| 구분 | 수정 전 | 수정 후 |
|------|---------|---------|
| 함수명 | 실제 코드와 다른 전처리 함수명 표기 | 실제 C++ 코드와 동일한 `LoadAndPreprocess()` |
| 실행 상태 문구 | 과거 로컬 실행 상태 중심 설명 | 현재 문서 근거와 최종 검증 방법 중심으로 설명 |
| 내부 화면 언급 | 내부 운영 화면 고유명 또는 전용 리소스 언급 | `기존 검사 운영 화면`, `내부 코드나 전용 리소스는 포함하지 않음` |
| 주관적 표현 | 평가형 문장 | `적절하다`, `적합하다`, `바람직하다` |
| TODO 설명 | 향후 시점을 구어체로 설명 | `추가 구현 시 무엇을 작성해야 하는지` |

## 파일별 수정 내용

### `01_Preprocessor_h_INPUT_SIZE.md`

- 질문형 표현을 검토 항목으로 변경했다.
- 실제 구현 함수명에 맞춰 `LoadAndPreprocess()`로 수정했다.
- 과거 로컬 실행 실패 문구를 제거하고, ONNX input metadata 또는 ONNX Runtime 세션 shape로 검증해야 한다고 정리했다.

### `02_Preprocessor_cpp_NCHW_and_Mat_Input.md`

- 실제 코드와 다른 전처리 함수명 표기를 `LoadAndPreprocess()`로 수정했다.
- 실제 `Preprocessor.h`, `Preprocessor.cpp`, `main.cpp`에서 사용하는 함수명과 맞췄다.

### `05_Inferencer_cpp_run_and_GetTensorMutableData.md`

- 전처리 함수 호출 흐름에서 `Preprocessor::LoadAndPreprocess()`로 수정했다.

### `06_Postprocessor_h_Monitoring_Functions.md`

- 평가형 표현을 객관적인 문장으로 바꿨다.
- WPF 연동, helper 함수 공개 범위, contour 옵션화 설명을 운영 문서 톤으로 정리했다.

### `07_Postprocessor_cpp_Monitoring_Functions_Implementation.md`

- WPF 전달 방식 설명을 "좋다" 중심이 아니라 가능한 연동 방식으로 정리했다.

### `10_CPP_Batch_and_WPF_Monitoring.md`

- 내부 운영 화면 이름을 직접 언급하지 않도록 수정했다.
- 외부 공개가 제한되는 내부 코드나 전용 리소스가 포함되지 않는다는 점을 명시했다.

### `11_StageController_cpp_Questions.md`

- 향후 구현 시점을 설명하는 표현을 `추가 구현 시`로 바꿔 문서 톤을 정리했다.

## Git 포함 주의

내부 운영 화면 관련 폴더는 현재 Git 상태에 나타나지 않는다.
내부 운영 화면 관련 코드는 Git에 추가하지 않아야 한다.
