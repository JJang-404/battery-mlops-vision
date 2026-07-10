# StageController.cpp 질문 정리

대상 파일:

- `include/StageController.h`
- `src/StageController.cpp`

## 1. `CartesianXYZ`가 무엇인가요?

`CartesianXYZ`는 직교 좌표계 기반의 `X/Y/Z 스테이지`를 뜻합니다.

여기서 `Cartesian`은 수학에서 말하는 직교 좌표계입니다. 즉, X축, Y축, Z축이 서로 직각으로 구성된 일반적인 3축 위치 제어 방식입니다.

```cpp
enum class StageType {
    CartesianXYZ,
    UvwAlignment
};
```

이 코드에서 `StageType`은 스테이지 종류를 구분합니다.

- `CartesianXYZ`: X/Y/Z를 각각 독립적인 직선축으로 이동하는 장비
- `UvwAlignment`: U/V/W 물리축을 이용해 논리적인 X/Y/θ 정렬을 수행하는 장비

따라서 `CartesianXYZ` 장비에서는 아래 함수 사용이 적절합니다.

```cpp
stage.moveXyz({10.0, 20.0, 5.0});
```

반대로 UVW 얼라인먼트 장비에서는 `moveXyz()`가 아니라 `moveXyTheta()`를 사용해야 합니다.

## 2. `waitUntilIdle`이 무엇인가요?

`waitUntilIdle()`은 스테이지가 움직인 뒤에 "이동이 끝났는지" 기다리는 함수입니다.

이름을 나누어 보면 다음과 같습니다.

- `wait`: 기다린다
- `until`: 어떤 조건이 될 때까지
- `idle`: 장비가 더 이상 움직이지 않는 대기 상태

즉, `waitUntilIdle(timeout)`은 지정한 시간 안에 장비가 이동 완료 상태가 되는지 확인하는 함수입니다.

현재 시뮬레이션 코드에서는 실제 장비가 없으므로 짧게 대기한 뒤 성공 처리합니다.

```cpp
if (simulate_) {
    const auto simulatedDelay =
        std::min(timeout, std::chrono::milliseconds(100));
    std::this_thread::sleep_for(simulatedDelay);
    return !emergencyStopped_;
}
```

실제 장비에서는 이 부분에 다음 확인 로직이 들어갑니다.

- 이동 완료 신호 확인
- In-position 신호 확인
- 모터 알람 발생 여부 확인
- 비상 정지 여부 확인
- 지정 시간 초과 여부 확인

그래서 `moveXyz()`나 `moveXyTheta()`는 이동 명령을 보낸 뒤 바로 성공으로 끝내지 않고, `waitUntilIdle()`을 호출해 이동 완료 여부를 확인합니다.

## 3. TODO 주석 정리

`src/StageController.cpp`의 영어 TODO 주석은 자연스러운 한국어로 바꾸었습니다.

예를 들어 아래 문장은:

```cpp
// TODO: Check U/V/W stroke limits before synchronized motion.
// TODO: Send synchronized U/V/W move commands to the real controller.
```

다음처럼 바꿨습니다.

```cpp
// TODO: 동기 이동 전에 U/V/W 각 축의 스트로크 한계를 확인합니다.
// TODO: 실제 모션 컨트롤러에 U/V/W 동기 이동 명령을 전송합니다.
```

현장 코드에서는 TODO가 단순 번역보다 "추가 구현 시 무엇을 작성해야 하는지" 바로 보이는 표현이어야 합니다.

## 4. StageController 전체 흐름도

```text
프로그램 시작
  |
  v
StageController 생성
  |
  +-- StageType::CartesianXYZ
  |     |
  |     v
  |   X/Y/Z 직교 스테이지로 사용
  |
  +-- StageType::UvwAlignment
        |
        v
      X/Y/θ 논리 좌표를 U/V/W 물리축으로 변환해서 사용


connect()
  |
  +-- simulate_ == true
  |     |
  |     v
  |   실제 장비 없이 연결된 것처럼 처리
  |
  +-- simulate_ == false
        |
        v
      제조사 SDK, 시리얼, TCP/IP, EtherCAT, PLC 통신 연결 필요


moveXyz(target)
  |
  +-- StageType이 CartesianXYZ인지 확인
  |
  +-- 연결 상태와 비상 정지 상태 확인
  |
  +-- X/Y/Z 이동 명령 전송
  |
  +-- waitUntilIdle(timeout)
  |
  +-- 현재 위치 xyz_ 갱신


moveXyTheta(target)
  |
  +-- StageType이 UvwAlignment인지 확인
  |
  +-- 연결 상태와 비상 정지 상태 확인
  |
  +-- xyThetaToUvw(target)
  |     |
  |     v
  |   X/Y/θ 논리 좌표를 U/V/W 물리축 위치로 변환
  |
  +-- U/V/W 동기 이동 명령 전송
  |
  +-- waitUntilIdle(timeout)
  |
  +-- 현재 논리 좌표와 물리축 위치 갱신


disconnect()
  |
  +-- 실제 장비면 SDK 또는 통신 연결 종료
  |
  +-- connected_ = false
```

## 핵심 정리

`StageController`는 장비를 직접 움직이는 코드와 운영 프로그램 사이에 놓이는 제어 클래스입니다.

- 운영 코드는 `moveXyz()` 또는 `moveXyTheta()`만 호출합니다.
- 실제 장비 연결 방식은 추가 구현 시 제조사 SDK나 PLC 통신 코드로 채웁니다.
- `waitUntilIdle()`은 이동 명령 후 장비가 멈추고 완료 상태가 되었는지 확인하는 안전 확인 단계입니다.
- UVW는 X/Y/Z처럼 독립 좌표가 아니라, X/Y/θ 정렬을 만들기 위한 물리 구동축입니다.
