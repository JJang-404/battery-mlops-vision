# C++ 데모 빌드·실행 명령

## 1. 프로젝트 폴더로 이동

```bash
cd /Users/jang/Desktop/git/battery-mlops-vision/cpp_demo
```

모든 명령은 `CMakeLists.txt`, `models/`, `test_images/`가 보이는 이 폴더에서 실행한다.

## 2. 최초 CMake 설정

```bash
cmake -S . -B build-mac \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++
```

정상 설정 메시지:

```text
-- Found OpenCV: ...
-- Configuring done
-- Generating done
```

## 3. 빌드

```bash
cmake --build build-mac -j$(sysctl -n hw.logicalcpu)
```

`.cpp`만 수정했을 때는 최초 설정 명령을 반복할 필요 없이 이 빌드 명령부터 실행하면 된다.

## 4. 기본 폴더 전체 검사

```bash
./build-mac/battery_demo
```

인자가 없으면 `test_images/` 안의 PNG, JPG, JPEG, BMP를 파일명순으로 모두 검사한다.
각 창에서 아무 키나 누르면 다음 이미지로 이동한다. 결과 이미지는
`monitoring_output/overlay_원본파일명`으로 저장된다.

## 5. 특정 파일 또는 다른 폴더 검사

```bash
./build-mac/battery_demo test_images/RGB_cell_cylindrical_0923_183.png
```

```bash
./build-mac/battery_demo /Users/jang/Pictures/battery_test
```

파일을 넘기면 한 장만, 폴더를 넘기면 폴더의 모든 지원 이미지를 처리한다.

## 6. ONNX Runtime 링크 확인

실행 시 `libonnxruntime.1.dylib` 오류가 나면 다음을 실행한다.

```bash
unlink third_party/onnxruntime/lib/libonnxruntime.1.dylib
ln -s libonnxruntime.dylib third_party/onnxruntime/lib/libonnxruntime.1.dylib
```

확인:

```bash
readlink third_party/onnxruntime/lib/libonnxruntime.1.dylib
```

결과가 `libonnxruntime.dylib`이면 정상이다.

## 7. 터미널 출력과 결과

터미널에는 이미지 번호, 파일명, 클래스별 픽셀 수, 처리 시간, 저장 경로, 최종 성공·실패
건수가 출력된다. OpenCV 창에는 원본과 오버레이가 좌우로 표시된다.

터미널 화면 지우기:

```bash
clear
```

단축키는 `Control + L`, VS Code 터미널 스크롤 기록까지 지우는 단축키는
`Command + K`다.
