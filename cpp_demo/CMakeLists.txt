cmake_minimum_required(VERSION 3.18)
project(BatteryDemoCpp CXX)
set(CMAKE_CXX_STANDARD 17)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# ONNX Runtime (압축 해제 경로 지정)
set(ORT_ROOT "${CMAKE_SOURCE_DIR}/third_party/onnxruntime")
include_directories(${ORT_ROOT}/include)
link_directories(${ORT_ROOT}/lib)

# 실행파일
file(GLOB SRCS "src/*.cpp")
add_executable(battery_demo ${SRCS})
include_directories(include)
if(APPLE)
    target_link_libraries(battery_demo
        ${OpenCV_LIBS}
        onnxruntime
        "-framework CoreML"
        "-framework Foundation"
    )
else()
    target_link_libraries(battery_demo
        ${OpenCV_LIBS}
        onnxruntime
    )
endif()
