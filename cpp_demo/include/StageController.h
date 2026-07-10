#pragma once

#include <chrono>
#include <functional>
#include <string>

enum class StageType {
    CartesianXYZ,
    UvwAlignment
};

struct XyzPosition {
    double xMm = 0.0;
    double yMm = 0.0;
    double zMm = 0.0;
};

// UVW 스테이지를 운영할 때 사용하는 논리 좌표
struct XyThetaPosition {
    double xMm = 0.0;
    double yMm = 0.0;
    double thetaDeg = 0.0;
};

// 모터 드라이버에서 읽거나 쓰는 UVW 물리축 위치
struct UvwMotorPosition {
    double uMm = 0.0;
    double vMm = 0.0;
    double wMm = 0.0;
};

using UvwTransform =
    std::function<UvwMotorPosition(const XyThetaPosition&)>;

class StageController {
public:
    explicit StageController(
        StageType type,
        bool simulate = true,
        UvwTransform uvwTransform = {});
    ~StageController();

    bool connect(const std::string& endpoint = "COM3");
    void disconnect();

    // XYZ 직교 스테이지 전용
    bool moveXyz(
        const XyzPosition& target,
        std::chrono::milliseconds timeout = std::chrono::seconds(30));

    // UVW 얼라인먼트 스테이지 전용:
    // 논리 X/Y/θ 명령을 내부에서 물리 U/V/W 명령으로 변환한다.
    bool moveXyTheta(
        const XyThetaPosition& target,
        std::chrono::milliseconds timeout = std::chrono::seconds(30));

    XyzPosition currentXyz() const;
    XyThetaPosition currentXyTheta() const;
    UvwMotorPosition currentUvwMotors() const;

    bool home(std::chrono::milliseconds timeout = std::chrono::seconds(60));
    void stop();
    void emergencyStop();

    bool isConnected() const { return connected_; }
    StageType type() const { return type_; }

private:
    UvwMotorPosition xyThetaToUvw(const XyThetaPosition& target) const;
    bool waitUntilIdle(std::chrono::milliseconds timeout);

    StageType type_;
    bool simulate_;
    bool connected_ = false;
    bool emergencyStopped_ = false;

    XyzPosition xyz_{};
    XyThetaPosition xyTheta_{};
    UvwMotorPosition uvw_{};
    UvwTransform uvwTransform_;
};
