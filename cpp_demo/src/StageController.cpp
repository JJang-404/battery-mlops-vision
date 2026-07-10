#include "StageController.h"
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <utility>
using namespace std;

StageController::StageController(
    StageType type,
    bool simulate,
    UvwTransform uvwTransform)
    : type_(type),
    simulate_(simulate),
    uvwTransform_(move(uvwTransform)) {}


StageController::~StageController() {
    disconnect();
}

bool StageController::connect(const string& endpoint) {
    if (connected_) return true;
    
    if (simulate_){
        connected_ = true;
        emergencyStopped_ = false;
        cout << "[Endpoint] : " << endpoint << '\n';
        return true;
    }

    // 실제 장비 연결부는 아직 임시 상태입니다.
    // 추후 제조사 모션 SDK, 시리얼, TCP/IP, EtherCAT, PLC 통신 코드로 교체합니다.
    cout << "[Stage] 실제 장비 연결은 아직 구현되지 않았습니다: " << endpoint<< '\n';
    return false;
}

void StageController::disconnect(){
    if (!connected_) return;

    if(!simulate_) {
        // TODO: 제조사 SDK 연결을 종료하거나 통신 핸들을 닫는 코드를 작성합니다.
    }
    connected_ = false;
    cout << "[Stage] disconnected\n";
}

bool StageController::moveXyz(const XyzPosition& target,
                            chrono::milliseconds timeout)
{
    if (type_ != StageType::CartesianXYZ) {
        throw std::logic_error("moveXyz() is only valid for a Cartesian XYZ stage.");
    }

    if(!connected_ || emergencyStopped_) return false;

    // TODO: 이동 전에 소프트 리미트와 기계적 간섭 가능성을 확인합니다.
    // TODO: 실제 모션 컨트롤러에 X/Y/Z 절대 이동 명령을 전송합니다.
    cout << "[XYZ] target = " << target.xMm << ", "
    << target.yMm << ", " << target.zMm << " mm\n";

    if(!waitUntilIdle(timeout)) return false;

    xyz_ = target;
    return true;
}

bool StageController::moveXyTheta(
    const XyThetaPosition& target,
    std::chrono::milliseconds timeout)
{
    if (type_ != StageType::UvwAlignment) {
        throw std::logic_error("moveXyTheta() is only valid for a UVW alignment stage.");
    }
    if (!connected_ || emergencyStopped_) return false;

    const UvwMotorPosition motorTarget = xyThetaToUvw(target);

    // TODO: 동기 이동 전에 U/V/W 각 축의 스트로크 한계를 확인합니다.
    // TODO: 실제 모션 컨트롤러에 U/V/W 동기 이동 명령을 전송합니다.
    std::cout << "[UVW] logical XYTheta = "
              << target.xMm << ", " << target.yMm << ", "
              << target.thetaDeg << " deg\n";
    std::cout << "[UVW] motor UVW = "
              << motorTarget.uMm << ", " << motorTarget.vMm << ", "
              << motorTarget.wMm << " mm\n";

    if (!waitUntilIdle(timeout)) return false;

    uvw_ = motorTarget;
    xyTheta_ = target;
    return true;
}

XyzPosition StageController::currentXyz() const {
    return xyz_;
}

XyThetaPosition StageController::currentXyTheta() const {
    return xyTheta_;
}

UvwMotorPosition StageController::currentUvwMotors() const {
    return uvw_;
}

bool StageController::home(std::chrono::milliseconds timeout) {
    if (!connected_ || emergencyStopped_) return false;

    // TODO: 제조사 장비에 맞는 원점 복귀 시퀀스를 실행하고 완료를 기다립니다.
    std::cout << "[Stage] homing\n";
    if (!waitUntilIdle(timeout)) return false;

    xyz_ = {};
    xyTheta_ = {};
    uvw_ = {};
    return true;
}

void StageController::stop() {
    if (!connected_) return;

    // TODO: 모션 컨트롤러에 감속 정지 명령을 전송합니다.
    std::cout << "[Stage] stop\n";
}

void StageController::emergencyStop() {
    if (!connected_) return;

    // TODO: 장비 안전 규칙에 맞게 즉시 정지 또는 서보 OFF 명령을 전송합니다.
    emergencyStopped_ = true;
    std::cout << "[Stage] emergency stop\n";
}

UvwMotorPosition StageController::xyThetaToUvw(
    const XyThetaPosition& target) const
{
    if (!uvwTransform_) {
        throw std::logic_error(
            "UVW transform is not configured. Inject the vendor API or "
            "a kinematic model verified with the real stage drawing.");
    }

    return uvwTransform_(target);
}

bool StageController::waitUntilIdle(std::chrono::milliseconds timeout) {
    if (timeout <= std::chrono::milliseconds::zero()) return false;

    if (simulate_) {
        const auto simulatedDelay =
            std::min(timeout, std::chrono::milliseconds(100));
        std::this_thread::sleep_for(simulatedDelay);
        return !emergencyStopped_;
    }

    // TODO: 컨트롤러에서 이동 완료, 알람, 타임아웃 신호를 주기적으로 확인합니다.
    return false;
}
