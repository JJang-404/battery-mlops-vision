#include "Preprocessor.h"

#include <stdexcept>

namespace {

constexpr float MEAN[3] = {0.485f, 0.456f, 0.406f};
constexpr float STD[3] = {0.229f, 0.224f, 0.225f};

} // namespace

std::vector<float> Preprocessor::LoadAndPreprocess(const std::string& imagePath)
{
    cv::Mat bgr = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (bgr.empty()) {
        throw std::runtime_error("이미지 로드 실패: " + imagePath);
    }
    return matToNCHW(bgr);
}

std::vector<float> Preprocessor::matToNCHW(const cv::Mat& bgr)
{
    cv::Mat rgb;
    cv::Mat resized;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    cv::resize(rgb, resized, {INPUT_SIZE, INPUT_SIZE}, 0, 0, cv::INTER_LINEAR);

    const int planeSize = INPUT_SIZE * INPUT_SIZE;
    std::vector<float> output(3 * planeSize);

    for (int y = 0; y < INPUT_SIZE; ++y) {
        for (int x = 0; x < INPUT_SIZE; ++x) {
            const cv::Vec3b pixel = resized.at<cv::Vec3b>(y, x);
            for (int channel = 0; channel < 3; ++channel) {
                float value = pixel[channel] / 255.0f;
                value = (value - MEAN[channel]) / STD[channel];
                output[channel * planeSize + y * INPUT_SIZE + x] = value;
            }
        }
    }
    return output;
}
