#include "Preprocessor.h"
#include "Inferencer.h"
#include "Postprocessor.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

bool isImageFile(const fs::path& path)
{
    if (!fs::is_regular_file(path)) return false;

    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });

    return extension == ".png" || extension == ".jpg" ||
           extension == ".jpeg" || extension == ".bmp";
}

std::vector<fs::path> findImages(const fs::path& inputPath)
{
    if (fs::is_regular_file(inputPath)) {
        if (!isImageFile(inputPath)) {
            throw std::runtime_error("지원하지 않는 이미지 파일입니다: " + inputPath.string());
        }
        return {inputPath};
    }

    if (!fs::is_directory(inputPath)) {
        throw std::runtime_error("파일 또는 폴더를 찾을 수 없습니다: " + inputPath.string());
    }

    std::vector<fs::path> imagePaths;
    for (const auto& entry : fs::directory_iterator(inputPath)) {
        if (isImageFile(entry.path())) imagePaths.push_back(entry.path());
    }

    std::sort(imagePaths.begin(), imagePaths.end());
    if (imagePaths.empty()) {
        throw std::runtime_error("폴더에 PNG/JPG/JPEG/BMP 이미지가 없습니다: " +
                                 inputPath.string());
    }
    return imagePaths;
}

} // namespace

int main(int argc, char* argv[])
{
    const fs::path modelPath = "models/battery_deeplab_v1.onnx";
    const fs::path inputPath = argc > 1 ? fs::path(argv[1]) : fs::path("test_images");
    const fs::path outputDirectory = "monitoring_output";

    try {
        const std::vector<fs::path> imagePaths = findImages(inputPath);
        fs::create_directories(outputDirectory);

        std::cout << "[demo] 입력: " << inputPath << '\n'
                  << "[demo] 이미지 " << imagePaths.size() << "장 발견\n"
                  << "[demo] ONNX 모델 로드\n";

        // macOS에서는 우선 CPU로 실행한다.
        Inferencer inferencer(modelPath.string(), false);

        int successCount = 0;
        for (std::size_t index = 0; index < imagePaths.size(); ++index) {
            const fs::path& imagePath = imagePaths[index];
            std::cout << "\n=== [" << index + 1 << '/' << imagePaths.size()
                      << "] " << imagePath.filename().string() << " ===\n";

            try {
                cv::Mat original = cv::imread(imagePath.string(), cv::IMREAD_COLOR);
                if (original.empty()) {
                    throw std::runtime_error("이미지를 불러올 수 없습니다.");
                }

                const auto startedAt = std::chrono::steady_clock::now();
                std::vector<float> input =
                    Preprocessor::LoadAndPreprocess(imagePath.string());
                std::vector<float> logits = inferencer.run(input);
                cv::Mat overlay = Postprocessor::buildOverlay(logits, original);
                const auto finishedAt = std::chrono::steady_clock::now();

                const double elapsedMs =
                    std::chrono::duration<double, std::milli>(finishedAt - startedAt).count();
                const fs::path outputPath =
                    outputDirectory / ("overlay_" + imagePath.filename().string());

                if (!cv::imwrite(outputPath.string(), overlay)) {
                    throw std::runtime_error("결과 이미지를 저장하지 못했습니다.");
                }

                cv::Mat beforeAfter;
                cv::hconcat(original, overlay, beforeAfter);

                std::cout << std::fixed << std::setprecision(1)
                          << "[완료] 처리 시간: " << elapsedMs << " ms\n"
                          << "[저장] " << outputPath << '\n'
                          << "[안내] 아무 키나 누르면 다음 이미지로 이동합니다.\n";

                cv::imshow("Battery Inspection - Original | Overlay", beforeAfter);
                cv::waitKey(0);
                cv::destroyAllWindows();
                ++successCount;
            }
            catch (const std::exception& error) {
                cv::destroyAllWindows();
                std::cerr << "[실패] " << imagePath.filename().string()
                          << ": " << error.what() << '\n';
            }
        }

        std::cout << "\n=== 검사 종료 ===\n"
                  << "성공: " << successCount << "장\n"
                  << "실패: " << imagePaths.size() - successCount << "장\n"
                  << "결과 폴더: " << outputDirectory << '\n';

        return successCount == static_cast<int>(imagePaths.size()) ? 0 : 1;
    }
    catch (const std::exception& error) {
        std::cerr << "실행 오류: " << error.what() << '\n';
        return 1;
    }
}
