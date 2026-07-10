// include/Preprocessor.h

#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
using namespace std;

class Preprocessor {
    public:
        static constexpr int INPUT_SIZE = 513; // shape of the input image for the model

        // 이미지 파일 -> NCHW float array (1, 3, INPUT_SIZE, INPUT_SIZE)
        // C#: Postprocessor.LoadAndPreprocess() 에 해당
        static vector<float> LoadAndPreprocess(const string& image_path);

        // 메모리 이미지 기반: 카메라 캡처 프레임 또는 이미 로드된 이미지에 사용
        static vector<float> matToNCHW(const cv::Mat& bgr);
    
};