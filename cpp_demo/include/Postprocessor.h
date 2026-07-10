#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
using namespace std;

class Postprocessor {
public:
    static constexpr int NUM_CLASSES = 3; // Number of classes for segmentation
    static constexpr int SIZE = 513;

    // C#: Postprocessor.BuildOverlay() 와 동일
    static cv::Mat buildOverlay(const vector<float>& logits, 
                                const cv::Mat& originalBgr,
                                float fillAlpha = 0.25f,
                                int contourThickness = 3);
private:
    static void drawLegend(cv::Mat& img);
    // BGR 클래스 색상 (C#: ClassColors[] 와 동일)
    static const cv::Scalar CLASS_COLORS[NUM_CLASSES];
    static cv::Mat getSegmentationMask(const vector<float>& logits);
    static cv::Mat getColorMap();
    static cv::Mat applyColorMapToMask(const cv::Mat& mask, const cv::Mat& colorMap);
    static void drawContours(cv::Mat& overlay, const cv::Mat& mask, const cv::Scalar& color, int thickness);
};                
