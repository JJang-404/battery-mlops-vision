#include "Postprocessor.h"
#include <iomanip>
#include <iostream>
using namespace std;

const cv::Scalar Postprocessor::CLASS_COLORS[NUM_CLASSES] = {
    {0,   0,   0},      // 0: background
    {0,   255, 255},    // 1: Pollution — 노랑
    {0,   0,   255},    // 2: Damaged   — 빨강
};

cv::Mat Postprocessor::buildOverlay(const vector<float>& logits,
                                    const cv::Mat& originalBgr,
                                    float fillAlpha,
                                    int contourThickness)
{
    cv::Mat mask513 = getSegmentationMask(logits);

    // 2. 원본 해상도로 리사이즈 (INTER_NEAREST - 클래스 인덱스 깨짐 방지)
    cv::Mat maskFull;
    cv::resize(mask513, maskFull, originalBgr.size(), 0, 0, cv::INTER_NEAREST);

    // 3. 클래스별 fill + contour 그리기
    cv::Mat overlay = originalBgr.clone();
    cv::Mat colorMap = getColorMap();
    cv::Mat colorMask = applyColorMapToMask(maskFull, colorMap);
    cv::Mat blended;
    cv::addWeighted(overlay, 1.0, colorMask, fillAlpha, 0, blended);

    for (int c=1; c<NUM_CLASSES; c++) {
        cv::Mat classMask;
        cv::compare(maskFull, c, classMask, cv::CMP_EQ);
        int px = cv::countNonZero(classMask);
        if (px == 0) continue; // 해당 클래스 픽셀 없으면 스킵

        blended.copyTo(overlay, classMask);

        // contour - 선명한 외곽선
        drawContours(overlay, classMask, CLASS_COLORS[c], contourThickness);

        int totalPixels = originalBgr.total();
        string name = (c == 1) ? "Pollution" : "Damaged";
        cout << fixed << setprecision(2)
             << "[mask] " << name << ": " << px << " px ("
             << 100.0 * px / totalPixels << "%)\n";
    }
    drawLegend(overlay);
    return overlay;
}

cv::Mat Postprocessor::getSegmentationMask(const vector<float>& logits)
{
    const int planeSize = SIZE * SIZE;
    cv::Mat mask513(SIZE, SIZE, CV_8UC1);

    for (int y=0; y<SIZE; y++){
        for (int x=0; x<SIZE; x++) {
            int bestClass = 0;
            float bestVal = logits[y * SIZE + x];
            for (int c=1; c<NUM_CLASSES; c++) {
                float val = logits[c * planeSize + y * SIZE + x];
                if (val > bestVal) {bestVal = val; bestClass = c;}
            }
            mask513.at<uchar>(y, x) = static_cast<uchar>(bestClass);
        }
    }
    return mask513;
}

cv::Mat Postprocessor::getColorMap()
{
    cv::Mat colorMap(1, NUM_CLASSES, CV_8UC3, cv::Scalar(0, 0, 0));
    for (int c=0; c<NUM_CLASSES; c++) {
        colorMap.at<cv::Vec3b>(0, c) = cv::Vec3b(
            static_cast<uchar>(CLASS_COLORS[c][0]),
            static_cast<uchar>(CLASS_COLORS[c][1]),
            static_cast<uchar>(CLASS_COLORS[c][2])
        );
    }
    return colorMap;
}

cv::Mat Postprocessor::applyColorMapToMask(const cv::Mat& mask, const cv::Mat& colorMap)
{
    cv::Mat colorMask(mask.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    for (int c=1; c<NUM_CLASSES; c++) {
        cv::Mat classMask;
        cv::compare(mask, c, classMask, cv::CMP_EQ);
        cv::Vec3b color = colorMap.at<cv::Vec3b>(0, c);
        colorMask.setTo(cv::Scalar(color[0], color[1], color[2]), classMask);
    }
    return colorMask;
}

void Postprocessor::drawContours(cv::Mat& overlay, const cv::Mat& mask, const cv::Scalar& color, int thickness)
{
    vector<vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::drawContours(overlay, contours, -1, color, thickness);
}

void Postprocessor::drawLegend(cv::Mat& img)
{
    const int width = 200, height = 80 , margin = 20;
    int x = img.cols - width - margin, y= img.rows - height - margin;
    
    cv::Mat roi = img(cv::Rect(x,y,width, height));
    cv::Mat dark(roi.size(), CV_8UC3, {0,0,0});
    cv::addWeighted(roi, 0.35, dark, 0.65, 0, roi);
    cv::rectangle(img, {x,y,width, height}, cv::Scalar::all(255), 1);

    const char* labels[] = {"Pollution", "Damaged"};
    for(int c=1; c<NUM_CLASSES; ++c){
        int rowY = y+12  + (c-1)*33;
        cv::rectangle(img, {x+10, rowY, 18,18 }, CLASS_COLORS[c], -1);
        cv::rectangle(img,{x+10, rowY, 18,18 }, cv::Scalar::all(255), 1);
        cv::putText(img, labels[c-1], {x+38, rowY+15},
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar::all(255), 1, cv::LINE_AA);
    }

}
