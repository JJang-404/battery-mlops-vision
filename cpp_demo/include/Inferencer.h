#pragma once
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
using namespace std;

// C# : Inferencer 클래스와 동일한 역할
// ONNX Runtime C++ API를 사용하여 ONNX 모델을 로드하고 추론을 수행하는 클래스
class Inferencer{
    public:
        explicit Inferencer(const string& onnxPath, bool useGPU=true);
        
        // 입력 : NCHW 형식의 이미지 데이터 (vector<float> 형식 (1x3x513x513)
        // 출력 : 추론 결과 (vector<float> 형식 (1x3x513x513)
        vector<float> run(const vector<float>& inputData);

        bool usingGPU() const { return gpu_; }

    private:
        Ort::Env env_;
        Ort::Session session_{nullptr};
        Ort::AllocatorWithDefaultOptions allocator_;
        std::string inputName_, outputName_;
        bool gpu_ = false;

        static constexpr int64_t INPUT_SHAPE[4] = {1,3,513,513};
      
};