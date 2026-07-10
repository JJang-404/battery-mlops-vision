#include "Inferencer.h"
#include <iostream>
#include <stdexcept>
using namespace std;

Inferencer::Inferencer(const string& onnxPath, bool useGPU)
    : env_(ORT_LOGGING_LEVEL_WARNING, "battery_demo")
{
    Ort::SessionOptions opts;
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    if(useGPU) {
        try{
            OrtCUDAProviderOptions cuda;
            cuda.device_id = 0;
            opts.AppendExecutionProvider_CUDA(cuda);
            gpu_ = true;
            std::cout << "[ORT] CUDA EP 등록 성공\n"; 
        }
        catch(const Ort::Exception& e) {
            std::cerr << "[ORT] CUDA EP 등록 실패: " << e.what() << "\n";
            std::cerr << "[ORT] CPU EP로 대체합니다.\n";
        }
    }

    session_ = Ort::Session(env_, onnxPath.c_str(), opts);

    // 입출력 이름 조회 (C# _session.InputMetadata 에 해당)
    auto inName = session_.GetInputNameAllocated(0, allocator_);
    auto outName = session_.GetOutputNameAllocated(0, allocator_);
    inputName_ = inName.get();
    outputName_ = outName.get();

    cout << "[ORT] Input Name: " << inputName_ << endl;
    cout << "[ORT] Output Name: " << outputName_ << endl;

}

vector<float> Inferencer::run(const vector<float>& inputData) {
    auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memInfo,
        const_cast<float*>(inputData.data()), inputData.size(),
        INPUT_SHAPE, 4
    );

    const char* inNames[] = {inputName_.c_str()};
    const char* outNames[] = {outputName_.c_str()};

    auto outputs = session_.Run(Ort::RunOptions{nullptr},
                                inNames, &inputTensor, 1, outNames, 1
    );

    float* dataPtr = outputs[0].GetTensorMutableData<float>();
    size_t count = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
    return vector<float>(dataPtr, dataPtr + count);
}