using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace BatteryDemo
{
    internal class Inferencer : IDisposable
    {
        private readonly InferenceSession _session;
        public string InputName { get; }
        public string OutputName { get; }
        public int[] InputShape { get; } = { 1, 3, 513, 513 };

        public bool UsingGpu { get; private set; }

        public Inferencer(string onnxPath, bool useGpu = true)
        {
            var opts = new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            };

            if (useGpu)
            {
                // CUDA EP 등록 시도. 실패 원인 2종:
                //  (1) Microsoft.ML.OnnxRuntime (CPU) 만 설치됨 → EntryPointNotFoundException
                //  (2) GPU 패키지는 맞지만 cuDNN/cuBLAS DLL 미발견 → OnnxRuntimeException
                // 어느 쪽이든 CPU 폴백으로 데모는 계속 동작시킨다.
                try
                {
                    opts.AppendExecutionProvider_CUDA(0);   // deviceId 0
                    UsingGpu = true;
                    Console.WriteLine("[ORT] CUDA EP 등록 성공");
                }
                catch (EntryPointNotFoundException)
                {
                    Console.WriteLine("[ORT] ⚠ CUDA EP 진입점 없음 — 'Microsoft.ML.OnnxRuntime.Gpu' NuGet 패키지 설치 필요. CPU로 폴백");
                    UsingGpu = false;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[ORT] ⚠ CUDA EP 등록 실패({ex.GetType().Name}): {ex.Message}. CPU로 폴백");
                    UsingGpu = false;
                }
            }
            // CPU 는 항상 fallback 으로 자동 추가됨

            _session = new InferenceSession(onnxPath, opts);
            InputName = _session.InputMetadata.Keys.First();
            OutputName = _session.OutputMetadata.Keys.First();

            // 실제로 어떤 provider 가 잡혔는지 확인
            var actualProviders = string.Join(", ", _session.InputMetadata.Keys);
            Console.WriteLine($"Session input  : {InputName} {string.Join(",", InputShape)}");
            Console.WriteLine($"Session output : {OutputName}");
        }
        public float[] Run(float[] inputData)
        {
            var tensor = new DenseTensor<float>(inputData, InputShape);
            var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(InputName, tensor)
        };

            using var results = _session.Run(inputs);
            var output = results.First().AsTensor<float>();
            return output.ToArray();   // shape: 1*3*513*513 flatten
        }

        public void Dispose() => _session?.Dispose();
    }
}
