using System.Diagnostics;
using System.Runtime.InteropServices;
using BatteryDemo;
using OpenCvSharp;

DllDir.Setup();

string baseDir = AppContext.BaseDirectory;
string onnxPath = Path.Combine(baseDir, "models", "battery_deeplab_v1.onnx");
string imgDir = Path.Combine(baseDir, "test_images");
List<string> imgPaths = ResolveImagePaths(args, imgDir);

if (!File.Exists(onnxPath)) throw new FileNotFoundException(onnxPath);

// ---- 이미지 경로 해석 ----
// 인자 없음 또는 폴더 → 폴더 내 모든 이미지 배치 처리
// 인자가 파일      → 단일 처리
static List<string> ResolveImagePaths(string[] args, string defaultDir)
{
    if (args.Length > 0)
    {
        if (File.Exists(args[0])) return new List<string> { args[0] };
        if (Directory.Exists(args[0])) return ListImages(args[0]);
        throw new FileNotFoundException($"인자가 파일도 폴더도 아님: {args[0]}");
    }
    return ListImages(defaultDir);
}

static List<string> ListImages(string dir)
{
    string[] exts = { "*.png", "*.jpg", "*.jpeg", "*.bmp" };
    var found = exts
        .SelectMany(p => Directory.EnumerateFiles(dir, p, SearchOption.TopDirectoryOnly))
        .OrderBy(f => f, StringComparer.OrdinalIgnoreCase)
        .ToList();

    if (found.Count == 0)
        throw new FileNotFoundException($"'{dir}' 폴더에 PNG/JPG/BMP 이미지가 없습니다.");

    Console.WriteLine($"[demo] {found.Count}장 이미지 발견");
    foreach (var f in found) Console.WriteLine($"   - {Path.GetFileName(f)}");
    return found;
}

// 1. ORT 세션 초기화 (모든 이미지에 재사용)
using var inf = new Inferencer(onnxPath, useGpu: true);

// 2. 워밍업 5회 — 첫 이미지로 한 번만
Console.WriteLine($"\n[batch] 워밍업 시작...");
float[] warmupData = Preprocessor.LoadAndPreprocess(imgPaths[0]);
var warmSw = Stopwatch.StartNew();
for (int i = 0; i < 5; i++) inf.Run(warmupData);
Console.WriteLine($"[batch] 워밍업 5회 완료: {warmSw.ElapsedMilliseconds} ms");

// 3. 배치 처리
var sw = new Stopwatch();
var latencies = new List<double>();

for (int idx = 0; idx < imgPaths.Count; idx++)
{
    string path = imgPaths[idx];
    string fileName = Path.GetFileName(path);
    Console.WriteLine($"\n=== [{idx + 1}/{imgPaths.Count}] {fileName} ===");

    // 전처리
    sw.Restart();
    float[] inputData = Preprocessor.LoadAndPreprocess(path);
    long prepMs = sw.ElapsedMilliseconds;

    // 추론 (20회 평균)
    sw.Restart();
    const int N = 20;
    float[] logits = new float[1];
    for (int i = 0; i < N; i++) logits = inf.Run(inputData);
    double avgInferMs = sw.ElapsedMilliseconds / (double)N;
    latencies.Add(avgInferMs);

    // 후처리 + 오버레이
    sw.Restart();
    using var original = Cv2.ImRead(path, ImreadModes.Color);
    using var overlay = Postprocessor.BuildOverlay(logits, original);
    long postMs = sw.ElapsedMilliseconds;

    Console.WriteLine($"전처리: {prepMs} ms | 추론(평균 {N}회): {avgInferMs:F1} ms | 후처리: {postMs} ms");

    // 저장
    string num = (idx + 1).ToString("D3");
    string ovPath = Path.Combine(baseDir, $"overlay_{num}.png");
    string baPath = Path.Combine(baseDir, $"before_after_{num}.png");

    using var beforeAfter = new Mat();
    Cv2.HConcat(original, overlay, beforeAfter);
    Cv2.ImWrite(ovPath, overlay);
    Cv2.ImWrite(baPath, beforeAfter);
    Console.WriteLine($"저장: {Path.GetFileName(ovPath)}, {Path.GetFileName(baPath)}");

    // 윈도우 표시 — 한 장씩 키 입력
    Cv2.ImShow($"[{idx + 1}/{imgPaths.Count}] {fileName} - Before | After", beforeAfter);
    Console.WriteLine($"(아무 키나 누르면 {(idx + 1 < imgPaths.Count ? "다음 이미지" : "종료")})");
    Cv2.WaitKey(0);
    Cv2.DestroyAllWindows();
}

// 4. 배치 통계 — latency 안정성 검증 (가이드 §5 합격선)
Console.WriteLine($"\n=== 배치 결과 ({imgPaths.Count}장) ===");
Console.WriteLine($"평균 latency : {latencies.Average():F1} ms/image");
Console.WriteLine($"min / max    : {latencies.Min():F1} / {latencies.Max():F1} ms");
Console.WriteLine($"변동폭       : {(latencies.Max() - latencies.Min()):F1} ms (가이드 합격선: ≤ 10 ms)");

static class DllDir
{
    [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
    private static extern bool SetDllDirectory(string lpPathName);

    public static void Setup()
    {
        // venv_battery 의 NVIDIA DLL 경로 (cuDNN 9 + cuBLAS + cuda_runtime)
        string[] dirs =
        {
            @"D:\02.study\part4_wj\Battery\Battery_Project\venv_battery\Lib\site-packages\nvidia\cudnn\bin",
            @"D:\02.study\part4_wj\Battery\Battery_Project\venv_battery\Lib\site-packages\nvidia\cublas\bin",
            @"D:\02.study\part4_wj\Battery\Battery_Project\venv_battery\Lib\site-packages\nvidia\cuda_runtime\bin",
        };

        // SetDllDirectory 는 마지막 호출만 유효 → PATH 환경변수에 prepend 하는 방식이 다중 경로에 안전
        string current = Environment.GetEnvironmentVariable("PATH") ?? "";
        foreach (var d in dirs)
        {
            if (Directory.Exists(d))
            {
                current = d + Path.PathSeparator + current;
                Console.WriteLine($"[dll] PATH += {d}");
            }
            else
            {
                Console.WriteLine($"[dll] ⚠ 경로 없음: {d}");
            }
        }
        Environment.SetEnvironmentVariable("PATH", current);
    }
}