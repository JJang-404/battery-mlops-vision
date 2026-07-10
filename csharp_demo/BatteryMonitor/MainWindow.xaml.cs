using System.Collections.ObjectModel;
using System.Diagnostics;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Threading;
using BatteryDemo;
using Microsoft.Win32;
using OpenCvSharp;

namespace BatteryMonitor;

public partial class MainWindow : Window
{
    private static readonly string[] ImageExtensions = [".png", ".jpg", ".jpeg", ".bmp"];
    private readonly DispatcherTimer _clockTimer;
    private CancellationTokenSource? _cancellation;
    private string _inputFolder;
    private int _selectedIndex = -1;

    public ObservableCollection<InspectionRecord> Records { get; } = [];

    public MainWindow()
    {
        InitializeComponent();
        DataContext = this;

        _inputFolder = Path.Combine(AppContext.BaseDirectory, "test_images");
        InputFolderText.Text = _inputFolder;

        _clockTimer = new DispatcherTimer { Interval = TimeSpan.FromSeconds(1) };
        _clockTimer.Tick += (_, _) => ClockText.Text = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");
        _clockTimer.Start();
        ClockText.Text = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");

        AddLog("모니터링 시스템 준비 완료");
    }

    private async void StartButton_Click(object sender, RoutedEventArgs e)
    {
        string modelPath = Path.Combine(AppContext.BaseDirectory, "models", "battery_deeplab_v1.onnx");
        List<string> images = FindImages(_inputFolder);

        if (!File.Exists(modelPath)) {
            MessageBox.Show($"모델을 찾을 수 없습니다.\n{modelPath}", "Model Error",
                MessageBoxButton.OK, MessageBoxImage.Error);
            return;
        }
        if (images.Count == 0) {
            MessageBox.Show("선택한 폴더에 이미지가 없습니다.", "Input Error",
                MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }

        Records.Clear();
        _selectedIndex = -1;
        _cancellation = new CancellationTokenSource();
        SetRunningState(true);
        TotalText.Text = $"Total: {images.Count}";
        BatchProgress.Maximum = images.Count;
        BatchProgress.Value = 0;
        AddLog($"{images.Count}장 배치 검사 시작");

        try {
            using var inferencer = await Task.Run(() => new Inferencer(modelPath, useGpu: true));

            for (int index = 0; index < images.Count; index++) {
                _cancellation.Token.ThrowIfCancellationRequested();
                string imagePath = images[index];
                StatusText.Text = $"INSPECTING  {index + 1}/{images.Count}";
                AddLog($"검사 시작: {Path.GetFileName(imagePath)}");

                InspectionRecord record = await Task.Run(
                    () => InspectOne(inferencer, imagePath, index + 1),
                    _cancellation.Token);

                Records.Add(record);
                HistoryGrid.SelectedItem = record;
                HistoryGrid.ScrollIntoView(record);
                ShowRecord(record);
                UpdateBatchStatus(images.Count);
            }

            StatusText.Text = "BATCH COMPLETED";
            AddLog("배치 검사 완료");
        }
        catch (OperationCanceledException) {
            StatusText.Text = "STOPPED";
            AddLog("사용자가 검사를 중지했습니다.");
        }
        catch (Exception error) {
            StatusText.Text = "ERROR";
            AddLog($"오류: {error.Message}");
            MessageBox.Show(error.ToString(), "Inspection Error",
                MessageBoxButton.OK, MessageBoxImage.Error);
        }
        finally {
            SetRunningState(false);
            _cancellation?.Dispose();
            _cancellation = null;
        }
    }

    private static InspectionRecord InspectOne(Inferencer inferencer, string imagePath, int number)
    {
        var stopwatch = Stopwatch.StartNew();
        float[] input = Postprocessor.LoadAndPreprocess(imagePath);
        float[] logits = inferencer.Run(input);

        using Mat original = Cv2.ImRead(imagePath, ImreadModes.Color);
        using Mat overlay = Postprocessor.BuildOverlay(logits, original);
        stopwatch.Stop();

        (int pollution, int damaged) = CountClasses(logits);
        string result = pollution == 0 && damaged == 0 ? "OK" : "NG";

        string outputDirectory = Path.Combine(AppContext.BaseDirectory, "monitoring_output");
        Directory.CreateDirectory(outputDirectory);
        string outputPath = Path.Combine(outputDirectory, $"overlay_{Path.GetFileName(imagePath)}");
        Cv2.ImWrite(outputPath, overlay);

        return new InspectionRecord {
            Number = number,
            FileName = Path.GetFileName(imagePath),
            Result = result,
            PollutionPixels = pollution,
            DamagedPixels = damaged,
            LatencyMs = stopwatch.Elapsed.TotalMilliseconds,
            OriginalPath = imagePath,
            OverlayPath = outputPath,
        };
    }

    private static (int Pollution, int Damaged) CountClasses(float[] logits)
    {
        const int size = 513;
        const int planeSize = size * size;
        int pollution = 0;
        int damaged = 0;

        for (int pixel = 0; pixel < planeSize; pixel++) {
            int bestClass = 0;
            float bestValue = logits[pixel];
            for (int classIndex = 1; classIndex < 3; classIndex++) {
                float value = logits[classIndex * planeSize + pixel];
                if (value > bestValue) {
                    bestValue = value;
                    bestClass = classIndex;
                }
            }
            if (bestClass == 1) pollution++;
            if (bestClass == 2) damaged++;
        }
        return (pollution, damaged);
    }

    private static List<string> FindImages(string directory)
    {
        if (!Directory.Exists(directory)) return [];

        return Directory.EnumerateFiles(directory)
            .Where(path => ImageExtensions.Contains(Path.GetExtension(path),
                StringComparer.OrdinalIgnoreCase))
            .OrderBy(Path.GetFileName, StringComparer.OrdinalIgnoreCase)
            .ToList();
    }

    private void ShowRecord(InspectionRecord record)
    {
        _selectedIndex = Records.IndexOf(record);
        OriginalImage.Source = LoadBitmap(record.OriginalPath);
        OverlayImage.Source = LoadBitmap(record.OverlayPath);
        CurrentFileText.Text = $"File: {record.FileName}";
        PollutionText.Text = $"Pollution: {record.PollutionPixels:N0} px";
        DamagedText.Text = $"Damaged: {record.DamagedPixels:N0} px";
        LatencyText.Text = $"Latency: {record.LatencyMs:F1} ms";
        ResultText.Text = record.Result;
        ResultPanel.Background = new SolidColorBrush(
            record.Result == "OK" ? Color.FromRgb(24, 123, 120) : Color.FromRgb(172, 57, 57));
    }

    private static BitmapImage LoadBitmap(string path)
    {
        var image = new BitmapImage();
        image.BeginInit();
        image.CacheOption = BitmapCacheOption.OnLoad;
        image.UriSource = new Uri(path, UriKind.Absolute);
        image.EndInit();
        image.Freeze();
        return image;
    }

    private void UpdateBatchStatus(int total)
    {
        int okCount = Records.Count(record => record.Result == "OK");
        int ngCount = Records.Count - okCount;
        ProcessedText.Text = $"Processed: {Records.Count}";
        OkText.Text = $"OK: {okCount}";
        NgText.Text = $"NG: {ngCount}";
        BatchProgress.Value = Records.Count;
    }

    private void SetRunningState(bool running)
    {
        StartButton.IsEnabled = !running;
        StopButton.IsEnabled = running;
    }

    private void AddLog(string message)
    {
        LogList.Items.Insert(0, $"{DateTime.Now:HH:mm:ss}  {message}");
    }

    private void StopButton_Click(object sender, RoutedEventArgs e) => _cancellation?.Cancel();

    private void FolderButton_Click(object sender, RoutedEventArgs e)
    {
        var dialog = new OpenFolderDialog {
            Title = "검사 이미지 폴더 선택",
            InitialDirectory = Directory.Exists(_inputFolder) ? _inputFolder : null,
        };
        if (dialog.ShowDialog() == true) {
            _inputFolder = dialog.FolderName;
            InputFolderText.Text = _inputFolder;
            AddLog($"입력 폴더 변경: {_inputFolder}");
        }
    }

    private void PreviousButton_Click(object sender, RoutedEventArgs e)
    {
        if (Records.Count == 0) return;
        _selectedIndex = Math.Max(0, _selectedIndex - 1);
        HistoryGrid.SelectedItem = Records[_selectedIndex];
        ShowRecord(Records[_selectedIndex]);
    }

    private void NextButton_Click(object sender, RoutedEventArgs e)
    {
        if (Records.Count == 0) return;
        _selectedIndex = Math.Min(Records.Count - 1, _selectedIndex + 1);
        HistoryGrid.SelectedItem = Records[_selectedIndex];
        ShowRecord(Records[_selectedIndex]);
    }

    private void HistoryGrid_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (HistoryGrid.SelectedItem is InspectionRecord record) ShowRecord(record);
    }

    private void ExitButton_Click(object sender, RoutedEventArgs e)
    {
        _cancellation?.Cancel();
        Close();
    }
}
