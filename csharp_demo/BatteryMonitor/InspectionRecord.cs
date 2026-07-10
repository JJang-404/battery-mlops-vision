namespace BatteryMonitor;

public sealed class InspectionRecord
{
    public int Number { get; init; }
    public string FileName { get; init; } = "";
    public string Result { get; init; } = "";
    public int PollutionPixels { get; init; }
    public int DamagedPixels { get; init; }
    public double LatencyMs { get; init; }
    public string OriginalPath { get; init; } = "";
    public string OverlayPath { get; init; } = "";
}
