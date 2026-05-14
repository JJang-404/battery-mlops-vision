using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCvSharp;

namespace BatteryDemo
{
    internal static class Postprocessor
    {
        // BGR→RGB, resize, normalize, NCHW
        private static readonly float[] Mean = { 0.485f, 0.456f, 0.406f };
        private static readonly float[] Std = { 0.229f, 0.224f, 0.225f };
        private const int Size = 513;

        /// <summary>이미지 파일 → ONNX 입력 float[] (NCHW, 1*3*513*513)</summary>
        public static float[] LoadAndPreprocess(string path)
        {
            using var bgr = Cv2.ImRead(path, ImreadModes.Color);          // BGR
            if (bgr.Empty()) throw new FileNotFoundException(path);

            using var rgb = new Mat();
            Cv2.CvtColor(bgr, rgb, ColorConversionCodes.BGR2RGB);          // BGR → RGB

            using var resized = new Mat();
            Cv2.Resize(rgb, resized, new OpenCvSharp.Size(Size, Size), 0, 0, InterpolationFlags.Linear);

            // HWC uint8 → NCHW float32 + ImageNet normalize
            var result = new float[1 * 3 * Size * Size];
            var indexer = resized.GetGenericIndexer<Vec3b>();
            int planeSize = Size * Size;

            for (int y = 0; y < Size; y++)
            {
                for (int x = 0; x < Size; x++)
                {
                    Vec3b pix = indexer[y, x];      // R,G,B (BGR2RGB 변환 후)
                    for (int c = 0; c < 3; c++)
                    {
                        float v = pix[c] / 255.0f;
                        v = (v - Mean[c]) / Std[c];
                        result[c * planeSize + y * Size + x] = v;
                    }
                }
            }
            return result;
        }

        // 클래스 정의 — 학습 시 인덱스와 1:1 일치해야 함
        // (0) background  — 투명 (그리지 않음)
        // (1) Pollution   — 노랑 (BGR: 0, 255, 255)
        // (2) Damaged     — 빨강 (BGR: 0, 0, 255)
        private const int NumClasses = 3;
        private static readonly Scalar[] ClassColors =
        {
            new Scalar(0, 0, 0),         // 0: background
            new Scalar(0, 255, 255),     // 1: Pollution (BGR 노랑)
            new Scalar(0, 0, 255),       // 2: Damaged   (BGR 빨강)
        };

        /// <summary>
        /// ORT 출력 (1*NumClasses*513*513 logits) → 원본 크기 결함 시각화 Mat
        /// 외곽선(contour) 방식 — 배터리 외관색에 무관하게 결함 영역이 또렷이 보임
        /// </summary>
        public static Mat BuildOverlay(float[] logits, Mat originalBgr, float fillAlpha = 0.25f, int contourThickness = 3)
        {
            int expected = NumClasses * Size * Size;
            if (logits.Length != expected)
                throw new InvalidOperationException(
                    $"logits 길이 {logits.Length} ≠ 기대값 {expected} (= {NumClasses}×{Size}×{Size}). " +
                    $"모델 출력 클래스 수가 {NumClasses} 가 아닙니다. NumClasses 상수를 모델에 맞춰 수정하세요.");

            int planeSize = Size * Size;

            // 1. pixel-wise argmax → 513x513 uint8 클래스 마스크
            using var mask513 = new Mat(Size, Size, MatType.CV_8UC1);
            var maskIndexer = mask513.GetGenericIndexer<byte>();

            for (int y = 0; y < Size; y++)
            {
                for (int x = 0; x < Size; x++)
                {
                    int bestC = 0;
                    float bestV = logits[y * Size + x];
                    for (int c = 1; c < NumClasses; c++)
                    {
                        float v = logits[c * planeSize + y * Size + x];
                        if (v > bestV) { bestV = v; bestC = c; }
                    }
                    maskIndexer[y, x] = (byte)bestC;
                }
            }

            // 2. 원본 해상도로 nearest-neighbor 리사이즈 (Linear 쓰면 클래스 인덱스 깨짐)
            using var maskFull = new Mat();
            Cv2.Resize(mask513, maskFull, originalBgr.Size(), 0, 0, InterpolationFlags.Nearest);

            // 3. 클래스별: 약한 fill (alpha 0.25) + 두꺼운 외곽선
            var overlay = originalBgr.Clone();
            int[] classPixelCount = new int[NumClasses];

            for (int c = 1; c < NumClasses; c++)
            {
                using var classMask = new Mat();
                Cv2.Compare(maskFull, (double)c, classMask, CmpTypes.EQ);
                classPixelCount[c] = Cv2.CountNonZero(classMask);
                if (classPixelCount[c] == 0) continue;

                // 3-a. 약한 fill — 영역 위치 인지용 (외관색 영향 최소)
                using var colorFill = new Mat(originalBgr.Size(), MatType.CV_8UC3, new Scalar(0, 0, 0));
                colorFill.SetTo(ClassColors[c], classMask);
                Cv2.AddWeighted(overlay, 1.0, colorFill, fillAlpha, 0, overlay);

                // 3-b. 두꺼운 외곽선 — 외관색 무관 항상 선명
                Cv2.FindContours(classMask, out OpenCvSharp.Point[][] contours, out _,
                    RetrievalModes.External, ContourApproximationModes.ApproxSimple);
                Cv2.DrawContours(overlay, contours, -1, ClassColors[c], thickness: contourThickness);
            }

            // 진단 로그
            int total = (int)(originalBgr.Total());
            Console.WriteLine($"[mask] Pollution(노랑): {classPixelCount[1]:N0} px ({100.0 * classPixelCount[1] / total:F2}%)");
            Console.WriteLine($"[mask] Damaged(빨강) : {classPixelCount[2]:N0} px ({100.0 * classPixelCount[2] / total:F2}%)");

            // 4. 범례 그리기
            DrawLegend(overlay);

            return overlay;
        }

        /// <summary>이미지 우측 하단에 클래스-색 범례(legend) 박스를 그림</summary>
        private static void DrawLegend(Mat img)
        {
            const int boxW = 200;
            const int boxH = 80;
            const int margin = 20;
            int x = img.Width - boxW - margin;
            int y = img.Height - boxH - margin;

            // 반투명 검정 배경 (ROI 에 알파 합성)
            using (var roi = new Mat(img, new Rect(x, y, boxW, boxH)))
            using (var dark = new Mat(roi.Size(), MatType.CV_8UC3, new Scalar(0, 0, 0)))
            {
                Cv2.AddWeighted(roi, 0.35, dark, 0.65, 0, roi);
            }

            // 외곽선
            Cv2.Rectangle(img, new Rect(x, y, boxW, boxH), Scalar.White, thickness: 1);

            // 클래스별 색 사각형 + 텍스트
            const int sq = 18;
            for (int c = 1; c < NumClasses; c++)
            {
                int rowY = y + 12 + (c - 1) * 33;
                Cv2.Rectangle(img, new Rect(x + 10, rowY, sq, sq), ClassColors[c], thickness: -1);
                Cv2.Rectangle(img, new Rect(x + 10, rowY, sq, sq), Scalar.White, thickness: 1);

                string label = c == 1 ? "Pollution" : "Damaged";
                Cv2.PutText(img, label,
                    new OpenCvSharp.Point(x + 38, rowY + 15),
                    HersheyFonts.HersheySimplex, 0.6, Scalar.White, 1, LineTypes.AntiAlias);
            }
        }
    }
}
