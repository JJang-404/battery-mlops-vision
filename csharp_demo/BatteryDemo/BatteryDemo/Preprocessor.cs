using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCvSharp;

namespace BatteryDemo
{
    public static class Preprocessor
    {
        // argmax + 원본 해상도 복원 + 컬러 오버레이
        private const int Size = 513;
        private const int NumClasses = 3;

        // 클래스별 오버레이 색상 (BGR — OpenCV 기본)
        private static readonly Scalar[] ClassColors =
        {
        new(0, 0, 0),         // 0: background — 투명
        new(0, 255, 255),     // 1: Pollution  — 노랑
        new(0, 0, 255),       // 2: Damaged    — 빨강
    };

        /// <summary>ORT 출력 (1*3*513*513 logits) → 원본 크기 컬러 오버레이 Mat</summary>
        public static Mat BuildOverlay(float[] logits, Mat originalBgr, float alpha = 0.4f)
        {
            // 1. pixel-wise argmax → 513x513 uint8 마스크
            var mask513 = new Mat(Size, Size, MatType.CV_8UC1);
            var maskIndexer = mask513.GetGenericIndexer<byte>();
            int planeSize = Size * Size;

            for (int y = 0; y < Size; y++)
            {
                for (int x = 0; x < Size; x++)
                {
                    int bestC = 0;
                    float bestV = logits[y * Size + x];     // c=0
                    for (int c = 1; c < NumClasses; c++)
                    {
                        float v = logits[c * planeSize + y * Size + x];
                        if (v > bestV) { bestV = v; bestC = c; }
                    }
                    maskIndexer[y, x] = (byte)bestC;
                }
            }

            // 2. 원본 해상도로 nearest-neighbor 리사이즈
            using var maskFull = new Mat();
            Cv2.Resize(mask513, maskFull, originalBgr.Size(), 0, 0, InterpolationFlags.Nearest);

            // 3. 컬러 오버레이 합성
            var overlay = originalBgr.Clone();
            var color = new Mat(originalBgr.Size(), MatType.CV_8UC3, new Scalar(0, 0, 0));

            for (int c = 1; c < NumClasses; c++)     // bg 제외
            {
                using var classMask = new Mat();
                Cv2.Compare(maskFull, c, classMask, CmpTypes.EQ);   // 해당 클래스만 255
                color.SetTo(ClassColors[c], classMask);
            }

            Cv2.AddWeighted(overlay, 1.0, color, alpha, 0, overlay);
            return overlay;
        }

        public static float[] LoadAndPreprocess(string imgPath)
        {
            // 1. 이미지 로드
            using var mat = Cv2.ImRead(imgPath);
            if (mat.Empty()) throw new System.Exception($"이미지를 불러올 수 없습니다: {imgPath}");

            // 2. 모델 입력 크기(513x513)로 리사이즈
            using var resized = new Mat();
            Cv2.Resize(mat, resized, new Size(513, 513));

            // 3. 정규화 및 float 배열 변환 (HWC -> CHW 방식)
            // 배터리 검사 모델 사양에 맞춰 0~1 사이로 정규화
            float[] floatBuffer = new float[1 * 3 * 513 * 513];
            int index = 0;

            // 채널별로 분리하여 저장 (DeepLab, YOLO 등에서 흔히 사용하는 방식)
            for (int c = 0; c < 3; c++) // B, G, R
            {
                for (int h = 0; h < 513; h++)
                {
                    for (int w = 0; w < 513; w++)
                    {
                        var pixel = resized.At<Vec3b>(h, w);
                        floatBuffer[index++] = pixel[c] / 255.0f;
                    }
                }
            }
            return floatBuffer;
        }

    }
}
