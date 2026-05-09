using System;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;

namespace Battery_Project.Tools{
    class LabelMigrator{
        static void Main(string[] args)
        {
            // 경로 설정
            string imgDir = @"D:\02.study\part4_wj\Battery\Battery_Project\battery_image";
            string sourceLabelDir = @"D:\02.study\part4_wj\Battery\103.배터리 불량 이미지 데이터\3.개방데이터\1.데이터\Training\01. 원천데이터\TL_Exterior_Img_Datasets_label";
            string destLabelDir = @"D:\02.study\part4_wj\Battery\Battery_Project\battery_label";

            Console.WriteLine("[START] 라벨 데이터 매칭 및 복사를 시작합니다.");

            try{
                // 목적지 폴더가 없으면 생성
                if(!Directory.Exists(destLabelDir))
                {
                    Directory.CreateDirectory(destLabelDir);
                    Console.WriteLine($"폴더 생성: {destLabelDir}");
                }
                
                // 이미지 폴더에서 모든 png 파일 목록 가져오기
                string[] imgFiles = Directory.GetFiles(imgDir, "*.png");
                Console.WriteLine($"찾은 이미지 수: {imgFiles.Length}");

                // 이미지 카운터 중에서 성공횟수와 스킵횟수 갯수 확인 
                int successCount = 0;
                int skipCount = 0;

                foreach(string imgPath in imgFiles)
                {
                    string fileName = Path.GetFileNameWithoutExtension(imgPath);
                    string labelFileName = fileName + ".json";

                    string sourcePath = Path.Combine(sourceLabelDir, labelFileName);
                    string destPath = Path.Combine(destLabelDir, labelFileName);

                    if (File.Exists(sourcePath))
                    {
                        File.Copy(sourcePath, destPath, true);
                        successCount++;
                        Console.WriteLine($"복사 완료: {labelFileName}");
                    }
                    else
                    {
                        skipCount++;
                    }
                }

                Console.WriteLine("\n-----------------------------------------");
                Console.WriteLine($"성공적으로 복사 됨: {successCount}개");
                Console.WriteLine($"매칭되는 라벨 없음 :{skipCount}개");
                Console.WriteLine("-----------------------------------------");

            }
            catch(Exception ex){
                Console.WriteLine($"오류 발생: {ex.Message}");
            }
            
            Console.WriteLine("\n아무 키나 누르면 종료됩니다.");
        }
    }
}