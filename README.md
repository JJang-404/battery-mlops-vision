# Battery MLOps Vision (배터리 불량 검출 및 추론 시스템 운영)

본 프로젝트는 AI 운영(MLOps) 관점에서 배터리 셀의 외관 불량을 실시간으로 탐지하고, 추론 엔진 최적화 및 운영 모니터링 체계를 구축하는 것을 목표로 합니다.

 ## Key MLOps Focus
- **추론 최적화**: PyTorch 모델의 ONNX 변환 및 가속화 (Target: FPS 개선)
- **실시간 모니터링**: MLflow를 활용한 학습 메트릭 관리 및 추론 로그 분석
- **비전 검사 자동화**: C#(Matrox MIL) 룰 기반 검사와 DeepLabV3+ 모델의 Hybrid 연동
- **운영 안정성**: 장애 대응을 위한 성능 프로파일링 및 데이터 재학습 파이프라인 설계

## Tech Stack
- **Languages**: Python (EDA, DL), C# (Vision UI, Rule-base)
- **Deep Learning**: PyTorch, DeepLabV3+, YOLO, Albumentations
- **MLOps/Ops**: ONNX, MLflow, Docker
- **Vision**: Matrox MIL Lite