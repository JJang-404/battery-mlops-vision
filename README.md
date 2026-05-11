# Battery MLOps Vision (배터리 불량 검출 및 추론 시스템 운영)

본 프로젝트는 AI 운영(MLOps) 관점에서 배터리 셀의 외관 불량을 실시간으로 탐지하고, 추론 엔진 최적화 및 운영 모니터링 체계를 구축하는 것을 목표로 합니다.

## 프로젝트 하이라이트 (EDA & Preprocess)
`notebooks/` 자료에 기반한 핵심 성과입니다.

- **데이터 무결성 검증**: 이미지 249장과 JSON 라벨 간의 100% 매칭 및 해상도 일치 확인.
- **클래스 불균형 분석**: Pollution(80%) vs Damaged(20%)의 분포를 파악하여 학습 시 클래스 가중치(Class Weight) 적용 전략 수립.
- **ROI 최적화**: Damaged 결함이 주로 하단에 편중되는 경향을 확인하여 룰 기반 ROI 가중치 프로토타입 설계.
- **전처리 파이프라인**: 
- 1920x1080 원본 영상을 DeepLabV3+ 최적 입력 규격인 640x640으로 리사이즈.
- 소수 클래스 보존을 위한 'Damaged 우선 덮어쓰기' 마스크 생성 로직 구현.

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