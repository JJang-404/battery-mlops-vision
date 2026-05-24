# Battery MLOps Vision — DeepLabV3+ 기반 배터리 외관 결함 검출 파이프라인

본 프로젝트는 **AI 운영 및 검증/테스트 관점**에서 리튬이온 원통형 배터리의 외관 결함을 검출하는 DeepLabV3+ 세그멘테이션 파이프라인을 구축하고, **학습 → 검증 → 재학습 → ONNX 변환 → C# 데모 배포**까지 운영 사이클 전 구간을 직접 구현·검증한 개인 프로젝트입니다.

## 프로젝트 개요

| 항목 | 내용 |
| :--- | :--- |
| 데이터 | AI허브 「리튬이온 배터리 불량 이미지 (2023)」 249장 (보안상 현장 데이터 접근 불가) |
| 클래스 | 3-class — background / Pollution(80%) / Damaged(20%) |
| 모델 | DeepLabV3+ (DRN-D-54 backbone) — AI허브 공개 사전학습 가중치 활용 |
| 배포 | ONNX Runtime CUDA EP + .NET 8 C# 데모 |

## 핵심 결과 (정량 검증)

| 지표 | AI허브 raw (v1) | fine-tune v2 + 후처리 (운영 채택) | 개선 |
| :--- | :--- | :--- | :--- |
| 정상품 과검율 | **100%** (24/24) | **4.2%** (1/24) | 약 24배 |
| Instance Recall | 92.9% | 92.9% | 동일 |
| 평균 오검 픽셀 | 6,831 | 10 | 약 680배 |
| 추론 latency (CUDA EP) | 532ms | **102ms** | **5.2× 가속** |

## 운영 사이클

### 1. EDA & 전처리 (`notebooks/01_eda.ipynb`, `02_preprocess.ipynb`)
- 데이터 무결성 100% 검증 (249장, JSON 폴리곤, 해상도 일치)
- 클래스 불균형 진단 (Pollution 80% / Damaged 20%, **픽셀 비율 0.45% / 0.12%**)
- JSON 폴리곤 → uint8 마스크 변환, train/val/test 분할

### 2. 사전 추론 검증 (`notebooks/03_inference_aihub.ipynb`)
- AI허브 공개 가중치(4-class)와 본 과제 라벨 체계(3-class) 불일치 진단
- raw 출력 → 정상품 100% 과검, **재학습 + 후처리 둘 다 필요** 결정

### 3. 1차 학습 (`notebooks/07_re_finetune_colab.ipynb`)
- 마지막 1×1 conv 헤드만 3-class로 재구성, Backbone + ASPP freeze
- Class-Weighted CrossEntropy 3차 보정 사이클
  - 1차: 30배 weight → 과검 100% (폐기)
  - 2차: 5배 weight + Dice → Instance Recall 0.116, 결함 88% 누락 (폐기)
  - **3차: 20배 상한 CE 단독 → 운영 채택 후보 (v2)**
- 학습 파라미터 1.3M / 전체 40.7M (3.2%) — 174장 소규모 데이터에 적합한 capacity 제한

### 4. ONNX 변환 (`notebooks/08_onnx_export.ipynb`)
- PyTorch → ONNX 수치 정합성 검증 (max diff 1.35e-05)
- CUDA EP 도입으로 latency **532ms → 102ms (5.2× 가속)**

### 5. 모델 예측 검증 (`notebooks/06_verification.ipynb`, `09_re_verification.ipynb`)
- v1·v2 **동일 후처리 조건 회귀 비교** (`postprocess_v2`, `min_conf` 0.70~0.97 스윕)
- 정상품 24장 / 결함 14장 전수 검증 — 정량 표 + 육안 확대 audit
- GT 라벨 오류·경계 케이스를 도메인 판정으로 분리 식별
- **운영점 확정: v2 + `min_conf=0.70` → 과검 4.2% / Instance Recall 92.9%**

### 6. C# 비전 데모 (`csharp_demo/`)
- .NET 8 + ONNX Runtime, CUDA EP + CPU 폴백
- 워밍업 후 latency 20회 평균 측정
- Before/After 시각화 + 픽셀 통계 로그

## 한계와 후속 개선 방향

| 한계 | 원인 | 후속 방향 |
| :--- | :--- | :--- |
| Damaged 클래스 노이즈 floor | 픽셀 비율 0.12% — 클래스 가중치 상한에 걸려 학습 신호 부족 | 데이터 추가 수집 후 3-class 재학습 |
| Pixel Recall 22.9% | 결함 영역의 부분만 마킹 | 2-class(bg/defect) 재학습 — 별도 노트북에서 학습 완료, 검증 적용은 후속 작업 |
| 캡 영역 과민 반응 | ColorJitter 증강이 캡 림 음영에 민감하게 학습 | ROI mask 도입 또는 증강 강도 감소 |
| 모델 아키텍처 단일 | DeepLabV3+ 외 비교 미실시 | **U-Net / SegFormer / YOLOv8-seg 비교 학습 (계획)** |

## Tech Stack

- **Python**: PyTorch, ONNX, OpenCV, scikit-image
- **C#**: .NET 8, ONNX Runtime (CUDA + CPU EP), WPF UI
- **Tooling**: Google Colab Pro (T4), Git, Jupyter

## 디렉터리

```
notebooks/        # EDA → 학습 → 검증 (01~09)
notebooks_docs/   # 노트북 → 마크다운 변환본
csharp_demo/      # .NET 8 + ONNX Runtime 추론 데모
models/           # battery_deeplab_v2.onnx 등
```
