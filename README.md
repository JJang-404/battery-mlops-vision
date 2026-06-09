
---
<div align="center">

# Battery MLOps Vision

### DeepLabV3+ 기반 리튬이온 배터리 외관 결함 검출 파이프라인

<br/>

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLabV3%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-005CED?style=for-the-badge&logo=onnx&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-EP-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![.NET](https://img.shields.io/badge/.NET-8.0-512BD4?style=for-the-badge&logo=dotnet&logoColor=white)
![C#](https://img.shields.io/badge/C%23-WPF_Demo-239120?style=for-the-badge&logo=csharp&logoColor=white)
![Colab](https://img.shields.io/badge/Colab_Pro-T4-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)
![Jupyter](https://img.shields.io/badge/Notebook-EDA%E2%86%92Verify-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

<br/>

```
  ┌────────────────────────────────────────────────────────────────────────┐
  │                                                                        │
   │    EDA   →    학습    →   ONNX 변환   →   회귀 검증   →   C# 데모  │ 
  │                                                                        │
  │  [249장]   [DeepLabV3+]   [CUDA EP]      [v1 ↔ v2]     [.NET 8 WPF]   │
  │                                                                        │
  └────────────────────────────────────────────────────────────────────────┘
```

</div>

---

## 한눈에 보기

> **Battery MLOps Vision**은 AI허브 리튬이온 배터리 데이터 249장으로
> **학습 → 검증 → 재학습 → ONNX 변환 → C# 비전 데모 배포**까지
> 운영 사이클 전 구간을 **구현·정량 검증**한 개인 포트폴리오 프로젝트입니다.

```text
  ╔══════════════════════╗          ╔══════════════════════╗          ╔══════════════════════╗
  ║                      ║          ║                      ║          ║                      ║
  ║       Notebook       ║  ──▶    ║        PyTorch       ║   ──▶   ║       C# Demo        ║
  ║   EDA · 학습 · 검증  ║          ║    → ONNX (CUDA)    ║          ║     비전 검사 UI     ║
  ║                      ║          ║                      ║          ║                      ║ 
  ╚══════════════════════╝          ╚══════════════════════╝          ╚══════════════════════╝
        notebooks/                         onnx_export                       csharp_demo/
```

---

## 정량 성과 (운영점: v2 + min_conf 0.70)

<div align="center">

| 지표 | 결과 | 비고 |
|:---|:---:|:---|
| **정상품 과검**              | 24장 중 **1장** (4.2%) | 미세 오염 경계 1건 |
| **결함 검출 (Instance Recall)** | 14장 중 **13장** (92.9%) | 미검 1건은 GT 472px 초미세 |
| **평균 오검 픽셀**            | **10 px / 장**         | 후처리(min_conf) 적용 후 |
| **추론 Latency**            | **102 ms** (CPU 532ms 대비 5.2× 가속) | ONNX Runtime CUDA EP |

</div>

> *왜 재학습했나* — AI허브 공개 가중치는 4-class 체계라 본 과제(3-class)에 그대로 쓰면
> 클래스 인덱스가 어긋나 정상품을 전부 결함으로 출력합니다(24/24). 이를 출발점으로
> 헤드 교체 + 3차 보정 재학습을 진행해 위 운영점에 도달했습니다.

---

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

  ---

## 모델 비교 실험 — Semantic vs Instance (`notebooks/10_yolov8_seg.ipynb`)

DeepLabV3+(Semantic) 단일 아키텍처의 타당성을 검증하기 위해 **YOLOv8-seg(Instance)** 를
동일 데이터·동일 분할로 직접 학습·비교했습니다.

<div align="center">

| 모델 | 유형 | 학습 파라미터 | Mask mAP50 | 결론 |
|:---|:---|:---:|:---:|:---|
| **DeepLabV3+** | Semantic | 1.3M (헤드) | 운영 채택 | 작은·고밀도 결함에 적합 |
| YOLOv8n-seg | Instance | 3.4M (전체) | 0.106 | floor — 부적합 |

</div>

- **직접 튜닝**: `imgsz` 640→1280, augmentation 완화(mosaic·mixup·scale↓), `cls` weight 0.3→0.5.
- 그럼에도 Mask mAP50이 **0.088 → 0.106 에 정체.** n-seg·s-seg 동일 실패 → **모델 크기·해상도 병목이
아님**을 확인.
- **원인**: 셀당 결함 폴리곤이 5~15개로 작고 밀집 → 작은 mask에서 IoU≥0.5 확보 불가, instance 탐지 구조의
한계.
- **결론**: 작은·고밀도 결함에는 **Semantic(DeepLabV3+)이 유리**함을 정량 확인.
  *"직접 학습 + 실패 모드 진단"* 으로 아키텍처 선택 근거를 데이터로 입증했습니다.

## 한계와 후속 개선 방향

| 한계 | 원인 | 후속 방향 |
| :--- | :--- | :--- |
| Damaged 클래스 노이즈 floor | 픽셀 비율 0.12% — 클래스 가중치 상한에 걸려 학습 신호 부족 | 데이터 추가 수집 후 3-class 재학습 |
| Pixel Recall 22.9% | 결함 영역의 부분만 마킹 | 2-class(bg/defect) 재학습 — 별도 노트북에서 학습 완료, 검증 적용은 후속 작업 |
| 캡 영역 과민 반응 | ColorJitter 증강이 캡 림 음영에 민감하게 학습 | ROI mask 도입 또는 증강 강도 감소 |
| 모델 아키텍처 단일 | DeepLabV3+ 외 비교 필요 | *YOLOv8-seg 비교 완료(위 "모델 비교 실험") — instance seg 한계 확인. U-Net / SegFormer는 후속** |


## 디렉터리

```
notebooks/        # EDA → 학습 → 검증 (01~09)
notebooks_docs/   # 노트북 → 마크다운 변환본
csharp_demo/      # .NET 8 + ONNX Runtime 추론 데모
models/           # battery_deeplab_v2.onnx 등
```
---
