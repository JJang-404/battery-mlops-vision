# 05. ONNX 변환 + GPU 배포 최적화 회고

> 노트북: `notebooks/05_onnx_export.ipynb`
> 입력: `models/best_finetuned.pt` (DeepLabV3+ fine-tuned, val mIoU 0.4038)
> 산출물: `models/battery_deeplab_v1.onnx` (+ `.onnx.data` 가중치 본체)
> 환경: Windows 11, Python 3.13 (venv_battery), ORT 1.26, GTX 1660 SUPER

---

## 1. 개요

C# 비전 검사 데모(Microsoft.ML.OnnxRuntime)에서 사용할 추론 모델을 PyTorch → ONNX로 변환하고, **제조 실시간 요구사항(≤ 100 ms/image)** 을 만족시키기까지의 의사결정·트러블슈팅 기록입니다. ONNX 변환 자체는 1시간 만에 끝났지만, **배포 환경에서 GPU 추론을 실제로 작동시키는 데** 3단계의 Silent Fallback 현상에 대해 계층별(커널-런타임-OS) 원인을 분석하고 최적화된 해결 방안을 도출하였습니다.

### 1-1. 결과 요약

| 환경 | latency | FPS | 가속비 | 비고 |
|---|---|---|---|---|
| ORT CPU FP32 | 532.2 ms | 1.9 | 1.0× | 베이스라인 (실시간 부적합) |
| **ORT GPU FP32 (CUDA EP)** | **102.1 ms** | **9.8** | **5.2×** | **제조 검사 환경의 실시간 처리 요구사항에 근접한 수준의 latency를 확보하였습니다. 정확도 100%** |

PyTorch ↔ ONNX 수치 정합성: max abs diff `1.35e-05`, 실제 결함 이미지 3장 argmax 100% 일치.

---

## 2. 의사결정 — 왜 양자화보다 GPU를 먼저 시도했는가

532 ms → 100 ms 이하를 위한 선택지는 두 갈래:

| 옵션 | 채택 여부 | 근거 |
|---|---|---|
| **Dynamic Quantization** | ✗ | DeepLabV3+ op 분포가 **Conv 67 / MatMul 0**. ORT Dynamic은 MatMul·Gemm 위주 양자화 → 이 모델에 효과가 제한적일 것으로 판단 + defect recall 저하 가능성 |
| **Static Quantization (INT8)** | 보류 | Calibration dataset 구축 + 정확도 trade-off(특히 Damaged class recall) 검증 필요. CPU-only 엣지 시나리오용 옵션으로 유보 |
| **ORT CUDA EP** | ✓ 채택 | 1660 SUPER 보유, 정확도 손실 0, 환경만 갖추면 즉시 효과 |

**제조업 비전 검사의 핵심 비용 비대칭**: 제조업 비전 검사의 특성상 False Negative(결함 미검) 발생 비용이 매우 높음을 고려하여, 정확도 손실이 없는 GPU 가속을 우선순위로 설정하였습니다. 양자화(Quantization)는 향후 CPU 전용 환경 대응을 위한 단계적 옵션으로 분류하였습니다.

---

## 3. 시행착오 — 세 번의 silent fallback

ORT CUDA Execution Provider는 초기화 실패 시 **명시적인 예외 없이 CPU provider로 fallback** 되는 특성을 확인하였습니다.

### 3-1. Jupyter 커널 mismatch — `[Azure, CPU]` 만 뜨고 CUDA 없음

**증상**:

```python
ort.get_available_providers()
# → ['AzureExecutionProvider', 'CPUExecutionProvider']
```

`onnxruntime-gpu` 가 venv에 설치돼 있는데도 CUDA가 빠진 상태. 530 ms latency 변화 없음.

**원인**: Jupyter 커널이 `Battery_Venv`(venv_battery 인터프리터)가 아니라 다른 conda env로 연결돼 있었음.<br/>해당 환경 내 CPU-only 패키지와의 라이브러리 충돌 가능성을 확인하고, 전용 가상환경(venv)으로 커널을 재매핑하여 해결하였습니다.

**진단 방법**: venv의 Python을 직접 호출해 비교하였습니다.

```powershell
$py = "...\venv_battery\Scripts\python.exe"
& $py -c "import onnxruntime as ort; print(ort.get_available_providers())"
# → ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']  ← 정상
```

→ venv 직접 호출 결과가 다르면 커널 환경이 의심됩니다.

**해결**: Jupyter 커널을 `Battery_Venv` 로 변경 + Restart Kernel + 셀 0부터 재실행.

### 3-2. CUDA 12 / cuDNN 9 런타임 DLL 부재 — silent fallback #2

**증상**: provider 목록에 CUDA가 떴지만 latency는 여전히 530 ms. `sess.get_providers()` 로 확인하니 `['CPUExecutionProvider']` 만 반환.

```
ort.get_available_providers()   →  ['Tensorrt, CUDA, CPU']   ← "등록 가능" 목록
sess.get_providers()            →  ['CPU']                    ← "실제 사용 중" 목록
```

**결정적 원인**: `get_available_providers()` 는 실제 사용 여부와 다를 수 있음을 확인하였습니다.<br/>실제 런타임 세션(sess.get_providers)의 할당 상태를 검증하는 프로세스를 확립하였습니다.

**원인**: ORT의 CUDA provider DLL이 `cudnn64_9.dll` 의존인데, NVIDIA 드라이버(581.57)는 GPU 통신만 담당하고 **CUDA 런타임 라이브러리(cudart, cublas, cudnn)는 별도 설치가 필요하였습니다.**.

ORT의 진짜 에러 메시지 (강제로 추출):

```
[E:onnxruntime] Error loading "onnxruntime_providers_cuda.dll"
which depends on "cudnn64_9.dll" which is missing.
Failed to create CUDAExecutionProvider.
Require cuDNN 9.* and CUDA 12.*
```

**해결** — DeepLab 최소 3종셋 pip 설치:

```powershell
pip install nvidia-cudnn-cu12 nvidia-cuda-runtime-cu12 nvidia-cublas-cu12
```

> **주의 — `-cu12` 접미사의 의미**: "CUDA 12용 빌드"이지 라이브러리 자체 버전 12가 아니였습니다. cuFFT는 SONAME 11.x, cuRAND는 10.x. 초기에 5개 패키지를 `==12.*` 로 일괄 핀했다가 `No matching distribution` 에러를 만났습니다. DeepLab(Conv 위주)엔 cuFFT/cuRAND가 불필요해 3종이면 충분하였습니다.

### 3-3. DLL 전이 의존성 — `os.add_dll_directory()` 만으론 부족 (silent fallback #3)

**증상**: 패키지 설치 후에도 같은 에러. DLL은 venv에 분명히 있음:

```
venv_battery\Lib\site-packages\nvidia\cudnn\bin\cudnn64_9.dll          ✓
venv_battery\Lib\site-packages\nvidia\cuda_runtime\bin\cudart64_12.dll ✓
venv_battery\Lib\site-packages\nvidia\cublas\bin\cublas64_12.dll       ✓
```

`os.add_dll_directory()` 로 세 경로를 등록해도 여전히 `Error 126: 모듈을 찾을 수 없음`. ORT는 다시 CPU로 폴백.

**원인: OS(Windows) 및 런타임 버전별 DLL 참조 방식의 차이로 인한 로딩 이슈 분석**:

| 메커니즘 | 역할 | 우리 케이스 |
|---|---|---|
| `os.add_dll_directory()` | Python 3.8+ 가 "**Python이 직접 호출해 로드하는** DLL"의 탐색 경로에 추가 | onnxruntime_providers_cuda.dll 로드 자체엔 영향 |
| `os.environ['PATH']` prepend | Windows가 "**이미 로드된 DLL이 또 다른 DLL을 의존성으로 부를 때**" 탐색하는 경로 | cuDNN/cuBLAS 가 *전이 의존성*으로 로드될 때 핵심 |

ORT 1.18+ CUDA EP는 **`onnxruntime_providers_cuda.dll` 을 직접 로드한 다음, 그 DLL이 또 cuDNN을 부르는 구조**. 즉 후자에 해당하므로 PATH 등록이 필수.

**해결** — 두 방식 병용:

```python
import os
from pathlib import Path

NVIDIA_BIN = Path(r'D:\...\venv_battery\Lib\site-packages\nvidia')
for sub in ['cudnn', 'cuda_runtime', 'cublas']:
    d = str(NVIDIA_BIN / sub / 'bin')
    os.add_dll_directory(d)                                       # Python 3.8+ 공식
    os.environ['PATH'] = d + os.pathsep + os.environ['PATH']      # 전이 의존성용

import onnxruntime as ort   # ← 반드시 위 등록 후에 import
```

→ 실측 결과: `sess.get_providers()` → `['CUDAExecutionProvider', 'CPUExecutionProvider']`. latency 530 ms → **102 ms (5.2× 가속)**.

---

## 4. 핵심 정리

| 주의사항 | 올바른 진단 |
|---|---|
| `get_available_providers()` 에 CUDA가 있다 = GPU가 작동한다 |  "등록 가능" ≠ "실제 사용". `sess.get_providers()` 별도 검증 필요합니다.  |
| `onnxruntime-gpu` 만 설치 | ORT 1.20+ 는 cuDNN 9 / CUDA 12 런타임 DLL을 **별도 설치** 필요합니다. |
| NVIDIA 드라이버 = CUDA 포함 | 드라이버는 GPU 통신만. 런타임 라이브러리는 별도로 필요합니다. |
| CUDA EP 실패 시 ORT가 예외를 던진다 | ORT 1.20+ 는 fallback provider가 있으면 **조용히 CPU로 폴백** + 로그 한 줄 표시됩니다. |
| `-cu12` = 라이브러리 버전 12 | "CUDA 12용 빌드"라는 의미. cuFFT는 11.x, cuRAND는 10.x |
| `os.add_dll_directory()` 만으로 DLL 문제 해결 | 전이 의존성 해석은 PATH도 같이 봐야 합니다. |

> 현장 배포 시 발생할 수 있는 환경 변수(DLL 누락, PATH 설정 등)를 매뉴얼화하여, 향후 인프라 구축 및 유지보수 과정에서 발생할 수 있는 기술적 리스크를 최소화하겠습니다.

**제조업 도메인 관점**:

- **Damaged class recall이 모델 가치의 핵심 지표** — false negative(결함 놓침) 비용이 false positive 대비 압도적. 양자화/최적화 acceptance에선 mIoU 평균만 보면 안 됨.
- **하드웨어 특성 고려한 의사결정** — 1660 SUPER(Turing TU116)는 INT8 Tensor Core가 없어 GPU INT8 가속이 크지 않다고 판단하였습니다. 그래서 FP32/FP16이 적합한 선택으로 판단하였습니다.
