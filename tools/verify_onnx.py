import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path

ONNX_PATH = Path(r'D:\02.study\part4_wj\Battery\Battery_Project\models\battery_deeplab_v1.onnx')

def verify_onnx():
    print(f"--- ONNX 검증 시작: {ONNX_PATH.name} ---")
    
    # 1. 파일 존재 확인
    if not ONNX_PATH.exists():
        print(f"Error: {ONNX_PATH} 파일이 없습니다.")
        return

    # 2. ONNX 로드 및 기본 검증
    try:
        model = onnx.load(str(ONNX_PATH))
        onnx.checker.check_model(model)
        print("1. ONNX Checker: 통과 (그래프 구조 정상)")
    except Exception as e:
        print(f"1. ONNX Checker: 실패\n{e}")
        return

    # 3. ONNX Runtime 세션 생성 (가중치 로드 확인)
    try:
        # 가중치 파일(.data)이 같은 위치에 있어야 함
        session = ort.InferenceSession(str(ONNX_PATH), providers=['CPUExecutionProvider'])
        print("2. ONNX Runtime 세션 생성: 성공 (가중치 로드 완료)")
    except Exception as e:
        print(f"2. ONNX Runtime 세션 생성: 실패\n{e}")
        return

    # 4. 입력/출력 정보 출력
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    
    print("\n--- 모델 명세 ---")
    for i in inputs:
        print(f"Input:  name='{i.name}', shape={i.shape}, type={i.type}")
    for o in outputs:
        print(f"Output: name='{o.name}', shape={o.shape}, type={o.type}")

    # 5. 더미 데이터 추론 테스트
    try:
        input_shape = inputs[0].shape
        # dynamic_axes가 None이므로 정확한 shape 필요 (1, 3, 513, 513)
        dummy_data = np.random.randn(*input_shape).astype(np.float32)
        
        preds = session.run([o.name for o in outputs], {inputs[0].name: dummy_data})
        print(f"\n3. 추론 테스트: 성공")
        print(f"   출력 shape: {preds[0].shape}")
        
    except Exception as e:
        print(f"\n3. 추론 테스트: 실패\n{e}")

if __name__ == "__main__":
    verify_onnx()
