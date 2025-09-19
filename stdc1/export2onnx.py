import numpy as np
import onnxruntime as rt
import tf2onnx
import tensorflow as tf

# --- 1. 저장된 STDC-Seg .keras 모델 불러오기 ---
# model.py에서 저장한 파일 이름과 정확히 일치해야 합니다.
MODEL_PATH = "stdc_seg_1024x1920.keras"
# 변환 후 저장될 ONNX 파일 이름
ONNX_PATH = "stdc_seg_1024x1920.onnx"
# 모델의 입력 형태 (높이, 너비, 채널)
INPUT_SHAPE = (1024, 1920, 3)
# 모델의 클래스 수
NUM_CLASSES = 19


try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ TensorFlow 모델 ('{MODEL_PATH}')을 성공적으로 불러왔습니다.")
except Exception as e:
    print(f"❌ 모델 로딩 중 오류 발생: {e}")
    print(f"'{MODEL_PATH}' 파일이 현재 폴더에 있는지 확인하세요.")
    exit() # 모델 로딩 실패 시 스크립트 종료


# --- 2. 모델을 ONNX로 변환 ---

# 모델의 입력 사양을 가변 배치(None)로 정의합니다.
input_signature = [tf.TensorSpec([None, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]], tf.float32, name="input")]

try:
    tf2onnx.convert.from_keras(
        model=model,
        input_signature=input_signature,
        output_path=ONNX_PATH,
        opset=16
    )
    print(f"✅ 모델을 '{ONNX_PATH}' 파일로 변환했습니다.")
except Exception as e:
    print(f"❌ ONNX 변환 중 오류 발생: {e}")
    exit()


# --- 3. 변환된 ONNX 모델 테스트 ---
print(f"\n🔬 변환된 ONNX 모델 ('{ONNX_PATH}')을 테스트합니다...")

try:
    # ONNX Runtime 세션 시작
    sess = rt.InferenceSession(ONNX_PATH)

    # 입력 이름과 출력 이름 확인
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    print(f"입력 레이어 이름: {input_name}")
    print(f"출력 레이어 이름: {output_name}")

    # 모델 입력 형식에 맞는 더미 데이터 생성 (배치 크기=1)
    dummy_input = np.random.randn(1, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]).astype(np.float32)

    # ONNX 모델로 추론 실행
    result = sess.run([output_name], {input_name: dummy_input})[0]

    # 결과 확인
    expected_shape = (1, INPUT_SHAPE[0], INPUT_SHAPE[1], NUM_CLASSES)
    print(f"\n추론 결과의 형태(Shape): {result.shape}")
    print(f"기대했던 형태(Shape): {expected_shape}")

    if result.shape == expected_shape:
        print("\n테스트 완료! ✅ 입출력 형태가 기대와 일치합니다.")
    else:
        print("\n테스트 실패! ❌ 입출력 형태가 기대와 다릅니다.")

except Exception as e:
    print(f"❌ ONNX 모델 테스트 중 오류 발생: {e}")
