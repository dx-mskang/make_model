import numpy as np
import onnxruntime as rt
import tf2onnx
import tensorflow as tf

# --- 1. 저장된 256x256 .keras 모델을 불러옵니다 ---
# 파일 이름을 새로 생성된 모델 파일 이름으로 변경
model_path = "unet_mobilenetv2_256.keras"
model = tf.keras.models.load_model(model_path)
print("✅ TensorFlow 모델을 성공적으로 불러왔습니다.")


# --- 2. 불러온 모델 객체를 사용해 ONNX로 변환합니다 ---

# 모델의 입력 사양을 256x256으로 직접 정의합니다.
input_signature = [tf.TensorSpec([1, 256, 256, 3], tf.float32, name="input")]

tf2onnx.convert.from_keras(
    model=model,
    input_signature=input_signature,  # 수정된 입력 사양 지정
    output_path="unet_mobilenetv2_256.onnx", # 출력 파일 이름도 변경
    opset=16
)
print("✅ 모델을 'unet_mobilenetv2_256.onnx' 파일로 변환했습니다.")


# --- 3. 변환된 ONNX 모델 테스트 ---
print("\n🔬 변환된 ONNX 모델을 테스트합니다...")

# ONNX Runtime 세션 시작
sess = rt.InferenceSession("unet_mobilenetv2_256.onnx")

# 입력 이름과 출력 이름 확인
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name
print(f"입력 레이어 이름: {input_name}")
print(f"출력 레이어 이름: {output_name}")

# 모델 입력 형식에 맞는 256x256 더미 데이터 생성
dummy_input = np.random.randn(1, 256, 256, 3).astype(np.float32)

# ONNX 모델로 추론 실행
result = sess.run([output_name], {input_name: dummy_input})[0]

# 결과 확인
print(f"\n추론 결과의 형태(Shape): {result.shape}")
# 최종 출력 형태가 (1, 256, 256, 3)이 됩니다.
print("테스트 완료! 이 형태가 (1, 256, 256, 3)으로 나오면 성공입니다.")
