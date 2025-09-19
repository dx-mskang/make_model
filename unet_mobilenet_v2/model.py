import tensorflow as tf
import tensorflow_datasets as tfds

# --- 데이터셋 정보 로드 ---
#dataset, info = tfds.load('oxford_iiit_pet:4.0.0', with_info=True)
#OUTPUT_CHANNELS = info.features['segmentation_mask'].shape[-1]
OUTPUT_CHANNELS = 3

# --- 1. 인코더(Encoder) 만들기: 입력 크기 변경 ---
# MobileNetV2 모델의 input_shape를 [256, 256, 3]으로 변경
base_model = tf.keras.applications.MobileNetV2(input_shape=[256, 256, 3], include_top=False)

# U-Net의 Skip Connection에 사용할 중간 레이어들의 출력을 지정합니다.
layer_names = [
    'block_1_expand_relu',   # 128x128
    'block_3_expand_relu',   # 64x64
    'block_6_expand_relu',   # 32x32
    'block_13_expand_relu',  # 16x16
    'block_16_project',      # 8x8
]
base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

# 인코더 모델을 정의합니다.
down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)
down_stack.trainable = False


# --- 2. 디코더(Decoder) 만들기 ---
def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)
  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))
  result.add(tf.keras.layers.BatchNormalization())
  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))
  result.add(tf.keras.layers.ReLU())
  return result

up_stack = [
    upsample(512, 3),  # 8x8 -> 16x16
    upsample(256, 3),  # 16x16 -> 32x32
    upsample(128, 3),  # 32x32 -> 64x64
    upsample(64, 3),   # 64x64 -> 128x128
]


# --- 3. U-Net 모델 조립: 입력 크기 변경 ---
def unet_model(output_channels:int):
  # U-Net 모델의 Input shape를 [256, 256, 3]으로 설정
  inputs = tf.keras.layers.Input(shape=[256, 256, 3])

  # 인코더(Down-sampling)
  skips = down_stack(inputs)
  x = skips[-1]
  skips = reversed(skips[:-1])

  # 디코더(Up-sampling)와 Skip Connection
  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate()
    x = concat([x, skip])

  # 최종 출력 레이어 (128x128 -> 256x256)
  last = tf.keras.layers.Conv2DTranspose(
      filters=output_channels,
      kernel_size=3,
      strides=2,
      padding='same')
  x = last(x)
  return tf.keras.Model(inputs=inputs, outputs=x)
#  last = tf.keras.layers.Conv2DTranspose(
#      filters=output_channels, kernel_size=3, strides=2,
#      padding='same')
#
#  x = last(x)
#
#  return tf.keras.Model(inputs=inputs, outputs=x)


# --- 최종 모델 생성 및 요약 출력 ---
model = unet_model(OUTPUT_CHANNELS)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print("🎉 U-Net 모델이 성공적으로 생성되었습니다!")
model.summary()

# 생성된 모델을 .keras 형식으로 저장
model.save("unet_mobilenetv2_256.keras")
print("\n✅ 모델을 'unet_mobilenetv2_256.keras' 파일에 저장했습니다.")
