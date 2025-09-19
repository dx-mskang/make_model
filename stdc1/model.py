import tensorflow as tf
from tensorflow.keras import layers, Model

# --- 모델의 기본 설정 ---
# Cityscapes 데이터셋 기준 (19개 클래스 + 1 배경)
NUM_CLASSES = 19
# 요청하신 1024x1920 크기로 입력 형태를 변경합니다.
INPUT_SHAPE = (1024, 1920, 3)

# --- 기본 빌딩 블록: Conv-BN-ReLU ---
def ConvBNReLU(x, filters, kernel_size=3, stride=1):
    """컨볼루션 - 배치 정규화 - ReLU 활성화 함수를 묶은 블록"""
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

# --- STDC 모델의 핵심: STDCModule ---
def STDCModule(x, filters, blocks, stride):
    """
    STDC 모듈 구현.
    첫 번째 블록을 제외한 나머지 블록은 채널 수가 절반씩 줄어듭니다.
    """
    out_layers = [ConvBNReLU(x, filters, stride=stride)]

    for i in range(1, blocks):
        inp = out_layers[i-1]
        f = filters // (2 ** i)
        if f > 0:
             out_layers.append(ConvBNReLU(inp, f, stride=1))

    if len(out_layers) > 1:
        x = layers.Concatenate()(out_layers)
    else:
        x = out_layers[0]
        
    return x

# --- STDC-Seg 모델 전체 정의 ---
def STDC_Seg(input_shape, num_classes):
    """STDC Segmentation 모델을 정의하는 함수"""
    
    inputs = layers.Input(shape=input_shape)

    # Backbone
    x = ConvBNReLU(inputs, 32, stride=2)
    x = ConvBNReLU(x, 64, stride=2)
    x = STDCModule(x, 256, blocks=4, stride=2)
    x = STDCModule(x, 512, blocks=4, stride=2)
    x = STDCModule(x, 1024, blocks=4, stride=1)
    
    # Segmentation Head
    x = layers.Conv2D(num_classes, 1, strides=1, padding='same')(x)
    
    #output_shape = tf.shape(inputs)[1:3]
    #outputs = tf.image.resize(x, size=output_shape, method='bilinear')
    outputs = layers.UpSampling2D(size=16, interpolation='bilinear')(x)

    model = Model(inputs=inputs, outputs=outputs, name='STDC-Seg')
    return model

# --- 스크립트 실행 시 모델 생성 및 요약 출력 ---
if __name__ == "__main__":
    model = STDC_Seg(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)
    
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    print(f"🎉 STDC-Seg 모델 (입력: {INPUT_SHAPE}, 클래스: {NUM_CLASSES})이 성공적으로 생성되었습니다!")
    model.summary()

    # 생성된 모델을 .keras 파일로 저장
    model.save(f"stdc_seg_{INPUT_SHAPE[0]}x{INPUT_SHAPE[1]}.keras")
    print(f"\n✅ 모델을 'stdc_seg_{INPUT_SHAPE[0]}x{INPUT_SHAPE[1]}.keras' 파일로 저장했습니다.")
