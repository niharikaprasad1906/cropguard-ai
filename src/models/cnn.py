
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

def build_model(num_classes):
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
    base.trainable = False

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(128, activation='relu')(x)
    output = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs=base.input, outputs=output)
