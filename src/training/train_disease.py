import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import json
import os

# ==========================================
# SETTINGS
# ==========================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

DATASET_PATH = "data/raw/disease/PlantVillage"

# ==========================================
# DATA PREPROCESSING
# ==========================================

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# ==========================================
# TRANSFER LEARNING MODEL
# ==========================================

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)


base_model.trainable = True

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(train_data.num_classes, activation="softmax")
])

# ==========================================
# COMPILE MODEL
# ==========================================

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==========================================
# TRAIN MODEL
# ==========================================

print("\nTraining CNN model...\n")

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# ==========================================
# SAVE MODEL
# ==========================================

os.makedirs("models", exist_ok=True)

model.save("models/disease_model.h5")

print("\nDisease model saved!")

# ==========================================
# SAVE CLASS LABELS
# ==========================================

with open("models/disease_classes.json", "w") as f:
    json.dump(train_data.class_indices, f)

print("Class labels saved!")

# ==========================================
# PLOT ACCURACY
# ==========================================

os.makedirs("outputs", exist_ok=True)

plt.figure(figsize=(8, 5))

plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title("CNN Disease Classification Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.savefig("outputs/cnn_accuracy.png")

plt.show()

# ==========================================
# PLOT LOSS
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("CNN Disease Classification Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.savefig("outputs/cnn_loss.png")

plt.show()

print("\nTraining completed successfully!")