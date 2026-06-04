import numpy as np
import json

from PIL import Image
from tensorflow.keras.models import load_model

# =====================================
# LOAD MODEL
# =====================================

model = load_model("models/disease_model.h5")

# =====================================
# LOAD CLASS LABELS
# =====================================

with open("models/disease_classes.json") as f:
    class_indices = json.load(f)

class_names = list(class_indices.keys())

# =====================================
# LOAD IMAGE
# =====================================

image_path = input("Enter image path: ")

image = Image.open(image_path).convert("RGB")

image = image.resize((224, 224))

img = np.array(image) / 255.0

img = np.expand_dims(img, axis=0)

# =====================================
# PREDICT
# =====================================

prediction = model.predict(img)

predicted_index = np.argmax(prediction)

confidence = np.max(prediction)

print("\nPrediction:")
print(class_names[predicted_index])

print(f"\nConfidence: {confidence:.2f}")