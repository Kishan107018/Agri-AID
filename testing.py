import tensorflow as tf
import numpy as np
from PIL import Image
import json

# Load model
model = tf.keras.models.load_model(r"C:\Users\Admin\Desktop\Python Project\Agri-AID\plant_disease_model_V2.keras")

# Load label map
with open("class_to_idx.json") as f:
    class_to_idx = json.load(f)
idx_to_class = {v: k for k, v in class_to_idx.items()}

# Preprocess test image
def load_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

# Test image path
image_path = r"C:\Users\Admin\Desktop\Python Project\Agri-AID\Dataset\valid\Grape___Black_rot\02a1df6c-97ec-41d4-b00c-9510741a39dc___FAM_B.Rot 0552_flipLR.JPG"
img = load_image(image_path)

# Predict
pred = model.predict(img)
pred_idx = int(np.argmax(pred[0]))
pred_class = idx_to_class[pred_idx]

# Split into plant & disease
plant, disease = pred_class.split("___")
print("Predicted Plant:", plant)
print("Predicted Disease:", disease)
