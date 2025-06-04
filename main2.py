import os
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Path to dataset
dataset_path = r"C:\Users\Admin\Desktop\Python Project\Agri-AID\Dataset\train"

# Get class names from folder names
class_names = sorted([
    name for name in os.listdir(dataset_path)
    if os.path.isdir(os.path.join(dataset_path, name))
])
num_classes = len(class_names)
print(f"Total classes (plant+disease): {num_classes}")

# Create label map
class_to_idx = {name: idx for idx, name in enumerate(class_names)}
idx_to_class = {idx: name for name, idx in class_to_idx.items()}

# Load image paths and labels
image_paths = []
labels = []

for class_name in class_names:
    folder = os.path.join(dataset_path, class_name)
    for file in os.listdir(folder):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_paths.append(os.path.join(folder, file))
            labels.append(class_to_idx[class_name])

print(f"Total images: {len(image_paths)}")

# Preprocessing function
def load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = img / 255.0
    return img, label

# Create tf.data.Dataset
ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
ds = ds.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)

# Define model
base_model = tf.keras.applications.MobileNetV2(include_top=False, input_shape=(224,224,3), weights='imagenet')
base_model.trainable = False

inputs = layers.Input(shape=(224,224,3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = models.Model(inputs, outputs)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Train
model.fit(ds, epochs=10)

# Save model and class map
model.save(r"C:\Users\Admin\Desktop\Python Project\Agri-AID\plant_disease_model_V2.keras")

import json
with open("class_to_idx.json", "w") as f:
    json.dump(class_to_idx, f)
