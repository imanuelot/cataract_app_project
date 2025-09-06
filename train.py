import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from collections import Counter

# ==========================
# Konfigurasi
# ==========================
img_size = (224, 224)
batch_size = 32
epochs = 20
train_dir = "dataset/train"
test_dir = "dataset/test"

# ==========================
# Data Augmentation
# ==========================
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v3.preprocess_input,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v3.preprocess_input
)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="sparse",
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="sparse",
    shuffle=False
)

# ==========================
# Debug Info Dataset
# ==========================
print("\n🔎 Class indices:", train_gen.class_indices)
print("🔎 Distribusi train:", Counter(train_gen.classes))
print("🔎 Distribusi val:", Counter(val_gen.classes))

# ==========================
# Simpan mapping label ke file
# ==========================
os.makedirs("models", exist_ok=True)
idx_to_class = {int(v): k for k, v in train_gen.class_indices.items()}
with open("models/labels.json", "w") as f:
    json.dump(idx_to_class, f)
print("🔎 Label mapping disimpan ke models/labels.json:", idx_to_class)

# ==========================
# Hitung class_weight
# ==========================
counter = Counter(train_gen.classes)
majority = max(counter.values())
class_weight = {cls: float(majority/count) for cls, count in counter.items()}
print("🔎 Class weight:", class_weight)

# ==========================
# Build Model
# ==========================
def build_model(input_shape=(224, 224, 3), num_classes=2):
    inputs = tf.keras.Input(shape=input_shape)
    base_model = MobileNetV3Small(
        input_shape=input_shape, include_top=False, weights="imagenet")
    
    # Fine-tuning sebagian
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)

model = build_model()

# ==========================
# Compile
# ==========================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ==========================
# Callbacks
# ==========================
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

checkpoint_best = tf.keras.callbacks.ModelCheckpoint(
    filepath="models/cataract_model_best.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max"
)

# ==========================
# Training
# ==========================
history = model.fit(
    train_gen,
    epochs=epochs,
    validation_data=val_gen,
    class_weight=class_weight,
    callbacks=[early_stop, checkpoint_best]
)

# ==========================
# Save final model
# ==========================
model.save("models/cataract_model_final.keras")
print("\n✅ Training selesai")
print("📂 Model terbaik ada di: models/cataract_model_best.keras")
print("📂 Model terakhir ada di: models/cataract_model_final.keras")
