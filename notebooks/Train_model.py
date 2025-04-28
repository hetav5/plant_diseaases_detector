# 0. Imports
import os
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# 1. Set paths
train_dir = r"f:\codes\plant_diseases_detector\data\train"
val_dir = r"f:\codes\plant_diseases_detector\data\val"

# 2. Create datasets (added shuffle and sparse label mode)
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    seed=123,
    image_size=(224, 224),
    batch_size=32,
    shuffle=True,
    label_mode='int'
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    seed=123,
    image_size=(224, 224),
    batch_size=32,
    shuffle=True,
    label_mode='int'
)

# 3. Read and save class names
class_names = sorted(os.listdir(train_dir))
print("✅ Class Names:", class_names)

label_map = {name: idx for idx, name in enumerate(class_names)}
os.makedirs("models", exist_ok=True)
with open("models/class_names.json", "w") as f:
    json.dump(label_map, f)

# 4. Compute class weights (to handle imbalance)
labels = []
for class_idx, class_dir in enumerate(class_names):
    class_path = os.path.join(train_dir, class_dir)
    labels += [class_idx] * len(os.listdir(class_path))

class_weights_array = compute_class_weight(class_weight='balanced',
                                            classes=np.unique(labels),
                                            y=labels)

class_weight = dict(enumerate(class_weights_array))
print("✅ Computed Class Weights:", class_weight)

# 5. Create base model
base_model = keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 6. Train top classifier
initial_epochs = 25
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=initial_epochs,
    class_weight=class_weight  # <-- important
)

# 7. Fine-tuning
base_model.trainable = True
fine_tune_at = 100

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Recompile
model.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

fine_tune_epochs = 10
total_epochs = initial_epochs + fine_tune_epochs

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1],
    class_weight=class_weight  # <-- still important
)

# 8. Save model
model.save("models/mobilenet_model.h5")
print("✅ Model saved at models/mobilenet_model.h5")

# 9. Plot graphs
# Combine histories
acc = history.history['accuracy'] + history_fine.history['accuracy']
val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']

loss = history.history['loss'] + history_fine.history['loss']
val_loss = history.history['val_loss'] + history_fine.history['val_loss']

# Accuracy plot
plt.figure(figsize=(8, 6))
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('models/accuracy_plot.png')
plt.show()

# Loss plot
plt.figure(figsize=(8, 6))
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('models/loss_plot.png')
plt.show()
