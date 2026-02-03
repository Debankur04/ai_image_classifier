# =========================
# Imports
# =========================
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.model_selection import train_test_split

# =========================
# Config
# =========================
DATASET_ROOT = "/kaggle/input/dalle-recognition-dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5
SEED = 42

import os
import numpy as np

DATASET_ROOT = "/kaggle/input/dalle-recognition-dataset"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")

paths = []
labels = []

# ---------- AI images ----------
fake_root = os.path.join(DATASET_ROOT, "fakeV2")
for root, _, files in os.walk(fake_root):
    for f in files:
        if f.lower().endswith(IMAGE_EXTS):
            paths.append(os.path.join(root, f))
            labels.append(1)

# ---------- Real images ----------
real_root = os.path.join(DATASET_ROOT, "real")
for root, _, files in os.walk(real_root):
    for f in files:
        if f.lower().endswith(IMAGE_EXTS):
            paths.append(os.path.join(root, f))
            labels.append(0)

paths = np.array(paths)
labels = np.array(labels)

print("Total images:", len(paths))
print("Class distribution:", np.bincount(labels))

assert np.sum(labels == 1) > 0
assert np.sum(labels == 0) > 0

IMG_SIZE = (224, 224)
MAX_FILE_SIZE = 15 * 1024 * 1024  # 15 MB

safe_paths = []
safe_labels = []

for p, y in zip(paths, labels):
    try:
        if os.path.getsize(p) <= MAX_FILE_SIZE:
            safe_paths.append(p)
            safe_labels.append(y)
    except:
        pass

paths = np.array(safe_paths)
labels = np.array(safe_labels)

print("After filtering:")
print("Total images:", len(paths))
print("Class distribution:", np.bincount(labels))

assert len(paths) == len(labels)
assert np.sum(labels == 0) > 0
assert np.sum(labels == 1) > 0

from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(
    paths, labels, test_size=0.2, stratify=labels, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

print("Train:", np.bincount(y_train))
print("Val  :", np.bincount(y_val))
print("Test :", np.bincount(y_test))

IMG_SIZE = (224, 224)

def load_image(path, label):
    image = tf.io.read_file(path)

    # Robust JPEG decode
    image = tf.image.decode_jpeg(
        image,
        channels=3,
        try_recover_truncated=True
    )

    # Resize EARLY to avoid huge memory usage
    image = tf.image.resize(image, IMG_SIZE)

    image = preprocess_input(image)
    return image, label
augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])
BATCH_SIZE = 32

def make_dataset(x, y, training=False):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        ds = ds.map(
            lambda img, lbl: (augment(img), lbl),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        ds = ds.shuffle(1000)

    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_dataset(X_train, y_train, training=True)
val_ds   = make_dataset(X_val, y_val)
test_ds  = make_dataset(X_test, y_test)
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weight_dict = {
    0: class_weights[0],  # Real
    1: class_weights[1],  # AI
}

print("Class weights:", class_weight_dict)

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB3

inputs = tf.keras.Input(shape=(224, 224, 3), name="input")

base_model = EfficientNetB3(
    include_top=False,
    weights="imagenet",
    input_tensor=inputs
)
base_model.trainable = False

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(512, activation="relu")(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

model = Model(inputs=inputs, outputs=outputs)

model.summary()

# =========================
# Compile & Train
# =========================
from tqdm.keras import TqdmCallback

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ]
)


history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weight_dict,
    callbacks=[TqdmCallback(verbose=1)]
)

# =========================
# Test
# =========================
test_loss, test_acc, test_precision, test_recall, test_auc = model.evaluate(test_ds)

print("Test Accuracy:", test_acc)
print("Precision:", test_precision)
print("Recall:", test_recall)
print("AUC:", test_auc)

import tf2onnx

model.trainable = False

input_signature = (
    tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),
)

tf2onnx.convert.from_keras(
    model,
    input_signature=input_signature,
    opset=17,
    output_path="/kaggle/working/ai_vs_real_cnn_frozen.onnx"
)

print("✅ ONNX export successful")
