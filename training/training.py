# =========================
# Standard Library
# =========================
import os

# =========================
# Third-party Libraries
# =========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# ML / DL
# =========================
import tensorflow as tf
from keras import layers, models

# =========================
# Sklearn
# =========================
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# =========================
# Utils
# =========================
from tqdm.keras import TqdmCallback

FOLDER_AI = {"ai", "fake", "fakev2", "synthetic", "generated"}
FOLDER_REAL = {"real", "human", "original"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

records = []

# ---------- PASS 1: Folder-based datasets ----------
for root, _, files in os.walk("/kaggle/input"):
    # ❗ skip CSV-based dataset completely here
    if "ai-vs-human-generated-dataset" in root:
        continue

    leaf = os.path.basename(root).lower()

    if leaf not in FOLDER_AI and leaf not in FOLDER_REAL:
        continue

    label = 1 if leaf in FOLDER_AI else 0

    for f in files:
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
            records.append({
                "path": os.path.join(root, f),
                "label": label
            })

# ---------- PASS 2: CSV-based dataset ----------
CSV_ROOT = "/kaggle/input/ai-vs-human-generated-dataset"

def load_csv_dataset(csv_path):
    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        img_path = os.path.join(CSV_ROOT, row["file_name"])
        label = int(row["label"])  # 0 = Real, 1 = AI

        if os.path.exists(img_path):
            records.append({
                "path": img_path,
                "label": label
            })

load_csv_dataset(f"{CSV_ROOT}/train.csv")

# ---------- FINAL DATAFRAME ----------
df = pd.DataFrame(records)

print("Total images:", len(df))
print(df["label"].value_counts())

df.to_csv("/kaggle/working/dataset_index_clean.csv", index=False)

print(df["label"].value_counts())


df = pd.read_csv("/kaggle/working/dataset_index_clean.csv")

MAX_FILE_SIZE = 15 * 1024 * 1024  # 15 MB

safe_rows = []

for _, row in df.iterrows():
    try:
        if os.path.getsize(row["path"]) <= MAX_FILE_SIZE:
            safe_rows.append(row)
    except:
        pass

safe_df = pd.DataFrame(safe_rows)

print("Original images:", len(df))
print("Safe images    :", len(safe_df))
print("Removed        :", len(df) - len(safe_df))

safe_df.to_csv("/kaggle/working/dataset_index_safe.csv", index=False)


df = pd.read_csv("/kaggle/working/dataset_index_safe.csv")

train_df, temp_df = train_test_split(
    df, test_size=0.2, stratify=df["label"], random_state=42
)

val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42
)

print("Train:", len(train_df))
print("Val  :", len(val_df))
print("Test :", len(test_df))

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def make_dataset(df, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices(
        (df["path"].values, df["label"].values)
    )

    if shuffle:
        ds = ds.shuffle(1000)

    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds

train_ds = make_dataset(train_df, shuffle=True)
val_ds   = make_dataset(val_df)
test_ds  = make_dataset(test_df)

model = models.Sequential([
    layers.Conv2D(32, 3, activation="relu", input_shape=(224,224,3)),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid")
])

model.summary()

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

EPOCHS = 5

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[TqdmCallback(verbose=1)]
)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

test_loss, test_acc = model.evaluate(test_ds)
print("✅ Test Accuracy:", test_acc)

model.save("/kaggle/working/ai_vs_real_cnn.keras")

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    preds = (preds > 0.5).astype(int).flatten()
    y_pred.extend(preds)
    y_true.extend(labels.numpy())

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)

print(classification_report(
    y_true, y_pred,
    target_names=["Real", "AI"]
))

from sklearn.metrics import roc_auc_score

y_scores = []

for images, _ in test_ds:
    scores = model.predict(images, verbose=0).flatten()
    y_scores.extend(scores)

auc = roc_auc_score(y_true, y_scores)
print("ROC-AUC:", auc)

for t in [0.3, 0.5, 0.7]:
    preds = (np.array(y_scores) > t).astype(int)
    acc = (preds == np.array(y_true)).mean()
    print(f"Threshold {t}: Accuracy {acc:.4f}")

shuffled_df = train_df.copy()
shuffled_df["label"] = np.random.permutation(shuffled_df["label"].values)

shuffled_ds = make_dataset(shuffled_df, shuffle=True)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(shuffled_ds, epochs=1)

import matplotlib.pyplot as plt

wrong = []

for images, labels in test_ds:
    preds = (model.predict(images, verbose=0) > 0.5).astype(int).flatten()
    for img, y, p in zip(images, labels, preds):
        if y != p:
            wrong.append((img.numpy(), y, p))
        if len(wrong) >= 5:
            break
    if len(wrong) >= 5:
        break

for img, y, p in wrong:
    plt.imshow(img)
    plt.title(f"True:{y} Pred:{p}")
    plt.axis("off")
    plt.show()

