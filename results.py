import glob
import os
import pickle

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from neural import load_dataset_from_file_list

print("Loading model and fitted artifacts...")
model = keras.models.load_model("gesture_model.h5")

with open("gesture_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("gesture_label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

file_paths = glob.glob("*_*.csv")
file_labels = [os.path.basename(f).rsplit("_", 1)[0] for f in file_paths]

_, val_files, _, val_file_labels = train_test_split(
    file_paths,
    file_labels,
    test_size=0.2,
    random_state=42,
    stratify=file_labels,
)

# Load raw validation data using imported helper
X_val_raw, y_val_raw = load_dataset_from_file_list(val_files)

# Transform using loaded encoder and scaler
y_val = label_encoder.transform(y_val_raw)
X_val = scaler.transform(X_val_raw)
x_val = np.expand_dims(X_val, axis=-1)

print("Evaluating predictions...")
y_pred_probs = model.predict(X_val) # type: ignore
y_pred = np.argmax(y_pred_probs, axis=1)

class_names = label_encoder.classes_

# Text Classification Report
print("\n" + "=" * 55)
print("              CLASSIFICATION REPORT")
print("=" * 55)
print(classification_report(y_val, y_pred, target_names=class_names))

# Confusion Matrix Computation
cm = confusion_matrix(y_val, y_pred)

print("\n" + "=" * 55)
print("            CONFUSION MATRIX (SUMMARY)")
print("=" * 55)
cm_df = pd.DataFrame(
    cm,
    index=[f"True: {c}" for c in class_names],
    columns=[f"Pred: {c}" for c in class_names],
)
print(cm_df)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
)
plt.title("Gesture Recognition - Confusion Matrix", fontsize=14)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()

# Save image and render plot
plt.savefig("confusion_matrix.png", dpi=300)
print("\nSaved confusion matrix plot to 'confusion_matrix.png'")
plt.show()
