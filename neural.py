import glob
import os
import pickle

import keras
import numpy as np
import pandas as pd
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

file_paths = glob.glob("*_*.csv")

# Extract gesture_name by taking everything before the final underscore '_'
# e.g., 'wave_12.csv' -> 'wave'
file_labels = [os.path.basename(f).rsplit('_', 1)[0] for f in file_paths]

# 2. Split entire file recordings into Train and Validation sets
train_files, val_files, train_file_labels, val_file_labels = train_test_split(
    file_paths,
    file_labels,
    test_size=0.2,
    random_state=42,
    stratify=file_labels
)

# 3. Helper function to read full recording files into feature and target arrays
def load_dataset_from_file_list(file_list):
    X_list, y_list = [], []

    for file_path in file_list:
        gesture_name = os.path.basename(file_path).rsplit('_', 1)[0]
        df = pd.read_csv(file_path)

        # Each row is a timestamp/sample of feature values (ax, ay, az, gx, gy, gz, d, etc.)
        X_list.append(df.values)
        y_list.append([gesture_name] * len(df))

    return np.vstack(X_list), np.concatenate(y_list)

# Load data split cleanly by recording trial
X_train_raw, y_train_raw = load_dataset_from_file_list(train_files)
X_val_raw, y_val_raw = load_dataset_from_file_list(val_files)

# 4. Encode gesture labels (e.g., 'wave' -> 0, 'ghost' -> 1)
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train_raw)
y_val = label_encoder.transform(y_val_raw)

# 5. Fit scale transform on training set only
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_val = scaler.transform(X_val_raw)

# Reshape for 1D CNN: (samples, feature_count, 1)
x_train = np.expand_dims(X_train, axis=-1)
x_val = np.expand_dims(X_val, axis=-1)

model = keras.Sequential([
    layers.Input(shape=(x_train.shape[1], 1)),

    layers.Conv1D(filters=16, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPooling1D(pool_size=2),
    layers.BatchNormalization(),

    layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPooling1D(pool_size=2),
    layers.BatchNormalization(),

    layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPooling1D(pool_size=2),
    layers.BatchNormalization(),

    layers.Flatten(),
    layers.Dropout(rate=0.3),
    layers.Dense(15, activation='softmax')
])
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
fit_model = model.fit(
    x_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(x_val, y_val),
    callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    ]
)
print(model.summary())
model.save("gesture_model.h5")

with open("gesture_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("gesture_label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
