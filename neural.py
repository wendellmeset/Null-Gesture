import keras
import numpy as np
import pandas as pd
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

pose_data = pd.read_csv('')
X = pose_data.drop(columns=['label']).values
y = pose_data['label'].values

x_train, x_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)

x_train = np.expand_dims(x_train, axis=-1)
x_val = np.expand_dims(x_val, axis=-1)

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
