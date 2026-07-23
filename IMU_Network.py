import tensorflow as tf
import tensorflow.keras as keras
from keras import layers
from keras import metrics as keras_metrics

SECONDS = 5 # Length of each sample collected.
HERTZ = 0.24 # 240 MHz
TIMESTAMPS = SECONDS * HERTZ

model = keras.Sequential([
    layers.Input(shape=(TIMESTAMPS, 6)), # shape (timestamps, # of IMU features (accel x, y, z; gyro x, y, z))

    layers.Conv1D(16, 3, padding='same'),
    layers.BatchNormalization(),      # ← normalize before activation
    layers.ReLU(),
    layers.MaxPooling1D(pool_size=2),

    layers.Conv1D(32, 3, padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling1D(pool_size=2),

    layers.GlobalAveragePooling1D(),
    layers.Dense(15, activation='softmax')
])
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=[
        keras_metrics.F1Score(average='weighted', name='f1_score')
    ]
)

# gesture_model = model.fit(x, y, epochs=10, validation_split=0.2)
