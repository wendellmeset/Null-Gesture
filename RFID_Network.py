import tensorflow as tf
import tensorflow.keras as keras
from keras import layers
from keras import metrics as keras_metrics

SECONDS = 3 # Length of each sample collected.
HERTZ = 30 # 30Hz packet rate
TIMESTAMPS = SECONDS * HERTZ

model = keras.Sequential([
    layers.Input(shape=(TIMESTAMPS, 2)), #2 features for RSSI and Tag Phase

    layers.Conv1D(16, 3, padding='same'),
    layers.BatchNormalization(),      # ← normalize before activation
    layers.ReLU(),
    layers.MaxPooling1D(pool_size=2),

    layers.Conv1D(32, 3, padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPooling1D(pool_size=2),

    layers.GlobalAveragePooling1D(),
    layers.Dense(1, activation='sigmoid')
])
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=[
        keras_metrics.F1Score(average='binary', name='f1_score')
    ]
)

# gesture_model = model.fit(x, y, epochs=10, validation_split=0.2)
