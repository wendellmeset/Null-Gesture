import keras
from keras import layers

SECONDS = 5
HERTZ = 50
TIMESTAMPS = SECONDS * HERTZ

# Shared input
inputs = keras.Input(shape=(TIMESTAMPS, 6))

# Shared Convolutional Backbone
x = layers.Conv1D(16, 3, padding='same')(inputs)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.MaxPooling1D(pool_size=2)(x)

x = layers.Conv1D(32, 3, padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.MaxPooling1D(pool_size=2)(x)

x = layers.GlobalAveragePooling1D()(x)

# Layer to handle the 5 orientations (flat, up, down, left, right)
orient_out = layers.Dense(5, activation='softmax', name='orientation')(x)


# Layer to handle the 6 movements (static, forward, backward, left, right,up, down)
motion_out = layers.Dense(6, activation='softmax', name='motion')(x)

model = keras.Model(inputs=inputs, outputs=[orient_out, motion_out])


model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss={
        'orientation': 'sparse_categorical_crossentropy',
        'motion': 'sparse_categorical_crossentropy'
    },
    metrics={
        'orientation': ['accuracy'],
        'motion': ['accuracy']
    }
)

# Training call:
# model.fit(X_train, {'orientation': y_orient, 'motion': y_motion}, epochs=15)
