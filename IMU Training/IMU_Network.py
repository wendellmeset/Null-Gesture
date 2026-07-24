import numpy as np
import keras
from keras import layers
from sklearn.model_selection import train_test_split

SECONDS = 5
HERTZ = 50
TIMESTAMPS = SECONDS * HERTZ

def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x
    x = layers.Conv1D(filters, kernel_size, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(filters, kernel_size, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = layers.Conv1D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    x = layers.add([x, shortcut])
    x = layers.ReLU()(x)
    return x

def build_model():
    inputs = keras.Input(shape=(TIMESTAMPS, 6))
    # Initial convolution
    x = layers.Conv1D(32, 7, strides=1, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Residual blocks with increasing filters and downsampling
    x = residual_block(x, 32, stride=1)
    x = residual_block(x, 64, stride=2)   # 250 → 125
    x = residual_block(x, 64, stride=1)
    x = residual_block(x, 128, stride=2)  # 125 → 62
    x = residual_block(x, 128, stride=1)
    x = residual_block(x, 256, stride=2)  # 62 → 31
    x = residual_block(x, 256, stride=1)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.4)(x)

    orient_out = layers.Dense(5, activation='softmax', name='orientation')(x)
    motion_out = layers.Dense(7, activation='softmax', name='motion')(x)

    model = keras.Model(inputs=inputs, outputs=[orient_out, motion_out])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss={
            'orientation': 'sparse_categorical_crossentropy',
            'motion': 'sparse_categorical_crossentropy'
        },
        metrics={
            'orientation': ['accuracy'],
            'motion': ['accuracy']
        }
    )
    return model

def main():
    X = np.load('X_imu.npy')
    y_orient = np.load('y_orient.npy')
    y_motion = np.load('y_motion.npy')

    print(f"X shape: {X.shape}")
    print(f"Orientation labels: {np.unique(y_orient)}")
    print(f"Motion labels: {np.unique(y_motion)}")

    X_train, X_val, y_orient_train, y_orient_val, y_motion_train, y_motion_val = train_test_split(
        X, y_orient, y_motion, test_size=0.2, random_state=42
    )

    model = build_model()
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
        keras.callbacks.ModelCheckpoint('best_imu_model.h5', save_best_only=True)
    ]

    history = model.fit(
        X_train,
        {'orientation': y_orient_train, 'motion': y_motion_train},
        validation_data=(X_val, {'orientation': y_orient_val, 'motion': y_motion_val}),
        epochs=80,
        batch_size=32,
        verbose=1,
        callbacks=callbacks
    )

    model.save('imu_primitive_model.h5')
    print("Model saved as imu_primitive_model.h5")

if __name__ == "__main__":
    main()
