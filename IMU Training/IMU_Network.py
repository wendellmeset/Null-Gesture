import numpy as np
import keras
from keras import layers
from sklearn.model_selection import train_test_split

SECONDS = 5
HERTZ = 50
TIMESTAMPS = SECONDS * HERTZ

def build_model():
    inputs = keras.Input(shape=(TIMESTAMPS, 6))
    x = layers.Conv1D(16, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(32, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.GlobalAveragePooling1D()(x)
    orient_out = layers.Dense(5, activation='softmax', name='orientation')(x)
    motion_out = layers.Dense(7, activation='softmax', name='motion')(x)  # FIXED to 7
    model = keras.Model(inputs=inputs, outputs=[orient_out, motion_out])
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
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
    # Load preprocessed data
    X = np.load('X_imu.npy')
    y_orient = np.load('y_orient.npy')
    y_motion = np.load('y_motion.npy')

    print(f"X shape: {X.shape}")
    print(f"Orientation labels: {np.unique(y_orient)}")
    print(f"Motion labels: {np.unique(y_motion)}")

    # Split
    X_train, X_val, y_orient_train, y_orient_val, y_motion_train, y_motion_val = train_test_split(
        X, y_orient, y_motion, test_size=0.2, random_state=42
    )

    model = build_model()
    model.summary()

    history = model.fit(
        X_train,
        {'orientation': y_orient_train, 'motion': y_motion_train},
        validation_data=(X_val, {'orientation': y_orient_val, 'motion': y_motion_val}),
        epochs=30,
        batch_size=32,
        verbose=1   # integer, not string
    )

    model.save('imu_primitive_model.h5')
    print("Model saved as imu_primitive_model.h5")

if __name__ == "__main__":
    main()
