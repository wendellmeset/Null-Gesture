#!/usr/bin/env python3
"""
Single‑file gesture recognition system.
- Loads pre‑trained IMU primitive model (imu_primitive_model.h5).
- Builds & trains fusion harness on synthetic data (no real fusion data needed).
- Receives IMU data over TCP (from esp32_reader.py on port 9999).
- Expects touch flag from your RFID system (set via set_touch_flag()).
- Outputs final gesture in real time.
"""

import time
import threading
import queue
import socket
import json
import sys
import select
import numpy as np
import tensorflow as tf
import keras
from keras import layers

# -------------------------------------------------------------------
# 1. Gesture definitions and fusion model
# -------------------------------------------------------------------

GESTURE_NAMES = [
    "Soli", "Push", "Pull", "Clockwise", "Anti-Clockwise",
    "Left", "Right", "Bye-Bye", "OneArmBoxing", "Clapping",
    "TwoArmBoxing", "T-Arms", "RaiseArms", "OpenCloseFist", "PalmUpDown"
]

# Rules: gesture_index -> (orientation_class, motion_class, touch_flag)
GESTURE_RULES = {
    0:  (0, 0, 1),   # Soli
    1:  (0, 1, 0),   # Push
    2:  (0, 2, 0),   # Pull
    3:  (0, 4, 0),   # Clockwise
    4:  (0, 3, 0),   # Anti-Clockwise
    5:  (0, 3, 0),   # Left
    6:  (0, 4, 0),   # Right
    7:  (0, 3, 0),   # Bye-Bye
    8:  (0, 1, 0),   # OneArmBoxing
    9:  (0, 1, 0),   # Clapping
    10: (0, 1, 0),   # TwoArmBoxing
    11: (3, 0, 0),   # T-Arms
    12: (1, 5, 0),   # RaiseArms
    13: (0, 0, 0),   # OpenCloseFist
    14: (1, 0, 0),   # PalmUpDown
}

def build_fusion_model():
    """Build the fusion MLP that maps primitives + touch to final gesture."""
    input_orient = keras.Input(shape=(5,), name='orientation_probs')
    input_motion = keras.Input(shape=(7,), name='motion_probs')
    input_touch = keras.Input(shape=(1,), name='touch_flag')
    concat = layers.Concatenate()([input_orient, input_motion, input_touch])
    x = layers.Dense(64, activation='relu')(concat)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation='relu')(x)
    output = layers.Dense(15, activation='softmax')(x)
    model = keras.Model(inputs=[input_orient, input_motion, input_touch], outputs=output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def generate_synthetic_fusion_data(num_samples_per_gesture=300):
    """Generate synthetic probability vectors for fusion training."""
    X_orient, X_motion, X_touch, y = [], [], [], []

    for gesture_idx, (orient, motion, touch) in GESTURE_RULES.items():
        for _ in range(num_samples_per_gesture):
            o = np.zeros(5)
            o[orient] = 0.9 + 0.1 * np.random.rand()
            o += np.random.normal(0, 0.03, size=5)
            o = np.clip(o, 0, 1)
            o /= o.sum()

            m = np.zeros(7)
            m[motion] = 0.9 + 0.1 * np.random.rand()
            m += np.random.normal(0, 0.03, size=7)
            m = np.clip(m, 0, 1)
            m /= m.sum()

            t = touch + np.random.normal(0, 0.05)
            t = np.clip(t, 0, 1)

            X_orient.append(o)
            X_motion.append(m)
            X_touch.append([t])
            y.append(gesture_idx)

    # Sequential mixes for Bye-Bye and Clapping
    for _ in range(num_samples_per_gesture // 2):
        m = np.zeros(7)
        m[3] = 0.45 + 0.1 * np.random.rand()
        m[4] = 0.45 + 0.1 * np.random.rand()
        m += np.random.normal(0, 0.03, size=7)
        m = np.clip(m, 0, 1)
        m /= m.sum()
        o = X_orient[0].copy()
        t = np.array([0.0 + np.random.normal(0, 0.05)]).clip(0,1)
        X_orient.append(o)
        X_motion.append(m)
        X_touch.append(t)
        y.append(7)

    for _ in range(num_samples_per_gesture // 2):
        m = np.zeros(7)
        m[1] = 0.45 + 0.1 * np.random.rand()
        m[2] = 0.45 + 0.1 * np.random.rand()
        m += np.random.normal(0, 0.03, size=7)
        m = np.clip(m, 0, 1)
        m /= m.sum()
        o = X_orient[0].copy()
        t = np.array([0.0 + np.random.normal(0, 0.05)]).clip(0,1)
        X_orient.append(o)
        X_motion.append(m)
        X_touch.append(t)
        y.append(9)

    return np.array(X_orient), np.array(X_motion), np.array(X_touch), np.array(y)

def get_or_train_fusion_model(force_retrain=False):
    """
    Load fusion model if saved, otherwise train from synthetic data and save.
    """
    model_path = 'fusion_harness_model.h5'
    if not force_retrain and tf.io.gfile.exists(model_path):
        try:
            print("Loading existing fusion model...")
            model = keras.models.load_model(model_path)
            # Quick test to ensure it's a valid model
            _ = model.predict([np.zeros((1,5)), np.zeros((1,7)), np.zeros((1,1))], verbose=0)
            print("Fusion model loaded successfully.")
            return model
        except Exception as e:
            print(f"Failed to load fusion model: {e}. Retraining...")
            force_retrain = True

    print("Training fusion model on synthetic data...")
    X_orient, X_motion, X_touch, y = generate_synthetic_fusion_data(300)
    print(f"Generated {len(y)} synthetic samples.")

    model = build_fusion_model()
    model.summary()
    model.fit(
        {'orientation_probs': X_orient, 'motion_probs': X_motion, 'touch_flag': X_touch},
        y,
        epochs=30,
        batch_size=32,
        validation_split=0.2,
        verbose="1"
    )
    model.save(model_path)
    print(f"Fusion model saved as {model_path}")
    return model

# -------------------------------------------------------------------
# 2. Real‑time pipeline
# -------------------------------------------------------------------

IMU_SECONDS = 5
IMU_HZ = 50
IMU_TIMESTAMPS = IMU_SECONDS * IMU_HZ

IMU_HOST = 'localhost'
IMU_PORT = 9999
BUFFER_SIZE = 4096

_touch_lock = threading.Lock()
_touch_flag = 0.0

def set_touch_flag(value):
    global _touch_flag
    with _touch_lock:
        _touch_flag = float(value)

def get_touch_flag():
    with _touch_lock:
        return _touch_flag

def imu_receiver(stop_event, imu_queue):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            s.connect((IMU_HOST, IMU_PORT))
            print(f"[IMU] Connected to {IMU_HOST}:{IMU_PORT}")
            while not stop_event.is_set():
                try:
                    data = s.recv(BUFFER_SIZE)
                    if not data:
                        break
                    for line in data.decode('utf-8').splitlines():
                        if not line.strip():
                            continue
                        try:
                            obj = json.loads(line)
                            if obj.get('type') == 'sample':
                                imu_queue.put((
                                    obj['timestamp'],
                                    obj['ax'], obj['ay'], obj['az'],
                                    obj['gx'], obj['gy'], obj['gz']
                                ))
                        except json.JSONDecodeError:
                            pass
                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"[IMU] Error: {e}")
                    break
    except Exception as e:
        print(f"[IMU] Connection failed: {e}")
    stop_event.set()

def main():
    # --- Load IMU model ---
    try:
        imu_model = tf.keras.models.load_model('imu_primitive_model.h5')
        print("IMU model loaded.")
    except Exception as e:
        print(f"Error loading IMU model: {e}")
        print("Please train and save imu_primitive_model.h5 first.")
        return

    # --- Load or train fusion model ---
    fusion_model = get_or_train_fusion_model(force_retrain=False)
    if fusion_model is None:
        print("Fusion model could not be obtained. Exiting.")
        return

    # --- Set up threading ---
    stop_event = threading.Event()
    imu_queue = queue.Queue()

    imu_thread = threading.Thread(target=imu_receiver, args=(stop_event, imu_queue))
    imu_thread.start()

    print("\n--- Real‑time gesture recognition started ---")
    print("Touch flag can be updated via set_touch_flag(value).")
    print("Press 't' in this terminal to toggle touch flag (demo), or Ctrl+C to quit.\n")

    imu_samples = []
    inference_interval = 0.05  # 20 Hz

    try:
        while not stop_event.is_set():
            while not imu_queue.empty():
                imu_samples.append(imu_queue.get())

            now = time.monotonic()
            imu_samples = [s for s in imu_samples if now - s[0] < 10.0]

            if select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.read(1)
                if key == 't':
                    new_val = 1.0 - get_touch_flag()
                    set_touch_flag(new_val)
                    print(f"Touch flag toggled to {new_val}")

            if len(imu_samples) >= IMU_TIMESTAMPS:
                start_time = now - IMU_SECONDS
                imu_win = []
                for t in np.linspace(start_time, now, IMU_TIMESTAMPS):
                    closest = min(imu_samples, key=lambda s: abs(s[0] - t))
                    imu_win.append(closest[1:])
                imu_win = np.array(imu_win, dtype=np.float32).reshape(1, IMU_TIMESTAMPS, 6)

                orient_probs, motion_probs = imu_model.predict(imu_win, verbose=0)
                orient_class = np.argmax(orient_probs[0])
                motion_class = np.argmax(motion_probs[0])

                touch_flag = get_touch_flag()

                # --- Ensure inputs are numpy arrays of correct shape ---
                fusion_input = {
                    'orientation_probs': orient_probs,          # shape (1,5)
                    'motion_probs': motion_probs,              # shape (1,7)
                    'touch_flag': np.array([[touch_flag]], dtype=np.float32)  # (1,1)
                }

                final_probs = fusion_model.predict(fusion_input, verbose="0")[0]
                gesture_idx = np.argmax(final_probs)
                confidence = final_probs[gesture_idx]

                print(f"Gesture: {GESTURE_NAMES[gesture_idx]} (conf: {confidence:.2f})  "
                      f"Orient: {orient_class}  Motion: {motion_class}  Touch: {touch_flag}")

                imu_samples = [s for s in imu_samples if now - s[0] < 1.0]

            time.sleep(inference_interval)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        stop_event.set()
        imu_thread.join(timeout=2)
        print("Stopped.")

if __name__ == "__main__":
    main()
