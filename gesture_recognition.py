#!/usr/bin/env python3
"""
Gesture Recognition System – Real‑time fusion of IMU primitives and RFID touch.

- Loads the pre‑trained IMU primitive model (imu_primitive_model.h5).
- Builds & trains a fusion MLP on synthetic data covering ALL primitive combinations.
- Receives IMU data over TCP (from esp32_reader.py on port 9999).
- Accepts touch flag from RFID via set_touch_flag().
- Outputs final gesture (one of 15 gestures or "Unknown") in real time.
"""

import time
import threading
import queue
import socket
import json
import sys
import numpy as np
import tensorflow as tf
import keras
from keras import layers

# -------------------------------------------------------------------
# 1. Gesture definitions and fusion model
# -------------------------------------------------------------------

# 15 gestures + "Unknown" (index 15)
GESTURE_NAMES = [
    "Soli", "Push", "Pull", "Clockwise", "Anti-Clockwise",
    "Left", "Right", "Bye-Bye", "OneArmBoxing", "Clapping",
    "TwoArmBoxing", "T-Arms", "RaiseArms", "OpenCloseFist", "PalmUpDown",
    "Unknown"
]

# Known mappings: gesture_index -> (orientation_class, motion_class, touch_flag)
# Any combination not listed will map to "Unknown" (15) in the synthetic data.
# You can extend this dictionary to add more mappings.
KNOWN_RULES = {
    0:  (0, 0, 1),   # Soli
    1:  (0, 1, 0),   # Push (Flat, Forward)
    2:  (0, 2, 0),   # Pull (Flat, Backward)
    3:  (0, 4, 0),   # Clockwise (Flat, SwingRight)
    4:  (0, 3, 0),   # Anti-Clockwise (Flat, SwingLeft)
    5:  (0, 3, 0),   # Left (Flat, SwingLeft)
    6:  (0, 4, 0),   # Right (Flat, SwingRight)
    7:  (0, 3, 0),   # Bye-Bye (alternating L/R – handled synthetically)
    8:  (0, 1, 0),   # OneArmBoxing (Flat, Forward repeated)
    9:  (0, 1, 0),   # Clapping (Flat, Forward/Backward alternating)
    10: (0, 1, 0),   # TwoArmBoxing (Flat, Forward repeated)
    11: (3, 0, 0),   # T-Arms (Left, Static)
    12: (1, 5, 0),   # RaiseArms (Up, SwingUp)
    13: (0, 0, 0),   # OpenCloseFist (Flat, Static)
    14: (1, 0, 0),   # PalmUpDown (Up, Static)
}

# Build a lookup for fast access during synthetic generation
KNOWN_LOOKUP = {}
for gest_idx, (o, m, t) in KNOWN_RULES.items():
    KNOWN_LOOKUP[(o, m, t)] = gest_idx

def build_fusion_model():
    """Fusion MLP: 5+7+1 → 16 classes."""
    input_orient = keras.Input(shape=(5,), name='orientation_probs')
    input_motion = keras.Input(shape=(7,), name='motion_probs')
    input_touch = keras.Input(shape=(1,), name='touch_flag')
    concat = layers.Concatenate()([input_orient, input_motion, input_touch])
    x = layers.Dense(64, activation='relu')(concat)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation='relu')(x)
    output = layers.Dense(16, activation='softmax')(x)   # 15 + Unknown
    model = keras.Model(inputs=[input_orient, input_motion, input_touch], outputs=output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def generate_synthetic_fusion_data(num_samples_per_combo=30):
    """
    Generate synthetic probability vectors for ALL 70 primitive combinations.
    Each combination is mapped to a known gesture or to "Unknown" (15).
    """
    X_orient, X_motion, X_touch, y = [], [], [], []
    for orient in range(5):
        for motion in range(7):
            for touch in [0, 1]:
                # Determine label
                if (orient, motion, touch) in KNOWN_LOOKUP:
                    label = KNOWN_LOOKUP[(orient, motion, touch)]
                else:
                    label = 15   # Unknown

                for _ in range(num_samples_per_combo):
                    # Orientation probabilities (noisy one‑hot)
                    o = np.zeros(5)
                    o[orient] = 0.9 + 0.1 * np.random.rand()
                    o += np.random.normal(0, 0.03, size=5)
                    o = np.clip(o, 0, 1)
                    o /= o.sum()

                    # Motion probabilities
                    m = np.zeros(7)
                    m[motion] = 0.9 + 0.1 * np.random.rand()
                    m += np.random.normal(0, 0.03, size=7)
                    m = np.clip(m, 0, 1)
                    m /= m.sum()

                    # Touch flag (binary with noise)
                    t = touch + np.random.normal(0, 0.05)
                    t = np.clip(t, 0, 1)

                    X_orient.append(o)
                    X_motion.append(m)
                    X_touch.append([t])
                    y.append(label)

    # Shuffle
    indices = np.random.permutation(len(y))
    X_orient = np.array(X_orient)[indices]
    X_motion = np.array(X_motion)[indices]
    X_touch = np.array(X_touch)[indices]
    y = np.array(y)[indices]
    return X_orient, X_motion, X_touch, y

def get_or_train_fusion_model(force_retrain=False):
    """
    Load existing fusion model or train a new one from synthetic data.
    """
    model_path = 'fusion_harness_model.h5'
    if not force_retrain and tf.io.gfile.exists(model_path):
        try:
            print("Loading existing fusion model...")
            model = keras.models.load_model(model_path)
            # Quick validation
            _ = model.predict([np.zeros((1,5)), np.zeros((1,7)), np.zeros((1,1))], verbose=0)
            print("Fusion model loaded successfully.")
            return model
        except Exception as e:
            print(f"Failed to load fusion model: {e}. Retraining...")
            force_retrain = True

    print("Training fusion model on synthetic data...")
    X_orient, X_motion, X_touch, y = generate_synthetic_fusion_data(num_samples_per_combo=40)
    print(f"Generated {len(y)} synthetic samples.")

    model = build_fusion_model()
    model.summary()
    model.fit(
        {'orientation_probs': X_orient, 'motion_probs': X_motion, 'touch_flag': X_touch},
        y,
        epochs=40,
        batch_size=32,
        validation_split=0.2,
        verbose=1
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

# Global touch flag (thread‑safe)
_touch_lock = threading.Lock()
_touch_flag = 0.0

def set_touch_flag(value):
    """Call this from your RFID thread to update the touch flag (0 or 1)."""
    global _touch_flag
    with _touch_lock:
        _touch_flag = float(value)

def get_touch_flag():
    with _touch_lock:
        return _touch_flag

# Keyboard listener (cross‑platform toggle)
def keyboard_listener(stop_event):
    """Reads keyboard input and toggles touch flag when 't' is pressed."""
    while not stop_event.is_set():
        try:
            key = sys.stdin.read(1)
            if key == 't':
                new_val = 1.0 - get_touch_flag()
                set_touch_flag(new_val)
                print(f"\nTouch flag toggled to {new_val}")
        except Exception:
            # stdin closed or other error
            break

# IMU TCP receiver
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

    keyboard_thread = threading.Thread(target=keyboard_listener, args=(stop_event,), daemon=True)
    keyboard_thread.start()

    print("\n--- Real‑time gesture recognition started ---")
    print("Touch flag can be updated via set_touch_flag(value).")
    print("Press 't' in this terminal to toggle touch flag (demo), or Ctrl+C to quit.\n")

    imu_samples = []
    inference_interval = 0.05  # 20 Hz

    try:
        while not stop_event.is_set():
            # Drain IMU queue
            while not imu_queue.empty():
                imu_samples.append(imu_queue.get())

            now = time.monotonic()
            imu_samples = [s for s in imu_samples if now - s[0] < 10.0]

            if len(imu_samples) >= IMU_TIMESTAMPS:
                # Resample to fixed 50 Hz over last 5 seconds
                start_time = now - IMU_SECONDS
                imu_win = []
                for t in np.linspace(start_time, now, IMU_TIMESTAMPS):
                    closest = min(imu_samples, key=lambda s: abs(s[0] - t))
                    imu_win.append(closest[1:])  # (ax, ay, az, gx, gy, gz)
                imu_win = np.array(imu_win, dtype=np.float32).reshape(1, IMU_TIMESTAMPS, 6)

                # --- IMU inference ---
                orient_probs, motion_probs = imu_model.predict(imu_win, verbose=0)
                orient_conf = np.max(orient_probs[0])
                motion_conf = np.max(motion_probs[0])

                # --- Confidence gate ---
                CONFIDENCE_THRESHOLD = 0.6
                if orient_conf < CONFIDENCE_THRESHOLD or motion_conf < CONFIDENCE_THRESHOLD:
                    gesture_idx = 15  # Unknown
                    confidence = min(orient_conf, motion_conf)
                    orient_class = np.argmax(orient_probs[0])
                    motion_class = np.argmax(motion_probs[0])
                    touch_flag = get_touch_flag()
                    print(f"⚠ Low confidence – Unknown (orient:{orient_conf:.2f}, motion:{motion_conf:.2f})")
                else:
                    orient_class = np.argmax(orient_probs[0])
                    motion_class = np.argmax(motion_probs[0])
                    touch_flag = get_touch_flag()

                    # --- Fusion prediction ---
                    fusion_input = {
                        'orientation_probs': orient_probs,
                        'motion_probs': motion_probs,
                        'touch_flag': np.array([[touch_flag]], dtype=np.float32)
                    }
                    final_probs = fusion_model.predict(fusion_input, verbose=0)[0]
                    gesture_idx = np.argmax(final_probs)
                    confidence = final_probs[gesture_idx]

                # --- Output ---
                gesture_name = GESTURE_NAMES[gesture_idx]
                print(f"Gesture: {gesture_name} (conf: {confidence:.2f})  "
                      f"Orient: {orient_class}  Motion: {motion_class}  Touch: {touch_flag}")

                # Keep last 1 second for continuity
                imu_samples = [s for s in imu_samples if now - s[0] < 1.0]

            time.sleep(inference_interval)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        stop_event.set()
        imu_thread.join(timeout=2)
        keyboard_thread.join(timeout=1)
        print("Stopped.")

if __name__ == "__main__":
    main()
