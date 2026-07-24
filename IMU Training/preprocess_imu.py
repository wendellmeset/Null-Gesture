import numpy as np
import pandas as pd
from scipy import interpolate
import os
import warnings
warnings.filterwarnings('ignore')

SECONDS = 5
HERTZ = 50
TIMESTAMPS = SECONDS * HERTZ

ORIENT_MAP = {
    "Flat Orientation": 0,
    "Up Orientation": 1,
    "Down Orientation": 2,
    "Left Orientation": 3,
    "Right Orientation": 4,
}

MOTION_MAP = {
    "Static": 0,
    "Forward": 1,
    "Backward": 2,
    "Swing Left": 3,
    "Swing Right": 4,
    "Swing Up": 5,
    "Swing Down": 6,
}

def find_columns(df):
    """Find sensor columns using fuzzy matching."""
    col_lower = {col.lower().strip(): col for col in df.columns}
    candidates = {
        'ax': ['ax', 'x', 'accelx', 'accel x'],
        'ay': ['ay', 'y', 'accely', 'accel y'],
        'az': ['az', 'z', 'accelz', 'accel z'],
        'gx': ['gx', 'gx', 'gyrox', 'gyro x'],
        'gy': ['gy', 'gy', 'gyroy', 'gyro y'],
        'gz': ['gz', 'gz', 'gyroz', 'gyro z'],
    }
    col_map = {}
    for key, cand_list in candidates.items():
        found = None
        for c in cand_list:
            if c in col_lower:
                found = col_lower[c]
                break
        if found is None:
            return None
        col_map[key] = found
    return col_map

def resample_window(df, col_map, target_hz=HERTZ, duration=SECONDS):
    """
    Resample to fixed time steps. Cleans data first.
    """
    # Use timestamp if available, else row index
    if 'timestamp' in df.columns:
        t = df['timestamp'].values
    else:
        t = np.arange(len(df))

    # Clean: remove duplicates and sort
    df_temp = pd.DataFrame({
        't': t,
        'ax': df[col_map['ax']],
        'ay': df[col_map['ay']],
        'az': df[col_map['az']],
        'gx': df[col_map['gx']],
        'gy': df[col_map['gy']],
        'gz': df[col_map['gz']]
    })
    # Remove rows with any NaN in sensor data
    df_temp = df_temp.dropna(subset=['ax','ay','az','gx','gy','gz'])
    if len(df_temp) < 2:
        return None
    # Sort by time
    df_temp = df_temp.sort_values('t')
    # Remove duplicate times (keep first)
    df_temp = df_temp.drop_duplicates(subset=['t'], keep='first')
    # Extract
    t_clean = df_temp['t'].values
    data = df_temp[['ax','ay','az','gx','gy','gz']].values

    # If there are still NaNs, interpolate within the cleaned data
    if np.isnan(data).any():
        # Interpolate column-wise
        data = pd.DataFrame(data).interpolate(method='linear', limit_direction='both').values

    # Check if we have at least 2 points for interpolation
    if len(t_clean) < 2:
        return None

    # Create regular grid
    t0 = t_clean.min()
    t_regular = np.linspace(t0, t0 + duration, target_hz * duration)

    resampled = []
    try:
        for i in range(6):
            f = interpolate.interp1d(t_clean, data[:, i], kind='linear', fill_value='extrapolate')
            resampled.append(f(t_regular))
    except Exception as e:
        print(f"Interpolation error: {e}")
        return None

    result = np.column_stack(resampled).astype(np.float32)
    # If any NaN in result, replace with column mean
    if np.isnan(result).any():
        for i in range(6):
            col_mean = np.nanmean(result[:, i])
            result[np.isnan(result[:, i]), i] = col_mean
    return result

def find_csv_files(base_dir):
    csv_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.csv'):
                full_path = os.path.join(root, file)
                path_parts = os.path.normpath(root).split(os.sep)
                if len(path_parts) >= 2:
                    motion_folder = path_parts[-1]
                    orient_folder = path_parts[-2]
                    if motion_folder in MOTION_MAP and orient_folder in ORIENT_MAP:
                        csv_files.append((full_path, ORIENT_MAP[orient_folder], MOTION_MAP[motion_folder]))
                    else:
                        print(f"Skipping {full_path}: unknown folder names")
                else:
                    print(f"Skipping {full_path}: not enough nesting")
    return csv_files

def main():
    base_dir = os.getcwd()
    print(f"Searching in: {base_dir}")
    csv_info = find_csv_files(base_dir)
    print(f"Found {len(csv_info)} CSV files.")

    if not csv_info:
        print("No CSV files found. Check your folder structure and names.")
        return

    X_list = []
    y_orient_list = []
    y_motion_list = []

    for idx, (file_path, orient_label, motion_label) in enumerate(csv_info):
        try:
            df = pd.read_csv(file_path)
            if len(df) < 10:
                print(f"Skipping {file_path}: too few rows ({len(df)})")
                continue

            col_map = find_columns(df)
            if col_map is None:
                print(f"Skipping {file_path}: could not find required columns. Available: {df.columns.tolist()}")
                continue

            # Resample
            data_win = resample_window(df, col_map)
            if data_win is None or data_win.shape[0] != TIMESTAMPS:
                print(f"Skipping {file_path}: resampled shape {data_win.shape if data_win is not None else 'None'}")
                continue

            X_list.append(data_win)
            y_orient_list.append(orient_label)
            y_motion_list.append(motion_label)

            if (idx+1) % 20 == 0:
                print(f"Processed {idx+1} files...")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    if not X_list:
        print("No valid windows created. Check your CSV content.")
        return

    X = np.array(X_list)
    y_orient = np.array(y_orient_list)
    y_motion = np.array(y_motion_list)

    print(f"Total valid windows: {X.shape[0]}")
    print(f"X shape: {X.shape}")
    print(f"Unique orientation labels: {np.unique(y_orient)}")
    print(f"Unique motion labels: {np.unique(y_motion)}")

    # Check for NaN in X
    if np.isnan(X).any():
        print("Warning: X contains NaN values. Replacing with column means...")
        for i in range(6):
            col_mean = np.nanmean(X[:, :, i])
            X[np.isnan(X[:, :, i]), i] = col_mean

    # ----- Normalize -----
    mean = X.mean(axis=(0, 1), keepdims=False)
    std = X.std(axis=(0, 1), keepdims=False) + 1e-8
    print(f"Mean per channel: {mean}")
    print(f"Std per channel: {std}")
    X_norm = (X - mean) / std

    # Shuffle
    indices = np.random.permutation(len(X_norm))
    X_norm = X_norm[indices]
    y_orient = y_orient[indices]
    y_motion = y_motion[indices]

    np.save('X_imu.npy', X_norm)
    np.save('y_orient.npy', y_orient)
    np.save('y_motion.npy', y_motion)

    print("Saved X_imu.npy, y_orient.npy, y_motion.npy")
    print("Now run IMU_Network.py to train!")

if __name__ == "__main__":
    main()
