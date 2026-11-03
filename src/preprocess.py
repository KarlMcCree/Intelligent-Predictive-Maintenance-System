import os
import pandas as pd
import numpy as np
from . import config

def extract_features(df):
    """Extracts time-domain features from a raw window of vibration data."""
    # Ensure we only use the Z-axis
    z = df["accel_z"].values
    
    rms = np.sqrt(np.mean(z**2))
    variance = np.var(z)
    peak = np.max(np.abs(z))
    crest_factor = peak / rms if rms > 0 else 0
    
    return {
        "rms": rms,
        "variance": variance,
        "peak": peak,
        "crest_factor": crest_factor
    }

def build_feature_dataset(csv_path, label):
    """Reads a CSV, applies sliding windows, and returns a DataFrame of features."""
    print(f"Processing {os.path.basename(csv_path)}...")
    df = pd.read_csv(csv_path)
    
    features_list = []
    
    for start in range(0, len(df) - config.WINDOW_SIZE, config.STEP_SIZE):
        window = df.iloc[start : start + config.WINDOW_SIZE]
        feats = extract_features(window)
        feats["label"] = label
        features_list.append(feats)
        
    return pd.DataFrame(features_list)

def load_and_merge_data():
    """Finds all normal and fault CSVs and merges them."""
    normal_files = [f for f in os.listdir(config.DATA_DIR) if f.startswith("normal_") and f.endswith(".csv")]
    fault_files = [f for f in os.listdir(config.DATA_DIR) if f.startswith("fault_") and f.endswith(".csv")]
    
    dfs = []
    for f in normal_files:
        dfs.append(build_feature_dataset(os.path.join(config.DATA_DIR, f), config.NORMAL_IDX))
    for f in fault_files:
        dfs.append(build_feature_dataset(os.path.join(config.DATA_DIR, f), config.FAULT_IDX))
        
    if not dfs:
        raise ValueError("No data found! Collect data first.")
        
    return pd.concat(dfs, ignore_index=True)