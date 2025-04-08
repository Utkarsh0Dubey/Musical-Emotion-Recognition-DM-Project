# src/feature_extraction.py

import os
import librosa
import pandas as pd
import yaml

def load_config():
    """Load configuration from config.yaml"""
    # Use a relative path from this script's location
    config_path = os.path.join(os.path.dirname(__file__), "../configs/config.yaml")
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            print(f"[DEBUG] Loaded configuration from {config_path}")
            return config
    except Exception as e:
        print(f"[ERROR] Unable to load config file at {config_path}: {e}")
        raise

def extract_audio_features(audio_path, sr=22050, n_mfcc=13, hop_length=512):
    """
    Extract MFCCs and other features from a single audio file.
    Returns a dictionary of features if successful; otherwise, returns None.
    """
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        if y is None or len(y) == 0:
            print(f"[WARNING] Audio file {audio_path} is empty or not loaded properly.")
            return None
    except Exception as e:
        print(f"[ERROR] Error loading {audio_path}: {e}")
        return None

    try:
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
        if mfccs.size == 0:
            print(f"[WARNING] No MFCC features extracted from {audio_path}.")
            return None
        mfccs_mean = mfccs.mean(axis=1)  # Average across time

        # Extract spectral contrast features
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        if spectral_contrast.size == 0:
            print(f"[WARNING] No spectral contrast features extracted from {audio_path}.")
            return None
        spectral_contrast_mean = spectral_contrast.mean(axis=1)
    except Exception as e:
        print(f"[ERROR] Failed to extract features from {audio_path}: {e}")
        return None

    features = {"filename": os.path.basename(audio_path)}
    # Store MFCC features
    for i in range(n_mfcc):
        features[f"mfcc_{i + 1}"] = mfccs_mean[i]
    # Store spectral contrast features
    for i, val in enumerate(spectral_contrast_mean):
        features[f"spectral_contrast_{i + 1}"] = val

    print(f"[DEBUG] Features extracted for {audio_path}")
    return features

def main():
    config = load_config()

    # Get configuration values with defaults if keys are missing
    audio_dir = config.get("data", {}).get("audio_dir", "data/audio")
    processed_data_dir = config.get("data", {}).get("processed_data_dir", "data/processed_data")
    sr = config.get("feature_extraction", {}).get("sampling_rate", 22050)
    n_mfcc = config.get("feature_extraction", {}).get("n_mfcc", 13)
    hop_length = config.get("feature_extraction", {}).get("hop_length", 512)

    # Ensure the processed data directory exists
    os.makedirs(processed_data_dir, exist_ok=True)
    print(f"[DEBUG] Ensured processed data directory exists at {processed_data_dir}")

    all_features = []
    audio_files_found = False

    # Iterate over audio files in the specified directory
    for root, dirs, files in os.walk(audio_dir):
        for file in files:
            if file.lower().endswith((".wav", ".mp3")):
                audio_files_found = True
                audio_path = os.path.join(root, file)
                print(f"[INFO] Processing file: {audio_path}")
                features_dict = extract_audio_features(audio_path, sr, n_mfcc, hop_length)
                if features_dict:
                    all_features.append(features_dict)
                else:
                    print(f"[DEBUG] No features extracted for {audio_path}")

    if not audio_files_found:
        print("[ERROR] No audio files found in the specified directory. Check your 'audio_dir' path in the config.")
        return

    if not all_features:
        print("No valid audio features extracted. Check your audio files.")
        return

    # Convert to DataFrame and save the features as CSV
    df_features = pd.DataFrame(all_features)
    output_csv = os.path.join(processed_data_dir, "audio_features.csv")
    df_features.to_csv(output_csv, index=False)
    print(f"✅ Audio features saved to: {output_csv}")

if __name__ == "__main__":
    main()