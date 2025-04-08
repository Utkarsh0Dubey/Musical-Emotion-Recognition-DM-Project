import os
import pandas as pd


def main():
    # --- Hard-coded absolute paths based on your directory structure ---
    # Preprocessed dynamic annotations CSV from DEAM (per second data)
    dynamic_csv = r"E:\DM LAB Proj\music_emotion_recognition\data\processed_data\dynamic_arousal_valence.csv"
    # Audio features CSV extracted from the audio files
    audio_features_csv = r"E:\DM LAB Proj\music_emotion_recognition\data\processed_data\audio_features.csv"
    # Output path for the combined CSV
    output_csv = r"E:\DM LAB Proj\music_emotion_recognition\data\processed_data\combined_dynamic_audio.csv"

    # --- Debug prints for file paths ---
    print(f"[DEBUG] Dynamic annotations CSV: {dynamic_csv}")
    print(f"[DEBUG] Audio features CSV:     {audio_features_csv}")

    # --- Verify file existence ---
    if not os.path.exists(dynamic_csv):
        print(f"[ERROR] Dynamic annotations CSV not found at: {dynamic_csv}")
        return
    if not os.path.exists(audio_features_csv):
        print(f"[ERROR] Audio features CSV not found at: {audio_features_csv}")
        return

    # --- Load the CSV files ---
    print("[INFO] Loading dynamic annotations...")
    df_dynamic = pd.read_csv(dynamic_csv)
    print(f"[DEBUG] Dynamic annotations shape: {df_dynamic.shape}")

    print("[INFO] Loading audio features...")
    df_audio = pd.read_csv(audio_features_csv)
    print(f"[DEBUG] Audio features shape: {df_audio.shape}")

    # --- Ensure the audio features DataFrame has a 'track_id' column ---
    if "filename" in df_audio.columns and "track_id" not in df_audio.columns:
        def extract_track_id(fn):
            try:
                return int(os.path.splitext(fn)[0])
            except Exception as e:
                print(f"[ERROR] Unable to extract track_id from '{fn}': {e}")
                return None

        df_audio["track_id"] = df_audio["filename"].apply(extract_track_id)
        print("[DEBUG] Sample audio features with track_id:")
        print(df_audio.head(3))

    # --- Check that both DataFrames have a 'track_id' column ---
    if "track_id" not in df_dynamic.columns:
        print("[ERROR] 'track_id' column not found in dynamic annotations CSV.")
        return
    if "track_id" not in df_audio.columns:
        print("[ERROR] 'track_id' column not found in audio features CSV after parsing.")
        return

    # --- Merge dynamic annotations with audio features on 'track_id'
    # Each time-slice row in the dynamic CSV will get the corresponding track's audio features.
    print("[INFO] Merging dynamic annotations with audio features on 'track_id'...")
    df_combined = pd.merge(df_dynamic, df_audio, on="track_id", how="inner")
    print(f"[DEBUG] Combined DataFrame shape: {df_combined.shape}")

    # --- Save the combined DataFrame ---
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_combined.to_csv(output_csv, index=False)
    print(f"✅ Combined dynamic annotations and audio features saved to: {output_csv}")


if __name__ == "__main__":
    main()
