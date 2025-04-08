import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def main():
    """
    1. For each CSV in arousal folder (e.g., '104.csv'), find matching valence CSV in valence folder.
    2. Read each file, which has columns like sample_15000ms, sample_15500ms, etc.
    3. Melt (pivot) each file to get a DataFrame with columns: [rater_row, time_ms, arousal/valence].
    4. (Optional) Average across all rater rows at each time_ms.
    5. Merge arousal and valence on time_ms.
    6. Add track_id (parsed from filename).
    7. Concatenate all tracks into one big DataFrame.
    8. (Optional) Merge with track-level audio features on track_id if desired.
    """

    # --- Hard-coded paths (adjust as needed) ---
    arousal_dir = r"E:\DM LAB Proj\music_emotion_recognition\dataset\DEAM\DEAM_Annotations\annotations\annotations per each rater\dynamic (per second annotations)\arousal"
    valence_dir = r"E:\DM LAB Proj\music_emotion_recognition\dataset\DEAM\DEAM_Annotations\annotations\annotations per each rater\dynamic (per second annotations)\valence"

    # Output for the merged per-second data
    dynamic_out_csv = r"E:\DM LAB Proj\music_emotion_recognition\data\processed_data\dynamic_arousal_valence.csv"

    # (Optional) If you want to merge with audio_features.csv:
    audio_features_csv = r"E:\DM LAB Proj\music_emotion_recognition\data\processed_data\audio_features.csv"
    final_out_csv = r"E:\DM LAB Proj\music_emotion_recognition\data\processed_data\dynamic_arousal_valence_features.csv"

    # --- Collect final results for all tracks ---
    all_tracks = []

    # List CSVs in the arousal folder
    arousal_files = [f for f in os.listdir(arousal_dir) if f.endswith(".csv")]

    for afile in arousal_files:
        # e.g. afile = "104.csv"
        arousal_path = os.path.join(arousal_dir, afile)
        valence_path = os.path.join(valence_dir, afile)  # matching filename in valence folder

        if not os.path.exists(valence_path):
            print(f"[WARNING] No matching valence file for {arousal_path}. Skipping.")
            continue

        # Parse track_id from filename (e.g. "104.csv" -> 104)
        track_id_str = os.path.splitext(afile)[0]
        try:
            track_id = int(track_id_str)
        except ValueError:
            print(f"[ERROR] Could not parse integer track_id from filename '{afile}'")
            continue

        # --- Read arousal CSV & pivot ---
        df_ar = pd.read_csv(arousal_path)
        if df_ar.empty:
            print(f"[WARNING] Arousal CSV {arousal_path} is empty. Skipping.")
            continue

        # Melt the DataFrame: each column -> row with column name as "variable" and cell value as "value"
        # Example: "sample_15000ms" -> one row with variable="sample_15000ms", value=some_arousal_value
        df_ar_melted = df_ar.melt(var_name="time_str", value_name="arousal")

        # Now parse time in ms from time_str like "sample_15000ms" -> 15000
        df_ar_melted["time_ms"] = df_ar_melted["time_str"].str.extract(r'sample_(\d+)ms').astype(float)

        # (Optional) If there are multiple rater rows, we can group by time_ms to get an average
        df_ar_melted = df_ar_melted.groupby("time_ms", as_index=False)["arousal"].mean()

        # --- Read valence CSV & pivot ---
        df_val = pd.read_csv(valence_path)
        if df_val.empty:
            print(f"[WARNING] Valence CSV {valence_path} is empty. Skipping.")
            continue

        df_val_melted = df_val.melt(var_name="time_str", value_name="valence")
        df_val_melted["time_ms"] = df_val_melted["time_str"].str.extract(r'sample_(\d+)ms').astype(float)
        df_val_melted = df_val_melted.groupby("time_ms", as_index=False)["valence"].mean()

        # --- Merge arousal & valence on time_ms ---
        df_merged = pd.merge(df_ar_melted, df_val_melted, on="time_ms", how="inner")
        df_merged["track_id"] = track_id

        # Sort by time if desired
        df_merged.sort_values(by="time_ms", inplace=True)
        df_merged.reset_index(drop=True, inplace=True)

        # Add to list
        all_tracks.append(df_merged)

    # --- Combine all tracks ---
    if not all_tracks:
        print("[ERROR] No valid per-second files processed.")
        return

    df_all = pd.concat(all_tracks, ignore_index=True)
    print(f"[INFO] Combined dynamic data shape: {df_all.shape}")

    # Save the dynamic (arousal + valence) data
    os.makedirs(os.path.dirname(dynamic_out_csv), exist_ok=True)
    df_all.to_csv(dynamic_out_csv, index=False)
    print(f"✅ Dynamic arousal+valence data saved to: {dynamic_out_csv}")

    # ----------------------------------------------------------------
    # (Optional) Merge with audio_features.csv if you want to attach
    # track-level features to each time_ms row
    # ----------------------------------------------------------------
    if os.path.exists(audio_features_csv):
        print(f"[INFO] Merging with audio features: {audio_features_csv}")
        df_audio = pd.read_csv(audio_features_csv)

        # If your audio_features.csv has "filename" -> parse track_id
        if "filename" in df_audio.columns:
            def parse_track_id(fn):
                try:
                    return int(os.path.splitext(fn)[0])
                except:
                    return None
            df_audio["track_id"] = df_audio["filename"].apply(parse_track_id)

        # Merge on track_id
        df_final = pd.merge(df_all, df_audio, on="track_id", how="inner")
        print(f"[INFO] After merging with audio features: {df_final.shape}")

        # Example: scale numeric columns except these
        skip_cols = {"track_id", "time_ms", "arousal", "valence", "filename"}
        feature_cols = [c for c in df_final.columns if c not in skip_cols]

        if feature_cols:
            print("[INFO] Scaling numeric audio features (repeated per time step).")
            scaler = StandardScaler()
            df_final[feature_cols] = scaler.fit_transform(df_final[feature_cols])
        else:
            print("[WARNING] No columns found for scaling. Skipping scaling.")

        df_final.to_csv(final_out_csv, index=False)
        print(f"✅ Dynamic + Audio features data saved to: {final_out_csv}")
    else:
        print("[INFO] audio_features.csv not found; skipping audio merge.")

if __name__ == "__main__":
    main()
