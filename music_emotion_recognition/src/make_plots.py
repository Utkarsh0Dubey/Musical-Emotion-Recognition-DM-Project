import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    # --- Hard-coded paths ---
    combined_csv = r"E:\DM LAB Proj\music_emotion_recognition\data\processed_data\combined_dynamic_audio.csv"
    plots_dir = r"E:\DM LAB Proj\music_emotion_recognition\plots"

    # Ensure the plots folder exists
    os.makedirs(plots_dir, exist_ok=True)

    # --- Load the combined CSV ---
    print(f"[INFO] Loading combined data from: {combined_csv}")
    df = pd.read_csv(combined_csv)
    print(f"[DEBUG] Data shape: {df.shape}")

    # --- 1. Correlation Heatmap ---
    plt.figure(figsize=(12, 10))
    # Select numeric columns for correlation
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    heatmap_path = os.path.join(plots_dir, "correlation_heatmap.png")
    plt.savefig(heatmap_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved correlation heatmap to: {heatmap_path}")

    # --- 2. Histograms for Arousal and Valence ---
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(df["arousal"], kde=True, color="skyblue")
    plt.title("Arousal Distribution")

    plt.subplot(1, 2, 2)
    sns.histplot(df["valence"], kde=True, color="salmon")
    plt.title("Valence Distribution")

    hist_path = os.path.join(plots_dir, "arousal_valence_histograms.png")
    plt.savefig(hist_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved arousal/valence histograms to: {hist_path}")

    # --- 3. Scatter Plot: Arousal vs Valence ---
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x="arousal", y="valence", data=df, alpha=0.6)
    plt.title("Arousal vs. Valence")
    plt.xlabel("Arousal")
    plt.ylabel("Valence")
    scatter_path = os.path.join(plots_dir, "arousal_vs_valence_scatter.png")
    plt.savefig(scatter_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved arousal vs. valence scatter plot to: {scatter_path}")

    # --- 4. Distribution of an Audio Feature (e.g., MFCC_1 if exists) ---
    # Check if there's at least one audio feature column (modify as needed)
    possible_audio_features = [col for col in df.columns if "mfcc" in col.lower()]
    if possible_audio_features:
        feature = possible_audio_features[0]
        plt.figure(figsize=(8, 6))
        sns.histplot(df[feature], kde=True, color="purple")
        plt.title(f"Distribution of {feature}")
        feature_path = os.path.join(plots_dir, f"{feature}_distribution.png")
        plt.savefig(feature_path, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Saved {feature} distribution plot to: {feature_path}")
    else:
        print("[WARNING] No audio feature columns found for distribution plot.")

    # --- Additional Plots as Needed ---
    # For example, boxplots or time series plots could be added here.

    print("[INFO] Visualization completed.")


if __name__ == "__main__":
    main()
