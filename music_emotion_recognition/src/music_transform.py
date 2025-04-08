import os
import numpy as np
import librosa
import soundfile as sf
import joblib

# Parameters (must match those used during feature extraction and model training)
SR = 22050
N_MFCC = 13
HOP_LENGTH = 512

# Load the trained emotion recognition model (adjust path as needed)
model_path = "E:/DM LAB Proj/music_emotion_recognition/models/random_forest_model.pkl"
model = joblib.load(model_path)


def extract_features_from_audio_segment(y_segment, sr=SR, n_mfcc=N_MFCC, hop_length=HOP_LENGTH):
    """
    Extract MFCCs and spectral contrast features from an audio segment.
    Returns a 1D NumPy array with 20 features (13 MFCCs + 7 spectral contrast values).
    """
    mfccs = librosa.feature.mfcc(y=y_segment, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    mfccs_mean = mfccs.mean(axis=1)
    spectral_contrast = librosa.feature.spectral_contrast(y=y_segment, sr=sr)
    spectral_contrast_mean = spectral_contrast.mean(axis=1)
    features_vector = np.concatenate([mfccs_mean, spectral_contrast_mean])
    return features_vector


def continuous_mapping(arousal, valence):
    """
    Map predicted emotion values to continuous transformation parameters.

    For pitch shift (in semitones), we map valence to a range of [-3, +3]:
      pitch_shift = 6 * (valence - 0.5)

    For time stretch, we map arousal to a range of [0.8, 1.2]:
      time_stretch = 1 + 0.4 * (arousal - 0.5)
    """
    pitch_shift = 6 * (valence - 0.5)
    time_stretch = 1 + 0.4 * (arousal - 0.5)
    return pitch_shift, time_stretch


def opposite_mapping(arousal, valence):
    """
    Compute the opposite transformation parameters.

    Here, we simply invert the effects:
      - For pitch, invert the sign.
      - For time stretch, subtract the effect from 1.

    Thus, if the normal mapping gives:
      pitch_shift = 6*(valence - 0.5) and time_stretch = 1 + 0.4*(arousal - 0.5),
    the opposite mapping becomes:
      pitch_shift_opposite = -6*(valence - 0.5)
      time_stretch_opposite = 1 - 0.4*(arousal - 0.5)
    """
    pitch_shift = -6 * (valence - 0.5)
    time_stretch = 1 - 0.4 * (arousal - 0.5)
    return pitch_shift, time_stretch


def process_segment(y_segment, sr, pitch_shift, time_stretch):
    """
    Apply pitch shifting and time stretching to the segment.
    """
    y_shifted = librosa.effects.pitch_shift(y_segment, sr=sr, n_steps=pitch_shift)
    y_transformed = librosa.effects.time_stretch(y_shifted, rate=time_stretch)
    return y_transformed


def main():
    # Input and output paths
    input_audio_path = "E:/DM LAB Proj/music_emotion_recognition/input_audio.wav"
    output_audio_normal = "E:/DM LAB Proj/music_emotion_recognition/output_audio_normal.wav"
    output_audio_opposite = "E:/DM LAB Proj/music_emotion_recognition/output_audio_opposite.wav"

    # Load the input audio file
    y, sr = librosa.load(input_audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    print(f"Loaded audio with duration: {duration:.2f} seconds, sample rate: {sr}")

    # Define segment length in seconds (e.g., 10-second segments)
    segment_length = 10
    num_segments = int(np.ceil(duration / segment_length))

    normal_segments = []
    opposite_segments = []

    for i in range(num_segments):
        start_time = i * segment_length
        end_time = min((i + 1) * segment_length, duration)
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        y_segment = y[start_sample:end_sample]

        # Extract features from the segment
        features_vector = extract_features_from_audio_segment(y_segment, sr, N_MFCC, HOP_LENGTH)
        if features_vector.shape[0] != 20:
            print(f"[WARNING] Expected 20 features, got {features_vector.shape[0]} for segment {i + 1}")

        # Prepare feature vector for prediction
        features_vector_reshaped = features_vector.reshape(1, -1)

        # Predict emotional parameters using the model
        predicted_emotion = model.predict(features_vector_reshaped)[0]
        # Assume the model outputs two values: [arousal, valence]
        predicted_arousal = predicted_emotion[0] if isinstance(predicted_emotion,
                                                               (list, np.ndarray)) else predicted_emotion
        predicted_valence = predicted_emotion[1] if isinstance(predicted_emotion, (list, np.ndarray)) and len(
            predicted_emotion) > 1 else predicted_emotion

        print(f"Segment {i + 1}: Predicted Arousal: {predicted_arousal:.2f}, Valence: {predicted_valence:.2f}")

        # Get transformation parameters for normal and opposite mappings
        norm_pitch, norm_time = continuous_mapping(predicted_arousal, predicted_valence)
        opp_pitch, opp_time = opposite_mapping(predicted_arousal, predicted_valence)
        print(f"Segment {i + 1}: Normal -> pitch shift: {norm_pitch:.2f}, time stretch: {norm_time:.2f}")
        print(f"Segment {i + 1}: Opposite -> pitch shift: {opp_pitch:.2f}, time stretch: {opp_time:.2f}")

        # Apply the transformations
        y_norm = process_segment(y_segment, sr, norm_pitch, norm_time)
        y_opp = process_segment(y_segment, sr, opp_pitch, opp_time)

        normal_segments.append(y_norm)
        opposite_segments.append(y_opp)

    # Concatenate segments for both outputs
    y_normal_full = np.concatenate(normal_segments)
    y_opposite_full = np.concatenate(opposite_segments)

    # Save both transformed outputs
    sf.write(output_audio_normal, y_normal_full, sr)
    sf.write(output_audio_opposite, y_opposite_full, sr)

    print(f"Normal transformed audio saved to {output_audio_normal}")
    print(f"Opposite transformed audio saved to {output_audio_opposite}")


if __name__ == "__main__":
    main()
