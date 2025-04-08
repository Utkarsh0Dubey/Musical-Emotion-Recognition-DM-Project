import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import joblib


def load_data():
    # Hard-coded path to your combined dynamic audio CSV
    data_path = r"E:\DM LAB Proj\music_emotion_recognition\data\processed_data\combined_dynamic_audio.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Combined CSV not found at: {data_path}")
    df = pd.read_csv(data_path)
    return df


def prepare_data(df):
    # Exclude identifier and target columns; assume targets are 'arousal' and 'valence'
    skip_cols = {"track_id", "time_ms", "arousal", "valence", "filename"}
    feature_cols = [col for col in df.columns if col not in skip_cols]
    target_cols = ["arousal", "valence"]

    if not feature_cols:
        raise ValueError("No feature columns found for training!")

    X = df[feature_cols]
    y = df[target_cols]

    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_mlp_regressor(X_train, y_train, X_test, y_test, models_dir):
    print("[MLP] Training MLP Regressor (scikit-learn deep learning)...")
    mlp = MLPRegressor(hidden_layer_sizes=(100, 50, 25),
                       activation='relu', solver='adam',
                       max_iter=500, random_state=42)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    mse_val = mean_squared_error(y_test, y_pred)
    r2_val = r2_score(y_test, y_pred)

    model_path = os.path.join(models_dir, "mlp_regressor_model.pkl")
    joblib.dump(mlp, model_path)
    print(f"[MLP] MLP Regressor - Test MSE: {mse_val:.4f}, R^2: {r2_val:.4f}, saved to: {model_path}")
    return mse_val, r2_val


def train_linear_regression(X_train, y_train, X_test, y_test, models_dir):
    print("[LR] Training Linear Regression model...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    mse_val = mean_squared_error(y_test, y_pred)
    r2_val = r2_score(y_test, y_pred)

    model_path = os.path.join(models_dir, "linear_regression_model.pkl")
    joblib.dump(lr, model_path)
    print(f"[LR] Linear Regression - Test MSE: {mse_val:.4f}, R^2: {r2_val:.4f}, saved to: {model_path}")
    return mse_val, r2_val


def train_decision_tree(X_train, y_train, X_test, y_test, models_dir):
    print("[DT] Training Decision Tree model...")
    dt = DecisionTreeRegressor(random_state=42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    mse_val = mean_squared_error(y_test, y_pred)
    r2_val = r2_score(y_test, y_pred)

    model_path = os.path.join(models_dir, "decision_tree_model.pkl")
    joblib.dump(dt, model_path)
    print(f"[DT] Decision Tree - Test MSE: {mse_val:.4f}, R^2: {r2_val:.4f}, saved to: {model_path}")
    return mse_val, r2_val


def train_random_forest(X_train, y_train, X_test, y_test, models_dir):
    print("[RF] Training Random Forest model...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    mse_val = mean_squared_error(y_test, y_pred)
    r2_val = r2_score(y_test, y_pred)

    model_path = os.path.join(models_dir, "random_forest_model.pkl")
    joblib.dump(rf, model_path)
    print(f"[RF] Random Forest - Test MSE: {mse_val:.4f}, R^2: {r2_val:.4f}, saved to: {model_path}")
    return mse_val, r2_val


def main():
    # Create models directory if it doesn't exist
    models_dir = r"E:\DM LAB Proj\music_emotion_recognition\models"
    os.makedirs(models_dir, exist_ok=True)

    print("[INFO] Loading and preparing data...")
    df = load_data()
    X_train, X_test, y_train, y_test = prepare_data(df)
    print(f"[DEBUG] Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

    print("\n===== MODEL COMPARISON (scikit-learn only) =====")

    # Train and evaluate each model using scikit-learn
    mlp_mse, mlp_r2 = train_mlp_regressor(X_train, y_train, X_test, y_test, models_dir)
    lr_mse, lr_r2 = train_linear_regression(X_train, y_train, X_test, y_test, models_dir)
    dt_mse, dt_r2 = train_decision_tree(X_train, y_train, X_test, y_test, models_dir)
    rf_mse, rf_r2 = train_random_forest(X_train, y_train, X_test, y_test, models_dir)

    print("\n===== SUMMARY OF RESULTS =====")
    print(f"MLP Regressor (scikit-learn) - Test MSE: {mlp_mse:.4f}, R^2: {mlp_r2:.4f}")
    print(f"Linear Regression          - Test MSE: {lr_mse:.4f}, R^2: {lr_r2:.4f}")
    print(f"Decision Tree              - Test MSE: {dt_mse:.4f}, R^2: {dt_r2:.4f}")
    print(f"Random Forest              - Test MSE: {rf_mse:.4f}, R^2: {rf_r2:.4f}")


if __name__ == "__main__":
    main()
