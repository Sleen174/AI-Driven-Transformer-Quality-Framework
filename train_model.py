"""
Machine Learning Anomaly Suite

Trains an Isolation Forest model for transformer manufacturing quality anomaly detection.
"""

from __future__ import annotations

import os

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


DATA_PATH = os.path.join("data", "transformer_data.csv")
MODEL_PATH = os.path.join("models", "isolation_forest_model.pkl")
METRICS_PATH = os.path.join("models", "metrics.pkl")

FEATURES = [
    "burr_microns",
    "humidity_rh",
    "winding_tension_n",
    "vpd_moisture_rate",
    "tank_bolt_torque_nm",
    "oil_fill_rate_lpm",
    "hydrogen_ppm",
    "acetylene_ppm",
    "ethylene_ppm",
    "transit_impact_g",
]


def train() -> None:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Dataset not found. Run: python generate_data.py")

    os.makedirs("models", exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    x = df[FEATURES]
    y_true = df["actual_label"]

    model = IsolationForest(
        n_estimators=150,
        contamination=0.05,
        random_state=42,
        max_samples="auto",
    )
    model.fit(x)

    y_pred = model.predict(x)

    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 3),
        "precision": round(float(precision_score(y_true, y_pred, pos_label=-1)), 3),
        "recall": round(float(recall_score(y_true, y_pred, pos_label=-1)), 3),
        "f1_score": round(float(f1_score(y_true, y_pred, pos_label=-1)), 3),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    joblib.dump({"model": model, "features": FEATURES}, MODEL_PATH)
    joblib.dump(metrics, METRICS_PATH)

    print("Model trained successfully.")
    print(f"Saved model: {MODEL_PATH}")
    print("Metrics:", metrics)


if __name__ == "__main__":
    train()
