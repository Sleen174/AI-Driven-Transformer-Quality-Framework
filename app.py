"""
Central Server Gateway Application

Flask dashboard for Zero-Trust verified transformer quality scoring.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

import joblib
import pandas as pd
from flask import Flask, render_template, request


DATA_PATH = os.path.join("data", "transformer_data.csv")
MODEL_PATH = os.path.join("models", "isolation_forest_model.pkl")
METRICS_PATH = os.path.join("models", "metrics.pkl")
PRIVATE_KEY = "factory-edge-node-private-key"

app = Flask(__name__)


def ensure_assets() -> None:
    if not os.path.exists(DATA_PATH):
        subprocess.check_call([sys.executable, "generate_data.py"])
    if not os.path.exists(MODEL_PATH) or not os.path.exists(METRICS_PATH):
        subprocess.check_call([sys.executable, "train_model.py"])


def sign_payload(payload: dict) -> str:
    message = json.dumps(payload, sort_keys=True) + PRIVATE_KEY
    return hashlib.sha256(message.encode("utf-8")).hexdigest()


def verify_signature(row: dict) -> bool:
    signature = row.get("signature")
    payload = {k: row[k] for k in row if k not in ["signature", "actual_label"]}
    return sign_payload(payload) == signature


def run_quality_compliance_check(values: dict) -> str:
    if values["burr_microns"] > 20:
        return "Quality Hold: CRGO burr exceeds 20 micron limit."
    if values["humidity_rh"] > 65:
        return "Quality Hold: Winding humidity exposure is high."
    if values["tank_bolt_torque_nm"] < 160:
        return "Quality Hold: Tanking torque below acceptable range."
    if values["hydrogen_ppm"] > 25 or values["acetylene_ppm"] > 3:
        return "Quality Hold: DGA gas indicator above normal range."
    if values["transit_impact_g"] > 3:
        return "Quality Hold: Transit impact above safe limit."
    return "Compliance Passed: No rule-based quality hold."


@app.route("/", methods=["GET", "POST"])
def dashboard():
    ensure_assets()
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    features = bundle["features"]
    metrics = joblib.load(METRICS_PATH)

    df = pd.read_csv(DATA_PATH)
    sample = df.tail(10)
    anomaly_count = int(sum(model.predict(df[features]) == -1))

    prediction = None
    status = None

    if request.method == "POST":
        form_values = {feature: float(request.form[feature]) for feature in features}
        input_data = pd.DataFrame([form_values], columns=features)
        raw_output = model.predict(input_data)[0]
        prediction = "Anomaly Detected / Quality Hold" if raw_output == -1 else "Normal / Release Passed"
        status = run_quality_compliance_check(form_values)

    return render_template(
        "dashboard.html",
        sample=sample.to_dict("records"),
        metrics=metrics,
        risk_count=anomaly_count,
        total=len(df),
        features=features,
        prediction=prediction,
        status=status,
    )


if __name__ == "__main__":
    app.run(debug=True)
