"""
Production Data Simulation Engine

Creates synthetic transformer manufacturing telemetry for:
1. Core stacking
2. Winding assembly
3. Vapor Phase Drying (VPD)
4. Tanking and oil filling
5. Final testing and dissolved gas analysis

Author: Neelam Pravin Rane
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


OUTPUT_PATH = os.path.join("data", "transformer_data.csv")
PRIVATE_KEY = "factory-edge-node-private-key"


def sign_payload(payload: dict) -> str:
    """Create SHA-256 signature for zero-trust telemetry validation."""
    message = json.dumps(payload, sort_keys=True) + PRIVATE_KEY
    return hashlib.sha256(message.encode("utf-8")).hexdigest()


def generate_dataset(rows: int = 1000, anomaly_rate: float = 0.05, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []
    start_time = datetime(2026, 1, 1, 8, 0, 0)

    for i in range(rows):
        is_anomaly = rng.random() < anomaly_rate

        # Normal operating ranges
        burr_microns = rng.normal(12, 3)
        humidity_rh = rng.normal(45, 8)
        winding_tension_n = rng.normal(520, 35)
        vpd_moisture_rate = rng.normal(0.45, 0.10)
        tank_bolt_torque_nm = rng.normal(185, 12)
        oil_fill_rate_lpm = rng.normal(85, 8)
        hydrogen_ppm = rng.normal(8, 3)
        acetylene_ppm = rng.normal(0.8, 0.4)
        ethylene_ppm = rng.normal(4, 1.5)
        transit_impact_g = rng.normal(1.2, 0.35)

        if is_anomaly:
            anomaly_type = rng.choice(
                ["core_burr", "humidity", "vpd", "tanking", "dga", "transit"]
            )
            if anomaly_type == "core_burr":
                burr_microns = rng.normal(28, 4)
            elif anomaly_type == "humidity":
                humidity_rh = rng.normal(72, 7)
                winding_tension_n = rng.normal(455, 30)
            elif anomaly_type == "vpd":
                vpd_moisture_rate = rng.normal(0.85, 0.12)
            elif anomaly_type == "tanking":
                tank_bolt_torque_nm = rng.normal(140, 15)
                oil_fill_rate_lpm = rng.normal(115, 10)
            elif anomaly_type == "dga":
                hydrogen_ppm = rng.normal(38, 8)
                acetylene_ppm = rng.normal(6, 2)
                ethylene_ppm = rng.normal(18, 5)
            elif anomaly_type == "transit":
                transit_impact_g = rng.normal(4.0, 0.9)

        payload = {
            "asset_id": f"TR-{2026}-{i + 1:04d}",
            "timestamp": (start_time + timedelta(minutes=i * 15)).isoformat(),
            "burr_microns": round(float(burr_microns), 3),
            "humidity_rh": round(float(humidity_rh), 3),
            "winding_tension_n": round(float(winding_tension_n), 3),
            "vpd_moisture_rate": round(float(vpd_moisture_rate), 3),
            "tank_bolt_torque_nm": round(float(tank_bolt_torque_nm), 3),
            "oil_fill_rate_lpm": round(float(oil_fill_rate_lpm), 3),
            "hydrogen_ppm": round(float(hydrogen_ppm), 3),
            "acetylene_ppm": round(float(acetylene_ppm), 3),
            "ethylene_ppm": round(float(ethylene_ppm), 3),
            "transit_impact_g": round(float(transit_impact_g), 3),
        }

        payload["signature"] = sign_payload(payload)
        payload["actual_label"] = -1 if is_anomaly else 1
        records.append(payload)

    return pd.DataFrame(records)


def main() -> None:
    os.makedirs("data", exist_ok=True)
    df = generate_dataset()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Dataset generated successfully: {OUTPUT_PATH}")
    print(df.head())


if __name__ == "__main__":
    main()
