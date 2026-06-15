# AI-Driven Transformer Quality Framework

This repository contains the source code for the MCA AI & ML major project:

**AI-Driven Transformer Quality Framework via Zero-Trust Verification and Isolation Forest Anomalies**

## Project Overview

The project demonstrates a prototype quality control system for power transformer manufacturing. It uses synthetic industrial telemetry, SHA-256 based zero-trust payload signing, an Isolation Forest anomaly detection model, and a Flask dashboard for supervisor quality-gating.

## Folder Structure

```text
AI-Driven-Transformer-Quality-Framework/
├── app.py
├── generate_data.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
├── models/
├── templates/
│   └── dashboard.html
├── static/
└── docs/
```

## Installation

```bash
pip install -r requirements.txt
```

## Run the Project

Generate data:

```bash
python generate_data.py
```

Train the model:

```bash
python train_model.py
```

Start the dashboard:

```bash
python app.py
```

Open the browser at:

```text
http://127.0.0.1:5000
```

## Key Modules

- `generate_data.py` creates synthetic transformer manufacturing telemetry.
- `train_model.py` trains the Isolation Forest anomaly detection model.
- `app.py` runs the Flask web dashboard and quality-gate router.
- `templates/dashboard.html` provides the supervisor interface.

## Author

Neelam Pravin Rane  
MCA - Machine Learning and Artificial Intelligence  
Amity University Online
