#!/bin/bash
# Simple end-to-end pipeline: full data → features → model

set -e

cd "$(dirname "$0")"

echo "=== Running full Zero-Fail Logistics pipeline ==="

echo "[0/4] Extracting product metadata (temperature/freshness)..."
.venv/bin/python src/00_extract_product_metadata.py

echo "[1/4] Data exploration & aggregates (full dataset)..."
.venv/bin/python src/01_data_exploration.py

echo "[2/4] Feature engineering..."
.venv/bin/python src/02_feature_engineering.py

echo "[3/4] Model training..."
.venv/bin/python src/03_train_model.py

echo "=== Pipeline complete. Model saved as stockout_prediction_model.pkl ==="

