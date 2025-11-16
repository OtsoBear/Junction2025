"""
Batch prediction script: identify high-risk order lines for the latest
requested delivery date in the historical dataset.

Usage:
    python src/predict_upcoming_orders.py

Outputs:
    artifacts/upcoming_risk_predictions.csv - sorted by stockout_probability desc
"""

import pandas as pd
from datetime import datetime
from pathlib import Path

from prediction_core import BASE_DIR, DATA_DIR, ARTIFACTS_DIR, load_model, engineer_features


def main():
    print("Loading trained model...")
    model = load_model()

    print("Loading full sales & deliveries data...")
    sales = pd.read_csv(
        DATA_DIR / "valio_aimo_sales_and_deliveries_junction_2025.csv"
    )

    # Parse dates
    sales["order_created_date"] = pd.to_datetime(
        sales["order_created_date"], format="%Y-%m-%d"
    )
    sales["requested_delivery_date"] = pd.to_datetime(
        sales["requested_delivery_date"], format="%Y-%m-%d"
    )

    # Use the latest requested delivery date as proxy for "upcoming"
    target_date = sales["requested_delivery_date"].max()
    print(f"Targeting requested delivery date: {target_date.date()}")

    upcoming = sales[sales["requested_delivery_date"] == target_date].copy()
    print(f"Found {len(upcoming):,} order lines for that delivery date.")

    # Compute lead time in days
    upcoming["lead_time_days"] = (
        upcoming["requested_delivery_date"] - upcoming["order_created_date"]
    ).dt.days

    # Build feature rows and predict
    records = []
    probs = []

    print("Scoring upcoming order lines...")
    for _, row in upcoming.iterrows():
        order_dt = datetime.combine(
            row["order_created_date"], datetime.min.time()
        )
        order_dt = order_dt.replace(
            hour=int(str(row["order_created_time"])[:2] or 0)
        )

        order_data = {
            "product_code": row["product_code"],
            "order_qty": float(row["order_qty"]),
            "customer_number": row["customer_number"],
            "order_date": order_dt.isoformat(),
            "lead_time_days": int(row["lead_time_days"]),
            "plant": row["plant"],
            "sales_unit": row["sales_unit"],
        }

        features = engineer_features(order_data)
        prob = float(model.predict(features)[0])

        probs.append(prob)
        records.append(
            {
                "order_number": row["order_number"],
                "order_row_number": row["order_row_number"],
                "customer_number": row["customer_number"],
                "product_code": row["product_code"],
                "order_qty": row["order_qty"],
                "sales_unit": row["sales_unit"],
                "requested_delivery_date": row["requested_delivery_date"],
                "plant": row["plant"],
                "storage_location": row["storage_location"],
                "stockout_probability": prob,
            }
        )

    result = pd.DataFrame(records)

    # Simple risk banding aligned with API
    def to_risk_level(p: float) -> str:
        if p < 0.10:
            return "LOW"
        elif p < 0.20:
            return "MEDIUM"
        else:
            return "HIGH"

    result["risk_level"] = result["stockout_probability"].apply(to_risk_level)

    # Sort by risk
    result = result.sort_values("stockout_probability", ascending=False)

    output_path = ARTIFACTS_DIR / "upcoming_risk_predictions.csv"
    result.to_csv(output_path, index=False)
    print(
        f"Saved upcoming risk predictions to {output_path}\n"
        f"Top 10 rows:\n{result.head(10)}"
    )


if __name__ == "__main__":
    main()


