import pandas as pd
import joblib
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Load reference data once (shared between API and batch prediction)
product_stats = pd.read_csv(ARTIFACTS_DIR / "product_stockout_stats.csv")
replacement_stats = pd.read_csv(ARTIFACTS_DIR / "replacement_stats.csv")
supplier_stats = pd.read_csv(ARTIFACTS_DIR / "supplier_reliability.csv")
primary_supplier = pd.read_csv(ARTIFACTS_DIR / "product_primary_supplier.csv")

product_stockout_rate = dict(
    zip(product_stats["product_code"], product_stats["stockout_rate"])
)
replacement_intensity_map = dict(
    zip(replacement_stats["product_code"], replacement_stats["replacement_intensity"])
)
replacement_stockout_rate_map = dict(
    zip(
        replacement_stats["product_code"],
        replacement_stats["replacement_stockout_rate"],
    )
)
supplier_reliability_map = dict(
    zip(supplier_stats["supplier_id"], supplier_stats["avg_fulfillment_rate"])
)
product_primary_supplier_map = dict(
    zip(primary_supplier["product_code"], primary_supplier["supplier_id"])
)


def load_model():
    """Load the trained LightGBM model."""
    return joblib.load(ARTIFACTS_DIR / "stockout_prediction_model.pkl")


def engineer_features(order_data: dict) -> pd.DataFrame:
    """
    Engineer model features from a single order line description.

    order_data expects keys like:
      - product_code, order_qty, order_date, lead_time_days
      - plant, sales_unit, customer_number (optional)
    """
    order_date = pd.to_datetime(order_data.get("order_date", datetime.now()))
    product_code = order_data.get("product_code")

    # Product & replacement stats
    prod_hist_stockout = product_stockout_rate.get(product_code, 0.05)
    repl_intensity = replacement_intensity_map.get(product_code, 0.0)
    repl_stockout_rate = replacement_stockout_rate_map.get(product_code, 0.0)

    # Supplier reliability (can be overridden in payload)
    supplier_reliability = order_data.get("supplier_reliability")
    if supplier_reliability is None:
        supplier_id = product_primary_supplier_map.get(product_code)
        supplier_reliability = supplier_reliability_map.get(supplier_id, 0.95)

    features = {
        # Time features
        "order_month": order_date.month,
        "order_day_of_week": order_date.dayofweek,
        "order_hour": order_date.hour,
        "order_week": order_date.isocalendar()[1],
        "lead_time_days": order_data.get("lead_time_days", 1),
        "is_weekend": int(order_date.dayofweek in [5, 6]),
        "is_monday": int(order_date.dayofweek == 0),
        "is_friday": int(order_date.dayofweek == 4),
        "quarter": (order_date.month - 1) // 3 + 1,
        "is_holiday_season": int(order_date.month in [11, 12, 1]),
        "is_rush_hour": int(order_date.hour in [7, 8, 9, 10, 11]),
        # Product & replacement features
        "product_historical_stockout_rate": prod_hist_stockout,
        "product_order_frequency": order_data.get("product_order_frequency", 100),
        "product_avg_order_qty": order_data.get("product_avg_order_qty", 10),
        "order_qty": order_data.get("order_qty", 1),
        "is_large_order": order_data.get("is_large_order", 0),
        "replacement_intensity": repl_intensity,
        "replacement_stockout_rate": repl_stockout_rate,
        # Customer features (fallback defaults)
        "customer_order_frequency": order_data.get("customer_order_frequency", 50),
        "customer_avg_lead_time": order_data.get("customer_avg_lead_time", 1.5),
        # Warehouse features (fallback defaults)
        "plant_stockout_rate": order_data.get("plant_stockout_rate", 0.05),
        "storage_stockout_rate": order_data.get("storage_stockout_rate", 0.05),
        "warehouse_stockout_rate": order_data.get("warehouse_stockout_rate", 0.05),
        # Supplier features
        "supplier_reliability": supplier_reliability,
    }

    ordered_cols = [
        "order_month",
        "order_day_of_week",
        "order_hour",
        "order_week",
        "lead_time_days",
        "is_weekend",
        "is_monday",
        "is_friday",
        "quarter",
        "is_holiday_season",
        "is_rush_hour",
        "product_historical_stockout_rate",
        "product_order_frequency",
        "product_avg_order_qty",
        "is_large_order",
        "order_qty",
        "replacement_intensity",
        "replacement_stockout_rate",
        "customer_order_frequency",
        "customer_avg_lead_time",
        "plant_stockout_rate",
        "storage_stockout_rate",
        "warehouse_stockout_rate",
        "supplier_reliability",
    ]

    df = pd.DataFrame([features])
    return df[ordered_cols]


