"""
Zero-Fail Logistics - Feature Engineering for Stockout Prediction
Junction 2025 Challenge - Valio Aimo

This script builds rich features for the ML model based on:
- Supplier reliability
- Seasonality patterns
- Time windows
- Historical product performance
- Customer patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

print("=" * 80)
print("FEATURE ENGINEERING FOR STOCKOUT PREDICTION")
print("=" * 80)

BASE_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = BASE_DIR / "artifacts"

print("\n[1/5] Loading processed data and aggregates...")
sales_df = pd.read_parquet(ARTIFACTS_DIR / "processed_sales_data.parquet")
print(f"✓ Loaded {len(sales_df):,} sales records")

# Load product- and replacement-level aggregates computed in exploration
product_stats = pd.read_csv(ARTIFACTS_DIR / "product_stockout_stats.csv")
replacement_stats = pd.read_csv(ARTIFACTS_DIR / "replacement_stats.csv")

product_stockout_map = dict(
    zip(product_stats["product_code"], product_stats["stockout_rate"])
)

print("\n[2/5] Engineering Time-Based Features...")

# Already have: order_month, order_day_of_week, order_hour, order_week, lead_time_days

# Add more temporal features
sales_df["is_weekend"] = sales_df["order_day_of_week"].isin([5, 6]).astype(int)
sales_df["is_monday"] = (sales_df["order_day_of_week"] == 0).astype(int)
sales_df["is_friday"] = (sales_df["order_day_of_week"] == 4).astype(int)

# Quarter and season
sales_df["quarter"] = sales_df["order_created_date"].dt.quarter
sales_df["is_holiday_season"] = sales_df["order_month"].isin([11, 12, 1]).astype(
    int
)  # Nov-Jan

# Morning/afternoon/evening
sales_df["time_of_day"] = pd.cut(
    sales_df["order_hour"],
    bins=[0, 6, 12, 18, 24],
    labels=["night", "morning", "afternoon", "evening"],
)
sales_df["is_rush_hour"] = sales_df["order_hour"].isin([7, 8, 9, 10, 11]).astype(int)

print("✓ Created time-based features")

print("\n[3/5] Engineering Product- and Replacement-Based Features...")

# Sort by date to make future time-based splits sensible
sales_df = sales_df.sort_values("order_created_date")

# Overall historical stockout rate per product (acts as long-term prior)
sales_df["product_historical_stockout_rate"] = sales_df["product_code"].map(
    product_stockout_map
)

# Product order frequency (popularity)
product_order_count = sales_df.groupby("product_code")["order_number"].transform(
    "count"
)
sales_df["product_order_frequency"] = product_order_count

# Average order quantity per product
product_avg_qty = sales_df.groupby("product_code")["order_qty"].transform("mean")
sales_df["product_avg_order_qty"] = product_avg_qty

# Is this order larger than average for this product?
sales_df["is_large_order"] = (
    sales_df["order_qty"] > sales_df["product_avg_order_qty"]
).astype(int)

# Merge replacement statistics at product level
sales_df = sales_df.merge(
    replacement_stats[
        ["product_code", "replacement_intensity", "replacement_stockout_rate"]
    ],
    on="product_code",
    how="left",
)
sales_df["replacement_intensity"] = sales_df["replacement_intensity"].fillna(0.0)
sales_df["replacement_stockout_rate"] = sales_df[
    "replacement_stockout_rate"
].fillna(0.0)

print("✓ Created product and replacement-based features")

print("\n[4/5] Engineering Customer-Based Features...")

# Customer order frequency
customer_order_count = (
    sales_df.groupby("customer_number")["order_number"].transform("count")
)
sales_df["customer_order_frequency"] = customer_order_count

# Customer segment (by order volume)
sales_df["customer_segment"] = pd.cut(
    sales_df["customer_order_frequency"],
    bins=[0, 10, 50, 200, float("inf")],
    labels=["small", "medium", "large", "vip"],
)

# Average lead time per customer
customer_avg_lead = sales_df.groupby("customer_number")["lead_time_days"].transform(
    "mean"
)
sales_df["customer_avg_lead_time"] = customer_avg_lead

print("✓ Created customer-based features")

print("\n[5/6] Engineering Warehouse/Plant Features...")

# Stockout rate by plant
plant_stockout_rate = sales_df.groupby("plant")["is_stockout"].transform("mean")
sales_df["plant_stockout_rate"] = plant_stockout_rate

# Stockout rate by storage location
storage_stockout_rate = sales_df.groupby("storage_location")["is_stockout"].transform(
    "mean"
)
sales_df["storage_stockout_rate"] = storage_stockout_rate

# Stockout rate by warehouse
warehouse_stockout_rate = sales_df.groupby("warehouse_number")["is_stockout"].transform(
    "mean"
)
sales_df["warehouse_stockout_rate"] = warehouse_stockout_rate

print("✓ Created warehouse/plant features")

print("\n[6/6] Engineering Freshness SLA Features...")

# Temperature-based features already in data from product metadata merge
# These include: temperature_condition, is_perishable, freshness_category

# Create interaction: short lead time + perishable = higher risk
sales_df["perishable_short_lead"] = (
    (sales_df["is_perishable"] == 1) & (sales_df["lead_time_days"] <= 1)
).astype(int)

# Perishable orders during rush hour (higher handling pressure)
sales_df["perishable_rush_hour"] = (
    (sales_df["is_perishable"] == 1) & (sales_df["is_rush_hour"] == 1)
).astype(int)

# Weekend perishable orders (may have limited staff/stock)
sales_df["perishable_weekend"] = (
    (sales_df["is_perishable"] == 1) & (sales_df["is_weekend"] == 1)
).astype(int)

print("✓ Created freshness SLA features")

print("\n" + "=" * 80)
print("FEATURE SUMMARY")
print("=" * 80)

# Select features for modeling (all numeric; categorical string cols excluded)
feature_columns = [
    # Time features
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
    # Product & replacement features
    "product_historical_stockout_rate",
    "product_order_frequency",
    "product_avg_order_qty",
    "is_large_order",
    "order_qty",
    "replacement_intensity",
    "replacement_stockout_rate",
    # Customer features
    "customer_order_frequency",
    "customer_avg_lead_time",
    # Warehouse features
    "plant_stockout_rate",
    "storage_stockout_rate",
    "warehouse_stockout_rate",
    # Supplier features
    "supplier_reliability",
    # Freshness SLA features
    "temperature_condition",
    "is_perishable",
    "perishable_short_lead",
    "perishable_rush_hour",
    "perishable_weekend",
]

# Keep only rows with all features available
sales_df_features = sales_df[
    feature_columns + ["is_stockout", "order_created_date"]
].copy()

# Handle missing values
print("\nMissing values before cleaning:")
print(sales_df_features.isnull().sum())

# Fill missing supplier reliability with median
sales_df_features["supplier_reliability"].fillna(
    sales_df_features["supplier_reliability"].median(), inplace=True
)

# Drop any remaining rows with missing values
sales_df_features.dropna(inplace=True)

print(f"\n✓ Final dataset shape: {sales_df_features.shape}")
print(f"  Features: {len(feature_columns)}")
print(f"  Samples: {len(sales_df_features):,}")
print(f"  Stockout rate: {sales_df_features['is_stockout'].mean()*100:.2f}%")

print("\n📋 Feature List:")
for i, feat in enumerate(feature_columns, 1):
    print(f"  {i:2d}. {feat}")

# Save feature-engineered dataset
print("\n💾 Saving feature-engineered dataset...")
sales_df_features.to_parquet(
    ARTIFACTS_DIR / "features_for_modeling.parquet", index=False
)
print("✓ Saved to artifacts/features_for_modeling.parquet")

print("\n" + "=" * 80)
print("✅ FEATURE ENGINEERING COMPLETE!")
print("=" * 80)
print("\nNext step: Run 03_train_model.py to build the prediction model")

