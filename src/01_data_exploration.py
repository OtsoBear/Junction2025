"""
Zero-Fail Logistics - Data Exploration & Stockout Analysis
Junction 2025 Challenge - Valio Aimo

This script explores the data to identify stockout patterns and build features.
It also computes supplier reliability, primary supplier per product, and
replacement-order statistics that are reused downstream for ML features and API.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

print("=" * 80)
print("VALIO AIMO - OUT-OF-STOCK PREDICTION - DATA EXPLORATION")
print("=" * 80)

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

print("\n[1/7] Loading Sales and Deliveries Data (full dataset)...")
# Load full sales data in chunks to handle large file
sales_chunks = []
chunk_size = 500_000
total_rows = 0

for i, chunk in enumerate(
    pd.read_csv(
        DATA_DIR / "valio_aimo_sales_and_deliveries_junction_2025.csv",
        chunksize=chunk_size,
    )
):
    sales_chunks.append(chunk)
    total_rows += len(chunk)
    if i % 10 == 0:
        print(f"  Loaded {total_rows:,} rows...")

sales_df = pd.concat(sales_chunks, ignore_index=True)
print(f"✓ Loaded {len(sales_df):,} sales order lines (full period)")
print(f"  Columns: {sales_df.columns.tolist()}")

print("\n[2/7] Loading Replacement Orders Data...")
replacement_df = pd.read_csv(
    DATA_DIR / "valio_aimo_replacement_orders_junction_2025.csv"
)
print(f"✓ Loaded {len(replacement_df):,} replacement orders")

print("\n[3/7] Loading Purchase Orders Data...")
purchase_df = pd.read_csv(
    DATA_DIR / "valio_aimo_purchases_junction_2025.csv"
)
print(f"✓ Loaded {len(purchase_df):,} purchase order lines")

print("\n[3.5/7] Loading Product Metadata (temperature/freshness)...")
product_metadata = pd.read_csv(ARTIFACTS_DIR / "product_metadata.csv")
print(f"✓ Loaded metadata for {len(product_metadata):,} products")

print("\n" + "=" * 80)
print("STOCKOUT ANALYSIS")
print("=" * 80)

# Identify stockouts in sales data
print("\n[4/7] Identifying Stockout Events in Sales...")

# Convert date columns
sales_df["order_created_date"] = pd.to_datetime(
    sales_df["order_created_date"], format="%Y-%m-%d"
)
sales_df["requested_delivery_date"] = pd.to_datetime(
    sales_df["requested_delivery_date"], format="%Y-%m-%d"
)
sales_df["picking_confirmed_date"] = pd.to_datetime(
    sales_df["picking_confirmed_date"], format="%Y-%m-%d"
)

# Define stockout: when picked quantity < ordered quantity
sales_df["is_stockout"] = (sales_df["picking_picked_qty"] < sales_df["order_qty"]).astype(
    int
)
sales_df["shortage_qty"] = sales_df["order_qty"] - sales_df["picking_picked_qty"]
sales_df["shortage_pct"] = (sales_df["shortage_qty"] / sales_df["order_qty"]) * 100

# Calculate additional features
sales_df["lead_time_days"] = (
    sales_df["requested_delivery_date"] - sales_df["order_created_date"]
).dt.days
sales_df["order_hour"] = pd.to_numeric(
    sales_df["order_created_time"].astype(str).str[:2], errors="coerce"
)
sales_df["order_day_of_week"] = sales_df["order_created_date"].dt.dayofweek
sales_df["order_month"] = sales_df["order_created_date"].dt.month
sales_df["order_week"] = sales_df["order_created_date"].dt.isocalendar().week

print(f"\n📊 STOCKOUT STATISTICS:")
print(f"   Total order lines: {len(sales_df):,}")
overall_stockout_rate = sales_df["is_stockout"].mean() * 100
print(
    f"   Stockout events: {sales_df['is_stockout'].sum():,} "
    f"({overall_stockout_rate:.2f}%)"
)
print(
    f"   Fully fulfilled: {(sales_df['is_stockout']==0).sum():,} "
    f"({(1 - sales_df['is_stockout'].mean())*100:.2f}%)"
)
print(
    f"   Total shortage quantity: {sales_df['shortage_qty'].sum():,.0f} units"
)

# Analyze by product
print("\n📦 TOP 20 PRODUCTS WITH HIGHEST STOCKOUT RATES (min 20 orders):")
product_stockout = (
    sales_df.groupby("product_code").agg(
        is_stockout_sum=("is_stockout", "sum"),
        total_orders=("is_stockout", "count"),
        stockout_rate=("is_stockout", "mean"),
        total_shortage=("shortage_qty", "sum"),
    )
).round(4)
product_stockout = product_stockout[
    product_stockout["total_orders"] >= 20
].sort_values("stockout_rate", ascending=False)
print(product_stockout.head(20))

print("\n[5/7] Analyzing Supplier Reliability (from Purchase Orders)...")

# Convert purchase dates
purchase_df["po_created_date"] = pd.to_datetime(
    purchase_df["po_created_date"], format="%Y-%m-%d"
)
purchase_df["requested_delivery_date"] = pd.to_datetime(
    purchase_df["requested_delivery_date"], format="%Y-%m-%d"
)

# Calculate supplier reliability metrics
purchase_df["is_shortage"] = (
    purchase_df["received_qty"] < purchase_df["ordered_qty"]
).astype(int)
purchase_df["fulfillment_rate"] = (
    purchase_df["received_qty"] / purchase_df["ordered_qty"]
)

supplier_reliability = (
    purchase_df.groupby("customer_number")
    .agg(
        shortage_rate=("is_shortage", "mean"),
        avg_fulfillment_rate=("fulfillment_rate", "mean"),
        total_orders=("order_number", "count"),
    )
    .round(4)
)
supplier_reliability = supplier_reliability[
    supplier_reliability["total_orders"] >= 10
]
supplier_reliability.index.name = "supplier_id"

print(f"\n📊 SUPPLIER RELIABILITY:")
print(f"   Total suppliers (>=10 orders): {len(supplier_reliability)}")
print(
    f"   Avg supplier fulfillment rate: "
    f"{supplier_reliability['avg_fulfillment_rate'].mean():.2%}"
)
print("\nTop 10 Most Reliable Suppliers:")
print(
    supplier_reliability.sort_values(
        "avg_fulfillment_rate", ascending=False
    ).head(10)
)
print("\nTop 10 Least Reliable Suppliers:")
print(
    supplier_reliability.sort_values(
        "avg_fulfillment_rate", ascending=True
    ).head(10)
)

# Determine primary supplier per product based on purchase frequency
print("\n[6/7] Deriving primary supplier per product...")
product_supplier_orders = (
    purchase_df.groupby(["product_code", "customer_number"])["order_number"]
    .count()
    .reset_index(name="order_count")
)

primary_supplier = (
    product_supplier_orders.sort_values(
        ["product_code", "order_count"], ascending=[True, False]
    )
    .drop_duplicates("product_code")
    .rename(columns={"customer_number": "supplier_id"})
)

print(f"   Computed primary supplier for {len(primary_supplier):,} products")

# Map supplier reliability back to products via primary supplier
supplier_reliability_reset = supplier_reliability.reset_index()
supplier_rel_map = dict(
    zip(
        supplier_reliability_reset["supplier_id"],
        supplier_reliability_reset["avg_fulfillment_rate"],
    )
)
product_to_supplier = dict(
    zip(primary_supplier["product_code"], primary_supplier["supplier_id"])
)

sales_df["supplier_id"] = sales_df["product_code"].map(product_to_supplier)
sales_df["supplier_reliability"] = sales_df["supplier_id"].map(supplier_rel_map)

print(
    f"   Supplier reliability mapped to "
    f"{sales_df['supplier_reliability'].notnull().sum():,} sales lines"
)

# Merge product metadata (temperature/freshness)
print("\n[6.5/7] Merging product metadata for freshness SLA features...")
sales_df = sales_df.merge(
    product_metadata,
    on="product_code",
    how="left"
)
print(
    f"   Product metadata merged: "
    f"{sales_df['is_perishable'].notnull().sum():,} sales lines with temperature data"
)
print(
    f"   Perishable products in orders: "
    f"{sales_df['is_perishable'].sum():,} ({sales_df['is_perishable'].mean()*100:.1f}%)"
)

print("\n[7/7] Analyzing Temporal & Replacement Patterns...")

# Stockout rate by month
monthly_stockout = sales_df.groupby("order_month")["is_stockout"].agg(
    stockouts="sum", total="count", rate="mean"
)
print("\n📅 MONTHLY STOCKOUT RATES:")
print(monthly_stockout)

# Stockout rate by day of week
dow_stockout = sales_df.groupby("order_day_of_week")["is_stockout"].agg(
    stockouts="sum", total="count", rate="mean"
)
dow_stockout.index = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
print("\n📅 DAY OF WEEK STOCKOUT RATES:")
print(dow_stockout)

# Stockout rate by order hour
hour_stockout = sales_df.groupby("order_hour")["is_stockout"].agg(
    stockouts="sum", total="count", rate="mean"
)
print("\n🕐 HOURLY STOCKOUT RATES:")
print(hour_stockout)

# Replacement-order based product statistics
print("\n[7/7] Computing replacement-order statistics...")

replacement_df["order_created_date"] = pd.to_datetime(
    replacement_df["order_created_date"], format="%Y-%m-%d"
)
replacement_df["requested_delivery_date"] = pd.to_datetime(
    replacement_df["requested_delivery_date"], format="%Y-%m-%d"
)

replacement_df["is_stockout"] = (
    replacement_df["picking_picked_qty"] < replacement_df["order_qty"]
).astype(int)
replacement_df["shortage_qty"] = (
    replacement_df["order_qty"] - replacement_df["picking_picked_qty"]
)

replacement_stats = (
    replacement_df.groupby("product_code")
    .agg(
        replacement_orders=("order_number", "count"),
        replacement_stockout_rate=("is_stockout", "mean"),
        replacement_total_shortage=("shortage_qty", "sum"),
    )
    .reset_index()
)

sales_counts = (
    sales_df.groupby("product_code")["order_number"]
    .count()
    .reset_index(name="sales_orders")
)
replacement_stats = replacement_stats.merge(
    sales_counts, on="product_code", how="left"
)
replacement_stats["replacement_intensity"] = (
    replacement_stats["replacement_orders"]
    / replacement_stats["sales_orders"].replace({0: np.nan})
)
replacement_stats["replacement_intensity"] = replacement_stats[
    "replacement_intensity"
].fillna(0.0)

print(
    f"   Replacement stats computed for "
    f"{len(replacement_stats):,} products with replacements"
)

print("\n" + "=" * 80)
print("SAVING PROCESSED DATA & AGGREGATES")
print("=" * 80)

# Save processed data for modeling
print("\nSaving processed sales data...")
sales_df.to_parquet(ARTIFACTS_DIR / "processed_sales_data.parquet", index=False)
print("✓ Saved to artifacts/processed_sales_data.parquet")

print("\nSaving supplier reliability metrics...")
supplier_reliability_reset.to_csv(
    ARTIFACTS_DIR / "supplier_reliability.csv", index=False
)
print("✓ Saved to artifacts/supplier_reliability.csv")

print("\nSaving primary supplier mapping...")
primary_supplier.to_csv(
    ARTIFACTS_DIR / "product_primary_supplier.csv", index=False
)
print("✓ Saved to artifacts/product_primary_supplier.csv")

print("\nSaving product stockout statistics...")
product_stockout.reset_index().to_csv(
    ARTIFACTS_DIR / "product_stockout_stats.csv", index=False
)
print("✓ Saved to artifacts/product_stockout_stats.csv")

print("\nSaving replacement statistics...")
replacement_stats.to_csv(ARTIFACTS_DIR / "replacement_stats.csv", index=False)
print("✓ Saved to artifacts/replacement_stats.csv")

print("\n" + "=" * 80)
print("✅ EXPLORATION COMPLETE!")
print("=" * 80)
print("\nKey Insights:")
print(f"1. Overall stockout rate: {overall_stockout_rate:.2f}%")
print(f"2. Analyzed {len(sales_df):,} order lines")
print(f"3. Identified {len(supplier_reliability):,} suppliers (>=10 orders)")
print(
    f"4. Replacement stats available for "
    f"{replacement_stats['product_code'].nunique():,} products"
)
print("\nNext step: Run 02_feature_engineering.py")

