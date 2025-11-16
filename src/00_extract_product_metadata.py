s"""
Extract product metadata from JSON for freshness SLA features
Junction 2025 Challenge - Valio Aimo
"""

import pandas as pd
import json
from pathlib import Path

print("=" * 80)
print("EXTRACTING PRODUCT METADATA FOR FRESHNESS FEATURES")
print("=" * 80)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

print("\n[1/2] Loading product data from JSON...")
with open(DATA_DIR / "valio_aimo_product_data_junction_2025.json", "r") as f:
    products = json.load(f)

print(f"✓ Loaded {len(products):,} products")

print("\n[2/2] Extracting temperature and freshness metadata...")

product_metadata = []

for product in products:
    product_code = product.get("salesUnitGtin")
    
    # Temperature data
    temp_condition = product.get("temperatureCondition")
    
    # From synkkaData
    synkka_data = product.get("synkkaData", {})
    min_temp = synkka_data.get("minTemperature")
    max_temp = synkka_data.get("maxTemperature")
    
    # Calculate temperature range (indicator of sensitivity)
    if min_temp is not None and max_temp is not None:
        temp_range = max_temp - min_temp
    else:
        temp_range = None
    
    # Determine freshness category based on temperature condition
    if temp_condition is not None:
        if temp_condition <= 3:  # Frozen/very cold
            freshness_category = "high_sensitivity"
        elif temp_condition <= 5:  # Refrigerated
            freshness_category = "medium_sensitivity"
        else:  # Room temp or ambient
            freshness_category = "low_sensitivity"
    else:
        freshness_category = "unknown"
    
    # Is perishable (requires temperature control)
    is_perishable = 1 if temp_condition and temp_condition <= 5 else 0
    
    product_metadata.append({
        "product_code": product_code,
        "temperature_condition": temp_condition if temp_condition else 8,  # Default to ambient
        "min_temperature": min_temp,
        "max_temperature": max_temp,
        "temperature_range": temp_range,
        "freshness_category": freshness_category,
        "is_perishable": is_perishable,
    })

metadata_df = pd.DataFrame(product_metadata)

print(f"\n✓ Extracted metadata for {len(metadata_df):,} products")
print(f"\nFreshness Category Distribution:")
print(metadata_df["freshness_category"].value_counts())
print(f"\nPerishable Products: {metadata_df['is_perishable'].sum():,} ({metadata_df['is_perishable'].mean()*100:.1f}%)")

print("\nTemperature Condition Distribution:")
print(metadata_df["temperature_condition"].value_counts().sort_index())

# Save metadata
output_path = ARTIFACTS_DIR / "product_metadata.csv"
metadata_df.to_csv(output_path, index=False)
print(f"\n💾 Saved product metadata to {output_path}")

print("\n" + "=" * 80)
print("✅ PRODUCT METADATA EXTRACTION COMPLETE!")
print("=" * 80)