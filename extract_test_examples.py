"""
Extract example data from the test set for the API
"""
import pandas as pd
import json
from pathlib import Path

# Load the feature-engineered data
df = pd.read_parquet("artifacts/features_for_modeling.parquet")

# Use the same time-based split as in training (last 10% as test set)
split_date = df["order_created_date"].quantile(0.9)
test_df = df[df["order_created_date"] >= split_date].copy()

print(f"Test set size: {len(test_df):,} samples")
print(f"Stockout rate in test set: {test_df['is_stockout'].mean()*100:.2f}%")

# Get a diverse set of examples
# 1. One example with low stockout risk (is_stockout = 0)
# 2. One example with high stockout risk (is_stockout = 1)
# 3. One borderline case

low_risk = test_df[test_df['is_stockout'] == 0].iloc[0]
high_risk = test_df[test_df['is_stockout'] == 1].iloc[0]
borderline = test_df.iloc[len(test_df)//2]

examples = {
    'low_risk': low_risk,
    'high_risk': high_risk,
    'borderline': borderline
}

# Check what columns are available
print("\nAvailable columns in test set:")
print(test_df.columns.tolist())

# Extract the fields needed for the API
api_fields = ['product_code', 'order_qty', 'customer_number', 'lead_time_days', 'plant', 'sales_unit', 'order_created_date']

print("\n" + "="*80)
print("EXAMPLE DATA EXTRACTED FROM TEST SET")
print("="*80)

for label, example in examples.items():
    print(f"\n{label.upper().replace('_', ' ')}:")
    for field in api_fields:
        if field in example.index:
            value = example[field]
            # Convert timestamps to ISO format
            if field == 'order_created_date':
                value = pd.Timestamp(value).isoformat()
            print(f"  {field}: {value}")
        else:
            print(f"  {field}: <not in dataset>")
    print(f"  Actual stockout: {example['is_stockout']}")

# Save to a JSON file for easy reference
output = {}
for label, example in examples.items():
    data = {}
    for field in api_fields:
        if field in example.index:
            value = example[field]
            if field == 'order_created_date':
                value = pd.Timestamp(value).isoformat()
            elif pd.isna(value):
                value = None
            elif isinstance(value, (int, float)):
                data[field] = float(value) if isinstance(value, float) else int(value)
            else:
                data[field] = str(value)
        else:
            # Use reasonable defaults for missing fields
            if field == 'product_code':
                data[field] = "8043409"  # Default example product
            elif field == 'customer_number':
                data[field] = "33345"  # Default customer
            elif field == 'plant':
                data[field] = "30588"  # Default plant
            elif field == 'sales_unit':
                data[field] = "kpl"  # Default unit
    data['actual_stockout'] = int(example['is_stockout'])
    output[label] = data

with open('test_examples.json', 'w') as f:
    json.dump(output, f, indent=2)

print("\n✓ Saved examples to test_examples.json")