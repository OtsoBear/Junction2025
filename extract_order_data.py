"""
Extract order data from the test set for the list-items-in-order endpoint
Only select orders with products that exist in the catalog
"""
import pandas as pd
import json
from pathlib import Path

# Load products catalog
print("Loading products catalog...")
products_df = pd.read_csv("products.csv", sep=";", dtype=str)
products_df['Tuotekoodi'] = products_df['Tuotekoodi'].astype(str).str.strip()
valid_product_codes = set(products_df['Tuotekoodi'].tolist())
print(f"✓ Loaded {len(valid_product_codes)} valid product codes from catalog")

# Load the processed sales data which has order information
print("\nLoading processed sales data...")
df = pd.read_parquet("artifacts/processed_sales_data.parquet")

print(f"Total orders: {len(df):,}")
print(f"Columns: {df.columns.tolist()}")

# Use time-based split to get test set (same as in training - last 10%)
split_date = df["order_created_date"].quantile(0.9)
test_df = df[df["order_created_date"] >= split_date].copy()

print(f"\nTest set: {len(test_df):,} records")
print(f"Date range: {test_df['order_created_date'].min()} to {test_df['order_created_date'].max()}")

# Filter for products in catalog
test_df['product_code_str'] = test_df['product_code'].astype(str).str.strip()
test_df_valid = test_df[test_df['product_code_str'].isin(valid_product_codes)].copy()
print(f"Records with products in catalog: {len(test_df_valid):,}")

# Check if we have order number information
if 'order_number' in test_df_valid.columns:
    print(f"\nUnique orders in test set (with catalog products): {test_df_valid['order_number'].nunique()}")
    
    # Get orders where ALL items are in the catalog
    order_counts = test_df_valid.groupby('order_number').size()
    multi_item_orders = order_counts[order_counts > 1].index
    
    # Filter to orders where all items have valid products
    orders_with_all_valid = []
    for order_num in multi_item_orders:
        order_items = test_df[test_df['order_number'] == order_num]
        all_items_valid = all(str(pc).strip() in valid_product_codes for pc in order_items['product_code'])
        if all_items_valid:
            orders_with_all_valid.append(order_num)
    
    print(f"Orders with ALL items in catalog: {len(orders_with_all_valid)}")
    
    if len(orders_with_all_valid) > 0:
        sample_order = orders_with_all_valid[0]
        print(f"\nSample multi-item order from test set: {sample_order}")
        order_items = test_df[test_df['order_number'] == sample_order]
        print(f"Items in order: {len(order_items)}")
        print("\nOrder details:")
        print(order_items[['product_code', 'order_qty', 'customer_number', 'order_created_date']].head())
        
        # Save sample orders to JSON - use orders from test set with catalog products
        orders_data = {}
        for i, order_num in enumerate(orders_with_all_valid[:5]):  # Get first 5 multi-item orders from test set
            order_items = test_df[test_df['order_number'] == order_num]
            items_list = []
            for _, item in order_items.iterrows():
                item_data = {
                    'product_code': str(item.get('product_code', '')),
                    'order_qty': float(item.get('order_qty', 0)),
                    'is_stockout': int(item.get('is_stockout', 0))
                }
                items_list.append(item_data)
            orders_data[str(order_num)] = {
                'order_number': str(order_num),
                'customer_number': str(order_items.iloc[0].get('customer_number', '')),
                'order_date': str(order_items.iloc[0].get('order_created_date', '')),
                'items': items_list,
                'total_items': len(items_list)
            }
        
        with open('order_examples.json', 'w') as f:
            json.dump(orders_data, f, indent=2)
        print(f"\n✓ Saved {len(orders_data)} order examples to order_examples.json")
    else:
        print("\nNo multi-item orders found")
else:
    # If no order_number, create synthetic order grouping
    print("\nNo order_number column found. Grouping by customer and date...")
    df['order_group'] = df['customer_number'].astype(str) + '_' + df['order_created_date'].astype(str)
    
    order_counts = df.groupby('order_group').size()
    multi_item_orders = order_counts[order_counts > 1].index
    
    print(f"Created {len(multi_item_orders)} multi-item order groups")
    
    # Save sample orders to JSON
    orders_data = {}
    for i, order_group in enumerate(multi_item_orders[:5]):  # Get first 5 multi-item orders
        order_items = df[df['order_group'] == order_group]
        items_list = []
        for _, item in order_items.iterrows():
            item_data = {
                'product_code': str(item.get('product_code', 'UNKNOWN')),
                'order_qty': float(item.get('order_qty', 0)),
                'is_stockout': int(item.get('is_stockout', 0))
            }
            items_list.append(item_data)
        
        order_num = f"ORD_{i+1:05d}"
        orders_data[order_num] = {
            'order_number': order_num,
            'customer_number': str(order_items.iloc[0].get('customer_number', '')),
            'order_date': str(order_items.iloc[0].get('order_created_date', '')),
            'items': items_list,
            'total_items': len(items_list)
        }
    
    with open('order_examples.json', 'w') as f:
        json.dump(orders_data, f, indent=2)
    print(f"\n✓ Saved {len(orders_data)} order examples to order_examples.json")