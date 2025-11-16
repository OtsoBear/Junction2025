"""
Zero-Fail Logistics - Real-time Stockout Prediction API
Junction 2025 Challenge - Valio Aimo

Flask API for real-time stockout probability prediction per order line.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import os
import warnings
import pandas as pd
import json

from prediction_core import load_model, engineer_features

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes and origins

print("Loading model and reference data...")
model = load_model()
print("✓ Model loaded successfully")

# Load products catalog
print("Loading products catalog...")
products_df = pd.read_csv("products.csv", sep=";", dtype=str)
# Convert product code to string and strip whitespace
products_df['Tuotekoodi'] = products_df['Tuotekoodi'].astype(str).str.strip()
print(f"✓ Loaded {len(products_df)} products from catalog")

# Load test examples from eval dataset
print("Loading test examples from eval dataset...")
try:
    with open("test_examples.json", "r") as f:
        test_examples = json.load(f)
    print("✓ Loaded test examples from eval dataset")
except FileNotFoundError:
    print("⚠ test_examples.json not found, using fallback examples")
    test_examples = None

# Load order examples from eval dataset
print("Loading order examples from eval dataset...")
try:
    with open("order_examples.json", "r") as f:
        order_examples = json.load(f)
    print(f"✓ Loaded {len(order_examples)} order examples from eval dataset")
except FileNotFoundError:
    print("⚠ order_examples.json not found")
    order_examples = {}


def get_product_info(product_code):
    """
    Retrieve product information from the products catalog.
    
    Args:
        product_code: Product code to look up
        
    Returns:
        dict: Product information or None if not found
    """
    product_code_str = str(product_code).strip()
    product_row = products_df[products_df['Tuotekoodi'] == product_code_str]
    
    if not product_row.empty:
        # Convert row to dict and clean up values
        product_info = product_row.iloc[0].to_dict()
        # Convert None/NaN values to empty strings for JSON serialization
        product_info = {k: (v if pd.notna(v) else "") for k, v in product_info.items()}
        return product_info
    return None


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model_loaded": True})


@app.route("/get-example-data", methods=["GET"])
def get_example_data():
    """
    Get example data that can be used with the prediction API.
    Returns examples including the cake bakery order scenario.
    """
    current_time = datetime.now().isoformat()
    
    # Load cake order example
    try:
        with open("cake_order_api_request.json", "r") as f:
            cake_order_data = json.load(f)
        cake_orders = cake_order_data.get("orders", [])
    except FileNotFoundError:
        cake_orders = []
    
    # Load multi-customer order example
    try:
        with open("multi_customer_orders.json", "r") as f:
            multi_customer_data = json.load(f)
        multi_customer_orders = multi_customer_data.get("orders", [])
    except FileNotFoundError:
        multi_customer_orders = []
    
    if order_examples and len(order_examples) >= 3:
        # Use actual orders from the eval dataset with real order numbers
        order_keys = list(order_examples.keys())
        
        # Get 3 different real orders for examples
        order_1 = order_examples[order_keys[0]]
        order_2 = order_examples[order_keys[1]]
        order_3 = order_examples[order_keys[2]]
        
        # Single order example - use first item from first order
        single_item = order_1["items"][0].copy()
        single_example = {
            "product_code": single_item["product_code"],
            "order_qty": single_item["order_qty"],
            "customer_number": order_1["customer_number"],
            "order_date": current_time,
            "lead_time_days": 1,
            "plant": "30588",
            "sales_unit": "kpl"
        }
        
        # Batch orders example - use first item from each of the 3 orders
        batch_orders = []
        for i, (order_key, order_data) in enumerate([(order_keys[0], order_1), (order_keys[1], order_2), (order_keys[2], order_3)]):
            item = order_data["items"][0].copy()
            batch_orders.append({
                "order_number": order_key,
                "product_code": item["product_code"],
                "order_qty": item["order_qty"],
                "customer_number": order_data["customer_number"],
                "order_date": current_time,
                "lead_time_days": 1,
                "plant": "30588",
                "sales_unit": "kpl"
            })
        
        example_data = {
            "multi_customer_orders": {
                "description": "🏢 Multiple customers with various order sizes - demonstrates risk across different customer types",
                "data": {
                    "orders": multi_customer_orders
                },
                "note": "Includes 5 customers (bakery, cafe, restaurant, hotel, school) with 16 total order lines. Shows how risk varies by customer type and order size. School district has highest risk items due to large milk orders.",
                "highlights": {
                    "total_customers": 5,
                    "total_order_lines": len(multi_customer_orders),
                    "customer_types": ["Bakery (7 items, 1 HIGH risk)", "Cafe (2 items)", "Restaurant (3 items)", "Hotel (2 items, 1 HIGH risk)", "School (2 items, 2 HIGH risk)"],
                    "expected_high_risk_count": 4
                }
            },
            "cake_bakery_order": {
                "description": "🎂 Single bakery order for 60-80 cakes - focused HIGH risk example",
                "data": {
                    "orders": cake_orders
                },
                "note": "Single customer (bakery) with 7 product lines. Butter order (95 units, 3x normal) triggers HIGH risk at 42.8%. Other ingredients show MEDIUM risk (15-30%).",
                "highlights": {
                    "high_risk_product": "Butter (product 8230) - 95 units",
                    "expected_high_risk_count": 1,
                    "total_products": len(cake_orders)
                }
            },
            "single_order_example": {
                "description": f"Example from real order {order_keys[0]} in eval dataset",
                "data": single_example,
                "note": f"This item is from real order {order_keys[0]}. Use /list-items-in-order to see all items in this order."
            },
            "batch_orders_example": {
                "description": "Example for /predict_batch endpoint with real order numbers from eval data",
                "data": {
                    "orders": batch_orders
                },
                "note": "These are real orders from eval dataset. Use /list-items-in-order with any order_number to see full order details."
            },
            "available_orders": {
                "description": "Sample of available order numbers you can query with /list-items-in-order",
                "order_numbers": order_keys[:10],
                "total_available": len(order_examples)
            },
            "usage_instructions": {
                "multi_customer_prediction": "POST the 'data' from multi_customer_orders to /predict_batch to analyze multiple customers",
                "cake_order_prediction": "POST the 'data' from cake_bakery_order to /predict_batch to see focused bakery risk",
                "single_prediction": "POST the 'data' from single_order_example to /predict",
                "batch_prediction": "POST the 'data' from batch_orders_example to /predict_batch",
                "list_order_items": "POST {\"order_number\": \"10000000\"} to /list-items-in-order to see all items in that order"
            }
        }
    else:
        # Fallback to hardcoded examples if test_examples.json not available
        example_data = {
            "single_order_example": {
                "description": "Example for /predict endpoint (single order)",
                "data": {
                    "product_code": "8043409",
                    "order_qty": 15.0,
                    "customer_number": "33345",
                    "order_date": datetime.now().isoformat(),
                    "lead_time_days": 1,
                    "plant": "30588",
                    "sales_unit": "kpl"
                }
            },
            "batch_orders_example": {
                "description": "Example for /predict_batch endpoint (multiple orders)",
                "data": {
                    "orders": [
                        {
                            "order_number": "ORD001",
                            "product_code": "8023043",
                            "order_qty": 20.0,
                            "customer_number": "33345",
                            "order_date": datetime.now().isoformat(),
                            "lead_time_days": 0,
                            "plant": "30588",
                            "sales_unit": "kpl"
                        },
                        {
                            "order_number": "ORD002",
                            "product_code": "8054679",
                            "order_qty": 1.0,
                            "customer_number": "12345",
                            "order_date": datetime.now().isoformat(),
                            "lead_time_days": 2,
                            "plant": "30588",
                            "sales_unit": "kpl"
                        },
                        {
                            "order_number": "ORD003",
                            "product_code": "8079322",
                            "order_qty": 5.0,
                            "customer_number": "54321",
                            "order_date": datetime.now().isoformat(),
                            "lead_time_days": 1,
                            "plant": "30588",
                            "sales_unit": "kpl"
                        }
                    ]
                }
            },
            "usage_instructions": {
                "single_prediction": "POST the 'data' from single_order_example to /predict",
                "batch_prediction": "POST the 'data' from batch_orders_example to /predict_batch"
            }
        }
    return jsonify(example_data)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict stockout probability for an order line.
    
    Expected JSON input:
    {
        "product_code": "410397",
        "order_qty": 5.0,
        "customer_number": "33345",
        "order_date": "2024-09-15T08:30:00",
        "lead_time_days": 1,
        "plant": "30588",
        "sales_unit": "ST"
    }
    """
    try:
        order_data = request.get_json()

        # Engineer features
        features = engineer_features(order_data)

        # Make prediction
        stockout_probability = model.predict(features)[0]

        # Determine risk level based on calibrated low-range probabilities
        if stockout_probability < 0.15:
            risk_level = "LOW"
            action = "Proceed normally"
        elif stockout_probability < 0.30:
            risk_level = "MEDIUM"
            action = "Monitor inventory"
        else:
            risk_level = "HIGH"
            action = "Proactive customer contact recommended"

        # Get product information
        product_code = order_data.get("product_code")
        product_info = get_product_info(product_code)
        
        response = {
            "stockout_probability": float(stockout_probability),
            "risk_level": risk_level,
            "recommended_action": action,
            "product_code": product_code,
            "order_qty": order_data.get("order_qty"),
            "timestamp": datetime.now().isoformat(),
        }
        
        # Add product information if found
        if product_info:
            response["product_info"] = product_info
        else:
            response["product_info"] = {"warning": f"Product code {product_code} not found in catalog"}
        
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    """
    Predict stockout probability for multiple order lines.
    
    Expected JSON input:
    {
        "orders": [
            {"product_code": "410397", "order_qty": 5.0, ...},
            {"product_code": "406468", "order_qty": 1.0, ...}
        ]
    }
    """
    try:
        data = request.get_json()
        orders = data.get("orders", [])

        predictions = []
        order_groups = {}  # Group predictions by order_number
        
        for order in orders:
            features = engineer_features(order)
            stockout_prob = model.predict(features)[0]

            if stockout_prob < 0.15:
                risk_level = "LOW"
            elif stockout_prob < 0.30:
                risk_level = "MEDIUM"
            else:
                risk_level = "HIGH"

            product_code = order.get("product_code")
            product_info = get_product_info(product_code)
            order_number = order.get("order_number", "UNKNOWN")
            
            prediction_result = {
                "product_code": product_code,
                "order_qty": order.get("order_qty"),
                "stockout_probability": float(stockout_prob),
                "risk_level": risk_level,
            }
            
            # Add product information if found
            if product_info:
                prediction_result["product_info"] = product_info
            else:
                prediction_result["product_info"] = {"warning": f"Product code {product_code} not found in catalog"}
            
            predictions.append(prediction_result)
            
            # Group by order_number
            if order_number not in order_groups:
                order_groups[order_number] = {
                    "order_number": order_number,
                    "customer_number": order.get("customer_number"),
                    "items": [],
                    "high_risk_count": 0,
                    "medium_risk_count": 0,
                    "low_risk_count": 0,
                    "max_risk_probability": 0.0
                }
            
            order_groups[order_number]["items"].append(prediction_result)
            order_groups[order_number]["max_risk_probability"] = max(
                order_groups[order_number]["max_risk_probability"],
                float(stockout_prob)
            )
            
            # Update risk counts
            if risk_level == "HIGH":
                order_groups[order_number]["high_risk_count"] += 1
            elif risk_level == "MEDIUM":
                order_groups[order_number]["medium_risk_count"] += 1
            else:
                order_groups[order_number]["low_risk_count"] += 1

        # Sort items within each order by risk (highest first)
        for order_data in order_groups.values():
            order_data["items"].sort(key=lambda x: x["stockout_probability"], reverse=True)
            order_data["total_items"] = len(order_data["items"])

        # Convert to list and sort by highest risk in each order
        orders_list = list(order_groups.values())
        orders_list.sort(key=lambda x: x["max_risk_probability"], reverse=True)

        # Overall statistics
        total_high_risk = sum(o["high_risk_count"] for o in orders_list)
        total_medium_risk = sum(o["medium_risk_count"] for o in orders_list)
        total_low_risk = sum(o["low_risk_count"] for o in orders_list)

        response = {
            "summary": {
                "total_orders": len(orders_list),
                "total_items": len(predictions),
                "high_risk_items": total_high_risk,
                "medium_risk_items": total_medium_risk,
                "low_risk_items": total_low_risk
            },
            "orders": orders_list,
            "timestamp": datetime.now().isoformat(),
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/list-items-in-order", methods=["POST"])
def list_items_in_order():
    """
    List all items in a specific order from the eval dataset.
    
    Expected JSON input:
    {
        "order_number": "10000000"
    }
    """
    try:
        data = request.get_json()
        order_number = str(data.get("order_number", ""))
        
        if not order_number:
            return jsonify({"error": "order_number is required"}), 400
        
        # Check if order exists in our examples
        if order_number not in order_examples:
            available_orders = list(order_examples.keys())
            return jsonify({
                "error": f"Order {order_number} not found in eval dataset",
                "available_orders": available_orders[:10],
                "total_available": len(available_orders)
            }), 404
        
        order_data = order_examples[order_number]
        
        # Enrich items with product information
        enriched_items = []
        for item in order_data["items"]:
            product_code = item["product_code"]
            product_info = get_product_info(product_code)
            
            enriched_item = {
                "product_code": product_code,
                "order_qty": item["order_qty"],
                "actual_stockout": bool(item["is_stockout"]),
                "stockout_status": "Stockout occurred" if item["is_stockout"] else "Fulfilled"
            }
            
            if product_info:
                enriched_item["product_name"] = product_info.get("Tuotenimi", "")
                enriched_item["product_info"] = product_info
            else:
                enriched_item["product_name"] = "Unknown product"
                enriched_item["product_info"] = {"warning": f"Product {product_code} not found in catalog"}
            
            enriched_items.append(enriched_item)
        
        # Calculate summary statistics
        total_items = len(enriched_items)
        stockout_items = sum(1 for item in enriched_items if item["actual_stockout"])
        fulfilled_items = total_items - stockout_items
        stockout_rate = (stockout_items / total_items * 100) if total_items > 0 else 0
        
        response = {
            "order_number": order_number,
            "customer_number": order_data["customer_number"],
            "order_date": order_data["order_date"],
            "summary": {
                "total_items": total_items,
                "fulfilled_items": fulfilled_items,
                "stockout_items": stockout_items,
                "stockout_rate_pct": round(stockout_rate, 2)
            },
            "items": enriched_items,
            "note": "This is real data from the eval/test dataset showing actual outcomes"
        }
        
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🚀 STOCKOUT PREDICTION API")
    print("=" * 80)
    print("\nAvailable endpoints:")
    print("  GET  /health                  - Health check")
    print("  GET  /get-example-data        - Get example data for testing")
    print("  POST /predict                 - Single order prediction")
    print("  POST /predict_batch           - Batch order predictions")
    print("  POST /list-items-in-order     - List all items in an order (eval data)")
    print("\n" + "=" * 80)
    port = int(os.environ.get("PORT", "5000"))
    print(f"\nStarting server on http://localhost:{port}")
    print("Press CTRL+C to stop")
    print("=" * 80 + "\n")

    app.run(host="0.0.0.0", port=port, debug=True)

