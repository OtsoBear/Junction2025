"""
Quick API Test Script
Tests the prediction API with sample orders
"""

import requests
import json
import os
from datetime import datetime

API_URL = os.environ.get("API_URL", "http://localhost:5000")

def test_health():
    """Test health endpoint"""
    print("\n" + "="*60)
    print("Testing Health Endpoint")
    print("="*60)
    
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        print("❌ Error: API is not running")
        print("Start the API with: python3 04_prediction_api.py")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_single_prediction():
    """Test single prediction endpoint"""
    print("\n" + "="*60)
    print("Testing Single Prediction")
    print("="*60)
    
    # High-risk order (large quantity, known historically problematic product)
    order = {
        "product_code": "410397",
        "order_qty": 15.0,
        "customer_number": "33345",
        "order_date": datetime.now().isoformat(),
        "lead_time_days": 0,  # Same-day = risky
        "plant": "30588",
        "sales_unit": "ST"
    }
    
    print(f"\nSending order:")
    print(json.dumps(order, indent=2))
    
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=order,
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        
        print(f"\nStatus: {response.status_code}")
        result = response.json()
        print(f"\nPrediction Result:")
        print(json.dumps(result, indent=2))
        
        # Interpret result
        prob = result.get('stockout_probability', 0)
        risk = result.get('risk_level', 'UNKNOWN')
        
        print(f"\n{'='*60}")
        print(f"Stockout Probability: {prob:.1%}")
        print(f"Risk Level: {risk}")
        print(f"Action: {result.get('recommended_action', 'N/A')}")
        print(f"{'='*60}")
        
        return response.status_code == 200
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_batch_prediction():
    """Test batch prediction endpoint"""
    print("\n" + "="*60)
    print("Testing Batch Prediction")
    print("="*60)
    
    orders = {
        "orders": [
            {
                "order_number": "TEST001",
                "product_code": "410397",
                "order_qty": 20.0,
                "sales_unit": "ST"
            },
            {
                "order_number": "TEST002",
                "product_code": "406468",
                "order_qty": 1.0,
                "sales_unit": "ST"
            },
            {
                "order_number": "TEST003",
                "product_code": "411433",
                "order_qty": 5.0,
                "sales_unit": "PAK"
            }
        ]
    }
    
    print(f"\nSending {len(orders['orders'])} orders...")
    
    try:
        response = requests.post(
            f"{API_URL}/predict_batch",
            json=orders,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        print(f"Status: {response.status_code}")
        result = response.json()
        
        print(f"\n{'='*60}")
        print(f"Total Orders: {result.get('total_orders', 0)}")
        print(f"High Risk Count: {result.get('high_risk_count', 0)}")
        print(f"{'='*60}")
        
        print("\nTop 5 Riskiest Orders:")
        for pred in result.get('predictions', [])[:5]:
            print(f"  Order {pred.get('order_number')}: "
                  f"{pred.get('stockout_probability', 0):.1%} - "
                  f"{pred.get('risk_level', 'UNKNOWN')}")
        
        return response.status_code == 200
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    print("\n" + "="*60)
    print("🧪 STOCKOUT PREDICTION API - TEST SUITE")
    print("="*60)
    
    results = {
        "health": test_health(),
        "single": False,
        "batch": False
    }
    
    if results["health"]:
        results["single"] = test_single_prediction()
        results["batch"] = test_batch_prediction()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Health Check:      {'✅ PASS' if results['health'] else '❌ FAIL'}")
    print(f"Single Prediction: {'✅ PASS' if results['single'] else '❌ FAIL'}")
    print(f"Batch Prediction:  {'✅ PASS' if results['batch'] else '❌ FAIL'}")
    print("="*60)
    
    if all(results.values()):
        print("\n🎉 All tests passed! API is working correctly.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the API server.")
        return 1


if __name__ == "__main__":
    exit(main())

