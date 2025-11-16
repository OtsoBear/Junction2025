# Junction 2025 - Stockout Prediction System

**AI-powered logistics intelligence for Valio Aimo**

## What It Does

Predicts product stockouts before they happen using machine learning.
- **ROC-AUC: 0.85+**
- **29 features** analyzing time, products, customers, warehouses & suppliers
- **<100ms API** response time

## The Intelligence

**Analyzes:**
- Temporal patterns (seasonality, rush hours, lead times)
- Product behavior (historical rates, replacements, popularity)
- Customer segments (VIP, order frequency, lead time preferences)
- Warehouse performance (plant capacity, storage locations)
- Supplier reliability (delivery consistency)
- Freshness risks (perishables + time pressure)

**Could analyze with more data:**
- Weather events → demand spikes
- Real-time truck GPS → early warnings
- Social trends → viral product detection
- Point-of-sale data → inventory optimization

## Quick Start

```bash
# Train model
bash run_pipeline.sh

# Start API
python src/04_prediction_api.py

# Make prediction
curl -X POST http://localhost:5555/predict \
  -H "Content-Type: application/json" \
  -d @cake_order_example.json
```

## Tech Stack

- **ML**: LightGBM with class imbalance optimization
- **API**: FastAPI with <100ms latency
- **Deployment**: Systemd + Nginx production-ready

Built for Junction 2025