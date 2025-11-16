# Prediction API Deployment Guide

This guide explains how to deploy the Stockout Prediction API to `otso.veistera.com/prediction-batch` without breaking the existing latex-to-calc service.

## What's Been Configured

### 1. Nginx Configuration
The nginx configuration at `/home/otso/LatexToCalc-Server/nginx/server.conf` has been updated to include:

- **`/prediction-batch`** - Batch predictions endpoint (proxies to `localhost:5000/predict_batch`)
- **`/prediction`** - Single prediction endpoint (proxies to `localhost:5000/predict`)
- **`/prediction-health`** - Health check endpoint (proxies to `localhost:5000/health`)

The existing latex-to-calc routes remain unchanged:
- **`/`** - Main latex-to-calc service (port 5002)
- **`/translate`** - Translation endpoint (port 5002)

### 2. Systemd Service
A systemd service file has been created to run the prediction API automatically.

## Deployment Steps

### Step 1: Install Python Dependencies
```bash
cd /home/otso/Junction
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Test the API Locally
```bash
python src/04_prediction_api.py
```

You should see:
```
🚀 STOCKOUT PREDICTION API
Starting server on http://localhost:5000
```

Test it with:
```bash
curl http://localhost:5000/health
```

Press Ctrl+C to stop the test server.

### Step 3: Set Up Systemd Service (Optional but Recommended)
```bash
# Copy service file to systemd directory
sudo cp prediction-api.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable service to start on boot
sudo systemctl enable prediction-api

# Start the service
sudo systemctl start prediction-api

# Check status
sudo systemctl status prediction-api
```

### Step 4: Update and Reload Nginx
```bash
# Run the setup script
./setup_nginx_prediction_api.sh
```

Or manually:
```bash
# Test nginx configuration
sudo nginx -t

# Reload nginx
sudo systemctl reload nginx
```

## Testing the Deployment

### 1. Test Health Endpoint
```bash
curl https://otso.veistera.com/prediction-health
```

Expected response:
```json
{"status": "healthy", "model_loaded": true}
```

### 2. Test Single Prediction
```bash
curl -X POST https://otso.veistera.com/prediction \
  -H "Content-Type: application/json" \
  -d '{
    "product_code": "410397",
    "order_qty": 5.0,
    "customer_number": "33345",
    "order_date": "2024-09-15T08:30:00",
    "lead_time_days": 1,
    "plant": "30588",
    "sales_unit": "ST"
  }'
```

### 3. Test Batch Prediction
```bash
curl -X POST https://otso.veistera.com/prediction-batch \
  -H "Content-Type: application/json" \
  -d '{
    "orders": [
      {
        "product_code": "410397",
        "order_qty": 5.0,
        "customer_number": "33345",
        "order_date": "2024-09-15T08:30:00",
        "lead_time_days": 1,
        "plant": "30588",
        "sales_unit": "ST"
      }
    ]
  }'
```

## Troubleshooting

### Check if Prediction API is Running
```bash
# If using systemd
sudo systemctl status prediction-api

# Check if port 5000 is in use
sudo lsof -i :5000
```

### Check Nginx Logs
```bash
# Error logs
sudo tail -f /var/log/nginx/error.log

# Access logs
sudo tail -f /var/log/nginx/access.log
```

### Restart Services
```bash
# Restart prediction API
sudo systemctl restart prediction-api

# Restart nginx
sudo systemctl restart nginx
```

### Test Nginx Configuration
```bash
sudo nginx -t
```

## Verify Existing Service Still Works

Make sure the latex-to-calc service still works:
```bash
curl https://otso.veistera.com/translate
```

## API Endpoints Summary

| Endpoint | Method | Description | Port |
|----------|--------|-------------|------|
| `/prediction-batch` | POST | Batch predictions | 5000 |
| `/prediction` | POST | Single prediction | 5000 |
| `/prediction-health` | GET | Health check | 5000 |
| `/` | GET/POST | LaTeX-to-Calc main | 5002 |


## Notes

- The prediction API runs on port 5000
- The latex-to-calc service runs on port 5002
- Both services are proxied through nginx on port 443 (HTTPS)
- CORS is enabled for all prediction endpoints
- SSL certificates are managed by Let's Encrypt