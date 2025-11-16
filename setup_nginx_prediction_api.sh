#!/bin/bash

echo "==================================================================="
echo "Setting up Nginx for Prediction API"
echo "==================================================================="

# Test nginx configuration
echo ""
echo "Testing nginx configuration..."
sudo nginx -t

if [ $? -eq 0 ]; then
    echo "✓ Nginx configuration is valid"
    
    # Reload nginx
    echo ""
    echo "Reloading nginx..."
    sudo systemctl reload nginx
    
    if [ $? -eq 0 ]; then
        echo "✓ Nginx reloaded successfully"
    else
        echo "✗ Failed to reload nginx"
        exit 1
    fi
else
    echo "✗ Nginx configuration has errors"
    exit 1
fi

echo ""
echo "==================================================================="
echo "Setup Complete!"
echo "==================================================================="
echo ""
echo "Your prediction API endpoints are now available at:"
echo "  • https://otso.veistera.com/prediction-batch  (batch predictions)"
echo "  • https://otso.veistera.com/prediction        (single prediction)"
echo "  • https://otso.veistera.com/prediction-health (health check)"
echo "  • https://otso.veistera.com/get-example-data  (example data for testing)"
echo ""
echo "The existing latex-to-calc service remains unchanged at:"
echo "  • https://otso.veistera.com/                  (main service)"
echo "  • https://otso.veistera.com/translate         (translate endpoint)"
echo ""
echo "==================================================================="
echo ""
echo "Next steps:"
echo "1. Make sure your prediction API is running: python src/04_prediction_api.py"
echo "2. Test the endpoint: curl https://otso.veistera.com/prediction-health"
echo ""