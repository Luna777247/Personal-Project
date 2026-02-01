"""
Test script for Fraud Detection API
"""
import requests
import json
from datetime import datetime

# API base URL
BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint"""
    print("Testing /health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")


def test_single_prediction():
    """Test single transaction prediction"""
    print("Testing /predict endpoint...")
    
    transaction = {
        "transaction_id": "TXN_TEST_001",
        "customer_id": 1234,
        "merchant_id": 567,
        "amount": 1250.50,
        "timestamp": datetime.now().isoformat(),
        "merchant_category": "electronics",
        "card_type": "credit",
        "transaction_type": "online",
        "country_code": "US",
        "device_id": "DEV_TEST_001",
        "ip_address": "192.168.1.100"
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=transaction)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")


def test_batch_prediction():
    """Test batch transaction prediction"""
    print("Testing /predict/batch endpoint...")
    
    transactions = {
        "transactions": [
            {
                "transaction_id": f"TXN_BATCH_{i:03d}",
                "customer_id": 1000 + i,
                "merchant_id": 500 + (i % 10),
                "amount": 50.0 + i * 10,
                "timestamp": datetime.now().isoformat(),
                "merchant_category": ["retail", "food", "gas"][i % 3],
                "card_type": "credit",
                "transaction_type": "online",
                "country_code": "US"
            }
            for i in range(5)
        ]
    }
    
    response = requests.post(f"{BASE_URL}/predict/batch", json=transactions)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")


def test_metrics():
    """Test metrics endpoint"""
    print("Testing /metrics endpoint...")
    response = requests.get(f"{BASE_URL}/metrics")
    print(f"Status: {response.status_code}")
    print(f"Metrics (first 500 chars):\n{response.text[:500]}...\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Fraud Detection API Test Suite")
    print("=" * 60 + "\n")
    
    try:
        test_health()
        test_single_prediction()
        test_batch_prediction()
        test_metrics()
        
        print("=" * 60)
        print("All tests completed!")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("Error: Cannot connect to API. Make sure the server is running.")
    except Exception as e:
        print(f"Error: {str(e)}")
