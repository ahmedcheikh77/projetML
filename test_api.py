# test_api.py - Test the API
import requests
import json

# API base URL
BASE_URL = "http://localhost:5000"


def test_health():
    """Test health endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    print("🔍 Health Check:", response.json())


def test_features():
    """Test features endpoint"""
    response = requests.get(f"{BASE_URL}/features")
    print("📋 Expected Features:", response.json())


def test_prediction():
    """Test single prediction"""
    # Sample customer data (adjust based on your features)
    customer_data = {
        'tenure': 12,
        'MonthlyCharges': 75.0,
        'TotalCharges': 900.0,
        'gender': 1,
        'Partner': 0,
        'Dependents': 0,
        'PhoneService': 1,
        'PaperlessBilling': 1
    }

    response = requests.post(f"{BASE_URL}/predict", json=customer_data)
    print("🎯 Single Prediction:", response.json())


def test_batch_prediction():
    """Test batch prediction"""
    customers = [
        {
            'customerID': '001',
            'tenure': 2,
            'MonthlyCharges': 85.0,
            'TotalCharges': 170.0,
            'gender': 0,
            'Partner': 1,
            'Dependents': 0,
            'PhoneService': 1,
            'PaperlessBilling': 1
        },
        {
            'customerID': '002',
            'tenure': 36,
            'MonthlyCharges': 45.0,
            'TotalCharges': 1620.0,
            'gender': 1,
            'Partner': 1,
            'Dependents': 1,
            'PhoneService': 1,
            'PaperlessBilling': 0
        }
    ]

    response = requests.post(f"{BASE_URL}/batch_predict", json={'customers': customers})
    print("📊 Batch Prediction:", json.dumps(response.json(), indent=2))


if __name__ == '__main__':
    print("🚀 Testing Churn Prediction API...")

    test_health()
    print()

    test_features()
    print()

    test_prediction()
    print()

    test_batch_prediction()