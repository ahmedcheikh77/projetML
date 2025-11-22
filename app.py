# app.py - Churn Prediction API
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)

# Load the trained model and scaler
try:
    model = joblib.load('best_churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("✅ Model and scaler loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    scaler = None

# Feature names expected by the model (update based on your actual features)
EXPECTED_FEATURES = [
    'tenure', 'MonthlyCharges', 'TotalCharges', 'gender', 'Partner', 'Dependents',
    'PhoneService', 'PaperlessBilling', 'charge_per_month', 'total_services',
    'high_spender', 'new_customer', 'simple_contract'
    # Add all your actual feature names here from df_final_encoded.columns
]


def preprocess_input(data):
    """Preprocess the input data to match training format"""
    try:
        # Create DataFrame from input
        df = pd.DataFrame([data])

        # Feature engineering (same as training)
        df['charge_per_month'] = df['TotalCharges'] / (df['tenure'] + 1)
        df['high_spender'] = (df['MonthlyCharges'] > 70).astype(int)  # Adjust threshold based on your data
        df['new_customer'] = (df['tenure'] <= 3).astype(int)
        df['simple_contract'] = 1  # Adjust based on your contract logic

        # Ensure all expected features are present
        for feature in EXPECTED_FEATURES:
            if feature not in df.columns:
                df[feature] = 0  # Default value for missing features

        # Reorder columns to match training
        df = df[EXPECTED_FEATURES]

        return df
    except Exception as e:
        raise Exception(f"Preprocessing error: {e}")


@app.route('/')
def home():
    return jsonify({
        "message": "Churn Prediction API is running!",
        "endpoints": {
            "predict": "/predict (POST)",
            "health": "/health (GET)",
            "features": "/features (GET)"
        },
        "timestamp": datetime.now().isoformat()
    })


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    status = "healthy" if model is not None else "unhealthy"
    return jsonify({
        "status": status,
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    })


@app.route('/features', methods=['GET'])
def get_features():
    """Return expected features for the model"""
    return jsonify({
        "expected_features": EXPECTED_FEATURES,
        "count": len(EXPECTED_FEATURES)
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        # Check if model is loaded
        if model is None or scaler is None:
            return jsonify({"error": "Model not loaded"}), 500

        # Get JSON data from request
        data = request.get_json()

        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Preprocess the input
        processed_data = preprocess_input(data)

        # Scale the features
        scaled_data = scaler.transform(processed_data)

        # Make prediction
        prediction = model.predict(scaled_data)[0]
        probability = model.predict_proba(scaled_data)[0][1]  # Probability of churn (class 1)

        # Determine risk level
        if probability > 0.7:
            risk_level = "High"
            action = "Immediate intervention required"
        elif probability > 0.4:
            risk_level = "Medium"
            action = "Proactive retention campaign"
        else:
            risk_level = "Low"
            action = "Monitor periodically"

        # Prepare response
        response = {
            "churn_prediction": int(prediction),
            "churn_probability": round(float(probability), 4),
            "risk_level": risk_level,
            "recommended_action": action,
            "confidence": round(float(probability) if prediction == 1 else float(1 - probability), 4),
            "timestamp": datetime.now().isoformat()
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Batch prediction for multiple customers"""
    try:
        if model is None or scaler is None:
            return jsonify({"error": "Model not loaded"}), 500

        data = request.get_json()

        if not data or 'customers' not in data:
            return jsonify({"error": "No customers data provided"}), 400

        results = []
        for customer_data in data['customers']:
            try:
                processed_data = preprocess_input(customer_data)
                scaled_data = scaler.transform(processed_data)

                prediction = model.predict(scaled_data)[0]
                probability = model.predict_proba(scaled_data)[0][1]

                risk_level = "High" if probability > 0.7 else "Medium" if probability > 0.4 else "Low"

                results.append({
                    "customer_id": customer_data.get('customerID', 'unknown'),
                    "churn_prediction": int(prediction),
                    "churn_probability": round(float(probability), 4),
                    "risk_level": risk_level
                })

            except Exception as e:
                results.append({
                    "customer_id": customer_data.get('customerID', 'unknown'),
                    "error": str(e)
                })

        return jsonify({
            "batch_id": f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "total_customers": len(results),
            "high_risk_count": len([r for r in results if 'risk_level' in r and r['risk_level'] == 'High']),
            "predictions": results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)