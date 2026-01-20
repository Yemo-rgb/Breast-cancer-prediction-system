"""
Breast Cancer Prediction System - Web Application
Flask-based web GUI for breast cancer prediction
"""

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the trained model and components
MODEL_PATH = 'model/breast_cancer_model.pkl'
SCALER_PATH = 'model/scaler.pkl'
FEATURES_PATH = 'model/feature_names.pkl'

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    print("✓ Model and components loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    scaler = None
    feature_names = None

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html', feature_names=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        if model is None or scaler is None:
            return jsonify({
                'error': 'Model not loaded. Please check server configuration.'
            }), 500
        
        # Get input data from form
        features = []
        feature_values = {}
        
        for feature in feature_names:
            value = float(request.form.get(feature, 0))
            features.append(value)
            feature_values[feature] = value
        
        # Convert to numpy array and reshape
        input_data = np.array(features).reshape(1, -1)
        
        # Scale the input data
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        
        # Prepare response
        result = {
            'prediction': 'Benign' if prediction == 1 else 'Malignant',
            'prediction_code': int(prediction),
            'probability_malignant': float(prediction_proba[0]),
            'probability_benign': float(prediction_proba[1]),
            'confidence': float(max(prediction_proba)) * 100,
            'input_features': feature_values
        }
        
        return jsonify(result)
    
    except ValueError as e:
        return jsonify({
            'error': f'Invalid input: {str(e)}'
        }), 400
    except Exception as e:
        return jsonify({
            'error': f'Prediction error: {str(e)}'
        }), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None
    })

if __name__ == '__main__':
    # Run the Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)