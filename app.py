from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import logging
import os

# Initialize Flask app
app = Flask(__name__, static_folder='assets', static_url_path='/assets')

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load the trained model
model_path = os.getenv('MODEL_PATH', 'random_forest_model.pkl')
if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
        app.logger.info("Model loaded successfully.")
    except Exception as e:
        app.logger.error(f"Error loading model from {model_path}: {e}")
        model = None
else:
    app.logger.error(f"Model file not found at {model_path}")
    model = None

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from request
        data = request.get_json()
        expected_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                             'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 
                             'ca', 'thal']
        
        # Validate input data
        if not isinstance(data, dict):
            raise ValueError("Input data should be a dictionary")
        if len(data) != len(expected_features):
            raise ValueError(f"Input data should contain 13 features, but got {len(data)}.")
        if not all(feature in data for feature in expected_features):
            missing_features = [feature for feature in expected_features if feature not in data]
            raise ValueError(f"Input data is missing the following features: {missing_features}")
        
        # Convert data into numpy array
        input_data = np.array([data[feature] for feature in expected_features]).reshape(1, -1)
        app.logger.debug(f"Input data for prediction: {input_data}")
        
        # Make prediction
        if model is not None:
            prediction = model.predict(input_data)
            app.logger.debug(f"Prediction: {prediction}")
            
            # Return prediction result
            return jsonify({'prediction': int(prediction[0])})
        else:
            return jsonify({'error': 'Model is not loaded. Please check the model path and ensure the model file exists.'}), 500
    except Exception as e:
        app.logger.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() in ['true', '1', 't']
    app.run(debug=debug_mode)
