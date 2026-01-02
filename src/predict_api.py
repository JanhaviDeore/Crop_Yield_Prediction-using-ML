from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os # Import os to check for file existence

app = Flask(__name__)

# --- Model Loading ---
# Define paths
MODEL_PATH = 'src\models\best_model.joblib'
PREPROCESSOR_PATH = 'src\models\preprocessor.joblib'

# Check if model files exist before loading
if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
    print(f"Error: Model or preprocessor file not found.")
    print(f"Please run 'train_model.py' first to generate these files in the 'models/' directory.")
    # In a real app, you might exit or raise an exception
    model = None
    preprocessor = None
else:
    print("Loading model and preprocessor...")
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    print("Model and preprocessor loaded successfully.")

@app.route('/')
def home():
    """Provides a simple welcome message for the API root."""
    return "Crop Yield Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles POST requests to /predict.
    Expects JSON data with features: 'temperature', 'rainfall_mm', 'soil_type', 'fertilizer_type'.
    Returns a JSON response with the predicted yield.
    """
    if model is None or preprocessor is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500

    try:
        data = request.get_json()
        
        # --- Input Validation ---
        # It's good practice to validate incoming data.
        # This assumes the synthetic data's columns are the expected features.
        required_features = ['temperature', 'rainfall_mm', 'soil_type', 'fertilizer_type']
        if not data:
            return jsonify({'error': 'No input data provided.'}), 400
        
        missing_features = [f for f in required_features if f not in data]
        if missing_features:
            return jsonify({'error': f'Missing features: {", ".join(missing_features)}'}), 400

        # Convert the single JSON object into a DataFrame
        # The [data] makes it a single-row DataFrame
        df = pd.DataFrame([data])
        
        # Ensure column order matches what the preprocessor was trained on
        # (Though ColumnTransformer handles this by name, it's safer)
        df = df[required_features] 

        # --- Preprocessing and Prediction ---
        # Use the loaded preprocessor to transform the data
        X_trans = preprocessor.transform(df)
        
        # Use the loaded model to make a prediction
        pred = model.predict(X_trans)
        
        # --- Return Response ---
        # Return the prediction as a JSON object
        return jsonify({'predicted_yield': float(pred[0])})

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    # Use 0.0.0.0 to make it accessible on your network, not just localhost
    app.run(debug=True, host='0.0.0.0', port=5000)
