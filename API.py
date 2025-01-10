from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import joblib
import numpy as np
import logging
import os
from pathlib import Path

# Initialize FastAPI app
app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load the trained model
model_path = os.getenv('MODEL_PATH', 'random_forest_model.pkl')
model = None

if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading model from {model_path}: {e}")
        raise RuntimeError(f"Failed to load model: {e}")
else:
    logging.error(f"Model file not found at {model_path}")
    raise FileNotFoundError(f"Model file not found at {model_path}")

# Serve static files (CSS, JS, images, etc.)
app.mount("/assets", StaticFiles(directory="assets"), name="assets")

# Serve HTML templates
templates_path = Path("templates/index.html")
if not templates_path.exists():
    raise FileNotFoundError(f"HTML template not found at {templates_path}")

# Home route
@app.get("/", response_class=HTMLResponse)
async def home():
    try:
        with open(templates_path, "r", encoding="utf-8") as file:
            return HTMLResponse(content=file.read())
    except Exception as e:
        logging.error(f"Error reading HTML template: {e}")
        raise HTTPException(status_code=500, detail="Failed to load the HTML template.")

# Prediction route
@app.post("/predict")
async def predict(request: Request):
    try:
        # Get input data from request
        data = await request.json()
        logging.debug(f"Received data: {data}")

        # Validate input data
        if not isinstance(data, dict):
            raise HTTPException(status_code=400, detail="Input data should be a dictionary")

        # Check if 'data' key exists and is a comma-separated string
        if 'data' not in data:
            raise HTTPException(status_code=400, detail="Input data must contain a 'data' key with a comma-separated string of feature values.")

        # Split the comma-separated string into a list of values
        feature_values = data['data'].split(',')
        logging.debug(f"Feature values: {feature_values}")

        # Ensure there are exactly 13 feature values
        if len(feature_values) != 13:
            raise HTTPException(status_code=400, detail=f"Input data should contain 13 features, but got {len(feature_values)}.")

        # Convert feature values to floats
        try:
            input_data = np.array([float(value) for value in feature_values]).reshape(1, -1)
            logging.debug(f"Input data for prediction: {input_data}")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid feature values: {e}")

        # Make prediction
        if model is not None:
            try:
                prediction = model.predict(input_data)
                logging.debug(f"Prediction: {prediction}")
            except Exception as e:
                logging.error(f"Error during prediction: {e}")
                raise HTTPException(status_code=500, detail="An error occurred while making the prediction.")

            # Return prediction result
            return JSONResponse(content={'prediction': int(prediction[0])})
        else:
            raise HTTPException(status_code=500, detail="Model is not loaded. Please check the model path and ensure the model file exists.")
    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"Unexpected error during prediction: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

# Run the FastAPI app
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=7000)