from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import requests
import os

# 1. Initialize the FastAPI Application

app = FastAPI(
    title="Plantro Smart Irrigation API",
    description="The AI Brain for Plantro System. Takes sensor data and returns irrigation decisions.",
    version="1.0.0"
)


# 2. Define the Data Input Structure (Hardware Payload)

class SensorPayload(BaseModel):
    soil_moisture: float
    current_temp: float
    current_humidity: float
    crop_type: str        
    growth_stage: str    


# 3. Weather Fetching Function

def get_weather_data(lat="26.1642", lon="32.7267"):
    api_key = "fd11766571ae34358aeda4adec495316" 
    url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        rain_probability = int(data['list'][1].get('pop', 0) * 100)
        forecast_temp = data['list'][8]['main']['temp_max']
        
        return {"rain_probability": rain_probability, "forecast_temp": forecast_temp}
    except Exception as e:
        print(f"Weather API Error: {e}")
        # Default fallback values so the system doesn't crash
        return {"rain_probability": 0, "forecast_temp": 35.0}


# 4. The AI Inference Pipeline Class

class PlantroInferencePipeline:
    def __init__(self):
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.classifier = joblib.load(os.path.join(base_dir, 'plantro_classifier.pkl'))
            self.regressor = joblib.load(os.path.join(base_dir, 'plantro_regressor.pkl'))
            self.crop_encoder = joblib.load(os.path.join(base_dir, 'crop_encoder.pkl'))
            self.stage_encoder = joblib.load(os.path.join(base_dir, 'stage_encoder.pkl'))
            print(" AI Models Loaded Successfully!")
        except Exception as e:
            print(f" Error loading models. Make sure .pkl files are in the same folder. Details: {e}")

    def predict(self, sensor_data: dict):
        try:
            # 1. Encode text features to numbers
            encoded_crop = self.crop_encoder.transform([sensor_data['crop_type']])[0]
            encoded_stage = self.stage_encoder.transform([sensor_data['growth_stage']])[0]
            
            # 2. Assemble feature array
            features = np.array([[
                sensor_data['soil_moisture'],
                sensor_data['current_temp'],
                sensor_data['current_humidity'],
                sensor_data['rain_probability'],
                sensor_data['forecast_temp'],
                encoded_crop,
                encoded_stage
            ]])
            
            # 3. Manager Model (Classifier)
            should_irrigate = int(self.classifier.predict(features)[0])
            
            if should_irrigate == 0:
                return {
                    "status": "success",
                    "irrigate": 0,
                    "pump_duration_minutes": 0,
                    "message": "Soil is optimal or rain is expected."
                }
            else:
                # 4. Employee Model (Regressor)
                predicted_time = self.regressor.predict(features)[0]
                final_duration = int(round(predicted_time))
                
                return {
                    "status": "success",
                    "irrigate": 1,
                    "pump_duration_minutes": final_duration,
                    "message": "Irrigation required."
                }
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"AI Processing Error: {str(e)}")

# Initialize the AI Brain globally so it loads only once
ai_brain = PlantroInferencePipeline()


# 5. API Endpoints (Routes)

@app.get("/")
def read_root():
    return {"message": "Welcome to Plantro API! Go to /docs to test the system."}

@app.post("/predict_irrigation")
def get_irrigation_decision(payload: SensorPayload):
    # 1. Convert the hardware data to a dictionary
    sensor_data = payload.dict()
    
    # 2. Fetch live weather data automatically
    weather_data = get_weather_data()
    
    # 3. Merge the weather data into our sensor data dictionary
    sensor_data['rain_probability'] = weather_data['rain_probability']
    sensor_data['forecast_temp'] = weather_data['forecast_temp']
    
    # 4. Send the complete data to the AI Brain
    result = ai_brain.predict(sensor_data)
    
    return result