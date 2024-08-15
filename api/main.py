''' 
Falso:
{
  "gender": "female",
  "age": 35,
  "hypertension": 0,
  "heart_disease": 0,
  "smoking_history": "never",
  "bmi": 35.42,
  "HbA1c_level": 4,
  "blood_glucose_level": 95
}

Verdadero:
{
  "gender": "male",
  "age": 64,
  "hypertension": 0,
  "heart_disease": 0,
  "smoking_history": "never",
  "bmi": 38.54,
  "HbA1c_level": 6.3,
  "blood_glucose_level": 155
}
'''

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, validator
import joblib
import pandas as pd
import numpy as np

model = joblib.load('model.pkl')
API_KEY = 'MLOps'
app = FastAPI()

# Mapeos para las columnas categóricas
gender_map = {"female": 0, "male": 1}
smoking_history_map = {
    "never": 4,
    "no Info": 0,
    "current": 1,
    "former": 3,
    "ever": 2,
    "not current": 5
}

class DiabetesRequest(BaseModel):
    gender: str
    age: int 
    hypertension: int
    heart_disease: int
    smoking_history: str
    bmi: float
    HbA1c_level: float
    blood_glucose_level: int

    @validator('gender')
    def validate_gender(cls, v):
        v = v.lower() 
        if v not in gender_map:
            raise ValueError('Invalid gender')
        return v
    
    @validator('smoking_history')
    def validate_smoking_history(cls, v):
        v = v.lower()
        if v not in smoking_history_map:
            raise ValueError('Invalid smoking history')
        return v
    
def preprocess_data(input_data: DiabetesRequest):

    try:
        # Mapeamos valores categóricos a sus correspondientes valores numéricos
        gender = gender_map[input_data.gender]
        smoking_history = smoking_history_map[input_data.smoking_history]

        # Convertimos la entrada en un array para pasar al modelo
        data = np.array([
            gender,
            input_data.age,
            input_data.hypertension,
            input_data.heart_disease,
            smoking_history,
            input_data.bmi,
            input_data.HbA1c_level,
            input_data.blood_glucose_level
        ]).reshape(1, -1)
        
        return data
    
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        raise HTTPException(status_code=500, detail="Error in preprocessing input data")

def get_api_key(api_key: str):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return api_key

@app.get("/")
def main():
    return {'message': 'diabetes prediction'}

@app.post('/diabetes_predict/')
def predict(input_data: DiabetesRequest, api_key: str = Depends(get_api_key)):
    try:
        # Preprocesar los datos
        processed_data = preprocess_data(input_data)

        print(f'Processed data: {processed_data}')

        # Realizar la predicción
        prediction = model.predict(processed_data)
        return {"prediction": int(prediction[0])} # 0 es NEGATIVO y 1 es POSITIVO
    
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail='Error during prediction')

