from fastapi import FastAPI
from pydantic import BaseModel
import os
import pickle

class EmployeeData(BaseModel):
    Age: int
    BusinessTravel: str
    Department: str
    EducationField: str
    Gender: str
    JobRole: str
    MaritalStatus: str
    TotalWorkingYears: float
    YearsAtCompany: int
    YearsWithCurrManager: int
    EnvironmentSatisfaction: float
    JobSatisfaction: float
    Mean_Work_Hours: float
    Overtime_Days_Count: int

app = FastAPI(title="Employee Attrition Prediction API")


model_path = os.path.join(os.path.dirname(__file__), "models", "random_forest_employee_attrition_v1.bin")
with open(model_path, "rb") as f_in:
    dv, model = pickle.load(f_in)  

@app.post("/predict")
def predict(data: EmployeeData):
 
    data_dict = data.dict()
    

    X = dv.transform([data_dict])
    
  
    pred_proba = model.predict_proba(X)[0, 1]
    pred_class = int(model.predict(X)[0])
    
    return {
        "prediction": pred_class,
        "probability": float(pred_proba)
    }

@app.get("/health", status_code=200)
def get_health_status():
    """
    This is health check endpoint.
    Returns a 200 OK status to indicate the service is running.
    """
    return {"status": "ok"}