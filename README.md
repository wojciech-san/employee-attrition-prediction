# 👩‍💼 Employee Attrition Prediction

This repository contains an end-to-end machine learning project that predicts **employee attrition** (whether an employee is likely to leave) based on HR-related attributes such as demographics, job/department information, satisfaction metrics, tenure, and overtime/work-hour indicators.

The project is packaged for reproducibility and deployment using **Pipenv** and **Docker**, and it is deployed to **Azure Web App** as a simple prediction API.

---

## ✨ Features

- **Exploratory Data Analysis (EDA)** and feature understanding in notebooks
- **Binary classification model** trained to predict employee attrition risk
- **Prediction API** exposed via `predict.py` with a `/predict` endpoint
- **Dependency management** with `Pipfile` / `Pipfile.lock` (and `requirements.txt`)
- **Containerization** with Docker for reproducible local and cloud execution
- **Cloud deployment**: Azure Web App endpoint available for live testing

---

## 📦 Dataset

The dataset used for model training is included in this repository under the `data/` directory (CSV format), so the project is reproducible without external downloads.

The model consumes a subset of features (see the API schema below), including:

- Demographics and job context: `Age`, `Gender`, `Department`, `JobRole`, `MaritalStatus`, `BusinessTravel`, `EducationField`
- Seniority/tenure: `TotalWorkingYears`, `YearsAtCompany`, `YearsWithCurrManager`
- Satisfaction indicators: `EnvironmentSatisfaction`, `JobSatisfaction`
- Work pattern indicators: `Mean_Work_Hours`, `Overtime_Days_Count`

> Note: If you replace the dataset with another version/source, ensure that the training pipeline and API schema remain aligned.

---

## 🗂️ Project Structure

```text
employee-attrition-prediction/
├── data/                      # Raw and reference datasets
│   ├── data_dictionary.xlsx   # Dataset feature descriptions
│   ├── employee_survey_data.csv
│   ├── general_data.csv
│   ├── in_time.csv
│   ├── manager_survey_data.csv
│   └── out_time.csv
│
├── docs/                      # Project documentation
│   ├── images/                # Images used in documentation
│   └── Docker Azure Deployment Guide.md  # Azure deployment instructions
│
├── models/                    # Trained model artifacts
│
├── notebooks/                 # Jupyter notebooks
│   ├── exploratory-data-analysis.ipynb   # EDA and feature understanding
│   └── predict-test.ipynb                 # API testing notebook
│
├── Dockerfile                 # Docker image definition
├── Pipfile                    # Pipenv dependencies
├── Pipfile.lock               # Locked dependency versions
├── requirements.txt           # Alternative pip dependencies
│
├── train.py                   # Model training script
├── predict.py                 # Prediction service (exposes /predict endpoint)
├── predict-test.py            # Script for testing the prediction API
└── README.md                  # Project documentation

```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+ (recommended)
- Git
- Docker (optional, but recommended for reproducibility)

## 🧪 Run Locally
### 1) Clone the repository
```bash
git clone https://github.com/wojciech-san/employee-attrition-prediction.git
cd employee-attrition-prediction
```
### 2) Install dependencies
### Option A — Pipenv (recommended)
```bash
pip install pipenv
pipenv install --dev
pipenv shell
```
### Option B — pip + requirements.txt
```bash
pip install -r requirements.txt
```

### 3) Start the API
This project exposes the web service via predict.py as app.

Depending on how predict.py is implemented in your environment, use one of the options below:

### Option A — FastAPI-style (uvicorn)
```bash
uvicorn predict:app --host 0.0.0.0 --port 9696
```

### Option B — Flask-style (gunicorn)
```bash
gunicorn --bind 0.0.0.0:9696 predict:app
```
After starting, the service should be reachable at:

* http://localhost:9696/predict

If Swagger is enabled (common for FastAPI), it’s typically available at:

* http://localhost:9696/docs

## 🐳 Docker Deployment
### Build the image
```bash
docker build -t employee-attrition-predictor .
```
### Run the container
```bash
docker run -it --rm -p 9696:9696 employee-attrition-predictor
```
Then test it locally at:
* http://localhost:9696/predict

### 📈 Model Training and Evaluation

Training and evaluation are documented in the notebooks inside notebooks/.

Typical steps covered there:

* Data cleaning & preprocessing

* Categorical encoding / feature engineering

* Train/validation split

* Model training + hyperparameter tuning (if applicable)

* Final model export to models/

For exact metrics and the final model selection rationale, refer to the training notebook(s).

## 🔌 API Usage
### Output format

The API returns:

* prediction: 0 or 1

* probability: probability of attrition (class 1)

Suggested interpretation:

* prediction = 1 → employee is likely to leave

* prediction = 0 → employee is likely to stay

## ✅ Example Request & Response (Sample)
### Example JSON (request)
```bash
{
  "Age": 36,
  "BusinessTravel": "Non-Travel",
  "Department": "Research & Development",
  "EducationField": "Medical",
  "Gender": "Male",
  "JobRole": "Research Scientist",
  "MaritalStatus": "Single",
  "TotalWorkingYears": 14.0,
  "YearsAtCompany": 3,
  "YearsWithCurrManager": 7,
  "EnvironmentSatisfaction": 2.0,
  "JobSatisfaction": 1.0,
  "Mean_Work_Hours": 5.201124,
  "Overtime_Days_Count": 0
}

```

### Example JSON (response)

```bash
{
  "prediction": 0,
  "probability": 0.2614375475797958
}
```

## ☁️ Azure Deployment

The prediction API is deployed to Azure Web App:

#### Endpoint:
* https://employee-attrition-app-fzerhkagebhqb9fw.polandcentral-01.azurewebsites.net/predict

### 1️⃣ Using curl (Linux/macOS/Windows Git Bash)
```bash
curl -X POST "https://employee-attrition-app-fzerhkagebhqb9fw.polandcentral-01.azurewebsites.net/predict" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 36,
    "BusinessTravel": "Non-Travel",
    "Department": "Research & Development",
    "EducationField": "Medical",
    "Gender": "Male",
    "JobRole": "Research Scientist",
    "MaritalStatus": "Single",
    "TotalWorkingYears": 14.0,
    "YearsAtCompany": 3,
    "YearsWithCurrManager": 7,
    "EnvironmentSatisfaction": 2.0,
    "JobSatisfaction": 1.0,
    "Mean_Work_Hours": 5.201124,
    "Overtime_Days_Count": 0
  }'

```

### 2️⃣ Using PowerShell (Invoke-RestMethod)
```bash
$uri = "https://employee-attrition-app-fzerhkagebhqb9fw.polandcentral-01.azurewebsites.net/predict"

$body = @{
  Age = 36
  BusinessTravel = "Non-Travel"
  Department = "Research & Development"
  EducationField = "Medical"
  Gender = "Male"
  JobRole = "Research Scientist"
  MaritalStatus = "Single"
  TotalWorkingYears = 14.0
  YearsAtCompany = 3
  YearsWithCurrManager = 7
  EnvironmentSatisfaction = 2.0
  JobSatisfaction = 1.0
  Mean_Work_Hours = 5.201124
  Overtime_Days_Count = 0
} | ConvertTo-Json

Invoke-RestMethod -Uri $uri -Method POST -Body $body -ContentType "application/json"

```

### 3️⃣ Using Python (requests)

```bash
import requests

url = "https://employee-attrition-app-fzerhkagebhqb9fw.polandcentral-01.azurewebsites.net/predict"

employee = {
  "Age": 36,
  "BusinessTravel": "Non-Travel",
  "Department": "Research & Development",
  "EducationField": "Medical",
  "Gender": "Male",
  "JobRole": "Research Scientist",
  "MaritalStatus": "Single",
  "TotalWorkingYears": 14.0,
  "YearsAtCompany": 3,
  "YearsWithCurrManager": 7,
  "EnvironmentSatisfaction": 2.0,
  "JobSatisfaction": 1.0,
  "Mean_Work_Hours": 5.201124,
  "Overtime_Days_Count": 0
}

resp = requests.post(url, json=employee, timeout=30)
print(resp.json())

```

#### Note: Azure Web Apps on free/basic tiers can experience a “cold start” after inactivity. The first request may take a few seconds.

### 📚 Documentation

This project includes a docs folder containing detailed instructions on how to deploy the Docker container to an Azure Web App.

Inside the docs folder, you will find step-by-step guidance, screenshots, and examples for:

* Pushing your Docker image to Docker Hub

* Creating and configuring an Azure Web App

* Setting environment variables and ports

* Verifying your deployment and testing the API

Tip: If you are new to deploying Docker containers on Azure, start with the instructions in the docs folder for a complete walkthrough.