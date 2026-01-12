import requests

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

url = 'http://localhost:9696/predict'
response = requests.post(url, json=employee)
print(response.json())