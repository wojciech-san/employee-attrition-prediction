FROM python:3.12-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]
RUN pipenv install --system --deploy

COPY ["predict.py", "./"]
 
COPY ["models/random_forest_employee_attrition_v1.bin", "./models/random_forest_employee_attrition_v1.bin"]

EXPOSE 9696

CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]

