# Real-Time Fraud Detection System

A machine learning project that detects potentially fraudulent credit card transactions using classification models and a FastAPI REST API.

## Features
- Data preprocessing and exploratory data analysis
- Logistic Regression and Random Forest models
- Imbalanced data evaluation using precision, recall, F1-score, and average precision
- MLflow experiment tracking
- Real-time fraud prediction through FastAPI
- Docker containerization

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, MLflow, FastAPI, Docker, Pytest.

## Run Locally

```bash
pip install -r requirements.txt
python -m src.train
uvicorn api.main:app --reload
```

API documentation: `http://localhost:8000/docs`

## Run with Docker

```bash
docker build -t fraud-detection-api .
docker run --rm -p 8000:8000 fraud-detection-api
```

## Dataset
Credit Card Fraud Detection dataset from Kaggle. The dataset contains anonymized transaction features and a binary fraud label.

