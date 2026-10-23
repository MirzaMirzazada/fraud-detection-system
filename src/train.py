from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import mlflow
from mlflow import sklearn as mlflow_sklearn

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "creditcard.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH).sort_values("Time").reset_index(drop=True)

target = "Class"
features = [c for c in df.columns if c != target]

# Preserve the chronological order.
n = len(df)
train = df.iloc[:int(0.60 * n)]
val = df.iloc[int(0.60 * n):int(0.80 * n)]

X_train, y_train = train[features], train[target]
X_val, y_val = val[features], val[target]

# Standardize Time and Amount; keep PCA features unchanged.
preprocessor = ColumnTransformer(
    transformers=[
        ("scale", StandardScaler(), ["Time", "Amount"]),
        ("pca", "passthrough", [f"V{i}" for i in range(1, 29)]),
    ],
    remainder="drop",
)

models = {
    "logistic_regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    ),
}

results = {}


from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)

# Create an experiment to group all model comparisons.
mlflow.set_experiment("Fraud Detection")

results = {}

for name, classifier in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    # Each model gets its own MLflow run.
    with mlflow.start_run(run_name=name):

        # 1. Train the model using training data only.
        pipeline.fit(X_train, y_train)

        # 2. Predict fraud probabilities on validation data.
        val_prob = pipeline.predict_proba(X_val)[:, 1]

        # 3. Calculate a metric suitable for rare fraud cases.
        ap = average_precision_score(y_val, val_prob)

        # 4. Select a threshold using validation data.
        candidates = []

        for threshold in np.arange(0.05, 0.96, 0.01):
            pred = (val_prob >= threshold).astype(int)

            precision = precision_score(
                y_val, pred, zero_division=0
            )
            recall = recall_score(
                y_val, pred, zero_division=0
            )

            if precision >= 0.50:
                candidates.append(
                    (recall, threshold, precision)
                )

        if candidates:
            _, best_threshold, _ = max(candidates)
        else:
            best_threshold = 0.50

        val_pred = (val_prob >= best_threshold).astype(int)

        # 5. Record parameters.
        mlflow.log_param("model_name", name)
        mlflow.log_param("training_rows", len(X_train))
        mlflow.log_param("validation_rows", len(X_val))
        mlflow.log_param("threshold_precision_target", 0.50)

        # 6. Record validation metrics.
        mlflow.log_metric("validation_average_precision", ap)
        mlflow.log_metric(
            "validation_precision",
            precision_score(y_val, val_pred, zero_division=0),
        )
        mlflow.log_metric(
            "validation_recall",
            recall_score(y_val, val_pred, zero_division=0),
        )
        mlflow.log_metric(
            "validation_f1",
            f1_score(y_val, val_pred, zero_division=0),
        )
        mlflow.log_metric("selected_threshold", best_threshold)

        # 7. Save a copy of this trained model in MLflow.
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            serialization_format="pickle",
        )

        # Keep the model available for selecting the winner.
        results[name] = {
            "average_precision": ap,
            "pipeline": pipeline,
            "threshold": float(best_threshold),
        }

        print(f"{name}: AP = {ap:.5f}")

# Select the best model using validation average precision.
best_name = max(
    results,
    key=lambda name: results[name]["average_precision"],
)

best_result = results[best_name]
best_pipeline = best_result["pipeline"]
best_threshold = best_result["threshold"]

# Save the artifact used by the FastAPI application.
joblib.dump(
    {
        "model": best_pipeline,
        "features": features,
        "threshold": best_threshold,
        "model_name": best_name,
    },
    MODEL_DIR / "fraud_model.joblib",
)

print("Best model:", best_name)
print("Selected threshold:", best_threshold)
print("Saved model to:", MODEL_DIR / "fraud_model.joblib")




# Choose a threshold using validation data only.
val_prob = best_pipeline.predict_proba(X_val)[:, 1]
thresholds = np.arange(0.05, 0.96, 0.01)

from sklearn.metrics import precision_score, recall_score

# Example policy: maximize recall subject to validation precision >= 50%.
candidates = []

for threshold in thresholds:
    pred = (val_prob >= threshold).astype(int)
    precision = precision_score(y_val, pred, zero_division=0)
    recall = recall_score(y_val, pred, zero_division=0)

    if precision >= 0.50:
        candidates.append((recall, threshold, precision))

if candidates:
    _, best_threshold, best_precision = max(candidates)
else:
    # Explicit fallback if no candidate meets the chosen constraint.
    best_threshold = 0.50
    best_precision = None

artifact = {
    "model": best_pipeline,
    "features": features,
    "threshold": float(best_threshold),
    "model_name": best_name,
}

joblib.dump(artifact, MODEL_DIR / "fraud_model.joblib")

print("Selected model:", best_name)
print(
    "Validation average precision:",
    best_result["average_precision"]
)
print("Selected threshold:", best_threshold)
print("Validation precision at threshold:", best_precision)
print("Saved model:", MODEL_DIR / "fraud_model.joblib")