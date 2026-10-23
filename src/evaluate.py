from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

ROOT = Path(__file__).resolve().parents[1]

df = pd.read_csv(
    ROOT / "data" / "raw" / "creditcard.csv"
).sort_values("Time").reset_index(drop=True)

n = len(df)
test = df.iloc[int(0.80 * n):]

artifact = joblib.load(ROOT / "models" / "fraud_model.joblib")
model = artifact["model"]
features = artifact["features"]
threshold = artifact["threshold"]

X_test = test[features]
y_test = test["Class"]

prob = model.predict_proba(X_test)[:, 1]
pred = (prob >= threshold).astype(int)

metrics = {
    "average_precision": float(
        average_precision_score(y_test, prob)
    ),
    "roc_auc": float(roc_auc_score(y_test, prob)),
    "precision": float(
        precision_score(y_test, pred, zero_division=0)
    ),
    "recall": float(
        recall_score(y_test, pred, zero_division=0)
    ),
    "f1": float(f1_score(y_test, pred, zero_division=0)),
    "threshold": float(threshold),
}

print(json.dumps(metrics, indent=2))
print("\nConfusion matrix:")
print(confusion_matrix(y_test, pred))
print("\nClassification report:")
print(classification_report(y_test, pred, zero_division=0))

(ROOT / "reports").mkdir(exist_ok=True)
with open(ROOT / "reports" / "test_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)