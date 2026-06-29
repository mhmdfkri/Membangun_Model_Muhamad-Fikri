import os
import joblib
import pandas as pd

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# =====================================
# MLflow Configuration
# =====================================

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Telco Churn Basic")

# Wajib untuk Basic
mlflow.autolog()

# =====================================
# Folder Output
# =====================================

os.makedirs("models", exist_ok=True)

# =====================================
# Load Dataset
# =====================================

df = pd.read_csv("data/telco_processed.csv")

# =====================================
# Feature & Target
# =====================================

X = df.drop("Churn Label", axis=1)
y = df["Churn Label"]

# =====================================
# One Hot Encoding
# =====================================

X = pd.get_dummies(
    X,
    drop_first=True
)

# =====================================
# Train Test Split
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================
# Training
# =====================================

with mlflow.start_run(run_name="RandomForest_Basic"):

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Prediction

    y_pred = model.predict(X_test)

    # Evaluation

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("=" * 50)
    print("MODEL PERFORMANCE")
    print("=" * 50)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    # Simpan model (untuk inference)

    joblib.dump(
        model,
        "models/random_forest.pkl"
    )

print("\nTraining selesai.")
print("Model disimpan di models/random_forest.pkl")
print("MLflow Tracking menggunakan SQLite")