import pandas as pd
import joblib
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
# MLflow Autolog (WAJIB untuk Basic)
# =====================================

mlflow.autolog()

# =====================================
# Load Dataset
# =====================================

df = pd.read_csv("../data/telco_processed.csv")

# =====================================
# Feature & Target
# =====================================

X = df.drop("Churn Label", axis=1)
y = df["Churn Label"]

# =====================================
# One Hot Encoding
# =====================================

X = pd.get_dummies(X, drop_first=True)

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
# MLflow Experiment
# =====================================

mlflow.set_experiment(
    "Telco Churn Basic"
)

with mlflow.start_run(run_name="RandomForest_Basic"):

    # =====================================
    # Model Training
    # =====================================

    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    rf.fit(X_train, y_train)

    # =====================================
    # Prediction
    # =====================================

    y_pred = rf.predict(X_test)

    # =====================================
    # Evaluation
    # =====================================

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    # =====================================
    # Save Model
    # =====================================

    joblib.dump(
        rf,
        "random_forest.pkl"
    )

print("Model berhasil disimpan")