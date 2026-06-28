import os
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import dagshub

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay
)

# =====================================
# Load Dataset
# =====================================

df = pd.read_csv("data/telco_processed.csv")

X = df.drop("Churn Label", axis=1)
y = df["Churn Label"]

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================
# DagsHub + MLflow
# =====================================

print("DAGSHUB TOKEN FOUND:", bool(os.getenv("DAGSHUB_TOKEN")))

token = os.getenv("DAGSHUB_TOKEN")

if token:
    dagshub.auth.add_app_token(token)

dagshub.init(
    repo_owner="mhmdfkri",
    repo_name="Membangun_Model_Muhamad-Fikri",
    mlflow=True
)

mlflow.set_experiment("Telco Churn Advanced")

with mlflow.start_run(run_name="RandomForest_Tuning_v1"):

    rf = RandomForestClassifier(
        random_state=42
    )

    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10]
    }

    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=10,
        cv=3,
        scoring="f1",
        random_state=42,
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Best Params :", search.best_params_)
    print("Accuracy    :", accuracy)
    print("Precision   :", precision)
    print("Recall      :", recall)
    print("F1 Score    :", f1)

    # ======================
    # Log Parameter
    # ======================

    mlflow.log_params(search.best_params_)

    # ======================
    # Log Metrics
    # ======================

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # ======================
    # Confusion Matrix
    # ======================

    fig, ax = plt.subplots()

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        ax=ax
    )

    plt.savefig("confusion_matrix.png")
    plt.close()

    mlflow.log_artifact("confusion_matrix.png")

    # ======================
    # Feature Importance
    # ======================

    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": best_model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False
    )

    importance_df.to_csv(
        "feature_importance.csv",
        index=False
    )

    mlflow.log_artifact("feature_importance.csv")

    # ======================
    # Log Model
    # ======================

    mlflow.sklearn.log_model(
        sk_model=best_model,
        name="model"
    )

print("Training selesai")