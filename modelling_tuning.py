import os
import joblib
import dagshub
import pandas as pd
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn

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

X = pd.get_dummies(
    X,
    drop_first=True
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================
# DagsHub
# =====================================

print("DAGSHUB TOKEN FOUND :", bool(os.getenv("DAGSHUB_TOKEN")))

token = os.getenv("DAGSHUB_TOKEN")

if token:
    dagshub.auth.add_app_token(token)

dagshub.init(
    repo_owner="mhmdfkri",
    repo_name="Membangun_Model_Muhamad-Fikri",
    mlflow=True
)

mlflow.set_experiment("Telco Churn Advanced")

# =====================================
# Output Folder
# =====================================

os.makedirs("models", exist_ok=True)

# =====================================
# Start MLflow Run
# =====================================

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

    print("=" * 50)
    print("BEST PARAMETER")
    print(search.best_params_)
    print("=" * 50)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    # =====================================
    # Manual Logging
    # =====================================

    mlflow.log_params(search.best_params_)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("best_cv_score", search.best_score_)

    # =====================================
    # Confusion Matrix
    # =====================================

    fig, ax = plt.subplots()

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        ax=ax
    )

    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close(fig)

    mlflow.log_artifact("confusion_matrix.png")

    # =====================================
    # Feature Importance
    # =====================================

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

    # =====================================
    # Best Parameter
    # =====================================

    pd.DataFrame([search.best_params_]).to_csv(
        "best_params.csv",
        index=False
    )

    mlflow.log_artifact("best_params.csv")

    # =====================================
    # Save Model
    # =====================================

    joblib.dump(
        best_model,
        "models/random_forest.pkl"
    )

    mlflow.sklearn.log_model(
        sk_model=best_model,
        name="model"
    )

print("\nTraining selesai.")
print("Model tersimpan di models/random_forest.pkl")