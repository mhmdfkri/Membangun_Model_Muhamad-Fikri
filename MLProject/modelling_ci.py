import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

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
# MLflow
# =====================================

mlflow.set_experiment("Telco Churn CI")

with mlflow.start_run():

    rf = RandomForestClassifier(
        random_state=42
    )

    param_dist = {
        "n_estimators":[100,200,300],
        "max_depth":[5,10,15,None],
        "min_samples_split":[2,5,10]
    }

    search = RandomizedSearchCV(
        rf,
        param_dist,
        cv=3,
        n_iter=10,
        random_state=42,
        scoring="f1",
        n_jobs=-1
    )

    search.fit(X_train,y_train)

    model = search.best_estimator_

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)

    mlflow.log_params(search.best_params_)

    mlflow.log_metric("accuracy",accuracy)
    mlflow.log_metric("precision",precision)
    mlflow.log_metric("recall",recall)
    mlflow.log_metric("f1_score",f1)

    fig,ax=plt.subplots()

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        ax=ax
    )

    plt.savefig("confusion_matrix.png")
    plt.close()

    mlflow.log_artifact("confusion_matrix.png")

    mlflow.sklearn.log_model(
        model,
        "model"
    )

print("Workflow CI selesai.")