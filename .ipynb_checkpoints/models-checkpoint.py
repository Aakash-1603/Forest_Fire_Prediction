# models.py
import joblib
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_squared_error, r2_score
)

# ----------------------------
# Train classification models
# ----------------------------
def train_classification(X_train, y_train):
    models = {}

    # Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    models["random_forest"] = rf

    # Optional: XGBoost Classifier
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, use_label_encoder=False, eval_metric="logloss")
        xgb.fit(X_train, y_train)
        models["xgboost"] = xgb
    except ImportError:
        pass  # if xgboost is not installed, skip

    return models


# ----------------------------
# Train regression models
# ----------------------------
def train_regression(X_train, y_train):
    models = {}

    # Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    models["random_forest_reg"] = rf

    # Optional: XGBoost Regressor
    try:
        from xgboost import XGBRegressor
        xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        xgb.fit(X_train, y_train)
        models["xgboost_reg"] = xgb
    except ImportError:
        pass

    return models


# ----------------------------
# Evaluate classification models
# ----------------------------
def evaluate_classification(model, X_test, y_test):
    preds = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, average="weighted", zero_division=0),
        "recall": recall_score(y_test, preds, average="weighted", zero_division=0),
        "f1": f1_score(y_test, preds, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, preds)
    }


# ----------------------------
# Evaluate regression models
# ----------------------------
def evaluate_regression(model, X_test, y_test):
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    return {"rmse": rmse, "r2": r2}


# ----------------------------
# Save & load models
# ----------------------------
def save_model(model, name, folder="models"):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{name}.joblib")
    joblib.dump(model, path)
    return path

def load_model(name, folder="models"):
    path = os.path.join(folder, f"{name}.joblib")
    if os.path.exists(path):
        return joblib.load(path)
    else:
        raise FileNotFoundError(f"Model {name} not found at {path}")
