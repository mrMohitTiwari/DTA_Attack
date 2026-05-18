"""
train_model.py
Topic: DecisionTreeClassifier, entropy criterion, max_depth,
       F1 score, classification_report, joblib model saving
What it does: trains a single decision tree, evaluates it on clean
              test data, saves model to disk
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy  as np
import joblib
from sklearn.tree    import DecisionTreeClassifier
from sklearn.metrics import f1_score, classification_report
from config import DT_PARAMS, MODELS_DIR, MAX_TRAIN_SAMPLES
from src.data_loader import load_processed


def train(X_train, y_train):
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Subsample if dataset is very large
    if len(X_train) > MAX_TRAIN_SAMPLES:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_train), MAX_TRAIN_SAMPLES, replace=False)
        X_tr, y_tr = X_train[idx], y_train[idx]
        print(f"Subsampled to {MAX_TRAIN_SAMPLES} rows")
    else:
        X_tr, y_tr = X_train, y_train

    print("Training Decision Tree...")
    dt = DecisionTreeClassifier(**DT_PARAMS)
    dt.fit(X_tr, y_tr)

    model_path = os.path.join(MODELS_DIR, "dt.pkl")
    joblib.dump(dt, model_path)
    print(f"Model saved: {model_path}")
    return dt


def evaluate_clean(dt, X_test, y_test):
    y_pred = dt.predict(X_test)
    f1     = f1_score(y_test, y_pred)
    print(f"\nF1 on clean test data: {f1:.4f}")
    print(classification_report(y_test, y_pred,
          target_names=['BENIGN', 'ATTACK']))
    return f1, y_pred


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_processed()
    dt = train(X_train, y_train)
    evaluate_clean(dt, X_test, y_test)