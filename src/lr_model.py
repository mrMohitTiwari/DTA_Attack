"""
lr_model.py
Topic: LogisticRegression, C parameter (inverse regularization),
       sigmoid function, decision boundary, weights and bias,
       F1 score, classification report, joblib saving
What it does: trains logistic regression with and without
              adversarial training, evaluates both, compares
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy   as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.metrics      import (f1_score, classification_report,
                                   confusion_matrix)
from config import MODELS_DIR, MAX_TRAIN_SAMPLES


# ── Hyperparameters from ELAT paper Table 3 ───────────────────────
LR_PARAMS = {
    "C":        202010.513969,
    # C is inverse of regularization strength
    # Large C = low regularization = model fits training data closely
    # Paper found this large value optimal — their dataset was
    # large enough that overfitting was not a concern
    "max_iter": 2000,
    "solver":   "lbfgs",
    # lbfgs = Limited-memory Broyden–Fletcher–Goldfarb–Shanno
    # An efficient quasi-Newton optimizer — good for large datasets
    "random_state": 42
}


def train_lr(X_train, y_train, params=None):
    """
    Train Logistic Regression on clean data.

    What LR learns:
    - One weight per feature: how much each network flow
      statistic influences the attack/benign decision
    - One bias term: baseline probability before seeing features
    - Decision rule: sigmoid(w·x + b) > 0.5 → predict attack

    Args:
        X_train : scaled training features
        y_train : training labels
        params  : hyperparameter dict, uses LR_PARAMS if None

    Returns:
        lr : trained LogisticRegression model
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    if params is None:
        params = LR_PARAMS

    # Subsample if too large
    if len(X_train) > MAX_TRAIN_SAMPLES:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_train), MAX_TRAIN_SAMPLES,
                         replace=False)
        X_tr, y_tr = X_train[idx], y_train[idx]
        print(f"Subsampled to {MAX_TRAIN_SAMPLES} rows")
    else:
        X_tr, y_tr = X_train, y_train

    print("Training Logistic Regression...")
    lr = LogisticRegression(**params)
    lr.fit(X_tr, y_tr)

    path = os.path.join(MODELS_DIR, "lr.pkl")
    joblib.dump(lr, path)
    print(f"LR saved: {path}")

    # Print weights insight
    w     = lr.coef_[0]
    b     = lr.intercept_[0]
    top5  = np.argsort(np.abs(w))[::-1][:5]
    print(f"\nTop 5 most influential features (by weight magnitude):")
    for rank, idx in enumerate(top5):
        print(f"  {rank+1}. feature_{idx:3d}  "
              f"weight={w[idx]:+.4f}  "
              f"({'pushes toward ATTACK' if w[idx]>0 else 'pushes toward BENIGN'})")
    print(f"Bias (intercept): {b:+.4f}")

    return lr


def evaluate_lr(lr, X_test, y_test, label="clean"):
    """
    Evaluate LR and return metrics dict.

    Args:
        lr     : trained LogisticRegression
        X_test : test features
        y_test : true labels
        label  : string for print output

    Returns:
        metrics dict with f1, precision, recall, confusion matrix
    """
    y_pred = lr.predict(X_test)
    f1     = f1_score(y_test, y_pred)
    cm     = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=['BENIGN','ATTACK'],
        output_dict=True
    )

    print(f"\nLR evaluation [{label}]:")
    print(f"  F1 score:   {f1:.4f}")
    print(f"  Precision:  {report['ATTACK']['precision']:.4f}")
    print(f"  Recall:     {report['ATTACK']['recall']:.4f}")
    print(classification_report(y_test, y_pred,
          target_names=['BENIGN','ATTACK']))

    return {
        "f1":        f1,
        "precision": report['ATTACK']['precision'],
        "recall":    report['ATTACK']['recall'],
        "cm":        cm,
        "y_pred":    y_pred,
        "report":    report
    }


def build_expanded_dataset_lr(X_train, y_train,
                               X_adv,   y_adv):
    """
    Stack clean + adversarial training data for LR retraining.

    For LR adversarial training:
    - Model sees both clean and perturbed samples
    - LR adjusts its weights to correctly classify
      adversarial samples as attacks
    - Because LR has a simple linear boundary,
      adversarial training shifts that boundary
      to be more robust to FGSM-style perturbations

    Args:
        X_train : clean training features
        y_train : clean training labels
        X_adv   : adversarial samples (from FGSM on training set)
        y_adv   : labels for adversarial samples

    Returns:
        X_expanded, y_expanded
    """
    X_expanded = np.vstack([X_train, X_adv])
    y_expanded = np.concatenate([y_train, y_adv])

    print(f"Clean training samples:     {X_train.shape[0]}")
    print(f"Adversarial samples added:  {X_adv.shape[0]}")
    print(f"Expanded total:             {X_expanded.shape[0]}")
    print(f"Attack ratio:               {y_expanded.mean():.2%}")

    return X_expanded, y_expanded


def train_defended_lr(X_expanded, y_expanded):
    """
    Retrain LR on expanded dataset — adversarial training.

    The defended LR learns weights that correctly classify
    both original AND adversarially perturbed samples.
    Because FGSM perturbations follow the gradient direction,
    the defended model learns to be less sensitive to
    gradient-direction perturbations.

    Returns:
        lr_defended : retrained LogisticRegression
    """
    print("Training defended LR on expanded dataset...")
    lr_defended = LogisticRegression(**LR_PARAMS)

    if len(X_expanded) > MAX_TRAIN_SAMPLES:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_expanded), MAX_TRAIN_SAMPLES,
                         replace=False)
        X_tr, y_tr = X_expanded[idx], y_expanded[idx]
    else:
        X_tr, y_tr = X_expanded, y_expanded

    lr_defended.fit(X_tr, y_tr)

    path = os.path.join(MODELS_DIR, "lr_defended.pkl")
    joblib.dump(lr_defended, path)
    print(f"Defended LR saved: {path}")

    return lr_defended


def compare_weights(lr_original, lr_defended, n_features=10):
    """
    Compare weights of original vs defended LR.

    This is a unique insight into LR — you can see
    exactly HOW adversarial training changed the model.
    Features whose weights changed most are the ones
    FGSM was exploiting most heavily.

    Args:
        lr_original : original trained LR
        lr_defended : adversarially trained LR
        n_features  : how many top features to show
    """
    w_orig = lr_original.coef_[0]
    w_def  = lr_defended.coef_[0]
    w_diff = np.abs(w_def - w_orig)
    top_idx = np.argsort(w_diff)[::-1][:n_features]

    print("\n── Weight changes from adversarial training ──")
    print(f"{'Feature':<15} {'Original W':>12} "
          f"{'Defended W':>12} {'Change':>12}")
    print("-" * 55)
    for i in top_idx:
        print(f"feature_{i:<7d} {w_orig[i]:>+12.4f} "
              f"{w_def[i]:>+12.4f} "
              f"{w_def[i]-w_orig[i]:>+12.4f}")

    return w_orig, w_def, top_idx


if __name__ == "__main__":
    from src.data_loader import load_processed
    X_train, X_test, y_train, y_test = load_processed()
    lr = train_lr(X_train, y_train)
    evaluate_lr(lr, X_test, y_test, label="clean test")