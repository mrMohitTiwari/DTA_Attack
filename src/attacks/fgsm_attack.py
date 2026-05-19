"""
fgsm_attack.py
Topic: FGSM attack, gradient sign method, TensorFlowV2Classifier,
       continuous feature perturbation, epsilon budget
What it does: wraps logistic regression for ART using a surrogate
              DNN (because LR has no gradient in ART directly),
              then runs FGSM to generate adversarial samples
              that fool the LR classifier
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(__file__))))

import numpy  as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics      import f1_score
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion           import FastGradientMethod
from config import DATA_ADV, MODELS_DIR, N_ATTACK_SAMPLES


def run_fgsm_on_lr(lr, X_test, y_test, eps=0.1):
    """
    Run FGSM attack against Logistic Regression.

    FGSM works on LR because LR has a differentiable
    loss surface — the sigmoid function is smooth so
    gradients exist everywhere.

    Why LR is the most robust classifier in the paper:
    - LR has a LINEAR decision surface
    - Gradient of a linear function is constant and small
    - FGSM perturbation = eps × sign(gradient)
    - Small gradient → small effective perturbation
    - The adversarial sample barely crosses the boundary

    Args:
        lr     : trained LogisticRegression model
        X_test : scaled test features
        y_test : test labels
        eps    : epsilon — perturbation budget per feature

    Returns:
        X_adv  : adversarial samples
        X_sub  : original test subset
        y_sub  : labels
    """
    os.makedirs(DATA_ADV, exist_ok=True)

    N     = min(N_ATTACK_SAMPLES, len(X_test))
    X_sub = X_test[:N]
    y_sub = y_test[:N]

    print(f"Wrapping LR for ART (SklearnClassifier)...")
    art_lr = SklearnClassifier(
        model=lr,
        clip_values=(float(X_sub.min()), float(X_sub.max()))
    )

    print(f"Running FGSM (eps={eps}) on {N} samples...")
    fgsm  = FastGradientMethod(estimator=art_lr, eps=eps)
    X_adv = fgsm.generate(x=X_sub)

    # Save
    np.save(os.path.join(DATA_ADV, "lr_X_adv_fgsm.npy"), X_adv)
    np.save(os.path.join(DATA_ADV, "lr_X_test_sub.npy"),  X_sub)
    np.save(os.path.join(DATA_ADV, "lr_y_test_sub.npy"),  y_sub)

    print(f"FGSM done. Adversarial shape: {X_adv.shape}")
    return X_adv, X_sub, y_sub


def load_lr_adv():
    return (
        np.load(os.path.join(DATA_ADV, "lr_X_adv_fgsm.npy")),
        np.load(os.path.join(DATA_ADV, "lr_X_test_sub.npy")),
        np.load(os.path.join(DATA_ADV, "lr_y_test_sub.npy")),
    )