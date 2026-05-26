"""
fgsm_attack.py
Topic: FGSM attack, gradient sign method, Logistic Regression,
       continuous feature perturbation, epsilon budget
What it does: runs the exact FGSM formula for sklearn Logistic
              Regression and stores the gradient/sign details used
              to perturb real dataset samples.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(__file__))))

import numpy  as np
from config import DATA_ADV, N_ATTACK_SAMPLES


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def lr_loss_gradient(lr, X, y):
    """
    Binary logistic regression cross-entropy gradient:

        p = sigmoid(w.x + b)
        loss = -y log(p) - (1-y) log(1-p)
        d loss / d x = (p - y) * w
    """
    w = lr.coef_[0]
    b = lr.intercept_[0]
    logits = X @ w + b
    probs = sigmoid(logits)
    gradients = (probs - y).reshape(-1, 1) * w.reshape(1, -1)
    losses = -(y * np.log(probs + 1e-12) +
               (1 - y) * np.log(1 - probs + 1e-12))
    return gradients, probs, losses


def run_fgsm_on_lr(lr, X_test, y_test, eps=0.1, save_prefix="lr"):
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

    print(f"Running analytical FGSM for LR (eps={eps}) on {N} samples...")
    gradients, _, _ = lr_loss_gradient(lr, X_sub, y_sub)
    perturbation = eps * np.sign(gradients)
    X_adv = X_sub + perturbation

    # Save
    np.save(os.path.join(DATA_ADV, f"{save_prefix}_X_adv_fgsm.npy"),
            X_adv)
    np.save(os.path.join(DATA_ADV, f"{save_prefix}_X_test_sub.npy"),
            X_sub)
    np.save(os.path.join(DATA_ADV, f"{save_prefix}_y_test_sub.npy"),
            y_sub)

    print(f"FGSM done. Adversarial shape: {X_adv.shape}")
    return X_adv, X_sub, y_sub


def load_lr_adv():
    return (
        np.load(os.path.join(DATA_ADV, "lr_X_adv_fgsm.npy")),
        np.load(os.path.join(DATA_ADV, "lr_X_test_sub.npy")),
        np.load(os.path.join(DATA_ADV, "lr_y_test_sub.npy")),
    )
