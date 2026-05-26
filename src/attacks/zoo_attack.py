"""
zoo_attack.py
Topic: Zeroth Order Optimization (ZOO), black-box attack,
       Logistic Regression, ART SklearnClassifier wrapper

What it does:
    Runs a small, query-based ZOO attack against Logistic Regression.
    ZOO estimates gradients using model queries instead of reading model
    weights, so it is much slower than FGSM and should use fewer samples.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(__file__))))

import numpy as np
from art.attacks.evasion import ZooAttack
from art.estimators.classification import SklearnClassifier

from config import DATA_ADV, N_ZOO_SAMPLES


def balanced_subset(X, y, n_samples):
    """Return a small benign/attack-balanced subset when possible."""
    benign_idx = np.where(y == 0)[0]
    attack_idx = np.where(y == 1)[0]
    half = n_samples // 2

    chosen = np.concatenate([
        benign_idx[:half],
        attack_idx[:n_samples - half],
    ])

    if len(chosen) < n_samples:
        remaining = np.setdiff1d(np.arange(len(y)), chosen,
                                 assume_unique=False)
        chosen = np.concatenate([
            chosen,
            remaining[:n_samples - len(chosen)]
        ])

    return X[chosen], y[chosen]


def run_zoo_on_lr(lr, X_test, y_test, n_samples=None,
                  max_iter=8, learning_rate=0.01):
    """
    Run ZOO against Logistic Regression through ART.

    Args:
        lr            : trained sklearn LogisticRegression
        X_test/y_test : scaled test data and labels
        n_samples     : small sample count; defaults to config
        max_iter      : ZOO optimizer iterations
        learning_rate : ZOO optimizer learning rate

    Returns:
        X_adv, X_sub, y_sub
    """
    os.makedirs(DATA_ADV, exist_ok=True)

    if n_samples is None:
        n_samples = N_ZOO_SAMPLES

    n_samples = min(n_samples, len(X_test))
    X_sub, y_sub = balanced_subset(X_test, y_test, n_samples)

    print(f"Wrapping LR for ZOO black-box attack on {len(X_sub)} samples...")
    art_lr = SklearnClassifier(
        model=lr,
        clip_values=(float(X_sub.min()), float(X_sub.max()))
    )

    print(
        "Running ZOO "
        f"(max_iter={max_iter}, learning_rate={learning_rate})..."
    )
    attack = ZooAttack(
        classifier=art_lr,
        max_iter=max_iter,
        binary_search_steps=1,
        learning_rate=learning_rate,
        initial_const=0.001,
        nb_parallel=32,
        batch_size=1,
        verbose=True,
    )
    X_adv = attack.generate(x=X_sub)

    np.save(os.path.join(DATA_ADV, "lr_X_adv_zoo.npy"), X_adv)
    np.save(os.path.join(DATA_ADV, "lr_X_test_zoo.npy"), X_sub)
    np.save(os.path.join(DATA_ADV, "lr_y_test_zoo.npy"), y_sub)

    print(f"ZOO done. Adversarial shape: {X_adv.shape}")
    return X_adv, X_sub, y_sub


def load_lr_zoo_adv():
    return (
        np.load(os.path.join(DATA_ADV, "lr_X_adv_zoo.npy")),
        np.load(os.path.join(DATA_ADV, "lr_X_test_zoo.npy")),
        np.load(os.path.join(DATA_ADV, "lr_y_test_zoo.npy")),
    )
