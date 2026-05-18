"""
dta_attack.py
Topic: ART library, SklearnClassifier wrapper, DecisionTreeAttack,
       white-box attack, .generate(), numpy save
What it does: wraps the trained DT for ART, runs DTA on a subset
              of test samples, saves adversarial array to .npy
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import joblib
from art.estimators.classification import SklearnClassifier
from art.attacks.evasion            import DecisionTreeAttack
from config import DATA_ADV, MODELS_DIR, N_ATTACK_SAMPLES


def run_dta(dt, X_test, y_test):
    os.makedirs(DATA_ADV, exist_ok=True)

    # Take a subset — DTA is slow on very large sets
    N     = min(N_ATTACK_SAMPLES, len(X_test))
    X_sub = X_test[:N]
    y_sub = y_test[:N]

    # Wrap the sklearn model for ART
    print("Wrapping model for ART...")
    art_dt = SklearnClassifier(
        model=dt,
        clip_values=(float(X_sub.min()), float(X_sub.max()))
    )

    # Run DTA
    print(f"Running DTA on {N} samples...")
    print("(This takes 3-10 minutes depending on your machine)\n")
    attack = DecisionTreeAttack(classifier=art_dt)
    X_adv  = attack.generate(x=X_sub)

    # Save
    np.save(os.path.join(DATA_ADV, "X_adv_dta.npy"), X_adv)
    np.save(os.path.join(DATA_ADV, "X_test_sub.npy"),  X_sub)
    np.save(os.path.join(DATA_ADV, "y_test_sub.npy"),  y_sub)

    print(f"Saved adversarial dataset: {X_adv.shape}")
    return X_adv, X_sub, y_sub


def load_adv():
    return (
        np.load(os.path.join(DATA_ADV, "X_adv_dta.npy")),
        np.load(os.path.join(DATA_ADV, "X_test_sub.npy")),
        np.load(os.path.join(DATA_ADV, "y_test_sub.npy")),
    )


if __name__ == "__main__":
    from src.data_loader import load_processed
    _, X_test, _, y_test = load_processed()
    dt = joblib.load(os.path.join(MODELS_DIR, "dt.pkl"))
    run_dta(dt, X_test, y_test)