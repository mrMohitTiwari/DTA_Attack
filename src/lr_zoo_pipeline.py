"""
lr_zoo_pipeline.py
Standalone ZOO black-box attack experiment for Logistic Regression.

This is separate from lr_pipeline.py because ZOO is query-based and much
slower than FGSM. Use it when you specifically want black-box LR results.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import joblib

from config import MODELS_DIR, DATA_ADV
from src.data_loader import load_processed
from src.lr_model import train_lr, evaluate_lr
from src.attacks.zoo_attack import run_zoo_on_lr, load_lr_zoo_adv
from src.lr_visualise import plot_lr_zoo_dashboard


def run_lr_zoo_pipeline(X_train=None, X_test=None,
                        y_train=None, y_test=None,
                        force=False):
    """
    Load/train LR, run ZOO on a small balanced subset, evaluate, plot.
    """
    if X_train is None or X_test is None or y_train is None or y_test is None:
        X_train, X_test, y_train, y_test = load_processed()

    print("\n" + "=" * 60)
    print("LR ZOO STEP 1: LOAD/TRAIN LOGISTIC REGRESSION")
    print("=" * 60)
    lr_path = os.path.join(MODELS_DIR, "lr.pkl")
    if os.path.exists(lr_path):
        print("Cached LR found — loading from disk")
        lr = joblib.load(lr_path)
    else:
        lr = train_lr(X_train, y_train)

    print("\n" + "=" * 60)
    print("LR ZOO STEP 2: RUN BLACK-BOX ZOO ATTACK")
    print("=" * 60)
    adv_path = os.path.join(DATA_ADV, "lr_X_adv_zoo.npy")
    if os.path.exists(adv_path) and not force:
        print("Cached ZOO adversarial data — loading")
        X_adv, X_sub, y_sub = load_lr_zoo_adv()
    else:
        X_adv, X_sub, y_sub = run_zoo_on_lr(lr, X_test, y_test)

    print("\n" + "=" * 60)
    print("LR ZOO STEP 3: EVALUATE")
    print("=" * 60)
    clean_metrics = evaluate_lr(lr, X_sub, y_sub,
                                label="ZOO clean subset")
    zoo_metrics = evaluate_lr(lr, X_adv, y_sub,
                              label="LR under ZOO")

    print("\n" + "=" * 60)
    print("LR ZOO STEP 4: VISUALISE")
    print("=" * 60)
    plot_lr_zoo_dashboard(clean_metrics, zoo_metrics,
                          X_sub, X_adv, y_sub)

    return {
        "clean": clean_metrics,
        "zoo": zoo_metrics,
        "X_clean": X_sub,
        "X_adv": X_adv,
        "y": y_sub,
    }


if __name__ == "__main__":
    run_lr_zoo_pipeline()
