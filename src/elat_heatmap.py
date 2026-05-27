"""
elat_heatmap.py
Generate a heatmap comparing this project's adversarial-trained F1
scores with the ELAT paper's reported average recovered F1 score.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import joblib
from sklearn.metrics import f1_score

from config import MODELS_DIR
from src.data_loader import load_processed
from src.dta_attack import load_adv
from src.attacks.fgsm_attack import load_lr_adv
from src.lr_visualise import plot_elat_f1_heatmap


def build_cached_results():
    _, X_test, _, y_test = load_processed()

    dt_def = joblib.load(os.path.join(MODELS_DIR, "dt_defended.pkl"))
    X_adv_dt, _, y_dt_sub = load_adv()
    dt_results = {
        "f1_defended": f1_score(y_dt_sub, dt_def.predict(X_adv_dt)),
    }

    lr_def = joblib.load(os.path.join(MODELS_DIR, "lr_defended.pkl"))
    X_adv_lr, _, y_lr_sub = load_lr_adv()
    lr_results = {
        "f1_defended": f1_score(y_lr_sub, lr_def.predict(X_adv_lr)),
    }

    return lr_results, dt_results


def main():
    lr_results, dt_results = build_cached_results()
    plot_elat_f1_heatmap(lr_results, dt_results)


if __name__ == "__main__":
    main()
