"""
evaluate.py
Topic: F1 score, confusion matrix, attack success rate,
       numpy boolean indexing
What it does: compares model predictions on clean vs adversarial
              data, prints full attack impact analysis
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, classification_report


def evaluate(dt, X_clean, X_adv, y_true):
    y_pred_clean = dt.predict(X_clean)
    y_pred_adv   = dt.predict(X_adv)

    f1_clean = f1_score(y_true, y_pred_clean)
    f1_adv   = f1_score(y_true, y_pred_adv)

    # Attack success — real attacks flipped to benign
    attack_idx = np.where(y_true == 1)[0]
    flipped    = np.where(
        (y_pred_clean[attack_idx] == 1) &
        (y_pred_adv[attack_idx]   == 0)
    )[0]

    print("=" * 55)
    print("DTA ATTACK RESULTS")
    print("=" * 55)
    print(f"F1 on clean data:            {f1_clean:.4f}")
    print(f"F1 on adversarial data:      {f1_adv:.4f}")
    print(f"F1 drop:                     {f1_clean - f1_adv:.4f}")
    print(f"\nTotal attack samples:        {len(attack_idx)}")
    print(f"Successfully fooled:          {len(flipped)}")
    print(f"Attack success rate:          {len(flipped)/len(attack_idx):.2%}")

    print(f"\nPerturbation stats:")
    norms = np.linalg.norm(X_adv - X_clean, axis=1)
    print(f"  Mean L2 norm:              {norms.mean():.6f}")
    print(f"  Max  L2 norm:              {norms.max():.6f}")
    feat_changed = np.sum(np.abs(X_adv - X_clean) > 1e-6, axis=1)
    print(f"  Mean features changed:     {feat_changed.mean():.2f}")

    print(f"\nClassification Report (adversarial):")
    print(classification_report(y_true, y_pred_adv,
          target_names=['BENIGN', 'ATTACK']))

    return {
        "f1_clean":    f1_clean,
        "f1_adv":      f1_adv,
        "f1_drop":     f1_clean - f1_adv,
        "n_attack":    len(attack_idx),
        "n_flipped":   len(flipped),
        "success_rate":len(flipped)/len(attack_idx),
        "y_pred_clean":y_pred_clean,
        "y_pred_adv":  y_pred_adv,
    }


if __name__ == "__main__":
    import joblib
    from src.data_loader import load_processed
    from src.dta_attack  import load_adv
    from config import MODELS_DIR
    _, X_test, _, y_test = load_processed()
    dt               = joblib.load(os.path.join(MODELS_DIR, "dt.pkl"))
    X_adv, X_sub, y_sub = load_adv()
    evaluate(dt, X_sub, X_adv, y_sub)