"""
lr_pipeline.py
Topic: full ML pipeline orchestration, model caching,
       step-by-step adversarial training workflow
What it does: runs the complete LR pipeline —
              train → attack → defend → compare → visualise
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import joblib

from config import MODELS_DIR, DATA_ADV, N_ATTACK_SAMPLES

# ── ALL imports at top level — no imports inside functions ─────────
from src.lr_model       import (train_lr, evaluate_lr,
                                 build_expanded_dataset_lr,
                                 train_defended_lr,
                                 compare_weights,
                                 LR_PARAMS)
from src.attacks.fgsm_attack import run_fgsm_on_lr, load_lr_adv
from src.lr_visualise   import (plot_lr_dashboard,
                                 plot_lr_vs_dt_comparison)


def run_lr_pipeline(X_train, X_test, y_train, y_test,
                    dt_results=None):
    """
    Full pipeline for LR adversarial training experiment.

    Steps:
      1. Train original LR on clean data
      2. Evaluate on clean test data
      3. Run FGSM attack on test set
      4. Evaluate original LR under attack
      5. Generate FGSM adversarial training samples
      6. Retrain LR on expanded dataset
      7. Evaluate defended LR under attack
      8. Compare weights original vs defended
      9. Visualise all results
      10. Compare LR vs DT if dt_results provided
    """

    # ── Step 1: Train original LR ──────────────────────────────────
    print("\n" + "="*60)
    print("LR STEP 1: TRAIN ORIGINAL LOGISTIC REGRESSION")
    print("="*60)
    lr_path = os.path.join(MODELS_DIR, "lr.pkl")
    if os.path.exists(lr_path):
        print("Cached LR found — loading from disk")
        lr_orig = joblib.load(lr_path)
    else:
        lr_orig = train_lr(X_train, y_train)

    # ── Step 2: Evaluate on clean data ────────────────────────────
    print("\n" + "="*60)
    print("LR STEP 2: EVALUATE ON CLEAN DATA")
    print("="*60)
    clean_metrics = evaluate_lr(
        lr_orig, X_test, y_test, label="clean test"
    )
    f1_clean = clean_metrics["f1"]

    # ── Step 3: Run FGSM attack on test set ───────────────────────
    print("\n" + "="*60)
    print("LR STEP 3: RUN FGSM ATTACK ON TEST SET")
    print("="*60)
    adv_path = os.path.join(DATA_ADV, "lr_X_adv_fgsm.npy")
    if os.path.exists(adv_path):
        print("Cached FGSM adversarial data — loading")
        X_adv_test, X_test_sub, y_test_sub = load_lr_adv()
    else:
        X_adv_test, X_test_sub, y_test_sub = run_fgsm_on_lr(
            lr_orig, X_test, y_test, eps=0.1
        )

    # ── Step 4: Evaluate original LR under attack ─────────────────
    print("\n" + "="*60)
    print("LR STEP 4: EVALUATE ORIGINAL LR UNDER FGSM")
    print("="*60)
    attack_metrics_orig = evaluate_lr(
        lr_orig, X_adv_test, y_test_sub,
        label="original LR under FGSM"
    )
    attack_metrics_orig["f1_clean"] = f1_clean

    # ── Step 5: Generate FGSM samples from training set ───────────
    print("\n" + "="*60)
    print("LR STEP 5: GENERATE FGSM ADVERSARIAL TRAINING SAMPLES")
    print("="*60)
    adv_train_path = os.path.join(DATA_ADV, "lr_X_adv_train.npy")
    adv_label_path = os.path.join(DATA_ADV, "lr_y_adv_train.npy")

    if os.path.exists(adv_train_path):
        print("Cached adversarial training data — loading")
        X_adv_train = np.load(adv_train_path)
        y_adv_train = np.load(adv_label_path)
    else:
        print("Generating FGSM on training set subset...")
        attack_idx  = np.where(y_train == 1)[0]
        N           = min(N_ATTACK_SAMPLES, len(attack_idx))
        X_tr_sub    = X_train[attack_idx[:N]]
        y_tr_sub    = y_train[attack_idx[:N]]

        # run_fgsm_on_lr is imported at top — no local import needed
        X_adv_train, _, _ = run_fgsm_on_lr(
            lr_orig, X_tr_sub, y_tr_sub, eps=0.1
        )
        y_adv_train = y_tr_sub

        np.save(adv_train_path, X_adv_train)
        np.save(adv_label_path, y_adv_train)
        print(f"Adversarial training samples saved.")

    # ── Step 6: Build expanded dataset and retrain ─────────────────
    print("\n" + "="*60)
    print("LR STEP 6: ADVERSARIAL TRAINING — RETRAIN ON EXPANDED DATA")
    print("="*60)
    defended_path = os.path.join(MODELS_DIR, "lr_defended.pkl")
    if os.path.exists(defended_path):
        print("Cached defended LR — loading")
        lr_def = joblib.load(defended_path)
    else:
        X_exp, y_exp = build_expanded_dataset_lr(
            X_train, y_train,
            X_adv_train, y_adv_train
        )
        lr_def = train_defended_lr(X_exp, y_exp)

    # ── Step 7: Evaluate defended LR ──────────────────────────────
    print("\n" + "="*60)
    print("LR STEP 7: EVALUATE DEFENDED LR UNDER FGSM")
    print("="*60)
    clean_def = evaluate_lr(
        lr_def, X_test, y_test,
        label="defended LR on clean data"
    )
    attack_metrics_def = evaluate_lr(
        lr_def, X_adv_test, y_test_sub,
        label="defended LR under FGSM"
    )
    attack_metrics_def["f1_clean"] = clean_def["f1"]

    # ── Step 8: Compare weights ────────────────────────────────────
    print("\n" + "="*60)
    print("LR STEP 8: WEIGHT ANALYSIS")
    print("="*60)
    compare_weights(lr_orig, lr_def, n_features=10)

    # ── Compute attack success rates ───────────────────────────────
    attack_idx   = np.where(y_test_sub == 1)[0]
    n_atk        = len(attack_idx)
    n_fool_orig  = int(np.sum(
        attack_metrics_orig["y_pred"][attack_idx] == 0
    ))
    n_fool_def   = int(np.sum(
        attack_metrics_def["y_pred"][attack_idx] == 0
    ))

    # ── Print final comparison table ──────────────────────────────
    print("\n" + "="*60)
    print("LR FINAL RESULTS")
    print("="*60)
    print(f"{'Metric':<35} {'Original':>10} {'Defended':>10}")
    print("-"*58)
    print(f"{'F1 on clean data':<35} "
          f"{f1_clean:>10.4f} "
          f"{clean_def['f1']:>10.4f}")
    print(f"{'F1 under FGSM attack':<35} "
          f"{attack_metrics_orig['f1']:>10.4f} "
          f"{attack_metrics_def['f1']:>10.4f}")
    print(f"{'F1 recovery':<35} "
          f"{'—':>10} "
          f"{attack_metrics_def['f1'] - attack_metrics_orig['f1']:>+10.4f}")
    print(f"{'Attacks fooled':<35} "
          f"{n_fool_orig:>10} "
          f"{n_fool_def:>10}")
    print(f"{'Attack success rate':<35} "
          f"{n_fool_orig/n_atk:>10.2%} "
          f"{n_fool_def/n_atk:>10.2%}")

    # ── Build results dict ─────────────────────────────────────────
    lr_results = {
        "f1_clean":      f1_clean,
        "f1_adv":        attack_metrics_orig["f1"],
        "f1_defended":   attack_metrics_def["f1"],
        "orig_asr":      n_fool_orig / n_atk * 100,
        "def_asr":       n_fool_def  / n_atk * 100,
        "n_fooled_orig": n_fool_orig,
        "n_fooled_def":  n_fool_def,
        "n_attack":      n_atk,
    }

    # ── Step 9: Visualise ──────────────────────────────────────────
    print("\n" + "="*60)
    print("LR STEP 9: GENERATE VISUALISATIONS")
    print("="*60)
    plot_lr_dashboard(
        lr_orig, lr_def,
        X_test_sub, X_adv_test, y_test_sub,
        attack_metrics_orig, attack_metrics_def
    )

    # ── Step 10: LR vs DT comparison ──────────────────────────────
    if dt_results is not None:
        print("\n" + "="*60)
        print("LR STEP 10: LR vs DT COMPARISON CHART")
        print("="*60)
        plot_lr_vs_dt_comparison(lr_results, dt_results)

    return lr_results


if __name__ == "__main__":
    from src.data_loader import load_processed
    X_train, X_test, y_train, y_test = load_processed()
    run_lr_pipeline(X_train, X_test, y_train, y_test)
