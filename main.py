"""
main.py — full DTA pipeline including adversarial training defence
Steps:
  1. Prepare data
  2. Train original Decision Tree
  3. Run DTA attack on test set
  4. Evaluate attack impact
  5. Adversarial training (retrain on expanded dataset)
  6. Compare original vs defended model
  7. Visualise everything
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import joblib
import numpy as np
from config                       import MODELS_DIR, DATA_ADV
from src.data_loader              import prepare, load_processed
from src.train_model              import train, evaluate_clean
from src.dta_attack               import run_dta, load_adv
from src.evaluate                 import evaluate
from src.adversarial_training     import (
    generate_adv_for_training,
    build_expanded_dataset,
    train_defended,
    compare_results
)
from src.visualise                import (
    plot_dashboard,
    export_csv,
    plot_comparison_dashboard
)


def main():

    # ── Step 1: Prepare data ───────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 1: PREPARE DATA")
    print("="*60)
    split_path = os.path.join("data", "processed", "split.pkl")
    if os.path.exists(split_path):
        print("Cached split found — loading from disk")
        X_train, X_test, y_train, y_test = load_processed()
    else:
        X_train, X_test, y_train, y_test = prepare()

    # ── Step 2: Train original Decision Tree ───────────────────────
    print("\n" + "="*60)
    print("STEP 2: TRAIN ORIGINAL DECISION TREE")
    print("="*60)
    model_path = os.path.join(MODELS_DIR, "dt.pkl")
    if os.path.exists(model_path):
        print("Cached model found — loading from disk")
        dt_original = joblib.load(model_path)
    else:
        dt_original = train(X_train, y_train)
    evaluate_clean(dt_original, X_test, y_test)

    # ── Step 3: Run DTA attack on test set ─────────────────────────
    print("\n" + "="*60)
    print("STEP 3: RUN DTA ATTACK ON TEST SET")
    print("="*60)
    adv_path = os.path.join(DATA_ADV, "X_adv_dta.npy")
    if os.path.exists(adv_path):
        print("Cached adversarial test data found — loading")
        X_adv_test, X_test_sub, y_test_sub = load_adv()
    else:
        X_adv_test, X_test_sub, y_test_sub = run_dta(
            dt_original, X_test, y_test
        )

    # ── Step 4: Evaluate attack impact ─────────────────────────────
    print("\n" + "="*60)
    print("STEP 4: EVALUATE ATTACK IMPACT")
    print("="*60)
    attack_results = evaluate(
        dt_original, X_test_sub, X_adv_test, y_test_sub
    )
    plot_dashboard(attack_results, X_test_sub, X_adv_test, y_test_sub)
    export_csv(
        X_test_sub, X_adv_test, y_test_sub,
        attack_results["y_pred_clean"],
        attack_results["y_pred_adv"]
    )

    # ── Step 5: Generate adversarial samples from TRAINING set ─────
    print("\n" + "="*60)
    print("STEP 5: GENERATE ADVERSARIAL TRAINING SAMPLES")
    print("="*60)
    adv_train_path = os.path.join(DATA_ADV, "X_adv_train.npy")
    if os.path.exists(adv_train_path):
        print("Cached adversarial training data found — loading")
        X_adv_train = np.load(adv_train_path)
        y_adv_train = np.load(os.path.join(DATA_ADV, "y_adv_train.npy"))
    else:
        X_adv_train, y_adv_train = generate_adv_for_training(
            dt_original, X_train, y_train
        )
        np.save(adv_train_path, X_adv_train)
        np.save(os.path.join(DATA_ADV, "y_adv_train.npy"), y_adv_train)

    # ── Step 6: Build expanded dataset and retrain ─────────────────
    print("\n" + "="*60)
    print("STEP 6: ADVERSARIAL TRAINING — RETRAIN ON EXPANDED DATA")
    print("="*60)
    defended_path = os.path.join(MODELS_DIR, "dt_defended.pkl")
    if os.path.exists(defended_path):
        print("Cached defended model found — loading")
        dt_defended = joblib.load(defended_path)
    else:
        X_expanded, y_expanded = build_expanded_dataset(
            X_train, y_train, X_adv_train, y_adv_train
        )
        dt_defended = train_defended(X_expanded, y_expanded)

    # ── Step 7: Compare original vs defended model ─────────────────
    print("\n" + "="*60)
    print("STEP 7: COMPARE ORIGINAL VS DEFENDED MODEL")
    print("="*60)
    comparison = compare_results(
        dt_original, dt_defended,
        X_test_sub, X_adv_test, y_test_sub
    )
    plot_comparison_dashboard(
        comparison, X_test_sub, X_adv_test, y_test_sub
    )

    # ── Final summary ───────────────────────────────────────────────
    print("\n" + "="*60)
    print("COMPLETE — CHECK results/ FOLDER")
    print("="*60)
    print("  dta_analysis.png                 — attack impact")
    print("  adversarial_training_comparison.png — defence results")
    print("  dta_results.csv                  — sample level data")


if __name__ == "__main__":
    main()