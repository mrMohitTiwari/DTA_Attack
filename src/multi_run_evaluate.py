"""
multi_run_evaluate.py
Topic: statistical evaluation, mean F1, std F1, model variance
What it does:
    Trains the Decision Tree 5 times with different random seeds.
    Each run generates its own adversarial samples and evaluates.
    Reports mean and std of F1 across all runs — this is what
    your supervisor wants to see. Single-run results could be
    lucky. Five runs with consistent results proves stability.

    Also filters to ATTACK-ONLY samples so you evaluate
    purely on how well the model detects adversarial attacks.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy  as np
import pandas as pd
import joblib

from sklearn.tree    import DecisionTreeClassifier
from sklearn.metrics import f1_score, classification_report

from art.estimators.classification import SklearnClassifier
from art.attacks.evasion            import DecisionTreeAttack

from config import (MODELS_DIR, DATA_ADV, RESULTS_DIR,
                    DT_PARAMS, MAX_TRAIN_SAMPLES,
                    N_ATTACK_SAMPLES, RANDOM_SEEDS)


def train_single(X_train, y_train, seed):
    """Train one DT with a specific random seed."""
    params = {**DT_PARAMS, "random_state": seed}

    if len(X_train) > MAX_TRAIN_SAMPLES:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X_train), MAX_TRAIN_SAMPLES, replace=False)
        X_tr, y_tr = X_train[idx], y_train[idx]
    else:
        X_tr, y_tr = X_train, y_train

    dt = DecisionTreeClassifier(**params)
    dt.fit(X_tr, y_tr)
    return dt


def run_dta_single(dt, X_test, y_test):
    """Run DTA on attack samples only, return adversarial array."""
    # Filter to attack samples only
    attack_idx = np.where(y_test == 1)[0]
    N          = min(N_ATTACK_SAMPLES, len(attack_idx))
    chosen     = attack_idx[:N]

    X_sub = X_test[chosen]
    y_sub = y_test[chosen]

    art_dt = SklearnClassifier(
        model       = dt,
        clip_values = (float(X_sub.min()), float(X_sub.max()))
    )
    attack = DecisionTreeAttack(classifier=art_dt)
    X_adv  = attack.generate(x=X_sub)

    return X_adv, X_sub, y_sub


def adversarial_train_single(dt, X_train, y_train, X_adv, y_adv, seed):
    """Retrain DT on clean + adversarial samples."""
    X_exp = np.vstack([X_train, X_adv])
    y_exp = np.concatenate([y_train, y_adv])

    params = {**DT_PARAMS, "random_state": seed}

    if len(X_exp) > MAX_TRAIN_SAMPLES:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X_exp), MAX_TRAIN_SAMPLES, replace=False)
        X_exp, y_exp = X_exp[idx], y_exp[idx]

    dt_def = DecisionTreeClassifier(**params)
    dt_def.fit(X_exp, y_exp)
    return dt_def


def run_multi_seed_evaluation(X_train, X_test, y_train, y_test):
    """
    Full pipeline across all seeds in RANDOM_SEEDS.

    For each seed:
      1. Train original DT
      2. Run DTA → get adversarial samples (attack-only)
      3. Evaluate original DT on adversarial samples
      4. Adversarial training → defended DT
      5. Evaluate defended DT on adversarial samples
      6. Record F1 scores

    Then compute mean and std across all seeds.

    Returns:
        summary dict with all per-run and aggregate stats
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR,  exist_ok=True)

    print("\n" + "="*65)
    print("MULTI-SEED EVALUATION — ATTACK-ONLY SAMPLES")
    print(f"Seeds: {RANDOM_SEEDS}")
    print(f"Each run: train → DTA attack → adversarial training → evaluate")
    print("="*65)

    per_run_results = []

    for run_idx, seed in enumerate(RANDOM_SEEDS):
        print(f"\n{'─'*65}")
        print(f"RUN {run_idx+1}/{len(RANDOM_SEEDS)}  seed={seed}")
        print(f"{'─'*65}")

        # ── 1. Train original DT ──────────────────────────────────
        print(f"  Training original DT...")
        dt_orig = train_single(X_train, y_train, seed)

        # Evaluate on clean test data
        y_pred_clean = dt_orig.predict(X_test)
        f1_clean     = f1_score(y_test, y_pred_clean)
        print(f"  F1 on clean data:        {f1_clean:.4f}")

        # ── 2. Run DTA — attack-only subset ───────────────────────
        print(f"  Running DTA (attack-only, N={N_ATTACK_SAMPLES})...")
        X_adv, X_sub, y_sub = run_dta_single(dt_orig, X_test, y_test)

        # ── 3. Evaluate original DT under attack ──────────────────
        y_pred_adv_orig = dt_orig.predict(X_adv)
        f1_orig_adv     = f1_score(y_sub, y_pred_adv_orig, zero_division=0)
        n_fooled_orig   = int(np.sum(y_pred_adv_orig == 0))
        print(f"  Original F1 under DTA:   {f1_orig_adv:.4f}"
              f"  ({n_fooled_orig}/{len(y_sub)} attacks fooled)")

        # ── 4. Generate adversarial training samples from train set ─
        print(f"  Generating adversarial training samples...")
        attack_train_idx = np.where(y_train == 1)[0]
        N_train          = min(N_ATTACK_SAMPLES, len(attack_train_idx))
        X_tr_sub         = X_train[attack_train_idx[:N_train]]
        y_tr_sub         = y_train[attack_train_idx[:N_train]]

        art_dt_tmp = SklearnClassifier(
            model       = dt_orig,
            clip_values = (float(X_tr_sub.min()), float(X_tr_sub.max()))
        )
        atk_tmp    = DecisionTreeAttack(classifier=art_dt_tmp)
        X_adv_train = atk_tmp.generate(x=X_tr_sub)

        # ── 5. Retrain defended DT ─────────────────────────────────
        print(f"  Adversarial training...")
        dt_def = adversarial_train_single(
            dt_orig, X_train, y_train, X_adv_train, y_tr_sub, seed
        )

        # ── 6. Evaluate defended DT under attack ───────────────────
        y_pred_adv_def  = dt_def.predict(X_adv)
        f1_def_adv      = f1_score(y_sub, y_pred_adv_def, zero_division=0)
        f1_def_clean    = f1_score(y_test, dt_def.predict(X_test))
        n_fooled_def    = int(np.sum(y_pred_adv_def == 0))
        print(f"  Defended F1 under DTA:   {f1_def_adv:.4f}"
              f"  ({n_fooled_def}/{len(y_sub)} attacks fooled)")
        print(f"  Defended F1 clean data:  {f1_def_clean:.4f}")

        per_run_results.append({
            "seed":          seed,
            "f1_clean_orig": f1_clean,
            "f1_adv_orig":   f1_orig_adv,
            "f1_clean_def":  f1_def_clean,
            "f1_adv_def":    f1_def_adv,
            "n_total":       len(y_sub),
            "n_fooled_orig": n_fooled_orig,
            "n_fooled_def":  n_fooled_def,
        })

    # ── Aggregate stats ────────────────────────────────────────────
    f1_orig_list = [r["f1_adv_orig"] for r in per_run_results]
    f1_def_list  = [r["f1_adv_def"]  for r in per_run_results]
    f1_clean_list = [r["f1_clean_orig"] for r in per_run_results]

    summary = {
        "per_run":          per_run_results,
        "mean_f1_clean":    float(np.mean(f1_clean_list)),
        "std_f1_clean":     float(np.std(f1_clean_list)),
        "mean_f1_orig_adv": float(np.mean(f1_orig_list)),
        "std_f1_orig_adv":  float(np.std(f1_orig_list)),
        "mean_f1_def_adv":  float(np.mean(f1_def_list)),
        "std_f1_def_adv":   float(np.std(f1_def_list)),
    }

    # ── Print final table ──────────────────────────────────────────
    print("\n" + "="*65)
    print("FINAL RESULTS ACROSS ALL SEEDS (ATTACK-ONLY EVALUATION)")
    print("="*65)
    print(f"\nPer-run breakdown:")
    print(f"{'Seed':<8} {'F1 clean':>10} {'F1 orig(adv)':>14} "
          f"{'F1 def(adv)':>13} {'Fooled orig':>13} {'Fooled def':>12}")
    print(f"{'─'*8} {'─'*10} {'─'*14} {'─'*13} {'─'*13} {'─'*12}")
    for r in per_run_results:
        print(f"{r['seed']:<8} {r['f1_clean_orig']:>10.4f} "
              f"{r['f1_adv_orig']:>14.4f} {r['f1_adv_def']:>13.4f} "
              f"{r['n_fooled_orig']:>10}/{r['n_total']} "
              f"{r['n_fooled_def']:>9}/{r['n_total']}")

    print(f"\nAggregate statistics:")
    print(f"{'Metric':<40} {'Mean':>8} {'Std':>8}")
    print(f"{'─'*58}")
    print(f"{'F1 on clean data':<40} "
          f"{summary['mean_f1_clean']:>8.4f} "
          f"{summary['std_f1_clean']:>8.4f}")
    print(f"{'F1 under DTA (original model)':<40} "
          f"{summary['mean_f1_orig_adv']:>8.4f} "
          f"{summary['std_f1_orig_adv']:>8.4f}")
    print(f"{'F1 under DTA (defended model)':<40} "
          f"{summary['mean_f1_def_adv']:>8.4f} "
          f"{summary['std_f1_def_adv']:>8.4f}")
    print(f"{'F1 recovery (def - orig)':<40} "
          f"{summary['mean_f1_def_adv'] - summary['mean_f1_orig_adv']:>+8.4f}")

    # ── Save CSV ───────────────────────────────────────────────────
    df = pd.DataFrame(per_run_results)
    csv_path = os.path.join(RESULTS_DIR, "multi_seed_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    return summary


if __name__ == "__main__":
    from src.data_loader import load_processed
    X_train, X_test, y_train, y_test = load_processed()
    run_multi_seed_evaluation(X_train, X_test, y_train, y_test)