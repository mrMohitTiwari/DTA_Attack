"""
adversarial_training.py
Topic: adversarial training, expanded dataset, model retraining,
       numpy vstack, decision tree robustness improvement
What it does: takes clean training data + adversarial samples,
              stacks them into one expanded dataset, retrains
              the decision tree on this expanded dataset so it
              learns to recognise adversarial patterns
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy  as np
import joblib
from sklearn.tree    import DecisionTreeClassifier
from sklearn.metrics import f1_score, classification_report
from config import DT_PARAMS, MODELS_DIR, MAX_TRAIN_SAMPLES


def build_expanded_dataset(X_train, y_train, X_adv, y_adv):
    """
    Stack clean training data with adversarial samples.

    Why we do this:
    - X_train contains normal clean network traffic
    - X_adv contains adversarially perturbed samples
    - Combining both forces the model to learn patterns
      of both normal AND adversarial inputs
    - The labels stay the same — adversarial samples
      are still attacks (label=1), just disguised ones

    Args:
        X_train : clean training features  (n_train, n_features)
        y_train : clean training labels    (n_train,)
        X_adv   : adversarial samples      (n_adv, n_features)
        y_adv   : labels for adv samples   (n_adv,)

    Returns:
        X_expanded : stacked features      (n_train + n_adv, n_features)
        y_expanded : stacked labels        (n_train + n_adv,)
    """
    X_expanded = np.vstack([X_train, X_adv])
    y_expanded = np.concatenate([y_train, y_adv])

    print(f"Clean training samples:      {X_train.shape[0]}")
    print(f"Adversarial samples added:   {X_adv.shape[0]}")
    print(f"Expanded dataset total:      {X_expanded.shape[0]}")
    print(f"Feature count:               {X_expanded.shape[1]}")
    print(f"Attack ratio in expanded:    {y_expanded.mean():.2%}")

    return X_expanded, y_expanded


def train_defended(X_expanded, y_expanded):
    """
    Retrain the Decision Tree on the expanded dataset.

    The model now sees adversarial examples during training,
    so it learns to classify them correctly as attacks
    even when their features have been perturbed.

    Args:
        X_expanded : combined clean + adversarial features
        y_expanded : combined labels

    Returns:
        dt_defended : retrained decision tree
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Subsample if too large
    if len(X_expanded) > MAX_TRAIN_SAMPLES:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_expanded), MAX_TRAIN_SAMPLES, replace=False)
        X_tr = X_expanded[idx]
        y_tr = y_expanded[idx]
        print(f"Subsampled expanded set to {MAX_TRAIN_SAMPLES} rows")
    else:
        X_tr, y_tr = X_expanded, y_expanded

    print("Retraining Decision Tree on expanded dataset...")
    dt_defended = DecisionTreeClassifier(**DT_PARAMS)
    dt_defended.fit(X_tr, y_tr)

    # Save with _defended suffix
    path = os.path.join(MODELS_DIR, "dt_defended.pkl")
    joblib.dump(dt_defended, path)
    print(f"Defended model saved: {path}")

    return dt_defended


def generate_adv_for_training(dt, X_train, y_train):
    """
    Generate adversarial samples from the TRAINING set
    (not test set) so we can include them in retraining.

    We use a subset of X_train to keep it manageable.

    Args:
        dt      : original trained decision tree
        X_train : clean training features
        y_train : clean training labels

    Returns:
        X_adv_train : adversarial version of training subset
        y_adv_train : corresponding labels
    """
    from art.estimators.classification import SklearnClassifier
    from art.attacks.evasion            import DecisionTreeAttack
    from config import N_ATTACK_SAMPLES

    # Only attack samples that are real attacks in training set
    attack_idx  = np.where(y_train == 1)[0]
    N           = min(N_ATTACK_SAMPLES, len(attack_idx))
    chosen_idx  = attack_idx[:N]

    X_sub = X_train[chosen_idx]
    y_sub = y_train[chosen_idx]

    print(f"Generating adversarial training samples from {N} attack rows...")

    art_dt = SklearnClassifier(
        model=dt,
        clip_values=(float(X_sub.min()), float(X_sub.max()))
    )
    attack = DecisionTreeAttack(classifier=art_dt)
    X_adv  = attack.generate(x=X_sub)

    print(f"Adversarial training samples generated: {X_adv.shape}")
    return X_adv, y_sub


def compare_results(dt_original, dt_defended,
                    X_clean, X_adv_test, y_true):
    """
    Compare original vs defended model on adversarial test data.
    This is the key result that shows adversarial training works.

    Args:
        dt_original  : model trained on clean data only
        dt_defended  : model trained on clean + adversarial data
        X_clean      : original test samples
        X_adv_test   : adversarial test samples (from DTA)
        y_true       : true labels

    Prints a comparison table and returns a results dict.
    """
    # Original model predictions
    y_orig_clean = dt_original.predict(X_clean)
    y_orig_adv   = dt_original.predict(X_adv_test)

    # Defended model predictions
    y_def_clean  = dt_defended.predict(X_clean)
    y_def_adv    = dt_defended.predict(X_adv_test)

    # F1 scores
    f1_orig_clean = f1_score(y_true, y_orig_clean)
    f1_orig_adv   = f1_score(y_true, y_orig_adv)
    f1_def_clean  = f1_score(y_true, y_def_clean)
    f1_def_adv    = f1_score(y_true, y_def_adv)

    # Attack success rates
    attack_idx = np.where(y_true == 1)[0]

    orig_flipped = np.sum(
        (y_orig_clean[attack_idx] == 1) &
        (y_orig_adv[attack_idx]   == 0)
    )
    def_flipped = np.sum(
        (y_def_clean[attack_idx] == 1) &
        (y_def_adv[attack_idx]   == 0)
    )

    print("\n" + "=" * 60)
    print("ADVERSARIAL TRAINING — COMPARISON RESULTS")
    print("=" * 60)
    print(f"{'Metric':<35} {'Original':>10} {'Defended':>10}")
    print("-" * 60)
    print(f"{'F1 on clean data':<35} {f1_orig_clean:>10.4f} {f1_def_clean:>10.4f}")
    print(f"{'F1 under DTA attack':<35} {f1_orig_adv:>10.4f} {f1_def_adv:>10.4f}")
    print(f"{'F1 recovery after defence':<35} {'—':>10} {f1_def_adv - f1_orig_adv:>+10.4f}")
    print(f"{'Samples fooled by DTA':<35} {orig_flipped:>10} {def_flipped:>10}")
    print(f"{'Attack success rate':<35} {orig_flipped/len(attack_idx):>10.2%} {def_flipped/len(attack_idx):>10.2%}")
    print("=" * 60)

    if f1_def_adv > f1_orig_adv:
        improvement = f1_def_adv - f1_orig_adv
        print(f"\nAdversarial training IMPROVED robustness by {improvement:.4f} F1 points")
    else:
        print("\nAdversarial training did not improve robustness on this subset")
        print("Try increasing N_ATTACK_SAMPLES in config.py")

    return {
        "f1_orig_clean": f1_orig_clean,
        "f1_orig_adv":   f1_orig_adv,
        "f1_def_clean":  f1_def_clean,
        "f1_def_adv":    f1_def_adv,
        "orig_flipped":  orig_flipped,
        "def_flipped":   def_flipped,
        "n_attack":      len(attack_idx),
        "y_orig_clean":  y_orig_clean,
        "y_orig_adv":    y_orig_adv,
        "y_def_clean":   y_def_clean,
        "y_def_adv":     y_def_adv,
    }


if __name__ == "__main__":
    from src.data_loader import load_processed
    from src.dta_attack  import load_adv
    _, X_test, _, y_test = load_processed()
    X_train_full, _, y_train_full, _ = load_processed()

    dt_original = joblib.load(os.path.join(MODELS_DIR, "dt.pkl"))
    X_adv_test, X_sub, y_sub = load_adv()

    # Generate adversarial samples from training data
    X_adv_train, y_adv_train = generate_adv_for_training(
        dt_original, X_train_full, y_train_full
    )

    # Build expanded dataset and retrain
    X_expanded, y_expanded = build_expanded_dataset(
        X_train_full, y_train_full, X_adv_train, y_adv_train
    )
    dt_defended = train_defended(X_expanded, y_expanded)

    # Compare
    compare_results(dt_original, dt_defended, X_sub, X_adv_test, y_sub)