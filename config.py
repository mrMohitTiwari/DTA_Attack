import os

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_RAW        = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED  = os.path.join(BASE_DIR, "data", "processed")
DATA_ADV        = os.path.join(BASE_DIR, "data", "adversarial")
MODELS_DIR      = os.path.join(BASE_DIR, "models")
RESULTS_DIR     = os.path.join(BASE_DIR, "results")

# Decision Tree hyperparameters (from ELAT paper Table 3)
DT_PARAMS = {
    "criterion":        "entropy",
    "max_depth":        9,
    "min_samples_leaf": 1,
    "min_samples_split":2,
    "random_state":     42
}

# Data settings
TEST_SIZE        = 0.20
RANDOM_STATE     = 42
MAX_TRAIN_SAMPLES = 200_000   # cap for speed
N_ATTACK_SAMPLES  = 2_000     # samples to run DTA on