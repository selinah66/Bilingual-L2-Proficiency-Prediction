from pathlib import Path

# Data paths
# This assumes config.py is located at src/l2_proficiency/config.py.
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "Data"
FILEPATH = DATA_DIR / "RawEyeMovement.xlsx"

# Data Processing Settings
TARGET_COLUMN = "L2 PROFICIENCY"

# Feature Selection
NUMERIC_COLUMNS = [
    'IA_FIRST_FIXATION_DURATION', 'IA_FIRST_RUN_DWELL_TIME', 'IA_REGRESSION_PATH_DURATION',
    'IA_DWELL_TIME', 'IA_FIRST_SACCADE_AMPLITUDE', 'IA_FIXATION_COUNT',
    'IA_FIRST_RUN_FIXATION_COUNT', 'IA_REGRESSION_IN_COUNT'
]

# Rows missing these measurements are removed during preprocessing.
CRITICAL_COLUMNS = [
    'IA_REGRESSION_PATH_DURATION', 'IA_FIRST_FIXATION_DURATION',
    'IA_FIRST_RUN_DWELL_TIME', 'IA_FIRST_SACCADE_AMPLITUDE'
]

# Features to protect during preprocessing
PROTECTED_COLUMNS = [
    'IA_FIRST_FIXATION_DURATION',
    'IA_REGRESSION_PATH_DURATION',
    'IA_DWELL_TIME'
]

# Groups used to filter rows whose measurements are entirely missing.
MISSINGNESS_FILTER_GROUPS = [
    ['IA_REGRESSION_PATH_DURATION', 'IA_FIRST_FIXATION_DURATION', 'IA_FIRST_RUN_DWELL_TIME'],
    ['IA_FIRST_SACCADE_AMPLITUDE']
]

# Model Settings
# Train-test split parameters
TEST_SIZE = 0.4
RANDOM_STATE = 42

# Random Forest hyperparameters
N_ESTIMATORS = 50
MAX_DEPTH = 10
MIN_SAMPLES_SPLIT = 10
MIN_SAMPLES_LEAF = 4
MAX_FEATURES = 0.3

# Cross-validation settings
CV_FOLDS = 5
N_SPLITS = 5
N_REPEATS = 3