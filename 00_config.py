# =============================================================================
# 00_config.py - UniFraud-GH Shared Configuration
# =============================================================================

import os

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.path.abspath('.')

DATA_DIR   = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
FIGURE_DIR = os.path.join(BASE_DIR, "figures")

for d in [OUTPUT_DIR, MODEL_DIR, FIGURE_DIR]:
    os.makedirs(d, exist_ok=True)

# --- Raw Data Files -----------------------------------------------------------

# Credit Card - Sparkov synthetic dataset (Kaggle: kartik2112/fraud-detection)
# To download: kaggle datasets download -d kartik2112/fraud-detection --unzip
CREDIT_CARD_TRAIN_CSV = os.path.join(DATA_DIR, "fraudTrain.csv")
CREDIT_CARD_TEST_CSV  = os.path.join(DATA_DIR, "fraudTest.csv")

CC_RAW_TARGET     = "is_fraud"
CC_AMOUNT_COL     = "amt"
CC_TIME_COL       = "trans_date_trans_time"
CC_CATEGORY_COL   = "category"
CC_MERCHANT_COL   = "merchant"
CC_DOB_COL        = "dob"
CC_LAT_COL        = "lat"
CC_LONG_COL       = "long"
CC_MERCH_LAT_COL  = "merch_lat"
CC_MERCH_LONG_COL = "merch_long"
CC_CITY_POP_COL   = "city_pop"

# Mobile Money - MoMTSim (IEEE Access 2024, DOI: 10.1109/ACCESS.2024.3439012)
MOMSIM_CSV     = os.path.join(DATA_DIR, "MoMTSim_transactions.csv")
MOMSIM_CSV_ALT = os.path.join(DATA_DIR, "paysim.csv")

# Health Insurance - Ghana NHIA G-DRG tariff reference and claims data
GD_CSV = os.path.join(DATA_DIR, "GD.csv")
REDATA_FILES = [
    os.path.join(DATA_DIR, "ReData1_100.csv"),
    os.path.join(DATA_DIR, "ReData2_300.csv"),
    os.path.join(DATA_DIR, "ReData3_500.csv"),
    os.path.join(DATA_DIR, "ReData4_750.csv"),
    os.path.join(DATA_DIR, "ReData5_1000.csv"),
]

# Health Insurance - NHIS claims with provided fraud labels (de-identified copy).
# Build it from the original NHIS workbook: names/DOB/real IDs removed, member/folder
# IDs replaced by random surrogates, fraud_label = 1 for FRAUDULENT. NEVER commit it.
HI_CLAIMS_CSV = os.path.join(DATA_DIR, "NHIS_claims_deidentified.csv")

# Which HI labelling is in use:
#   "provided"   -> labels come from the NHIS workbook (use 01_health_insurance_claims.ipynb)
#   "rule_based" -> labels derived from G-DRG tariff rules (01_health_insurance_labeling.ipynb);
#                   billing-derived features then leak the label and are removed in NB05/07/08.
HI_LABEL_SOURCE = "provided"

# In the NHIS workbook, ICD-10 is missing for 60 of the 87 fraudulent claims and for NO legitimate claim.
# A "has_icd10" flag therefore acts as a near-rule for most fraud (grouped-CV AP about 0.58 with it,
# about 0.04 without it). Default False so the headline results are not driven by it; report both.
HI_USE_MISSING_ICD_FLAG = False

# --- Intermediate and Output Files -------------------------------------------
HI_LABELED_CSV   = os.path.join(OUTPUT_DIR, "health_insurance_labeled.csv")
HI_GROUPS_CSV    = os.path.join(OUTPUT_DIR, "health_insurance_groups.csv")   # row_id, member_id, date (for grouped / temporal splits)
CC_PROCESSED_CSV = os.path.join(OUTPUT_DIR, "credit_card_processed.csv")
MM_PROCESSED_CSV = os.path.join(OUTPUT_DIR, "mobile_money_processed.csv")
HI_PROCESSED_CSV = os.path.join(OUTPUT_DIR, "health_insurance_processed.csv")
CC_FEATURES_CSV  = os.path.join(OUTPUT_DIR, "credit_card_features.csv")
MM_FEATURES_CSV  = os.path.join(OUTPUT_DIR, "mobile_money_features.csv")
HI_FEATURES_CSV  = os.path.join(OUTPUT_DIR, "health_insurance_features.csv")

# --- Target Column -----------------------------------------------------------
TARGET_COL = "fraud_label"

# --- Ghana Economic Constants ------------------------------------------------
# Exchange rate used throughout the project for GHS conversions
USD_TO_GHS = 15.8
GHS_TO_USD = 1 / USD_TO_GHS

GHANA_PEAK_HOURS = {
    "morning": (7,  9),
    "lunch":   (12, 14),
    "evening": (17, 20),
}

# --- Random Seed -------------------------------------------------------------
RANDOM_STATE = 42

# --- Model Hyperparameters ---------------------------------------------------
RF_PARAMS = {
    "n_estimators": 100,
    "max_depth":    10,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs":       -1,
}

XGB_PARAMS = {
    "n_estimators":  100,
    "learning_rate": 0.1,
    "max_depth":     6,
    "eval_metric":   "logloss",
    "random_state":  RANDOM_STATE,
    "n_jobs":        -1,
}

ISOLATION_FOREST_PARAMS = {
    "n_estimators": 100,
    "random_state": RANDOM_STATE,
    "n_jobs":       -1,
}

# --- Preprocessing Settings --------------------------------------------------
MISSING_DROP_THRESHOLD = 0.20
KNN_IMPUTE_K           = 5
SMOTE_RATIO            = 0.30
OUTLIER_CAP_PERCENTILE = 99

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# --- DANN (Domain-Adversarial Neural Network) Settings -----------------------
DANN_HIDDEN_DIMS = [128, 64]
DANN_EPOCHS      = 50
DANN_LR          = 0.001
DANN_LAMBDA_MAX  = 1.0

# --- VAE (Variational Autoencoder) Settings ----------------------------------
VAE_LATENT_DIM  = 16
VAE_HIDDEN_DIMS = [64, 32]
VAE_EPOCHS      = 50
VAE_BATCH_SIZE  = 128
VAE_LR          = 0.001
VAE_BETA        = 1.0

# --- Evaluation Settings -----------------------------------------------------
CV_FOLDS          = 5
BOOTSTRAP_SAMPLES = 1000
ALPHA             = 0.05

# --- Domain Labels -----------------------------------------------------------
DOMAIN_CREDIT_CARD  = 0
DOMAIN_MOBILE_MONEY = 1
DOMAIN_HEALTH_INS   = 2
DOMAIN_NAMES = {
    DOMAIN_CREDIT_CARD:  "Credit Card",
    DOMAIN_MOBILE_MONEY: "Mobile Money",
    DOMAIN_HEALTH_INS:   "Health Insurance",
}

print("Config loaded. BASE_DIR:", BASE_DIR)
