# UniFraud-GH: Cross-Domain Fraud Detection for Ghana

A multi-domain fraud detection research system covering three financial sectors in Ghana: **credit card transactions**, **mobile money transfers**, and **health insurance claims**. This project explores domain adaptation techniques to transfer fraud knowledge across sectors, addressing the challenge of limited labelled data in any single domain.

Developed as part of a thesis project at Ashesi University.

Financial fraud in Ghana operates across multiple sectors simultaneously. Fraudsters who exploit credit card systems often employ similar behavioural patterns in mobile money and insurance claim fraud. This project builds on that observation by:

1. Labelling health insurance fraud using Ghana NHIA G-DRG tariff compliance rules (Sowah et al., 2019)
2. Engineering domain-specific and universal features across all three sectors
3. Training baseline classifiers (Logistic Regression, Random Forest, XGBoost, Isolation Forest, Autoencoder) per domain
4. Applying three domain adaptation methods: DANN, Subdomain Adaptation with MMD, and HEN
5. Evaluating a stacking ensemble and cross-domain transfer across all six source-target pairs
6. Explaining model decisions using SHAP and LIME

---

## Repository Structure

```
UniFraud-GH/
|
|-- 00_config.py                        <- Shared configuration (paths, hyperparameters, constants)
|-- 01_health_insurance_labeling.ipynb  <- Fraud labelling using G-DRG tariff rules (legacy rule-based labels)
|-- 01c_health_insurance_claims.ipynb   <- NHIS claims with PROVIDED fraud labels (use this if you have the NHIS workbook)
|-- 01b_EDA.ipynb                       <- Exploratory data analysis across all three domains
|-- 02_preprocessing.ipynb              <- 7-step preprocessing pipeline
|-- 03_feature_engineering.ipynb        <- Domain-specific and universal feature engineering
|-- 05_baseline_model.ipynb             <- Baseline models per domain
|-- 06_domain_adaptation.ipynb          <- DANN, Subdomain Adaptation, and HEN transfer
|-- 07_ensemble_evaluation.ipynb        <- 5 experiments + stacking ensemble + statistical tests
|-- 08_explainability.ipynb             <- SHAP and LIME explainability
|
|-- tools/deidentify_nhis.py            <- Removes names/DOB/real IDs from the NHIS workbook (run locally)
|-- requirements.txt                    <- Python dependencies
|-- data/                               <- (not committed) Raw datasets (see below)
|-- outputs/                            <- Generated CSVs and results (auto-created)
|-- models/                             <- Saved model files (auto-created)
|-- figures/                            <- Saved plots (auto-created)
```

---

## Datasets

The raw data files are **not included in this repository** due to size and licensing. Download them and place them in the `data/` folder.

| Dataset | Source | Filename(s) |
|---|---|---|
| Credit Card (Sparkov) | Kaggle: kartik2112/fraud-detection | `fraudTrain.csv`, `fraudTest.csv` |
| Mobile Money (MoMTSim) | IEEE Access 2024 / GitHub | `MoMTSim_transactions.csv` |
| Health Insurance (NHIA, legacy rule-based labels) | Ghana NHIA (provided) | `GD.csv`, `ReData1_100.csv` ... `ReData5_1000.csv` |
| Health Insurance (NHIS claims, provided labels) | NHIS workbook (provided; **not public**) | `NHIS_claims_deidentified.csv` (made from `NHIS_CLAIM_DATASET.xlsx`, see below) |

### Health insurance claims: privacy and preparation

The NHIS workbook contains patient names, dates of birth and member numbers. **Never commit it, and never commit anything derived from it that still carries identifiers.** `data/` is git-ignored, and `.gitignore` also blocks `*.xlsx` and `NHIS*`.

Create the de-identified file locally (the original never leaves your machine):

```
python tools/deidentify_nhis.py "C:/path/to/NHIS_CLAIM_DATASET.xlsx"
```

This drops names, DOB and CCC, replaces member/folder numbers with random surrogate IDs, derives `age_at_admission` and `length_of_stay`, and recodes the label so that `fraud_label = 1` means fraud (the workbook column `LEGITIMATE=1, FRAUDULENT=0` is the opposite). Then run `01c_health_insurance_claims.ipynb` and keep `HI_LABEL_SOURCE = "provided"` in `00_config.py`.

Things to know about these labels: fraud is rare (87 of 9,721 claims, 0.9%); 77 of the 87 sit in two outpatient G-DRG codes; ICD-10 is missing on 60 of the 87 fraudulent claims and on no legitimate claim, so that flag is excluded by default (`HI_USE_MISSING_ICD_FLAG`) and results should be reported with and without it; 50 members have claims of both classes, so split by `member_id` (see `outputs/health_insurance_groups.csv`), not by row. With 87 positives a single split is very noisy: use repeated grouped cross-validation and report intervals.

To download the credit card dataset using the Kaggle CLI:
```
kaggle datasets download -d kartik2112/fraud-detection --unzip -p data/
```

---

## Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/sedemabla/UniFraud-GH.git
cd UniFraud-GH
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add data files

Create a `data/` folder in the project root and place all dataset files there (see table above).

---

## How to Run

Run the notebooks **in order**. Each notebook saves its outputs to the `outputs/` folder so the next notebook can read them.

| Step | Notebook | Description |
|---|---|---|
| 1 | `01_health_insurance_labeling.ipynb` **or** `01c_health_insurance_claims.ipynb` | Labels HI fraud from G-DRG tariff rules (legacy), **or** loads the NHIS claims with provided labels |
| 2 | `01b_EDA.ipynb` | Exploratory analysis of all three datasets |
| 3 | `02_preprocessing.ipynb` | Cleans and prepares all three datasets |
| 4 | `03_feature_engineering.ipynb` | Builds domain features and selects top 25 per domain |
| 5 | `05_baseline_model.ipynb` | Trains 5 baseline models per domain |
| 6 | `06_domain_adaptation.ipynb` | Runs DANN, Subdomain, and HEN cross-domain transfer |
| 7 | `07_ensemble_evaluation.ipynb` | Evaluates 5 experiments and stacking ensemble |
| 8 | `08_explainability.ipynb` | Generates SHAP beeswarm, force, dependence, and LIME plots |

Open each notebook in Jupyter and run all cells from top to bottom:

```bash
jupyter notebook
```

---

## Generated Figures

All figures are saved automatically to the `figures/` folder when each notebook is run. Below is a complete list of every plot produced and which notebook generates it.

### Notebook 01 - Health Insurance Fraud Labeling
| File | Description |
|---|---|
| `01_health_insurance_labels.png` | Fraud distribution, fraud rule triggers, and fraud rate by MDC specialty |

### Notebook 01b - Exploratory Data Analysis
| File | Description |
|---|---|
| `eda_01_class_distribution.png` | Class balance (legitimate vs fraud) across all three domains |
| `eda_02_credit_card.png` | Credit card: transaction amount, fraud by merchant category, transaction hour, cardholder-to-merchant distance |
| `eda_03_mobile_money.png` | Mobile money: fraud rate by transaction type, amount distribution, originator balance vs new balance, simulation hour |
| `eda_04_health_insurance.png` | Health insurance: fraud rate by MDC specialty, fraud rules triggered, billing deviation distribution, patient age |
| `eda_05_cross_domain.png` | Cross-domain comparison: fraud rates, dataset sizes, summary table |
| `eda_06_correlations.png` | Feature correlation heatmaps for each domain (top 12 features) |

### Notebook 02 - Preprocessing
| File | Description |
|---|---|
| `02_class_distributions.png` | Class distributions across all three domains after preprocessing |

### Notebook 03 - Feature Engineering
| File | Description |
|---|---|
| `03_feature_sets.png` | Top 15 selected features per domain ranked by Random Forest importance |

### Notebook 05 - Baseline Models
| File | Description |
|---|---|
| `baseline_results.png` | Baseline model performance per domain: Logistic Regression, Random Forest, XGBoost, Isolation Forest, Autoencoder |

### Notebook 06 - Domain Adaptation
| File | Description |
|---|---|
| `domain_adaptation_heatmaps.png` | Zero-shot PR-AUC lift over chance for all 6 source-target scenarios, per method |
| `domain_adaptation_label_efficiency.png` | PR-AUC lift versus number of labelled target frauds (zero-shot, few-shot, full), against a Target-only RF |

### Notebook 07 - Ensemble Evaluation
| File | Description |
|---|---|
| `07_experiment_summary.png` | F1 score comparison across all baseline models and the stacking ensemble per domain |

### Notebook 08 - Explainability (SHAP and LIME)
| File | Description |
|---|---|
| `08_shap_global.png` | Global SHAP feature importance bar chart for all three domains side by side |
| `08_shap_summary_credit_card.png` | SHAP beeswarm plot for credit card: top 15 features with direction of effect on fraud probability |
| `08_shap_summary_mobile_money.png` | SHAP beeswarm plot for mobile money: top 15 features with direction of effect |
| `08_shap_summary_health_insurance.png` | SHAP beeswarm plot for health insurance: top 15 features with direction of effect |
| `08_shap_force_credit_card.png` | SHAP force plot breaking down a single fraud prediction in the credit card domain |
| `08_shap_force_mobile_money.png` | SHAP force plot breaking down a single fraud prediction in the mobile money domain |
| `08_shap_force_health_insurance.png` | SHAP force plot breaking down a single fraud prediction in the health insurance domain |
| `08_shap_dependence_credit_card.png` | SHAP dependence plot showing how the top feature drives fraud probability in credit card |
| `08_shap_dependence_mobile_money.png` | SHAP dependence plot for the top feature in mobile money |
| `08_shap_dependence_health_insurance.png` | SHAP dependence plot for the top feature in health insurance |
| `08_explainability_dashboard.png` | Combined dashboard: SHAP top 8 features per domain and a cross-domain feature importance heatmap |

All figures are generated fresh each time the notebooks are run and are not committed to the repository (they live only in your local `figures/` folder after running).

---

## Domain Adaptation Methods

### DANN - Domain-Adversarial Neural Network
Trains a feature extractor that learns domain-invariant representations by simultaneously optimising a fraud classifier and a domain discriminator with a gradient reversal layer. The adversarial training forces the encoder to produce features that are indistinguishable across source and target domains.

### Subdomain Adaptation with MMD
Minimises the Maximum Mean Discrepancy (MMD) between source and target feature distributions in the encoder's hidden layer, using unlabelled target rows.

### HEN - Hierarchical Ensemble Network
Uses an attention mechanism to weight features by relevance before encoding them, pre-trains on the labelled source domain, then (in the few-shot regime) fine-tunes on a small labelled target sample. The attention weights provide built-in interpretability.

---

## Semantic Feature Alignment

A key contribution of this project is the semantic alignment layer that maps domain-specific column names to three universal fraud concepts (plus features that share a name in both domains) before cross-domain transfer:

- **Amount Anomaly**: transaction size relative to population distribution
- **Counterparty Risk**: properties of the recipient account, merchant, or healthcare provider
- **Structural Distance**: temporal, geographic, or billing pattern irregularity

This allows the models to exploit common fraud signals across sectors even when raw feature names differ.

---

## Results Summary

All detailed results are saved to `outputs/all_experiment_results.csv` after running notebook 07.

- **Experiment 1**: Within-domain baseline
- **Experiment 2**: Direct cross-domain transfer (no adaptation)
- **Experiment 3**: Domain adaptation, zero-shot rows (DANN, Subdomain, HEN; few-shot and full are in notebook 06)
- **Experiment 4**: Universal feature identification
- **Experiment 5**: Stacking ensemble

Statistical validation uses paired t-tests (alpha = 0.05) comparing the ensemble against XGBoost baseline and comparing HEN against DANN and Subdomain Adaptation.

---

## Evaluation Protocol and Metrics

**Splits are fixed in notebook 02, before anything is fitted**, and stored in a `split` column that every later notebook uses:

| Domain | Split | Why |
|---|---|---|
| Credit Card | Temporal: `fraudTest` (later period) is the test set; the last 18% of `fraudTrain` by time is validation | a random split lets the model see the future and the same cardholders |
| Mobile Money | Temporal by simulation `step` (70 / 15 / 15) | same reason; the fraud rate drifts over time in MoMTSim, so the test rate differs from the train rate |
| Health Insurance | Grouped by `member_id` (no member in two splits), stratified by label | 50 members have claims of both classes |

Scalers, imputers, outlier caps, target encodings, quantile thresholds, DRG fraud rates and feature selection are all fitted on **train rows only**. Identifiers (card number, transaction number, account names) and outcome fields (`isFlaggedFraud`, `isSuccessful`, `isUnauthorizedOverdraft`) are dropped.

**Metrics.** PR-AUC is the headline metric in every domain (fraud is rare everywhere, so accuracy and ROC-AUC flatter the model). Next to it, **recall at a review budget** (`REVIEW_BUDGET = 5%` of records in `00_config.py`) answers the operational question: if investigators can review the top 5% of claims, what share of the fraud do they find? Compare PR-AUC with the test fraud rate (that is chance level), not with 0.5. In notebooks 05 and 07 F1 / MCC use a threshold tuned on the same domain's validation rows; in notebook 06 they use the budget operating point (top 5% flagged), so no target labels set a threshold.

**Domain adaptation (notebook 06)** reports three separate regimes:

| Regime | Target labels used | Reference |
|---|---|---|
| zero-shot | none (DANN / MMD may use unlabelled target features) | Source-only model without adaptation |
| few-shot | 5 / 20 / 50 target fraud cases (+10x legitimate), 3 random draws | Target-only RF trained on the same labels |
| full | all target train rows | Target-only RF (ceiling) |

Model selection uses source validation only and the target test rows are used once, so a transfer method only "works" if it beats Source-only (zero-shot) or Target-only RF (few-shot).

**Health insurance caveat.** ICD-10 is missing on 60 of the 87 fraudulent claims and on no legitimate one, so the `has_icd10` flag is excluded by default; report results with `HI_USE_MISSING_ICD_FLAG = True` as well. With 87 positives (about 13 in the test split) a single split is very noisy: quote intervals from repeated grouped cross-validation.

---

## References

- Sowah, R. A. et al. (2019). "Design and Development of Fraud Detection Systems of National Health Insurance Claims using Machine Learning." *IEEE*.
- MoMTSim dataset: IEEE Access 2024, DOI: 10.1109/ACCESS.2024.3439012
- Ganin, Y. et al. (2016). "Domain-Adversarial Training of Neural Networks." *JMLR*.

---

## Author

Sedem Agudetse  
Ashesi University, Ghana  
GitHub: [sedemabla](https://github.com/sedemabla)
