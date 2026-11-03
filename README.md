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
|-- 01_health_insurance_labeling.ipynb  <- Fraud labelling using G-DRG tariff rules
|-- 01b_EDA.ipynb                       <- Exploratory data analysis across all three domains
|-- 02_preprocessing.ipynb              <- 7-step preprocessing pipeline
|-- 03_feature_engineering.ipynb        <- Domain-specific and universal feature engineering
|-- 05_baseline_model.ipynb             <- Baseline models per domain
|-- 06_domain_adaptation.ipynb          <- DANN, Subdomain Adaptation, and HEN transfer
|-- 07_ensemble_evaluation.ipynb        <- 5 experiments + stacking ensemble + statistical tests
|-- 08_explainability.ipynb             <- SHAP and LIME explainability
|
|-- requirements.txt                    <- Python dependencies
|-- data/                               <- (not committed) Raw datasets (see below)
|-- outputs/                            <- Generated CSVs and results (auto-created)
|-- models/                             <- Saved model files (auto-created)
|-- figures/                            <- Saved plots (auto-created)
```

---

## Leakage-safe pipeline (v2, in progress)

The original notebooks 01-03 fit target encoding, outlier caps, scalers and feature selection on the
full dataset before splitting, labelled *missing* drug charges as fraud, and kept label-defining
billing columns in the health-insurance features. The `unifraud/` package replaces that logic:

| Module | Purpose |
|---|---|
| `unifraud/labels_hi.py` | HI rule labels; missing data is a data-quality flag, not fraud; untariffed claims are `NaN` |
| `unifraud/leakage.py` | registry of rule-derived / identifier columns + single-feature-AUC tripwire |
| `unifraud/splits.py` | chronological (CC, MM) and patient-grouped (HI) splits |
| `unifraud/preprocess.py` | train-only fit: winsorise, robust-scale, out-of-fold target encoding, time features |
| `scripts/prepare_data.py` | end-to-end preparation -> `outputs/clean/*.csv` + manifests |
| `unifraud/concepts.py` | oriented z-score concept map (amount / counterparty / structure), train-only fit, concept-validity check |
| `unifraud/transfer.py` | concept-space baselines: target-only, source-only (+ prior correction via BBSE/Saerens), pooled with up-weighted target labels, transfer gain |
| `unifraud/metrics.py` | AP-skill, cost threshold, ECE, precision@k, bootstrap CI, Nadeau-Bengio corrected t-test, Holm |
| `scripts/run_transfer_baselines.py` | all six directed pairs x label budgets x seeds -> `outputs/transfer/*.csv` |

```bash
pip install -r requirements.txt
python -m pytest tests            # unit tests (synthetic data)
python scripts/prepare_data.py --data data --out outputs/clean
python scripts/run_transfer_baselines.py --clean outputs/clean --out outputs/transfer
```

The concept specs in `unifraud/concepts.py` are provisional: inspect `outputs/transfer/concept_validity.csv` and `manifest.json` (which columns fed each concept) before trusting any transfer number. DANN, class-conditional MMD and the gated network are not yet implemented.

Downstream notebooks (04-08) have **not** yet been migrated to `outputs/clean`; results from the old
notebooks should not be cited. See `paper/REVISION_NOTES.md` (branch `claude/paper-draft`) for the full defect list.

---

## Datasets

The raw data files are **not included in this repository** due to size and licensing. Download them and place them in the `data/` folder.

| Dataset | Source | Filename(s) |
|---|---|---|
| Credit Card (Sparkov) | Kaggle: kartik2112/fraud-detection | `fraudTrain.csv`, `fraudTest.csv` |
| Mobile Money (MoMTSim) | IEEE Access 2024 / GitHub | `MoMTSim_transactions.csv` |
| Health Insurance (NHIA) | Ghana NHIA (provided) | `GD.csv`, `ReData1_100.csv` ... `ReData5_1000.csv` |

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
| 1 | `01_health_insurance_labeling.ipynb` | Labels HI fraud using G-DRG tariff compliance |
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
| `domain_adaptation_heatmaps.png` | Heatmaps of MCC and ROC-AUC for all 6 source-target transfer scenarios across DANN, HEN, and Subdomain Adaptation |

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
Clusters transactions into subdomains and minimises Maximum Mean Discrepancy (MMD) between source and target cluster distributions. This is more fine-grained than global domain alignment and handles the heterogeneous fraud patterns across sectors.

### HEN - Hierarchical Ensemble Network
Uses an attention mechanism to weight features by relevance before encoding them, pre-trains on the labelled source domain, then fine-tunes on the (partially labelled) target domain. The attention weights provide built-in interpretability.

---

## Semantic Feature Alignment

A key contribution of this project is the semantic alignment layer that maps domain-specific column names to three universal fraud concepts before cross-domain transfer:

- **Amount Anomaly**: transaction size relative to population distribution
- **Counterparty Risk**: properties of the recipient account, merchant, or healthcare provider
- **Structural Distance**: temporal, geographic, or billing pattern irregularity

This allows the models to exploit common fraud signals across sectors even when raw feature names differ.

---

## Results Summary

All detailed results are saved to `outputs/all_experiment_results.csv` after running notebook 07.

- **Experiment 1**: Within-domain baseline
- **Experiment 2**: Direct cross-domain transfer (no adaptation)
- **Experiment 3**: Domain adaptation (DANN, Subdomain, HEN)
- **Experiment 4**: Universal feature identification
- **Experiment 5**: Stacking ensemble

Statistical validation uses paired t-tests (alpha = 0.05) comparing the ensemble against XGBoost baseline and comparing HEN against DANN and Subdomain Adaptation.

---

## Evaluation Metrics

Because the three datasets have very different fraud rates, different primary metrics are used:

| Domain | Primary Metric | Rationale |
|---|---|---|
| Credit Card | ROC-AUC | 0.52% fraud rate makes F1 unreliable |
| Mobile Money | F1 | Near-balanced dataset |
| Health Insurance | MCC | ~40% fraud rate; MCC is robust to label imbalance |

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
