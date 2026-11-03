# UniFraud-GH: paper draft and required code fixes

`main.tex` (IEEEtran, compiles to `main.pdf`) is a **method-complete draft**. All numeric results are
`TBD`: the notebooks' existing numbers are not reportable because of the defects below
(found by reading every notebook and by an independent model review; each cites notebook/cell).

## Defects that invalidate the current results (must fix before any number is published)
1. **HI label circularity** (NB01 c11/c13, NB03 c9/c11, NB06 never calls `remove_hi_leaks`):
   rule inputs and derived features (`dev_*`, `overbill_*`, `is_overbilled`, `drg_fraud_rate`) stay in the
   features used by all DA experiments, so HEN MM->HI MCC 0.98 is rule reproduction, not transfer.
   NB02 c11 drops `overbill_medical`, which does not exist; the real column is `overbill_drug`.
2. **HI label artefact**: `impossible_amount` fires on `fillna(0)<=0`. `Medical` has 629 missing of 2655,
   and the rule fires 632 times, so it labels *missing drug charge* as fraud.
3. **Full-dataset target encoding before the split** (NB02 step 4) for merchant/category/job/dob (CC),
   `recipient` (MM), Folder No / NHIS ID / G-DRG (HI). Also: capping, scaling, imputation and RF feature
   selection (NB03 c13) are fit on all rows. `drg_fraud_rate` uses the label on all rows.
4. **Transfer is not unsupervised and has no target-only baseline** (NB06): epoch and threshold chosen on
   labelled target data; HEN fine-tunes on 20% of target labels. No negative-transfer check.
5. **"Semantic alignment" is positional truncation** (`src_X[:, :shared]` in NB06 `align_features`).
6. **DANN applies lambda twice** (`lam * dloss` plus GRL `-lam*grad`).
7. **"Subdomain Adaptation" is global MMD** (KMeans imported, never used); "HEN" is an MLP with softmax gating.
8. **Stats are mis-specified** (NB07 c15): the "ensemble vs XGB" test compares RF with XGB; paired t-test over six
   heterogeneous scenarios; no multiple-comparison correction; 5-fold "95% CI" is a percentile of 5 numbers.
9. **CC time features are dead**: the timestamp is target-encoded to a float before `to_datetime`.
10. **Inconsistent/unexecuted outputs**: NB06 and NB07 report different DA means (0.5426 vs 0.5894 DANN F1);
    NB05 output stops after the first model; NB08 has no outputs; notebook 04 is missing.
11. CC train/test is merged and split randomly (no temporal holdout); no RNG seeds for torch/numpy in NB06;
    PR-AUC via trapezoid is optimistic; IsolationForest normalisation uses test-set max.

## Mapping paper -> code to implement
| Paper element | Code change |
|---|---|
| Sec. V-A leak-free preprocessing | sklearn `Pipeline`/`ColumnTransformer`, train-only fit, out-of-fold target encoding, drop IDs |
| Eq. (concept), Prop. 1 | oriented z-score averaging over explicit `G_kc`; use only R^3 (+ optional universal features) |
| DANN, Prop. 3 | single lambda application; report proxy A-distance |
| LMMD | class-conditional multi-bandwidth MMD with soft target weights |
| HEN, Eq. (L2-SP) | pretrain, then fine-tune with L2-SP on b target labels; compare to target-only |
| Sec. VII protocol | chronological CC/MM splits, grouped HI split, seeds x budgets, Holm, Nadeau-Bengio |

## Items marked VERIFY in the draft
Author list/order, Sowah et al. 2019 venue, MoMTSim reference title/authors, Sparkov authorship/licence,
NHIA data-use statement. Sparkov/MoMTSim bibliographic entries were written from the README, not checked.

## Venue assessment (from review)
Realistic after the fixes: IEEE Access or ACM TMIS (or ESWA/DSS). TKDE/TNNLS unlikely (DA components
are standard; the concept map is a thin novelty). TIFS is a poor fit.
