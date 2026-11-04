"""Create data/NHIS_claims_deidentified.csv from the original NHIS claims workbook.

Usage (run locally; the original file never leaves your machine):
    python tools/deidentify_nhis.py "C:/path/NHIS_CLAIM_DATASET.xlsx"

Removes: SURNAME, OTHER NAMES, DOB, CCC.
Replaces: MEMBER NO. and FOLDER NO. by random surrogate IDs (not reversible without the original).
Derives: age_at_admission, length_of_stay.  Recodes the label: fraud_label = 1 means FRAUDULENT
(the source column 'LEGITIMATE=1, FRAUDULENT=0' is inverted relative to this pipeline).
Admission/discharge dates and diagnoses are kept (needed for modelling) and are quasi-identifiers:
keep the output private and out of Git (data/ is git-ignored).
"""
import os
import sys

import numpy as np
import pandas as pd

LABEL = "LEGITIMATE=1, FRAUDULENT=0"
REQUIRED = ["SURNAME", "OTHER NAMES", "DOB", "GENDER", "ADMISSION", "DISCHARGE", "FOLDER NO.", "MEMBER NO.", "CCC",
            "PROCEDURES", "ICD10", "DIAGNOSES", "GDRG", "SERVICES (GHC)", "PHARMACY (GHC)", "TOTAL (GHC)", LABEL]


def main(src, dst="data/NHIS_claims_deidentified.csv", seed=20261104):
    d = pd.read_excel(src)
    d.columns = [c.strip() for c in d.columns]
    missing = [c for c in REQUIRED if c not in d.columns]
    if missing:
        raise SystemExit(f"Missing expected columns: {missing}")
    adm = pd.to_datetime(d["ADMISSION"], dayfirst=True, errors="coerce")
    dis = pd.to_datetime(d["DISCHARGE"], dayfirst=True, errors="coerce")
    dob = pd.to_datetime(d["DOB"], dayfirst=True, errors="coerce")
    rng = np.random.default_rng(seed)

    def surrogate(s, prefix):
        codes, uniq = pd.factorize(s)
        perm = rng.permutation(len(uniq))
        return [f"{prefix}{perm[c]:05d}" for c in codes]

    out = pd.DataFrame({
        "row_id": np.arange(1, len(d) + 1),
        "member_id": surrogate(d["MEMBER NO."], "M"),
        "folder_id": surrogate(d["FOLDER NO."], "F"),
        "age_at_admission": ((adm - dob).dt.days / 365.25).round(1),
        "gender": d["GENDER"],
        "admission_date": adm.dt.strftime("%Y-%m-%d"),
        "discharge_date": dis.dt.strftime("%Y-%m-%d"),
        "length_of_stay": (dis - adm).dt.days,
        "procedures": d["PROCEDURES"],
        "icd10": d["ICD10"],
        "diagnoses": d["DIAGNOSES"],
        "gdrg": d["GDRG"],
        "services_ghc": d["SERVICES (GHC)"],
        "pharmacy_ghc": d["PHARMACY (GHC)"],
        "total_ghc": d["TOTAL (GHC)"],
        "fraud_label": (d[LABEL] == 0).astype(int),
    })
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    out.to_csv(dst, index=False)
    print(f"Wrote {dst}: {out.shape}, fraud={int(out.fraud_label.sum())} ({out.fraud_label.mean():.2%})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1], *(sys.argv[2:3]))
