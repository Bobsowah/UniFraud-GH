#!/usr/bin/env bash
# Fetch external datasets listed in docs/external_datasets.md. Run on a machine with network access.
# Requires: curl, and for Kaggle items the `kaggle` CLI with credentials in ~/.kaggle/kaggle.json
# (or KAGGLE_USERNAME / KAGGLE_KEY). NOT executed in the cloud session (hosts were blocked) - untested.
set -euo pipefail
OUT="${1:-data/external}"
mkdir -p "$OUT"

echo "== UCI default of credit card clients (CC BY 4.0)"
curl -fL "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip" \
  -o "$OUT/uci_credit_default.zip" || echo "UCI download failed"

echo "== Kaggle datasets"
for slug in ealaxi/paysim1 mlg-ulb/creditcardfraud kartik2112/fraud-detection \
            ealtman2019/ibm-transactions-for-anti-money-laundering-aml ellipticco/elliptic-data-set; do
  d="$OUT/kaggle_$(echo "$slug" | tr '/' '_')"; mkdir -p "$d"
  kaggle datasets download -d "$slug" -p "$d" --unzip || echo "skip $slug (check slug/credentials)"
done
# BAF suite: see https://github.com/feedzai/bank-account-fraud for the current Kaggle slug.
# IEEE-CIS (accept rules on the website first):
#   kaggle competitions download -c ieee-fraud-detection -p "$OUT/ieee_cis"

echo "== Mendeley MoMTSim v2 (DOI 10.17632/zhj366m53p.2): download manually from"
echo "   https://data.mendeley.com/datasets/zhj366m53p/2  into $OUT/momtsim_v2/"
echo "Done. Record licences and SHA-256 of what you keep."
