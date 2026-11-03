"""UniFraud-GH: leakage-safe building blocks for cross-sector fraud detection.

Modules
-------
labels_hi   Health-insurance rule labels with explicit missing-data semantics.
leakage     Registry of label-defining inputs and automatic leakage checks.
splits      Chronological, grouped and stratified splitting.
preprocess  Train-only-fit transformers (out-of-fold target encoding, winsorising, ...).
"""

__all__ = ["labels_hi", "leakage", "splits", "preprocess"]
