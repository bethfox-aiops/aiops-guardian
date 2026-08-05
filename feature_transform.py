#!/usr/bin/env python3
"""
feature_transform.py

Log-transform for heavy-tailed features, shared by the retrain scripts and
the live watchdogs so training and live scoring apply the exact identical
transform -- a training/serving skew here would silently make the model's
scaler and distance metric meaningless.

Why this exists (2026-08-05): disk_w_kbps and net_kbps are legitimately
bursty -- promtail's write pattern can spike past 500,000 kB/s against a
typical baseline under 1,000. Comparing the actual 2000-row (~19h) training
window against the last 30 days of aiops_data/metrics.csv found the
training window had never seen disk_w_kbps above ~61,000 - 8x lower than
real, recurring bursts. KNN's raw-scale Euclidean distance was getting
dominated by whichever value the narrow training window happened to catch,
so every larger-but-normal burst outside that window looked
catastrophically anomalous. log1p compresses the dynamic range so an
unseen-but-real burst magnitude lands much closer to previously-seen
values in transformed space, rather than an unbounded outlier every time.

This is a complement to, not a replacement for, widening/adjusting
RECENT_ROWS -- the transform makes the model robust to magnitudes it
hasn't exactly seen; it doesn't by itself guarantee good scaler statistics
from a too-narrow window.
"""

import numpy as np

BURSTY_FEATURES = {"net_kbps", "disk_w_kbps"}


def transform_bursty_features(row, columns):
    """For the live watchdogs: row is a list of raw feature values in
    `columns` order (matches DATA_FEATURES/FEATURES). Values in
    BURSTY_FEATURES are always >= 0 (kB/s rates), so log1p needs no sign
    handling."""
    return [
        float(np.log1p(v)) if col in BURSTY_FEATURES else v
        for col, v in zip(columns, row)
    ]


def transform_bursty_features_df(df):
    """For the retrain scripts: log1p-transforms the bursty columns of a
    features DataFrame. Returns a copy; does not mutate the input."""
    df = df.copy()
    for col in BURSTY_FEATURES:
        if col in df.columns:
            df[col] = np.log1p(df[col])
    return df
