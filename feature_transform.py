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

Clipped to >= 0 before log1p (2026-08-16): these are rate deltas
(bytes-since-last-sample / elapsed), assumed always >= 0 since they're
kB/s throughput -- true in practice except right after a very long
suspend (33h, this incident), where a NIC counter reset (kernel-logged
"igc ... Timeout reading IGC_PTM_STAT register" at the exact same
timestamp) made net_kbps go deeply negative for one sample. log1p(x) is
undefined for x <= -1, so that one bad sample crashed KNN and autoencoder
outright (ValueError: Input contains NaN, sklearn's check_array correctly
rejecting it) rather than just scoring wrong -- both restarted cleanly via
systemd's Restart=on-failure, but the crash itself was avoidable. A
negative reading here is already invalid data (a real rate can't be
negative), so clamping to 0 treats it as "no throughput measured this
tick" -- conservative and honest, not a fabricated plausible-looking
value standing in for bad data.
"""

import numpy as np

BURSTY_FEATURES = {"net_kbps", "disk_w_kbps"}


def transform_bursty_features(row, columns):
    """For the live watchdogs: row is a list of raw feature values in
    `columns` order (matches DATA_FEATURES/FEATURES). BURSTY_FEATURES
    values are clamped to >= 0 before log1p -- see module docstring."""
    return [
        float(np.log1p(max(v, 0))) if col in BURSTY_FEATURES else v
        for col, v in zip(columns, row)
    ]


def transform_bursty_features_df(df):
    """For the retrain scripts: log1p-transforms the bursty columns of a
    features DataFrame, clamped to >= 0 first (see module docstring).
    Returns a copy; does not mutate the input."""
    df = df.copy()
    for col in BURSTY_FEATURES:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))
    return df
