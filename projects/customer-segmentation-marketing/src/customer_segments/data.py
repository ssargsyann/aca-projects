"""Explicit numeric schema, fitted imputation, and quantile clipping."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


def load_data(path, separator="auto"):
    if separator == "auto":
        with Path(path).open(encoding="utf-8-sig") as handle:
            separator = "\t" if "\t" in handle.readline() else ","
    frame = pd.read_csv(path, sep=separator)
    if frame.empty:
        raise ValueError("Input must contain at least one row.")
    return frame


class NumericFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, features, clip_quantiles=(0.01, 0.99)):
        self.features = features
        self.clip_quantiles = clip_quantiles

    def _numeric(self, frame):
        missing = sorted(set(self.features) - set(frame.columns))
        if missing:
            raise ValueError(f"Missing required features: {', '.join(missing)}")
        try:
            numeric = frame.loc[:, list(self.features)].apply(pd.to_numeric, errors="raise").astype(float)
        except (TypeError, ValueError) as error:
            raise ValueError("Configured features must be numeric or missing.") from error
        if np.isinf(numeric.to_numpy()).any():
            raise ValueError("Infinite feature values are not supported.")
        return numeric

    def fit(self, X, y=None):
        low, high = self.clip_quantiles
        if not 0 <= low < high <= 1:
            raise ValueError("clip_quantiles must satisfy 0 <= low < high <= 1.")
        if not self.features or len(set(self.features)) != len(self.features):
            raise ValueError("Provide a nonempty list of unique features.")
        values = self._numeric(X)
        self.medians_ = values.median()
        if self.medians_.isna().any():
            raise ValueError("Configured features cannot be entirely missing during fitting.")
        self.lower_ = values.quantile(low)
        self.upper_ = values.quantile(high)
        return self

    def transform(self, X):
        check_is_fitted(self, "medians_")
        values = self._numeric(X).fillna(self.medians_)
        return values.clip(self.lower_, self.upper_, axis="columns").to_numpy()
