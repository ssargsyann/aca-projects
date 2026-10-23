"""Input validation and train-fitted feature selection."""
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


def validate_target(frame, target):
    if target not in frame:
        raise ValueError(f"Required target column is missing: {target}")
    labels = frame[target]
    if labels.isna().any() or set(labels.unique()) != {0, 1}:
        raise ValueError("Target must contain both binary classes 0 and 1, with no missing labels.")
    if labels.value_counts().min() < 2:
        raise ValueError("Each target class requires at least two rows for a stratified split.")
    return labels.astype(int)


class FeatureSelector(TransformerMixin, BaseEstimator):
    """Learn missingness on training rows only and enforce inference schema."""

    def __init__(self, missing_threshold=0.65, drop_columns=()):
        self.missing_threshold = missing_threshold
        self.drop_columns = drop_columns

    def fit(self, X, y=None):
        if not 0 <= self.missing_threshold < 1:
            raise ValueError("missing_threshold must be in [0, 1).")
        candidates = X.drop(columns=list(self.drop_columns), errors="ignore")
        self.columns_ = candidates.columns[candidates.isna().mean() <= self.missing_threshold].tolist()
        if not self.columns_:
            raise ValueError("No usable features remain after column filtering.")
        return self

    def transform(self, X):
        check_is_fitted(self, "columns_")
        missing = sorted(set(self.columns_) - set(X.columns))
        if missing:
            raise ValueError(f"Missing input features: {', '.join(missing)}")
        return X.loc[:, self.columns_].copy()
