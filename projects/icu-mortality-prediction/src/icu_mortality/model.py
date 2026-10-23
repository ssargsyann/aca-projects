"""Construct one serializable preprocessing and classification pipeline."""
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from .data import FeatureSelector


def build_pipeline(config, labels):
    positives = int(labels.sum())
    classifier = XGBClassifier(
        **config["model"],
        scale_pos_weight=(len(labels) - positives) / positives,
        random_state=config["random_state"], eval_metric="auc",
    )
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocess = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median", keep_empty_features=True), make_column_selector(dtype_include="number")),
        ("categorical", categorical, make_column_selector(dtype_exclude="number")),
    ])
    return Pipeline([
        ("select", FeatureSelector(config["missing_threshold"], config["drop_columns"])),
        ("preprocess", preprocess),
        ("model", classifier),
    ])
