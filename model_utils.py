"""Training, evaluation and prediction logic (kept free of Streamlit so it can be tested)."""
import numpy as np
import pandas as pd
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier,
                              VotingClassifier)
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

from features import RAW_FEATURES, TARGET, engineer

POS_LABEL = "High"          # High rating  -> PASS
LABEL_MAP = {"High": "Pass", "Low": "Fail"}


def make_model(kind: str = "ensemble", n_estimators: int = 600, min_samples_leaf: int = 1,
               max_features="sqrt", seed: int = 0):
    """kind="ensemble": soft vote of gradient boosting + class-weighted RF + extra trees.
    kind="rf": class-weighted random forest only."""
    rf = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                max_features=max_features, class_weight="balanced", random_state=seed, n_jobs=-1)
    if kind == "rf":
        return rf
    et = ExtraTreesClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                              max_features=max_features, random_state=seed, n_jobs=-1)
    hgb = HistGradientBoostingClassifier(learning_rate=0.02, max_leaf_nodes=63, max_iter=1000,
                                         min_samples_leaf=5, l2_regularization=1.0, random_state=seed)
    return VotingClassifier([("hgb", hgb), ("rf", rf), ("et", et)], voting="soft")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Validate/clean a dataframe of student records (raw features + rating)."""
    missing = [c for c in RAW_FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    out = df[RAW_FEATURES + [TARGET]].copy()
    for c in RAW_FEATURES:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out[TARGET] = out[TARGET].astype(str).str.strip().str.capitalize()
    out[TARGET] = out[TARGET].replace({"Pass": "High", "Fail": "Low", "1": "High", "0": "Low"})
    out = out.dropna()
    out = out[out[TARGET].isin(["High", "Low"])]
    out[RAW_FEATURES] = out[RAW_FEATURES].clip(lower=0).round().astype(int)
    return out.reset_index(drop=True)


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop exact duplicate rows (augmentation copies) and feature-identical rows with conflicting labels."""
    key = df[RAW_FEATURES].astype(str).agg("|".join, axis=1)
    n_labels = df.groupby(key)[TARGET].transform("nunique")
    conflicting = int((n_labels > 1).sum())
    out = df[n_labels == 1].drop_duplicates().reset_index(drop=True)
    return out, {"original": len(df), "conflicting": conflicting,
                 "duplicates": len(df) - conflicting - len(out), "kept": len(out)}


def _groups(df: pd.DataFrame):
    # identical feature rows share a group so duplicates never straddle train/test folds
    return df[RAW_FEATURES].astype(str).agg("|".join, axis=1).factorize()[0]


def evaluate(df: pd.DataFrame, params: dict | None = None, folds: int = 5) -> dict:
    """Leakage-safe cross-validated evaluation."""
    params = params or {}
    X, y = engineer(df), (df[TARGET] == POS_LABEL).astype(int).values
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=42)
    proba = cross_val_predict(make_model(**params), X, y, cv=cv, groups=_groups(df),
                              method="predict_proba")[:, 1]
    pred = (proba >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y, proba)
    return {
        "n": len(df),
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
        "confusion": confusion_matrix(y, pred, labels=[1, 0]),  # rows: actual Pass, Fail
        "oof_proba": proba,
        "roc": pd.DataFrame({"False positive rate": fpr, "True positive rate": tpr}),
    }


def train(df: pd.DataFrame, params: dict | None = None):
    model = make_model(**(params or {}))
    model.fit(engineer(df), (df[TARGET] == POS_LABEL).astype(int))
    return model


def importances(model) -> pd.DataFrame:
    if hasattr(model, "named_estimators_"):
        forests = [model.named_estimators_[k] for k in ("rf", "et")]
    else:
        forests = [model]
    imp = np.mean([f.feature_importances_ for f in forests], axis=0)   # mean impurity importance of the forests
    return (pd.DataFrame({"feature": forests[0].feature_names_in_, "importance": imp})
            .sort_values("importance", ascending=False).reset_index(drop=True))


def predict(model, df: pd.DataFrame) -> pd.DataFrame:
    p = model.predict_proba(engineer(df))[:, 1]
    return pd.DataFrame({"Pass probability": p, "Prediction": np.where(p >= 0.5, "Pass", "Fail")},
                        index=df.index)
