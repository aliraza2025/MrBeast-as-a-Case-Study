# models.py
import json, numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression

def run_models(df):
    num_feats = ["duration_seconds", "title_len"]
    cat_feats = ["publish_hour", "publish_dow", "peak_hour"]

    X = df[num_feats + cat_feats]
    y_reg = df["log_views"]
    y_cls = (df["views"] >= df["views"].median()).astype(int)

    pre = ColumnTransformer([
        ("num", StandardScaler(), num_feats),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats)
    ])

    reg_model = Pipeline([("pre", pre), ("lr", LinearRegression())])
    cls_model = Pipeline([("pre", pre),
                          ("logit", LogisticRegression(max_iter=200, class_weight="balanced"))])

    n = len(df)
    # For classification, require at least 1 sample of each class per fold
    cls_counts = y_cls.value_counts()
    min_per_class = int(cls_counts.min()) if not cls_counts.empty else 0

    # Choose folds safely
    cls_folds = max(2, min(5, min_per_class))  # e.g., with 3 per class -> 3-fold
    reg_folds = max(2, min(5, n))              # can’t exceed n samples

    # If still too small, just use holdout-free CV=2
    if cls_folds < 2: cls_folds = 2
    if reg_folds < 2: reg_folds = 2

    # CV splitters
    skf = StratifiedKFold(n_splits=cls_folds, shuffle=True, random_state=42)
    kf  = KFold(n_splits=reg_folds, shuffle=True, random_state=42)

    r2  = float(np.mean(cross_val_score(reg_model, X, y_reg, cv=kf,  scoring="r2")))
    auc = float(np.mean(cross_val_score(cls_model, X, y_cls, cv=skf, scoring="roc_auc")))

    return {"cv_r2_regression": round(r2, 4),
            "cv_roc_auc_classification": round(auc, 4)}
