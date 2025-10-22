# advanced_models.py
import os, json, joblib
import pandas as pd, numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier
)

DATA_CSV = "data/youtube_dataset.csv"
METRICS_JSON = "figs/metrics_advanced.json"
RF_MODEL_PATH = "data/rf_classifier_for_shap.joblib"

os.makedirs("data", exist_ok=True)
os.makedirs("figs", exist_ok=True)

# ---------- 1) Load base dataset ----------
df = pd.read_csv(DATA_CSV)

# ---------- 2) Feature enrichment (idempotent) ----------
def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Add engineered columns if missing
    if "is_short" not in df:
        df["is_short"] = (df["duration_seconds"] <= 60).astype(int)
    if "duration_bin" not in df:
        df["duration_bin"] = pd.qcut(
            df["duration_seconds"], q=5, labels=False, duplicates="drop"
        )
    if "has_howto" not in df:
        df["has_howto"] = df["title"].str.contains(r"\bhow to\b", case=False, regex=True).astype(int)
    if "has_vs" not in df:
        df["has_vs"] = df["title"].str.contains(r"\bvs\b", case=False, regex=True).astype(int)
    return df

df = enrich(df)

# Predictors/targets
num = ["duration_seconds", "title_len"]
cat = ["publish_hour", "publish_dow", "peak_hour", "is_short", "duration_bin", "has_howto", "has_vs"]
X = df[num + cat]
y_reg = df["log_views"]
y_cls = (df["views"] >= df["views"].median()).astype(int)

# ---------- 3) Preprocessor ----------
pre = ColumnTransformer([
    ("num", StandardScaler(), num),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
])

# ---------- 4) Models ----------
rf_r = Pipeline([("pre", pre), ("m", RandomForestRegressor(n_estimators=400, random_state=42))])
gb_r = Pipeline([("pre", pre), ("m", GradientBoostingRegressor(random_state=42))])
rf_c = Pipeline([("pre", pre), ("m", RandomForestClassifier(n_estimators=600, class_weight="balanced", random_state=42))])
gb_c = Pipeline([("pre", pre), ("m", GradientBoostingClassifier(random_state=42))])

# ---------- 5) Cross-validation ----------
kf  = KFold(n_splits=5, shuffle=True, random_state=42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

r2_rf  = float(np.mean(cross_val_score(rf_r, X, y_reg, cv=kf,  scoring="r2")))
r2_gb  = float(np.mean(cross_val_score(gb_r, X, y_reg, cv=kf,  scoring="r2")))
auc_rf = float(np.mean(cross_val_score(rf_c, X, y_cls, cv=skf, scoring="roc_auc")))
auc_gb = float(np.mean(cross_val_score(gb_c, X, y_cls, cv=skf, scoring="roc_auc")))

metrics = {
    "rf_reg_r2": round(r2_rf, 4),
    "rf_cls_roc_auc": round(auc_rf, 4),
    "gb_reg_r2": round(r2_gb, 4),
    "gb_cls_roc_auc": round(auc_gb, 4),
}

# ---------- 6) Persist metrics & one fitted classifier for SHAP ----------
with open(METRICS_JSON, "w") as f:
    json.dump(metrics, f, indent=2)

rf_c.fit(X, y_cls)
joblib.dump((rf_c, X.columns.tolist()), RF_MODEL_PATH)

print("✅ Advanced model metrics saved to", METRICS_JSON)
print("✅ Saved RF classifier for SHAP at", RF_MODEL_PATH)
print(metrics)
