# shap_make_figs.py  (robust for SHAP ≥0.20)
import os, joblib, shap, pandas as pd, numpy as np
from scipy import sparse

import matplotlib
matplotlib.use("Agg")  # headless saving
import matplotlib.pyplot as plt
from shap import plots as shap_plots

DATA_CSV  = "data/youtube_dataset.csv"
MODEL_PKL = "data/rf_classifier_for_shap.joblib"
OUT_DIR   = "figs"
FIG5_PATH = os.path.join(OUT_DIR, "shap_summary_rf_classifier.png")
FIG6_PATH = os.path.join(OUT_DIR, "shap_waterfall_rf_sample.png")  # waterfall instead of force
SAMPLE_ID = 10  # change to any 0..N-1

os.makedirs(OUT_DIR, exist_ok=True)

def log(msg): print(f"▶ {msg}", flush=True)

# 1) Load
log("Loading data & model…")
df = pd.read_csv(DATA_CSV)
rf_pipeline, feature_names = joblib.load(MODEL_PKL)

# 2) Ensure engineered columns exist
def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "is_short" not in df:
        df["is_short"] = (df["duration_seconds"] <= 60).astype(int)
    if "duration_bin" not in df:
        df["duration_bin"] = pd.qcut(df["duration_seconds"], q=5, labels=False, duplicates="drop")
    if "has_howto" not in df:
        df["has_howto"] = df["title"].str.contains(r"\bhow to\b", case=False, regex=True).astype(int)
    if "has_vs" not in df:
        df["has_vs"] = df["title"].str.contains(r"\bvs\b", case=False, regex=True).astype(int)
    return df

df = enrich(df)

# 3) Transform with the pipeline preprocessor
log("Transforming features with pipeline preprocessor…")
X = df[feature_names]
pre = rf_pipeline.named_steps["pre"]
clf = rf_pipeline.named_steps["m"]
Xp = pre.transform(X)

log(f"Feature matrix type: {type(Xp)}")
Xp_dense = Xp.toarray() if sparse.issparse(Xp) else np.asarray(Xp)
feat_names_out = pre.get_feature_names_out()

# 4) SHAP explainer + values
log("Building SHAP TreeExplainer…")
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(Xp_dense)

# For classifiers, pick positive class
def select_pos(values, expected):
    if isinstance(values, list):
        idx = 1 if len(values) >= 2 else len(values) - 1
        v = values[idx]
        e = expected[idx] if isinstance(expected, (list, np.ndarray)) else expected
        return v, e
    return values, expected

pos_shap_values, pos_expected = select_pos(shap_values, explainer.expected_value)

# 5) FIGURE 5 — SHAP summary (global)
log("Saving Figure 5 (summary)…")
plt.figure()
shap.summary_plot(pos_shap_values, Xp_dense, feature_names=feat_names_out, show=False)
plt.title("SHAP Summary — Random Forest Classifier", fontsize=12)
plt.tight_layout()
plt.savefig(FIG5_PATH, dpi=300, bbox_inches="tight")
plt.close()
log(f"Saved: {FIG5_PATH}")

# 6) FIGURE 6 — Waterfall (local explanation for one sample)
sample = int(np.clip(SAMPLE_ID, 0, Xp_dense.shape[0] - 1))
log(f"Saving Figure 6 (waterfall) for sample index {sample}…")

# Ensure we pass a 1D vector for ONE sample and ONE class
vals = pos_shap_values
base = pos_expected

v = np.asarray(vals)
if v.ndim == 3:        # (n_samples, n_features, n_classes)
    v = v[:, :, -1]    # take positive class
v_sample = v[sample]
if v_sample.ndim > 1:  # e.g., (n_features, 2)
    v_sample = v_sample[:, -1]

if isinstance(base, (list, np.ndarray)):  # base value for positive class
    base = base[-1]

ex = shap.Explanation(
    values        = v_sample,                 # (n_features,)
    base_values   = float(base),              # scalar
    data          = Xp_dense[sample, :],      # transformed features
    feature_names = feat_names_out
)

plt.figure()
shap_plots.waterfall(ex, max_display=20, show=False)
plt.title(f"SHAP Waterfall — Example Video #{sample}", fontsize=12)
plt.tight_layout()
plt.savefig(FIG6_PATH, dpi=300, bbox_inches="tight")
plt.close()
log(f"Saved: {FIG6_PATH}")

print("✅ Done.")
