from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pygam import LinearGAM, f, l, s
from pygam.terms import TermList
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

from helpers import lag_corr_scan, lift_curve, permutation_pvalue_for_var, roi_curve

# %% -
# config
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True, parents=True)
PLOT_DIR = Path("plots")
PLOT_DIR.mkdir(exist_ok=True, parents=True)

# %% -
synthetic_df = pd.read_csv(DATA_DIR / "synthetic_data.csv")
channel_cols = synthetic_df.filter(like="spend_").columns.tolist()
control_cols = ["seasonality", "competitor_discount", "price_index"]

# %% -
X = synthetic_df[list(channel_cols) + control_cols]
y = synthetic_df["conversions"]

# %% -
# do automatic discovery of lags
summary = lag_corr_scan(X, y, lags=range(-7, 8), method="pearson")
summary["p_value"] = [
    permutation_pvalue_for_var(X[c].values, y.values, lags=range(-7, 8), n_perm=1000)[0] for c in summary["variable"]
]
_, pvals_corrected, _, _ = multipletests(summary["p_value"], method="fdr_bh")
summary["p_value_adj"] = pvals_corrected
summary.drop(columns=["p_value"], inplace=True)

# %% -
# shift the tv data by the lag manually for now
X["spend_tv"] = X["spend_tv"].shift(1).fillna(0.0)

# optional: if competitor is not int, coerce to int for factor term
if "competitor_discount" in X:
    X["competitor_discount"] = X["competitor_discount"].astype(int)

# %% -
# basic linear predictor with regularization
lasso_pipe = Pipeline([("scaler", StandardScaler()), ("lasso", LassoCV(cv=5, random_state=42))])
lasso_pipe.fit(X, y)
coefs_dict = dict(zip(lasso_pipe.feature_names_in_, lasso_pipe.named_steps["lasso"].coef_))
coef_df = (
    pd.DataFrame.from_dict(coefs_dict, orient="index", columns=["Coefficient"])
    .reset_index()
    .rename(columns={"index": "Feature"})
)

# %% -
# configure LinearGAM terms

# fmt: off
term_spec = {
    # channels (splines with custom smoothness)
    "spend_search_ads":     {"kind": "s", "n_splines": 15, "lam": 0.3},
    "spend_social_media":   {"kind": "s", "n_splines": 20, "lam": 0.6},
    "spend_tv":             {"kind": "s", "n_splines": 15, "lam": 0.5},  # even for lagged tv column
    "spend_email":          {"kind": "s", "n_splines": 10, "lam": 0.3},
    "spend_affiliates":     {"kind": "s", "n_splines": 15, "lam": 0.6},

    # controls
    "seasonality":          {"kind": "l", "lam": 0.0},   # linear control
    "price_index":          {"kind": "l", "lam": 0.0},   # linear control
    "competitor_discount":  {"kind": "f", "lam": 0.0},   # factor for binary event
}
# fmt: on

# build a safe name->index map
col_index = {name: i for i, name in enumerate(X.columns)}

# construct terms from the spec
terms = []
for name, cfg in term_spec.items():
    i = col_index[name]
    kind = cfg["kind"]
    if kind == "s":
        terms.append(s(i, n_splines=cfg.get("n_splines", 20), lam=cfg.get("lam")))
    elif kind == "l":
        terms.append(l(i))
    elif kind == "f":
        terms.append(f(i))
    else:
        raise ValueError(kind)

term_list = TermList(*terms)

# %% -
gam = LinearGAM(term_list).fit(X.values, y)
gam.summary()

# %% -
# tune the penalty paramter to smooth out the fits
do_tune = True
if do_tune:
    lam = np.logspace(-3, 5, 5)
    lams = [lam] * 5 + [[0.0]] * len(control_cols)  # zero penalty for controls
    gam.gridsearch(X, y, lam=lams)
    gam.summary()

# %% -
# make partial dependence plots
fig, axes = plt.subplots(3, 2, figsize=(8, 8))
if len(channel_cols) == 1:
    axes = [axes]  # ensure iterable
axes = axes.flatten()  # flatten in case of multiple subplots

for i, col in enumerate(channel_cols):
    # grid + partial dependence curve
    XX = gam.generate_X_grid(term=i)
    pdep, confi = gam.partial_dependence(term=i, width=0.95, X=XX)  # returns the smooth for term i on the grid
    axes[i].plot(XX[:, i], pdep, lw=2, label="GAM partial effect")
    axes[i].plot(XX[:, i], confi[:, 0], ls="--", color="grey", alpha=0.7)
    axes[i].plot(XX[:, i], confi[:, 1], ls="--", color="grey", alpha=0.7)

    # partial residuals for observed data
    f_all = gam.predict(X.values)  # total prediction
    f_i_obs = gam.partial_dependence(term=i, X=X.values)  # contribution of term i at each row
    partial_resid = y.values - (f_all - f_i_obs)

    axes[i].scatter(X[col].values, partial_resid, s=10, alpha=0.35, label="partial residuals")

    axes[i].set_title(f"{col} → partial effect & residuals")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("effect on conversions")
    axes[i].legend(loc="best")

plt.tight_layout()
fig.savefig(PLOT_DIR / "gam_fits.png")

# %% -
# caculate and plot lift and ROI curves
lc_search = lift_curve(gam, X, "spend_search_ads")
roi_search = roi_curve(lc_search, value_per_conv=100, spend_name="Search")

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(8, 3))

# Lift curve
axes[0].plot(roi_search["feature_value"], roi_search["lift"], marker="o", color="blue")
axes[0].set_title("Lift vs Spend")
axes[0].set_xlabel("Spend")
axes[0].set_ylabel("Lift")

# ROI curve
axes[1].plot(roi_search["feature_value"], roi_search["roi"], marker="o", color="green")
axes[1].set_title("ROI vs Spend")
axes[1].set_xlabel("Spend")
axes[1].set_ylabel("ROI")

plt.tight_layout()
fig.savefig(PLOT_DIR / "lift_roi_curves.png")
