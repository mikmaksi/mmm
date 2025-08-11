import numpy as np
import pandas as pd


def lift_curve(gam, X_df: pd.DataFrame, feature: str, grid=None):
    # hold others at median; vary one feature
    X_ref = X_df.median(numeric_only=True).to_frame().T
    if grid is None:
        grid = np.linspace(X_df[feature].min(), X_df[feature].max(), 60)

    # build a prediction grid
    Xg = pd.concat([X_ref] * len(grid), ignore_index=True)
    Xg[feature] = grid

    # total prediction at grid and at baseline (feature=0)
    y_hat = gam.predict(Xg.values)
    X0 = X_ref.copy()
    X0[feature] = 0
    y0 = gam.predict(X0.values)[0]

    lift = y_hat - y0
    return pd.DataFrame({"feature_value": grid, "lift": lift})


def roi_curve(lift_df: pd.DataFrame, value_per_conv: float, spend_name: str):
    spend = lift_df["feature_value"].values
    value = lift_df["lift"].values * value_per_conv
    with np.errstate(divide="ignore", invalid="ignore"):
        roi = np.where(spend > 0, value / spend, np.nan)
    out = lift_df.copy()
    out["roi"] = roi
    out["channel"] = spend_name
    return out


def lag_corr_scan(X: pd.DataFrame, y: pd.Series, lags=range(-7, 8), method="pearson"):
    """
    Returns a tidy DataFrame with best lag and correlations for each column in X.
    lags: iterable of ints (negative means predictor leads y)
    method: 'pearson' or 'spearman'
    """
    rows = []
    for col in X.columns:
        xc = X[col].values
        yc = y.values
        cors = []
        for k in lags:
            if k < 0:  # predictor leads: shift predictor forward
                xk = np.roll(xc, -k)
            else:  # predictor lags: shift predictor backward
                xk = np.roll(xc, -k)  # consistent roll direction
            # mask the wrapped edges to avoid using rolled-over points
            mask = np.ones_like(xk, dtype=bool)
            if k != 0:
                m = abs(k)
                mask[:m] = False
                mask[-m:] = False
            xv = xk[mask]
            yv = yc[mask]
            if method == "spearman":
                rk = pd.Series(xv).rank().corr(pd.Series(yv).rank())
            else:
                rk = np.corrcoef(xv, yv)[0, 1]
            cors.append(rk)
        cors = np.array(cors)
        # pick best by absolute correlation (you can switch to signed if you prefer)
        idx = int(np.nanargmax(np.abs(cors)))
        best_lag = list(lags)[idx]
        corr_best = cors[idx]
        corr_0 = cors[list(lags).index(0)]
        rows.append({"variable": col, "best_lag": best_lag, "corr_best": corr_best, "corr_at_0": corr_0})
    return pd.DataFrame(rows).sort_values(by="corr_best", key=lambda s: np.abs(s), ascending=False)


def permutation_pvalue_for_var(x, y, lags=range(-7, 8), n_perm=1000, method="pearson", random_state=42):
    """
    Time-series safe null via circular shifts: randomly roll x by a uniform offset,
    re-compute the *max absolute* lagged correlation; p = Pr(null >= observed).
    """
    rng = np.random.default_rng(random_state)
    # observed max |corr|
    obs = np.max(np.abs(_lag_corrs(x, y, lags, method)))
    # null distribution
    null = np.empty(n_perm)
    n = len(x)
    for b in range(n_perm):
        # roll by a random offset; avoid trivial 0 shift
        offset = rng.integers(1, n - 1)
        x_roll = np.roll(x, offset)
        null[b] = np.max(np.abs(_lag_corrs(x_roll, y, lags, method)))
    # p-value with +1 smoothing
    p = (1 + np.sum(null >= obs)) / (n_perm + 1)
    return p, obs, null


def _lag_corrs(x, y, lags, method):
    out = []
    for k in lags:
        xr = np.roll(x, -k)
        mask = np.ones_like(xr, dtype=bool)
        if k != 0:
            m = abs(k)
            mask[:m] = False
            mask[-m:] = False
        xv = xr[mask]
        yv = y[mask]
        if method == "spearman":
            val = pd.Series(xv).rank().corr(pd.Series(yv).rank())
        else:
            val = np.corrcoef(xv, yv)[0, 1]
        out.append(val)
    return np.array(out)
