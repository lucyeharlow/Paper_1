"""
October 2 m temperature: regression and autocorrelation diagnostics for the trend model.

Loads the main trend script (main_script below) and reuses its loading, land mask,
anomalies, year ranges and completeness rule, so the diagnostics describe exactly the
October data behind the trend figures. Everything below is October only.

MODEL
  Y_t  --remove linear trend (OLS)-->  eps_t  --fit AR(p), p = 0, 1, 2-->  eta_t
  The AR models are fitted to the regression residuals eps_t by conditional least squares,
  all on the same sample (the first 2 years are conditioned on), so their AICc values are
  comparable. AICc = n ln(sigma^2) + 2k + 2k(k+1)/(n-k-1), k = p + 1. AR(0) means eps_t is
  already white noise and needs no autocorrelation correction.
  Adequacy: if AR(p) captures the dependence, its innovations eta_t look like white noise.
  This is checked with the ACF of eta_t and a Ljung-Box test,
  Q = n(n+2) sum_{k=1..h} r_k^2 / (n-k) ~ chi2(h - p), h = min(10, n // 5).

FIGURES (written to the main script's out_dir), one row per dataset
  october_regression_diagnostics.png
      Regional (area-weighted) October series: residuals vs time | residuals vs fitted |
      normal Q-Q of the residuals with a Shapiro-Wilk test.
      With a single predictor (year) the fitted values are a linear function of time, so
      panel 2 is panel 1 rescaled; it is kept because it is the standard check.
  october_ar_order_diagnostics.png
      Regional October series: ACF of eps | PACF of eps | ACF of AR(1) innovations |
      ACF of AR(2) innovations | table of phi, AICc, delta-AICc and Ljung-Box p for
      AR(0), AR(1), AR(2). Grey bands: 95 % white-noise limits +-1.96/sqrt(n).
  october_ar_order_maps.png
      The same tests for every complete land grid cell (the cells behind the trend maps):
      (top) AR order with the lowest AICc; (middle) Ljung-Box rejects white noise for the
      OLS residuals, i.e. some correction is needed; (bottom) Ljung-Box rejects white noise
      for the AR(1) innovations, i.e. AR(1) is not enough. Text: % of land area.
      At the 5 % level about 5 % of cells reject by chance even if every cell is white noise.
  october_ar_order_summary.csv
      Per dataset: regional-series results and per-cell area fractions.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
import cartopy.feature as cfeature

main_script = Path(__file__).with_name("lag_autocorr.py")
spec = importlib.util.spec_from_file_location("trends", main_script)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

month = 10
orders = (0, 1, 2)
max_lag = 10
alpha = 0.05


# ---------------------------------------------------------------- data
def prepare():
    """Same steps as the main script: land mask, anomalies, year ranges; then October."""
    data = {n: m.anomalies(n, da) for n, da in m.land_only(m.load_all()).items()}
    shared = set.intersection(*(m.years_of(da) for n, da in data.items() if n != "CHIRTS"))
    periods = {n: sorted(shared) for n in data}
    periods["CHIRTS"] = sorted(m.years_of(data["CHIRTS"]) & m.chirts_years)
    out = {}
    for n, da in data.items():
        da = m.select_years(da, periods[n])
        octo = m.complete(da.sel(time=da.time.dt.month == month))
        years = octo.time.dt.year.values                       # gap-free year axis, so lag k
        out[n] = octo.assign_coords(time=years).reindex(       # always means k years
            time=np.arange(years.min(), years.max() + 1))
    return out


def regional(da):
    """Area-weighted regional October series (complete cells), as (T, 1)."""
    s = da.weighted(np.cos(np.deg2rad(da.lat))).mean(("lat", "lon"))
    return s.values[:, None]


def cells(da):
    """(T, C) array of the cells that have data, plus their positions and area weights."""
    st = da.stack(cell=("lat", "lon"))
    keep = st.notnull().any("time").values
    st = st.isel(cell=keep)
    return st.values, st.cell, np.cos(np.deg2rad(st.lat.values))


# ---------------------------------------------------------------- statistics on (T, C)
def detrend(y, t):
    """OLS on year per column; returns residuals and fitted values (NaN where y is NaN)."""
    ok = np.isfinite(y)
    tt = np.where(ok, t[:, None], np.nan)
    tm, ym = np.nanmean(tt, 0), np.nanmean(y, 0)
    b = np.nansum((tt - tm) * (y - ym), 0) / np.nansum((tt - tm) ** 2, 0)
    fit = np.where(ok, ym + b * (t[:, None] - tm), np.nan)
    return y - fit, fit


def acf(e, nlags=max_lag):
    """Sample ACF, lags 1..nlags; pairs with a missing year are skipped."""
    e = e - np.nanmean(e, 0)
    d = np.nansum(e ** 2, 0)
    return np.array([np.nansum(e[k:] * e[:-k], 0) / d for k in range(1, nlags + 1)])


def pacf(r):
    """PACF from the ACF (Durbin-Levinson); r has shape (L, C) for lags 1..L."""
    L = r.shape[0]
    phi = np.zeros((L, L) + r.shape[1:])
    for k in range(L):
        if k == 0:
            phi[0, 0] = r[0]
            continue
        num = r[k] - sum(phi[k - 1, j] * r[k - 1 - j] for j in range(k))
        den = 1 - sum(phi[k - 1, j] * r[j] for j in range(k))
        phi[k, k] = num / den
        for j in range(k):
            phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - 1 - j]
    return np.array([phi[k, k] for k in range(L)])


def ar_fits(e):
    """Fit AR(0), AR(1), AR(2) to residuals e (T, C) on a common sample.

    Returns {p: (phi (C, p), innovations (T-2, C), AICc (C,))}.
    """
    pmax = max(orders)
    T = e.shape[0]
    y = e[pmax:]
    lags = np.stack([e[pmax - k:T - k] for k in range(1, pmax + 1)], -1)   # (T-pmax, C, pmax)
    ok = np.isfinite(y) & np.isfinite(lags).all(-1)                       # common sample
    n = ok.sum(0)
    out = {}
    for p in orders:
        if p == 0:
            phi = np.zeros((y.shape[1], 0))
            innov = np.where(ok, y, np.nan)
        else:
            X = np.where(ok[..., None], lags[..., :p], 0.0)
            yz = np.where(ok, y, 0.0)
            phi = np.linalg.solve(np.einsum("tcp,tcq->cpq", X, X),
                                  np.einsum("tcp,tc->cp", X, yz)[..., None])[..., 0]
            innov = np.where(ok, y - np.einsum("tcp,cp->tc", X, phi), np.nan)
        k = p + 1
        sigma2 = np.nansum(innov ** 2, 0) / n
        out[p] = (phi, innov, n * np.log(sigma2) + 2 * k + 2 * k * (k + 1) / (n - k - 1))
    return out


def ljung_box(innov, p):
    """Ljung-Box test of the innovations of an AR(p) model; returns (Q, p-value, h)."""
    n = np.isfinite(innov).sum(0)
    h = int(min(max_lag, n.min() // 5))
    r = acf(innov, h)
    Q = n * (n + 2) * np.sum(r ** 2 / (n - np.arange(1, h + 1)[:, None]), 0)
    return Q, stats.chi2.sf(Q, h - p), h


# ---------------------------------------------------------------- plotting helpers
def acf_panel(ax, r, n, title, colour, pac=False):
    lags = np.arange(1, len(r) + 1)
    band = 1.96 / np.sqrt(n)
    ax.axhspan(-band, band, color="0.88", zorder=0)
    ax.bar(lags, r, width=0.6, color=colour)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set(xlim=(0.4, len(r) + 0.6), ylim=(-0.75, 0.75), xticks=lags[::2])
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)


def no_data(row, name, y):
    """Blank a row (with a note) when a dataset has no complete October cells."""
    if np.isfinite(y).sum() > 5:
        return False
    for ax in row:
        ax.axis("off")
    row[0].text(0, 0.5, f"{m.labels[name]}: no complete October cells", fontsize=10,
                fontweight="bold", transform=row[0].transAxes)
    return True


def save(fig, name):
    out = m.out_dir / name
    fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ---------------------------------------------------------------- figures
def regression_figure(series):
    fig, axes = plt.subplots(len(series), 3, figsize=(13, 2.6 * len(series)), squeeze=False)
    for row, (name, (t, y)) in zip(axes, series.items()):
        c = m.colours[name]
        if no_data(row, name, y):
            continue
        e, fit = detrend(y, t)
        e, fit = e[:, 0], fit[:, 0]
        ok = np.isfinite(e)
        sd = np.nanstd(e, ddof=2)
        for ax, x, xl in ((row[0], t, "Year"), (row[1], fit, "Fitted anomaly (°C)")):
            ax.axhline(0, color="black", linewidth=0.7)
            for s in (-2, 2):
                ax.axhline(s * sd, color="0.6", linewidth=0.6, linestyle="--")
            ax.plot(x[ok], e[ok], "o", ms=3.5, color=c)
            if xl == "Year":
                ax.plot(t[ok], e[ok], color=c, linewidth=0.8, alpha=0.6)
            ax.set_xlabel(xl, fontsize=8)
            ax.tick_params(labelsize=7)
        row[0].set_ylabel(f"{m.labels[name]}\nresidual (°C)", fontsize=9, fontweight="bold")
        z = np.sort(e[ok] / sd)
        q = stats.norm.ppf((np.arange(z.size) + 0.5) / z.size)
        row[2].axline((0, 0), slope=1, color="black", linewidth=0.7, linestyle="--")
        row[2].plot(q, z, "o", ms=3.5, color=c)
        sw = stats.shapiro(e[ok]).pvalue
        row[2].text(0.04, 0.95, f"Shapiro-Wilk p = {sw:.2f}", transform=row[2].transAxes,
                    fontsize=8, va="top")
        row[2].set_xlabel("Theoretical normal quantile", fontsize=8)
        row[2].set_ylabel("Standardised residual", fontsize=8)
        row[2].tick_params(labelsize=7)
    for ax, title in zip(axes[0], ("Residuals vs time (±2σ dashed)", "Residuals vs fitted",
                                   "Normal Q-Q of residuals")):
        ax.set_title(title, fontsize=11, fontweight="bold")
    fig.suptitle("October regional series: OLS trend residuals", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save(fig, "october_regression_diagnostics.png")


def ar_figure(series, rows):
    fig, axes = plt.subplots(len(series), 5, figsize=(19, 2.6 * len(series)), squeeze=False,
                             gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.25]})
    for row, (name, (t, y)) in zip(axes, series.items()):
        c = m.colours[name]
        if no_data(row, name, y):
            continue
        e = detrend(y, t)[0]
        n = int(np.isfinite(e).sum())
        fits = ar_fits(e)
        r = acf(e)[:, 0]
        acf_panel(row[0], r, n, "", c)
        acf_panel(row[1], pacf(acf(e))[:, 0], n, "", c)
        for ax, p in ((row[2], 1), (row[3], 2)):
            acf_panel(ax, acf(fits[p][1])[:, 0], n - 2, "", c)
        row[0].set_ylabel(f"{m.labels[name]}\n(n = {n})", fontsize=9, fontweight="bold")
        aicc = {p: float(fits[p][2][0]) for p in orders}
        best = min(aicc, key=aicc.get)
        lines = [f"{'model':<6}{'phi':>14}{'AICc':>8}{'dAICc':>7}{'LB p':>7}"]
        for p in orders:
            phi = fits[p][0][0]
            Q, pv, h = ljung_box(fits[p][1], p)
            lines.append(f"AR({p}) {', '.join(f'{v:+.2f}' for v in phi) or '-':>13}"
                         f"{aicc[p]:8.1f}{aicc[p] - aicc[best]:7.1f}{float(pv[0]):7.2f}")
            rows.append({"dataset": name, "scope": "regional series", "model": f"AR({p})",
                         "phi": ";".join(f"{v:.3f}" for v in phi), "AICc": aicc[p],
                         "delta_AICc": aicc[p] - aicc[best], "ljung_box_Q": float(Q[0]),
                         "ljung_box_lags": h, "ljung_box_p": float(pv[0]), "n_years": n})
        lines += ["", f"lowest AICc: AR({best})", f"Ljung-Box h = {h} lags (df = h - p)",
                  "dAICc < 2: models equally supported"]
        row[4].axis("off")
        row[4].text(0, 0.95, "\n".join(lines), family="monospace", fontsize=7.5, va="top",
                    transform=row[4].transAxes)
    for ax, title in zip(axes[0], ("ACF of residuals ε", "PACF of residuals ε",
                                   "ACF of AR(1) innovations η", "ACF of AR(2) innovations η",
                                   "Model comparison")):
        ax.set_title(title, fontsize=11, fontweight="bold")
    for ax in axes[-1, :4]:
        ax.set_xlabel("Lag (years)", fontsize=8)
    fig.suptitle("October regional series: does AR(1) describe the temporal dependence?\n"
                 "(grey: 95% white-noise band; bars outside it at several lags in η = structure "
                 "the model has missed)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "october_ar_order_diagnostics.png")


def map_figure(data, rows):
    fig, axes = plt.subplots(3, len(data), figsize=(3.1 * len(data), 9.5), squeeze=False,
                             subplot_kw={"projection": m.pc})
    order_cmap = ListedColormap(["#d9d9d9", "#6baed6", "#08519c"])
    order_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3)
    rej_cmap = ListedColormap(["#d9d9d9", "#cb181d"])
    rej_norm = BoundaryNorm([-0.5, 0.5, 1.5], 2)
    for j, (name, da) in enumerate(data.items()):
        t = da.time.values.astype(float)
        y, cell, w = cells(da)
        result = {}
        if y.shape[1]:
            e = detrend(y, t)[0]
            fits = ar_fits(e)
            aicc = np.stack([fits[p][2] for p in orders])
            result["best"] = np.array(orders)[np.argmin(aicc, 0)]
            result["lb0"] = ljung_box(fits[0][1], 0)[1] < alpha
            result["lb1"] = ljung_box(fits[1][1], 1)[1] < alpha
        for i, (key, cmap, norm, label) in enumerate((
                ("best", order_cmap, order_norm, "lowest-AICc order"),
                ("lb0", rej_cmap, rej_norm, "LB rejects white noise: OLS residuals"),
                ("lb1", rej_cmap, rej_norm, "LB rejects white noise: AR(1) innovations"))):
            ax = axes[i, j]
            if key in result:
                field = xr.DataArray(result[key].astype(float), coords={"cell": cell},
                                     dims="cell").unstack("cell").reindex_like(da.isel(time=0))
                ax.pcolormesh(field.lon, field.lat, field, transform=m.pc, cmap=cmap,
                              norm=norm, shading="nearest")
                if key == "best":
                    note = "\n".join(f"AR({p}): {100 * w[result[key] == p].sum() / w.sum():.0f}%"
                                     for p in orders)
                else:
                    note = f"{100 * w[result[key]].sum() / w.sum():.0f}% of area\n(~5% by chance)"
            else:
                note = "no complete cells"
            ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.7, zorder=5)
            ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.5, zorder=5)
            ax.set_extent([m.lon_min, m.lon_max, m.lat_min, m.lat_max], crs=m.pc)
            ax.text(0.03, 0.04, note, transform=ax.transAxes, fontsize=7.5, zorder=8,
                    bbox=dict(boxstyle="square,pad=0.2", facecolor="white", edgecolor="none",
                              alpha=0.85))
            if i == 0:
                ax.set_title(m.labels[name], fontsize=10, fontweight="bold")
            if j == 0:
                ax.text(-0.12, 0.5, label, transform=ax.transAxes, rotation=90, fontsize=8.5,
                        ha="center", va="center", fontweight="bold")
        if result:
            area = lambda mask: 100 * w[mask].sum() / w.sum()
            rows.append({"dataset": name, "scope": "grid cells (% of land area)",
                         "n_cells": y.shape[1],
                         **{f"pct_area_best_AR{p}": area(result["best"] == p) for p in orders},
                         "pct_area_LB_rejects_OLS_residuals": area(result["lb0"]),
                         "pct_area_LB_rejects_AR1_innovations": area(result["lb1"])})
    handles = [matplotlib.patches.Patch(color=order_cmap(i), label=f"AR({p}) lowest AICc")
               for i, p in enumerate(orders)]
    handles.append(matplotlib.patches.Patch(color=rej_cmap(1), label="Ljung-Box p < 0.05"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("October, every complete land grid cell: which AR order, and is AR(1) enough?",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "october_ar_order_maps.png")


def main():
    data = prepare()
    series = {n: (da.time.values.astype(float), regional(da)) for n, da in data.items()}
    rows = []
    print("\nOctober regression and AR-order diagnostics...")
    regression_figure(series)
    ar_figure(series, rows)
    map_figure(data, rows)
    table = pd.DataFrame(rows)
    table.to_csv(m.out_dir / "october_ar_order_summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()