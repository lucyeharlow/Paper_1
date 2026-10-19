"""
October vertical velocity (omega): checks of the assumptions behind the OLS trends and their
AR(1)-corrected significance, for ERA5, MERRA-2 and JRA-3Q.

Loads omega_regime_change_october.py (same folder) and reuses its loading, common 1.25-degree
grid and completeness rule, so the checks describe exactly the October data and boxes behind
the regime / trend maps. Run for each level in diag_levels.

MODEL
  Y_t  --remove linear trend (OLS)-->  eps_t  --fit AR(p), p = 0, 1, 2-->  eta_t
  AR models are fitted to eps_t by conditional least squares on a common sample (the first 2
  years conditioned on), so their AICc values are comparable:
  AICc = n ln(sigma^2) + 2k + 2k(k+1)/(n-k-1), k = p + 1. AR(0): eps_t is already white noise.
  Adequacy: if AR(p) captures the dependence, its innovations eta_t are white noise; checked
  with their ACF and the Ljung-Box test Q = n(n+2) sum_{k=1..h} r_k^2/(n-k) ~ chi2(h - p),
  h = min(10, n // 5).

REGIONAL SERIES (cos-latitude area-weighted means of the complete boxes)
  Congo box (10-32 E, 10 S-20 N) and the whole domain south of 10 S, the regions used in the
  printed statistics of the trend script.

FIGURES per level (in <out_dir>/october/assumption_checks)
  omega_oct_<lev>hPa_regression_diagnostics.png   one row per dataset x region:
      residuals vs time | residuals vs fitted | normal Q-Q (Shapiro-Wilk p)
  omega_oct_<lev>hPa_ar_order_diagnostics.png     same rows: ACF eps | PACF eps |
      ACF of AR(1) innovations | ACF of AR(2) innovations | AICc / Ljung-Box table
  omega_oct_<lev>hPa_ar_order_maps.png            every complete box, one column per dataset:
      lowest-AICc AR order | Ljung-Box rejects white noise for eps (a correction is needed) |
      Ljung-Box rejects for the AR(1) innovations (AR(1) is not enough) | Shapiro-Wilk rejects
      normality of eps. Text: % of the domain area (about 5 % reject by chance at the 5 % level).
  omega_oct_assumption_summary_regional.csv, omega_oct_assumption_summary_boxes.csv
"""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

spec = importlib.util.spec_from_file_location(
    "omega_oct", Path(__file__).with_name("october_trends.py"))
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m                        # lets parallel readers find it
spec.loader.exec_module(m)

diag_levels = [200.0, 500.0, 850.0]
orders = (0, 1, 2)
max_lag = 10
alpha = 0.05
colours = {"ERA5": "#FF7C7E", "MERRA-2": "#FEC300", "JRA-3Q": "#016ff2"}
regions = {"Congo box": lambda la, lo: (la >= m.congo[2]) & (la <= m.congo[3])
           & (lo >= m.congo[0]) & (lo <= m.congo[1]),
           f"South of {abs(m.s_africa_lat_n):g}°S": lambda la, lo: la < m.s_africa_lat_n}


# ---------------------------------------------------------------- data
def october(data, lev):
    """{name: (year, lat, lon)} on a gap-free year axis, so lag k always means k years."""
    out = {}
    for n, d in data.items():
        da = d.sel(lev=lev).isel(month=0)
        out[n] = da.reindex(year=np.arange(int(da.year.min()), int(da.year.max()) + 1))
    return out


def regional(da, region):
    la, lo = xr.broadcast(da.lat, da.lon)
    sub = da.where(region(la, lo))
    s = sub.weighted(np.cos(np.deg2rad(sub.lat))).mean(("lat", "lon"))
    return s.where(da.notnull().any(("lat", "lon"))).values[:, None]


def cells(da):
    st = da.stack(cell=("lat", "lon"))
    keep = st.notnull().any("year").values
    st = st.isel(cell=keep)
    return st.transpose("year", "cell").values, st.cell, np.cos(np.deg2rad(st.lat.values))


# ---------------------------------------------------------------- statistics on (T, C)
def detrend(y, t):
    ok = np.isfinite(y)
    tt = np.where(ok, t[:, None], np.nan)
    tm, ym = np.nanmean(tt, 0), np.nanmean(y, 0)
    b = np.nansum((tt - tm) * (y - ym), 0) / np.nansum((tt - tm) ** 2, 0)
    fit = np.where(ok, ym + b * (t[:, None] - tm), np.nan)
    return y - fit, fit


def acf(e, nlags=max_lag):
    e = e - np.nanmean(e, 0)
    d = np.nansum(e ** 2, 0)
    return np.array([np.nansum(e[k:] * e[:-k], 0) / d for k in range(1, nlags + 1)])


def pacf(r):
    """PACF from the ACF (Durbin-Levinson); r (L, C) for lags 1..L."""
    L = r.shape[0]
    phi = np.zeros((L, L) + r.shape[1:])
    for k in range(L):
        if k == 0:
            phi[0, 0] = r[0]
            continue
        phi[k, k] = ((r[k] - sum(phi[k - 1, j] * r[k - 1 - j] for j in range(k)))
                     / (1 - sum(phi[k - 1, j] * r[j] for j in range(k))))
        for j in range(k):
            phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - 1 - j]
    return np.array([phi[k, k] for k in range(L)])


def ar_fits(e):
    """AR(0), AR(1), AR(2) on a common sample: {p: (phi (C, p), innovations, AICc (C,))}."""
    pmax, T = max(orders), e.shape[0]
    y = e[pmax:]
    lags = np.stack([e[pmax - k:T - k] for k in range(1, pmax + 1)], -1)
    ok = np.isfinite(y) & np.isfinite(lags).all(-1)
    n = ok.sum(0)
    out = {}
    for p in orders:
        if p == 0:
            phi, innov = np.zeros((y.shape[1], 0)), np.where(ok, y, np.nan)
        else:
            X = np.where(ok[..., None], lags[..., :p], 0.0)
            phi = np.linalg.solve(np.einsum("tcp,tcq->cpq", X, X),
                                  np.einsum("tcp,tc->cp", X, np.where(ok, y, 0.0))[..., None])[..., 0]
            innov = np.where(ok, y - np.einsum("tcp,cp->tc", X, phi), np.nan)
        k = p + 1
        sigma2 = np.nansum(innov ** 2, 0) / n
        out[p] = (phi, innov, n * np.log(sigma2) + 2 * k + 2 * k * (k + 1) / (n - k - 1))
    return out


def ljung_box(innov, p):
    n = np.isfinite(innov).sum(0)
    h = int(min(max_lag, n.min() // 5))
    r = acf(innov, h)
    Q = n * (n + 2) * np.sum(r ** 2 / (n - np.arange(1, h + 1)[:, None]), 0)
    return Q, stats.chi2.sf(Q, h - p), h


# ---------------------------------------------------------------- plotting helpers
def acf_panel(ax, r, n, colour):
    lags = np.arange(1, len(r) + 1)
    band = 1.96 / np.sqrt(n)
    ax.axhspan(-band, band, color="0.88", zorder=0)
    ax.bar(lags, r, width=0.6, color=colour)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set(xlim=(0.4, len(r) + 0.6), ylim=(-0.75, 0.75), xticks=lags[::2])
    ax.tick_params(labelsize=7)


def no_data(row, label, y):
    if np.isfinite(y).sum() > 5:
        return False
    for ax in row:
        ax.axis("off")
    row[0].text(0, 0.5, f"{label}: no complete boxes", fontsize=10, fontweight="bold",
                transform=row[0].transAxes)
    return True


def save(fig, out, name):
    fig.savefig(out / name, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}", flush=True)


# ---------------------------------------------------------------- figures
def regression_figure(series, lev, out):
    fig, axes = plt.subplots(len(series), 3, figsize=(13, 2.5 * len(series)), squeeze=False)
    for row, ((n, region), (t, y)) in zip(axes, series.items()):
        c, label = colours[n], f"{n}\n{region}"
        if no_data(row, label, y):
            continue
        e, fit = (a[:, 0] for a in detrend(y, t))
        ok = np.isfinite(e)
        sd = np.nanstd(e, ddof=2)
        for ax, x, xl in ((row[0], t, "Year"), (row[1], fit, "Fitted omega (Pa s⁻¹)")):
            ax.axhline(0, color="black", linewidth=0.7)
            for k in (-2, 2):
                ax.axhline(k * sd, color="0.6", linewidth=0.6, linestyle="--")
            ax.plot(x[ok], e[ok], "o", ms=3.5, color=c)
            if xl == "Year":
                ax.plot(t[ok], e[ok], color=c, linewidth=0.8, alpha=0.6)
            ax.set_xlabel(xl, fontsize=8)
            ax.tick_params(labelsize=7)
        row[0].set_ylabel(f"{label}\nresidual (Pa s⁻¹)", fontsize=8.5, fontweight="bold")
        z = np.sort(e[ok] / sd)
        row[2].axline((0, 0), slope=1, color="black", linewidth=0.7, linestyle="--")
        row[2].plot(stats.norm.ppf((np.arange(z.size) + 0.5) / z.size), z, "o", ms=3.5, color=c)
        row[2].text(0.04, 0.95, f"Shapiro-Wilk p = {stats.shapiro(e[ok]).pvalue:.2f}",
                    transform=row[2].transAxes, fontsize=8, va="top")
        row[2].set_xlabel("Theoretical normal quantile", fontsize=8)
        row[2].set_ylabel("Standardised residual", fontsize=8)
        row[2].tick_params(labelsize=7)
    for ax, title in zip(axes[0], ("Residuals vs time (±2σ dashed)", "Residuals vs fitted",
                                   "Normal Q-Q of residuals")):
        ax.set_title(title, fontsize=11, fontweight="bold")
    fig.suptitle(f"October {lev:.0f} hPa omega, regional series: OLS trend residuals",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    save(fig, out, f"omega_oct_{lev:.0f}hPa_regression_diagnostics.png")


def ar_figure(series, lev, out, rows):
    fig, axes = plt.subplots(len(series), 5, figsize=(19, 2.5 * len(series)), squeeze=False,
                             gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.3]})
    for row, ((n, region), (t, y)) in zip(axes, series.items()):
        c, label = colours[n], f"{n}\n{region}"
        if no_data(row, label, y):
            continue
        e = detrend(y, t)[0]
        nn = int(np.isfinite(e).sum())
        fits = ar_fits(e)
        acf_panel(row[0], acf(e)[:, 0], nn, c)
        acf_panel(row[1], pacf(acf(e))[:, 0], nn, c)
        for ax, p in ((row[2], 1), (row[3], 2)):
            acf_panel(ax, acf(fits[p][1])[:, 0], nn - 2, c)
        row[0].set_ylabel(f"{label}\n(n = {nn})", fontsize=8.5, fontweight="bold")
        aicc = {p: float(fits[p][2][0]) for p in orders}
        best = min(aicc, key=aicc.get)
        lines = [f"{'model':<6}{'phi':>14}{'AICc':>9}{'dAICc':>7}{'LB p':>7}"]
        for p in orders:
            phi = fits[p][0][0]
            Q, pv, h = ljung_box(fits[p][1], p)
            lines.append(f"AR({p}) {', '.join(f'{v:+.2f}' for v in phi) or '-':>13}"
                         f"{aicc[p]:9.1f}{aicc[p] - aicc[best]:7.1f}{float(pv[0]):7.2f}")
            rows.append({"level_hPa": lev, "dataset": n, "scope": region, "model": f"AR({p})",
                         "phi": ";".join(f"{v:.3f}" for v in phi), "AICc": aicc[p],
                         "delta_AICc": aicc[p] - aicc[best], "ljung_box_Q": float(Q[0]),
                         "ljung_box_lags": h, "ljung_box_p": float(pv[0]), "n_years": nn})
        lines += ["", f"lowest AICc: AR({best})", f"Ljung-Box h = {h} (df = h - p)",
                  "dAICc < 2: equally supported"]
        row[4].axis("off")
        row[4].text(0, 0.95, "\n".join(lines), family="monospace", fontsize=7.5, va="top",
                    transform=row[4].transAxes)
    for ax, title in zip(axes[0], ("ACF of residuals ε", "PACF of residuals ε",
                                   "ACF of AR(1) innovations η", "ACF of AR(2) innovations η",
                                   "Model comparison")):
        ax.set_title(title, fontsize=11, fontweight="bold")
    for ax in axes[-1, :4]:
        ax.set_xlabel("Lag (years)", fontsize=8)
    fig.suptitle(f"October {lev:.0f} hPa omega, regional series: does AR(1) describe the "
                 "temporal dependence?\n(grey: 95% white-noise band)", fontsize=13,
                 fontweight="bold")
    fig.tight_layout()
    save(fig, out, f"omega_oct_{lev:.0f}hPa_ar_order_diagnostics.png")


def map_figure(octo, lev, out, rows):
    fig, axes = plt.subplots(4, 3, figsize=(15, 15), squeeze=False,
                             subplot_kw={"projection": m.pc})
    order_cmap = ListedColormap(["#d9d9d9", "#6baed6", "#08519c"])
    order_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3)
    rej_cmap, rej_norm = ListedColormap(["#d9d9d9", "#cb181d"]), BoundaryNorm([-0.5, 0.5, 1.5], 2)
    labels = ("lowest-AICc AR order", "LB rejects white noise:\nOLS residuals",
              "LB rejects white noise:\nAR(1) innovations", "Shapiro-Wilk rejects\nnormality of residuals")
    for j, (n, da) in enumerate(octo.items()):
        t = da.year.values.astype(float)
        y, cell, w = cells(da)
        result = {}
        if y.shape[1]:
            e = detrend(y, t)[0]
            fits = ar_fits(e)
            result["best"] = np.array(orders)[np.argmin(np.stack([fits[p][2] for p in orders]), 0)]
            result["lb0"] = ljung_box(fits[0][1], 0)[1] < alpha
            result["lb1"] = ljung_box(fits[1][1], 1)[1] < alpha
            ef = e[np.isfinite(e).all(1)]                       # complete years only
            result["sw"] = stats.shapiro(ef, axis=0).pvalue < alpha
        for i, key in enumerate(("best", "lb0", "lb1", "sw")):
            ax = axes[i, j]
            if key in result:
                field = xr.DataArray(result[key].astype(float), coords={"cell": cell},
                                     dims="cell").unstack("cell").reindex_like(da.isel(year=0))
                cmap, norm = (order_cmap, order_norm) if key == "best" else (rej_cmap, rej_norm)
                ax.pcolormesh(field.lon, field.lat, field, transform=m.pc, cmap=cmap, norm=norm,
                              shading="nearest")
                area = lambda mask: 100 * w[mask].sum() / w.sum()
                note = ("\n".join(f"AR({p}): {area(result[key] == p):.0f}%" for p in orders)
                        if key == "best" else f"{area(result[key]):.0f}% of area\n(~5% by chance)")
            else:
                note = "no complete boxes"
            m.add_geo(ax, labels=(j == 0, i == 3))
            ax.text(0.02, 0.03, note, transform=ax.transAxes, fontsize=8, zorder=9,
                    bbox=dict(boxstyle="square,pad=0.2", facecolor="white", edgecolor="none",
                              alpha=0.85))
            if i == 0:
                ax.set_title(n, fontsize=12, fontweight="bold")
            if j == 0:
                ax.text(-0.14, 0.5, labels[i], transform=ax.transAxes, rotation=90, fontsize=9.5,
                        ha="center", va="center", fontweight="bold")
        if result:
            area = lambda mask: 100 * w[mask].sum() / w.sum()
            rows.append({"level_hPa": lev, "dataset": n, "scope": "boxes (% of domain area)",
                         "n_boxes": y.shape[1],
                         **{f"pct_area_best_AR{p}": area(result["best"] == p) for p in orders},
                         "pct_area_LB_rejects_OLS_residuals": area(result["lb0"]),
                         "pct_area_LB_rejects_AR1_innovations": area(result["lb1"]),
                         "pct_area_SW_rejects_normality": area(result["sw"])})
    handles = [matplotlib.patches.Patch(color=order_cmap(i), label=f"AR({p}) lowest AICc")
               for i, p in enumerate(orders)]
    handles.append(matplotlib.patches.Patch(color=rej_cmap(1), label="test rejects (p < 0.05)"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=10)
    fig.suptitle(f"October {lev:.0f} hPa omega, every complete box: which AR order, is AR(1) "
                 "enough, are residuals normal?", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    save(fig, out, f"omega_oct_{lev:.0f}hPa_ar_order_maps.png")


def main():
    out = m.out_dir / "october" / "assumption_checks"
    out.mkdir(parents=True, exist_ok=True)
    m.levels = sorted(set(m.levels) | set(diag_levels))
    data = m.load_all()
    rows = []
    for lev in diag_levels:
        print(f"\n--- {lev:.0f} hPa ---", flush=True)
        octo = october(data, lev)
        series = {(n, r): (da.year.values.astype(float), regional(da, f))
                  for n, da in octo.items() for r, f in regions.items()}
        regression_figure(series, lev, out)
        ar_figure(series, lev, out, rows)
        map_figure(octo, lev, out, rows)
    table = pd.DataFrame(rows)
    boxes = table.scope.str.startswith("boxes")
    for name, part in (("regional", table[~boxes]), ("boxes", table[boxes])):
        part = part.dropna(axis=1, how="all")
        part.to_csv(out / f"omega_oct_assumption_summary_{name}.csv", index=False)
        with pd.option_context("display.width", 220, "display.max_columns", 20):
            print(f"\n{part.round(3).to_string(index=False)}")


if __name__ == "__main__":
    main()