"""
October SW-down: checks of the assumptions behind the OLS trends and their AR(1)-corrected
significance, for ERA5, MERRA-2, CLARA-A3 and SARAH-3 (same as the temperature and vertical
velocity checks).

Loads sw_trends_monthly.py (same folder) and reuses its readers, so the checks describe exactly
the October data (1983-2024, MERRA-2 grid) behind the trend maps.

MODEL
  Y_t  --remove linear trend (OLS)-->  eps_t  --fit AR(p), p = 0, 1, 2-->  eta_t
  AR models are fitted to eps_t by conditional least squares on a common sample (the first 2
  years conditioned on), so their AICc values are comparable:
  AICc = n ln(sigma^2) + 2k + 2k(k+1)/(n-k-1), k = p + 1. AR(0): eps_t is already white noise.
  Adequacy: if AR(p) captures the dependence, its innovations eta_t are white noise; checked
  with their ACF and the Ljung-Box test Q = n(n+2) sum_{k=1..h} r_k^2/(n-k) ~ chi2(h - p),
  h = min(10, n // 5).

REGIONAL SERIES  cos-latitude area-weighted mean over the southern Africa box (10-40 E,
  35-10 S, as in the temperature scripts), one per dataset.

FIGURES (in <OUT_DIR>/assumption_checks)
  sw_oct_regression_diagnostics.png   one row per dataset: residuals vs time | residuals vs
                                      fitted | normal Q-Q (Shapiro-Wilk p)
  sw_oct_ar_order_diagnostics.png     same rows: ACF eps | PACF eps | ACF of AR(1) innovations |
                                      ACF of AR(2) innovations | AICc / Ljung-Box table
  sw_oct_ar_order_maps.png            every grid point of the map domain (Africa, 40 S - 20 N,
                                      20 W - 55 E), one column per dataset: lowest-AICc AR order |
                                      Ljung-Box rejects white noise for eps | Ljung-Box rejects for
                                      the AR(1) innovations | Shapiro-Wilk rejects normality.
                                      Text: % of the domain area (about 5 % reject by chance).
  sw_oct_assumption_summary_regional.csv, sw_oct_assumption_summary_gridpoints.csv
"""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import matplotlib.patches
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
import cartopy.crs as ccrs
import cartopy.feature as cfeature

spec = importlib.util.spec_from_file_location(
    "sw_trends", Path(__file__).with_name("shortwave_comparison_monthly.py"))
s = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = s
spec.loader.exec_module(s)

orders = (0, 1, 2)
max_lag = 10
alpha = 0.05
box = (10.0, 40.0, -35.0, -10.0)                     # regional series: W, E, S, N
map_domain = (-20.0, 55.0, -40.0, 20.0)              # maps: W, E, S, N
region_name = "southern Africa (10-40°E, 35-10°S)"
colours = {"ERA5": "#FF7C7E", "MERRA-2": "#FEC300", "CLARA-A3": "#016ff2",
           "SARAH-3": "#31BCBA"}
pc = ccrs.PlateCarree()


def geo(ax, labels=(True, True)):
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.6, zorder=5)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.4, zorder=5)
    ax.set_extent(map_domain, crs=pc)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="grey", alpha=0.5, linestyle="--")
    gl.top_labels = gl.right_labels = False
    gl.left_labels, gl.bottom_labels = labels
    gl.xlabel_style = gl.ylabel_style = {"size": 8}


def in_box(la, lo):
    return (lo >= box[0]) & (lo <= box[1]) & (la >= box[2]) & (la <= box[3])


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
    row[0].text(0, 0.5, f"{label}: no data", fontsize=10, fontweight="bold",
                transform=row[0].transAxes)
    return True


def save(fig, out, name, tight=True):
    # maps are laid out by hand: bbox_inches="tight" / tight_layout can squash cartopy axes
    # (shapely "LinearRing ... not closed" errors on some cartopy versions)
    fig.savefig(out / name, dpi=200, facecolor="white", bbox_inches="tight" if tight else None)
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
        for ax, x, xl in ((row[0], t, "Year"), (row[1], fit, "Fitted SW↓ (W m⁻²)")):
            ax.axhline(0, color="black", linewidth=0.7)
            for k in (-2, 2):
                ax.axhline(k * sd, color="0.6", linewidth=0.6, linestyle="--")
            ax.plot(x[ok], e[ok], "o", ms=3.5, color=c)
            if xl == "Year":
                ax.plot(t[ok], e[ok], color=c, linewidth=0.8, alpha=0.6)
            ax.set_xlabel(xl, fontsize=8)
            ax.tick_params(labelsize=7)
        row[0].set_ylabel(f"{label}\nresidual (W m⁻²)", fontsize=8.5, fontweight="bold")
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
    fig.suptitle(f"October SW↓ ({s.START}-{s.END}), {region_name}: OLS trend residuals",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    save(fig, out, "sw_oct_regression_diagnostics.png")


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
            rows.append({"dataset": n, "scope": region, "model": f"AR({p})",
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
    fig.suptitle(f"October SW↓ ({s.START}-{s.END}), {region_name}: does AR(1) describe the "
                 "temporal dependence?\n(grey: 95% white-noise band)", fontsize=13,
                 fontweight="bold")
    fig.tight_layout()
    save(fig, out, "sw_oct_ar_order_diagnostics.png")


def map_figure(octo, lev, out, rows):
    fig, axes = plt.subplots(4, len(octo), figsize=(4.6 * len(octo), 15), squeeze=False,
                             subplot_kw={"projection": pc})
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
                ax.pcolormesh(field.lon, field.lat, field, transform=pc, cmap=cmap, norm=norm,
                              shading="nearest")
                area = lambda mask: 100 * w[mask].sum() / w.sum()
                note = ("\n".join(f"AR({p}): {area(result[key] == p):.0f}%" for p in orders)
                        if key == "best" else f"{area(result[key]):.0f}% of area\n(~5% by chance)")
            else:
                note = "no complete grid points"
            geo(ax, labels=(j == 0, i == 3))
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
            rows.append({"dataset": n, "scope": "grid points (% of map-domain area)",
                         "n_boxes": y.shape[1],
                         **{f"pct_area_best_AR{p}": area(result["best"] == p) for p in orders},
                         "pct_area_LB_rejects_OLS_residuals": area(result["lb0"]),
                         "pct_area_LB_rejects_AR1_innovations": area(result["lb1"]),
                         "pct_area_SW_rejects_normality": area(result["sw"])})
    handles = [matplotlib.patches.Patch(color=order_cmap(i), label=f"AR({p}) lowest AICc")
               for i, p in enumerate(orders)]
    handles.append(matplotlib.patches.Patch(color=rej_cmap(1), label="test rejects (p < 0.05)"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=10)
    fig.suptitle(f"October SW↓ ({s.START}-{s.END}), every grid point of the map domain: which AR order, is AR(1) "
                 "enough, are residuals normal?", fontsize=13, fontweight="bold")
    fig.subplots_adjust(left=0.07, right=0.99, top=0.94, bottom=0.07, wspace=0.12, hspace=0.1)
    save(fig, out, "sw_oct_ar_order_maps.png", tight=False)


def main():
    out = Path(s.OUT_DIR) / "assumption_checks"
    out.mkdir(parents=True, exist_ok=True)
    octo = {}
    for name in s.NAMES:
        print(f"\n--- {name} ---", flush=True)
        da = s.LOADERS[name]()
        if da is None:
            continue
        da = da.reindex(year=np.arange(int(da.year.min()), int(da.year.max()) + 1))
        octo[name] = da.sel(lat=slice(map_domain[2], map_domain[3]),
                            lon=slice(map_domain[0], map_domain[1]))
    series = {(n, "S. Africa box"): (da.year.values.astype(float), regional(da, in_box))
              for n, da in octo.items()}
    rows = []
    regression_figure(series, None, out)
    ar_figure(series, None, out, rows)
    map_figure(octo, None, out, rows)
    table = pd.DataFrame(rows)
    grid = table.scope.str.startswith("grid")
    for name, part in (("regional", table[~grid]), ("gridpoints", table[grid])):
        part = part.dropna(axis=1, how="all")
        part.to_csv(out / f"sw_oct_assumption_summary_{name}.csv", index=False)
        with pd.option_context("display.width", 220, "display.max_columns", 20):
            print(f"\n{part.round(3).to_string(index=False)}")


if __name__ == "__main__":
    main()