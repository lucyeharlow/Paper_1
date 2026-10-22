"""
October 300 hPa vertical velocity (omega): leading modes of interannual variability (EOFs), how
they relate to the omega trend, and how PV co-varies with omega at every grid point.

Reuses (same folder):
  omega_regime_change_october.py         omega readers (ERA5, MERRA-2, JRA-3Q), box-averaging to
                                         the 1.25-degree grid, completeness rule, cache
  era5_merra2_pv_monthly_trends_ssa.py   PV readers (ERA5 pv, MERRA-2 EPV), in PVU
Domain: the PV script's sub-Saharan Africa domain (35 S - 15 N, 20 W - 55 E). Omega is read
afresh for this domain (its own cache folder, cache_pv_domain); PV is box-averaged onto the same
1.25-degree grid. Only years and boxes complete in all omega datasets are used (no NaNs).

1. EOFs of omega (per dataset)
   omega'(x, y, t) = October anomaly, linearly DETRENDED at each box, so the EOFs describe
   year-to-year variability only. Each box weighted by sqrt(cos lat) (area weighting of the
   covariance). EOFs from the SVD of the weighted (year x box) matrix.
     EOF_k map  = regression of omega' onto the standardised PC_k (Pa s-1 per 1 s.d. of PC_k),
     variance   = fraction of the weighted variance explained,
     separation = North et al. (1982) rule of thumb: lambda_k +- lambda_k sqrt(2 / n);
                  modes whose ranges overlap are not well separated (may mix).
   Comparison with the trend:
     r_trend    = area-weighted (centred) pattern correlation between EOF_k and the OLS trend
                  map of the undetrended data. Each EOF's sign is chosen so r_trend >= 0.
     projection = the undetrended anomalies projected onto EOF_k (same weights); its OLS trend
                  (s.d. per decade, AR(1)-corrected p, as in the trend scripts) says how much
                  the long-term change projects onto that mode.
   A mode that resembles the trend (large r_trend) AND whose projection has a significant trend
   suggests the trend is (partly) a shift in that mode of variability. Pattern correlations
   between smooth fields are high by chance more easily than the number of boxes suggests, so
   treat r_trend as descriptive.

2. Grid-point regression of PV on omega (ERA5 and MERRA-2, each with its own omega)
   PV'(x, y, t) = a(x, y) + b(x, y) omega'(x, y, t) + e,  both detrended October anomalies.
   b in PVU per 0.01 Pa s-1, r = correlation. p: t-test on b with n_eff - 3 degrees of freedom
   (n_eff from the lag-1 autocorrelation of e, capped at n; 3 = intercept, trend, slope);
   black dots: Benjamini-Hochberg FDR q < alpha_fdr per map.
   Also: PV' regressed onto the standardised PC1 of that dataset's omega (PVU per s.d.), i.e.
   the PV pattern that goes with the leading omega mode.
   PV < 0 south of the equator: there, b < 0 means more subsidence (omega > 0) goes with PV of
   larger magnitude.

FIGURES (out_dir)
  eof_omega_oct_<lev>hPa_maps.png       rows: datasets; cols: trend map, EOF1..EOF<n_modes>
  eof_omega_oct_<lev>hPa_pcs.png        projections of the undetrended anomalies onto each EOF
  pv_omega_regression_oct_<lev>hPa.png  rows: ERA5, MERRA-2; cols: r, b, PV on PC1
CSV  eof_omega_oct_<lev>hPa_summary.csv
"""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod                           # lets the parallel omega readers find it
    spec.loader.exec_module(mod)
    return mod


om = load_module("omega_oct", "october_trends.py")
pvm = load_module("pv_era5_merra2", "era5_merra2_potential_vorticity.py")

# ---------------------------------------------------------------- settings
out_dir = Path("/soge-home/users/jesu4127/Paper_1/pv_omega_eof")
level = 300                                   # hPa
month = 10
n_modes = 3
alpha_fdr = 0.10
lat_s, lat_n, lon_w, lon_e = pvm.lat_min, pvm.lat_max, pvm.lon_min, pvm.lon_max   # PV domain
omega_names = ["ERA5", "MERRA-2", "JRA-3Q"]
pv_names = ["ERA5", "MERRA-2"]
colours = {"ERA5": "#FF7C7E", "MERRA-2": "#FEC300", "JRA-3Q": "#016ff2"}


# ---------------------------------------------------------------- loading
def configure_readers():
    """Point the omega and PV readers at this domain, level and month."""
    om.lat_s, om.lat_n, om.lon_w, om.lon_e = lat_s, lat_n, lon_w, lon_e
    om.grid_lat = np.arange(lat_s, lat_n + 1e-6, om.grid_step)
    om.grid_lon = np.arange(lon_w, lon_e + 1e-6, om.grid_step)
    om.half = om.grid_step / 2
    om.levels, om.months = [float(level)], [month]
    om.cache_dir = om.out_dir / "cache_pv_domain"      # the cache is per domain
    half = om.half                                     # read PV to the outer box edges
    pvm.lat_min, pvm.lat_max = lat_s - half, lat_n + half
    pvm.lon_min, pvm.lon_max = lon_w - half, lon_e + half
    pvm.levels, pvm.months = [level], np.array([month])


def load_omega():
    data = om.load_all()                               # common years, complete boxes only
    return {n: d.sel(lev=float(level), month=month, drop=True) for n, d in data.items()}


def load_pv(years):
    pv = pvm.load()
    out = {}
    for n in pv_names:
        da = pv[n].sel(level=level, month=month, drop=True)
        da = om.box_mean(da).sel(year=[y for y in years if y in da.year.values])
        out[n] = da
    return out


# ---------------------------------------------------------------- statistics
def gap_free(da):
    return da.reindex(year=np.arange(int(da.year.min()), int(da.year.max()) + 1))


def detrend(da):
    """OLS-detrended anomalies along 'year' (NaN years stay NaN)."""
    ok = da.notnull()
    x = da.year.astype(float).where(ok)
    xa, ya = x - x.mean("year"), da - da.mean("year")
    return ya - (xa * ya).sum("year") / (xa ** 2).sum("year") * xa


def trend_test(y, t):
    """OLS slope per decade and AR(1)-corrected p for one series; missing years are skipped,
    and lag 1 always means the previous calendar year."""
    full = np.full(int(t.max() - t.min()) + 1, np.nan)
    full[(t - t.min()).astype(int)] = y
    tt = np.arange(full.size) + t.min()
    ok = np.isfinite(full)
    ta, ya = tt[ok] - tt[ok].mean(), full[ok] - full[ok].mean()
    b = (ta * ya).sum() / (ta ** 2).sum()
    e = np.full(full.size, np.nan)
    e[ok] = ya - b * ta
    n = int(ok.sum())
    r1 = np.nansum(e[1:] * e[:-1]) / np.nansum(e ** 2)
    e = e[ok]
    n_eff = min(n * (1 - r1) / (1 + r1), n)
    se = np.sqrt((e ** 2).sum() / (n_eff - 2) / (ta ** 2).sum())
    return 10 * b, 2 * stats.t.sf(abs(b / se), n_eff - 2)


def bh(p):
    """Benjamini-Hochberg over one map: True where the adjusted p < alpha_fdr."""
    sig, ok = np.zeros(p.shape, bool), np.isfinite(p)
    if ok.any():
        sig[ok] = stats.false_discovery_control(p[ok]) < alpha_fdr
    return sig


def wcorr(a, b, w):
    """Area-weighted centred pattern correlation over boxes where both are finite."""
    ok = np.isfinite(a) & np.isfinite(b)
    a, b, w = a[ok], b[ok], w[ok]
    a, b = a - np.average(a, weights=w), b - np.average(b, weights=w)
    return float((w * a * b).sum() / np.sqrt((w * a * a).sum() * (w * b * b).sum()))


def eofs(da):
    """EOF analysis of one dataset's (year, lat, lon) October omega."""
    da = da.dropna("year", how="all")                          # years with data
    t = da.year.values.astype(float)
    st = da.stack(box=("lat", "lon"))
    keep = st.notnull().all("year").values                     # complete boxes only
    st = st.isel(box=keep)
    rows = np.ones(st.sizes["year"], bool)
    raw = st.values                                            # (year, box)
    anom = raw - raw.mean(0)
    det = detrend(st.isel(year=rows)).values                   # detrended anomalies
    coslat = np.cos(np.deg2rad(st.lat.values))
    sw = np.sqrt(coslat)
    u, s, vt = np.linalg.svd(det * sw, full_matrices=False)
    n = det.shape[0]
    lam = s ** 2 / (n - 1)
    frac = lam / lam.sum()
    trend = np.array([trend_test(anom[:, j], t[rows])[0] for j in range(anom.shape[1])])
    out = {"t": t[rows], "frac": frac[:n_modes], "lam": lam[:n_modes], "n": n,
           "trend": unstack(trend, st, da), "modes": []}
    for k in range(n_modes):
        pc = (det * sw) @ vt[k]                                # detrended PC
        sd = pc.std(ddof=1)
        eof = det.T @ (pc / sd) / (n - 1)                      # Pa s-1 per 1 s.d. of PC
        r = wcorr(eof, trend, coslat)
        sign = 1 if r >= 0 else -1                             # orient towards the trend
        proj = sign * ((anom * sw) @ vt[k]) / sd               # undetrended projection, s.d.
        slope, p = trend_test(proj, t[rows])
        out["modes"].append({"eof": unstack(sign * eof, st, da), "pc": sign * pc / sd,
                             "proj": proj, "r_trend": abs(r), "proj_trend": slope,
                             "proj_p": p})
    return out


def unstack(values, st, like):
    """(box,) values back onto the (lat, lon) grid of `like`."""
    field = xr.DataArray(values, coords={"box": st.box}, dims="box").unstack("box")
    return field.reindex(lat=like.lat, lon=like.lon)


def north_separated(lam, n):
    """True for modes whose sampling range (North et al. 1982) overlaps no neighbour."""
    err = lam * np.sqrt(2 / n)
    sep = []
    for k in range(len(lam)):
        gaps = [abs(lam[k] - lam[j]) > err[k] + err[j] for j in (k - 1, k + 1)
                if 0 <= j < len(lam)]
        sep.append(all(gaps))
    return sep


def regress_pv_on_omega(pv, w):
    """PV' = a + b omega' + e at every box (both detrended): b, r, p (AR(1)-corrected)."""
    pv, w = gap_free(pv), gap_free(w).reindex(year=gap_free(pv).year)
    ok = pv.notnull() & w.notnull()
    pd_, wd = detrend(pv.where(ok)), detrend(w.where(ok))
    n = ok.sum("year")
    sxx = (wd ** 2).sum("year")
    b = (wd * pd_).sum("year") / sxx
    r = (wd * pd_).sum("year") / np.sqrt(sxx * (pd_ ** 2).sum("year"))
    e = pd_ - b * wd
    sse = (e ** 2).sum("year")
    r1 = (e * e.shift(year=1)).sum("year") / sse
    n_eff = np.minimum(n * (1 - r1) / (1 + r1), n)
    se = np.sqrt((sse / (n_eff - 3) / sxx).where(n_eff > 3))
    p = xr.apply_ufunc(lambda tt, df: 2 * stats.t.sf(np.abs(tt), df), b / se, n_eff - 3)
    valid = (n >= 8) & (n_eff > 3)
    return b.where(valid), r.where(valid), p.where(valid)


def regress_on_pc(field, pc, years):
    """Detrended field regressed onto a standardised PC: units of field per 1 s.d."""
    f = detrend(gap_free(field)).sel(year=years)
    pcx = xr.DataArray(pc, coords={"year": years}, dims="year")
    return (f * pcx).sum("year") / (pcx ** 2).sum("year")


# ---------------------------------------------------------------- plotting
def geo(ax, labels=(True, True)):
    om.add_geo(ax, labels)
    ax.set_extent([lon_w, lon_e, lat_s, lat_n], crs=om.pc)


def sym_norm(fields, pct=98):
    vals = np.concatenate([np.abs(np.asarray(f)).ravel() for f in fields])
    lim = float(np.nanpercentile(vals, pct)) or 1e-9
    return mcolors.TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)


def mesh(ax, field, norm, cmap="RdBu_r"):
    return ax.pcolormesh(field.lon, field.lat, field, cmap=cmap, norm=norm, shading="nearest",
                         transform=om.pc, zorder=2, rasterized=True)


def stipple(ax, field, sig):
    lon2, lat2 = np.meshgrid(field.lon, field.lat)
    ax.scatter(lon2[sig], lat2[sig], s=3, color="black", linewidths=0, transform=om.pc,
               zorder=4)


def label_rows_cols(fig, axes, row_labels, col_labels, W, H):
    fig.canvas.draw()
    for j, lab in enumerate(col_labels):
        b = axes[0, j].get_position()
        fig.text((b.x0 + b.x1) / 2, b.y1 + 0.035, lab, ha="center", va="bottom",
                 fontsize=13, fontweight="bold")
    for i, lab in enumerate(row_labels):
        b = axes[i, 0].get_position()
        fig.text(b.x0 - 0.045, (b.y0 + b.y1) / 2, lab, rotation=90, ha="center",
                 va="center", fontsize=15, fontweight="bold")


def eof_maps_figure(res, path):
    names = list(res)
    nr, nc = len(names), n_modes + 1
    fig, axes = plt.subplots(nr, nc, figsize=(5.2 * nc, 3.9 * nr + 1.2),
                             subplot_kw={"projection": om.pc}, squeeze=False)
    fig.subplots_adjust(left=0.06, right=0.9, top=0.86, bottom=0.05, wspace=0.12, hspace=0.32)
    tnorm = sym_norm([res[n]["trend"] for n in names])
    enorm = sym_norm([m["eof"] for n in names for m in res[n]["modes"]])
    for i, n in enumerate(names):
        r = res[n]
        sep = north_separated(r["lam"], r["n"])
        geo(axes[i, 0], (True, i == nr - 1))
        tm = mesh(axes[i, 0], r["trend"], tnorm)
        for k, m in enumerate(r["modes"]):
            ax = axes[i, k + 1]
            geo(ax, (False, i == nr - 1))
            em = mesh(ax, m["eof"], enorm)
            ax.set_title(f"{100 * r['frac'][k]:.0f}% var{'' if sep[k] else ' (not separated)'}; "
                         f"r$_{{trend}}$ = {m['r_trend']:.2f}", fontsize=10)
    label_rows_cols(fig, axes, names, ["Trend (undetrended)"] +
                    [f"EOF{k + 1} (detrended)" for k in range(n_modes)], *fig.get_size_inches())
    for m_, nrm, lab, y0 in ((tm, tnorm, "Trend (Pa s$^{-1}$ decade$^{-1}$)", 0.55),
                             (em, enorm, "EOF (Pa s$^{-1}$ per s.d. of PC)", 0.08)):
        cax = fig.add_axes([0.915, y0, 0.012, 0.32])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=nrm, cmap="RdBu_r"), cax=cax)
        cb.set_label(lab, fontsize=11)
    fig.suptitle(f"October {level} hPa ω: trend and leading EOFs of detrended interannual "
                 f"variability (area-weighted; EOF sign chosen so r$_{{trend}}$ ≥ 0; "
                 "ω > 0 subsidence)", fontsize=15, fontweight="bold", y=0.975)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"  Saved {path}", flush=True)


def pc_figure(res, path):
    fig, axes = plt.subplots(n_modes, 1, figsize=(13, 2.9 * n_modes), sharex=True, squeeze=False)
    for k, ax in enumerate(axes[:, 0]):
        for n, r in res.items():
            m = r["modes"][k]
            t = r["t"]
            ax.plot(t, m["proj"], color=colours[n], linewidth=1.8,
                    label=f"{n}: trend {m['proj_trend']:+.2f} s.d./decade, "
                          f"{'p < 0.001' if m['proj_p'] < 0.001 else f'p = {m['proj_p']:.3f}'}")
            fit = np.polyval(np.polyfit(t, m["proj"], 1), t)
            ax.plot(t, fit, color=colours[n], linewidth=1, linestyle="--")
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_ylabel(f"Projection on EOF{k + 1}\n(s.d. of PC{k + 1})", fontsize=10)
        ax.legend(fontsize=9, frameon=False, loc="upper left", ncol=3)
        ax.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("Year")
    fig.suptitle(f"October {level} hPa ω: undetrended anomalies projected onto each EOF "
                 "(dashed: OLS fit; p AR(1)-corrected)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"  Saved {path}", flush=True)


def regression_figure(reg, path):
    names = list(reg)
    fig, axes = plt.subplots(len(names), 3, figsize=(16.5, 4.1 * len(names) + 1.2),
                             subplot_kw={"projection": om.pc}, squeeze=False)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.84, bottom=0.12, wspace=0.12, hspace=0.3)
    norms = [mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1),
             sym_norm([reg[n]["b"] for n in names]), sym_norm([reg[n]["pc1"] for n in names])]
    for i, n in enumerate(names):
        g = reg[n]
        sig = bh(g["p"].values)
        for j, key in enumerate(("r", "b", "pc1")):
            ax = axes[i, j]
            geo(ax, (j == 0, i == len(names) - 1))
            mesh(ax, g[key], norms[j], "PuOr_r")
            if key != "pc1":
                stipple(ax, g[key], sig)
    label_rows_cols(fig, axes, names, ["r(PV′, ω′)", "b: PV′ on ω′",
                                       "PV′ on ω PC1"], *fig.get_size_inches())
    for j, (nrm, lab) in enumerate(zip(norms, ("correlation", "PVU per 0.01 Pa s$^{-1}$",
                                                "PVU per s.d. of ω PC1"))):
        b = axes[-1, j].get_position()
        cax = fig.add_axes([b.x0 + 0.1 * b.width, 0.06, 0.8 * b.width, 0.015])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=nrm, cmap="PuOr_r"), cax=cax,
                          orientation="horizontal")
        cb.set_label(lab, fontsize=11)
    fig.suptitle(f"October {level} hPa: grid-point regression of PV′ on ω′ (both detrended; "
                 f"dots: BH-FDR q < {alpha_fdr}, AR(1)-corrected)\nPV < 0 south of the "
                 "equator: there b < 0 means more subsidence goes with stronger (more negative) PV",
                 fontsize=14, fontweight="bold", y=0.975)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"  Saved {path}", flush=True)


# ---------------------------------------------------------------- main
def main():
    out_dir.mkdir(parents=True, exist_ok=True)
    configure_readers()
    omega = load_omega()
    years = sorted(set.intersection(*(set(d.year.values.tolist()) for d in omega.values())))
    print("\nEOFs of detrended October omega ...", flush=True)
    res = {n: eofs(omega[n]) for n in omega_names}
    rows = []
    for n, r in res.items():
        sep = north_separated(r["lam"], r["n"])
        for k, m in enumerate(r["modes"]):
            rows.append({"dataset": n, "mode": k + 1, "n_years": r["n"],
                         "variance_pct": 100 * r["frac"][k], "north_separated": sep[k],
                         "pattern_corr_with_trend": m["r_trend"],
                         "projection_trend_sd_per_decade": m["proj_trend"],
                         "projection_trend_p": m["proj_p"]})
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / f"eof_omega_oct_{level}hPa_summary.csv", index=False)
    print(table.round(3).to_string(index=False))
    eof_maps_figure(res, out_dir / f"eof_omega_oct_{level}hPa_maps.png")
    pc_figure(res, out_dir / f"eof_omega_oct_{level}hPa_pcs.png")

    print("\nPV on omega regressions ...", flush=True)
    pv = load_pv(years)
    reg = {}
    for n in pv_names:
        b, r, p = regress_pv_on_omega(pv[n], omega[n])
        pc1 = regress_on_pc(pv[n], res[n]["modes"][0]["pc"], res[n]["t"].astype(int))
        reg[n] = {"b": 0.01 * b, "r": r, "p": p, "pc1": pc1}
        sig = bh(p.values)
        print(f"  {n}: median r = {float(r.median()):+.2f}; {100 * sig.mean():.0f}% of boxes "
              f"FDR-significant", flush=True)
    regression_figure(reg, out_dir / f"pv_omega_regression_oct_{level}hPa.png")
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()