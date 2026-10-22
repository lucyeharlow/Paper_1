"""
CERES EBAF Ed4.2.1: global maps of the October trends in
  (a) cloud area fraction (day + night)        cldarea_total_daynight_mon   % per decade
  (b) surface downward shortwave, all-sky      sfc_sw_down_all_mon          W m-2 per decade
  (c) surface downward longwave, all-sky       sfc_lw_down_all_mon          W m-2 per decade
over all Octobers in the files (2000-2025).

DATA  data/ceres/CERES_EBAF_Ed4.2.1_Subset_*.nc  (200003-202407 and 202408-202605; 1 x 1 deg)

METHOD (as in the SW, temperature and vertical velocity scripts)
  OLS trend per grid point; lag-1 autocorrelation r1 of the residuals gives
  n_eff = n (1 - r1) / (1 + r1), capped at n; two-sided t-test with n_eff - 2 degrees of
  freedom; Benjamini-Hochberg FDR q < FDR_ALPHA over each map. A point needs MIN_YEARS years.

FIGURE  <OUT_DIR>/CERES_October_trends_<y0>-<y1>.png
  All trends plotted (land and ocean); markers (white centre, black edge) show FDR-significant
  points, thinned to one every DOT_EVERY grid points. Each map has its own colour bar,
  symmetric about zero at the CLIM_PCTL percentile of |trend|.
"""
from __future__ import annotations
import calendar
import glob
import os
import numpy as np
import xarray as xr
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter

CERES_GLOB = "/soge-home/users/jesu4127/data/ceres/CERES_EBAF_Ed4.2.1_Subset_*.nc"
OUT_DIR = "/soge-home/users/jesu4127/Paper_1/ceres/plots"

MONTH = 10
FDR_ALPHA = 0.10
MIN_YEARS = 8
DECADE = 10
CLIM_PCTL = 98
DOT_EVERY = 3                      # one marker per 3 x 3 deg
DOT_SIZE = 5
VARS = [  # (variable, title, unit label, scale to plotted unit)
    ("cldarea_total_daynight_mon", "Cloud area fraction (day + night)", "% decade$^{-1}$", 1.0),
    ("sfc_sw_down_all_mon", "Surface downward shortwave (all-sky)", r"W m$^{-2}$ decade$^{-1}$", 1.0),
    ("sfc_lw_down_all_mon", "Surface downward longwave (all-sky)", r"W m$^{-2}$ decade$^{-1}$", 1.0),
]


def log(msg):
    print(msg, flush=True)


def load_ceres(variables, month=MONTH):
    """{var: (year, lat, lon)} for one calendar month from all CERES files, lon -180..180."""
    files = sorted(glob.glob(CERES_GLOB))
    if not files:
        raise SystemExit(f"no files match {CERES_GLOB}")
    parts = []
    for f in files:
        with xr.open_dataset(f) as ds:
            sub = ds[variables]
            parts.append(sub.isel(time=np.flatnonzero(sub.time.dt.month.values == month)).load())
    ds = xr.concat(parts, "time").sortby("time")
    ds = ds.isel(time=np.unique(ds.time.dt.year.values, return_index=True)[1])
    ds = ds.assign_coords(time=ds.time.dt.year.values).rename(time="year")
    if float(ds.lon.max()) > 180:
        ds = ds.assign_coords(lon=(ds.lon + 180) % 360 - 180).sortby("lon")
    ds = ds.sortby("lat")
    log(f"CERES {calendar.month_name[month]}: {ds.year.size} years """"
October downwelling SW trends for ERA5, MERRA-2, CLARA-A3, SARAH-3 and CERES-EBAF over the
years ALL FIVE have (CERES starts in March 2000, so at most 2000-2024).

Reuses the loaders, statistics and map drawing of sw_trends_monthly.py (keep this file in the
same folder): OLS trend per grid point on the MERRA-2 0.5 x 0.625 deg grid, AR(1)-corrected
t-test, Benjamini-Hochberg FDR per map, markers on FDR-significant land points, oceans white.
The common years are the years present in every dataset (a year missing from any one dataset
is dropped from all), so every panel covers exactly the same Octobers.

    python sw_down_october_common_period.py

FIGURE  <OUT_DIR>/sw_down/SW_down_trends_10_Oct_common_<y0>-<y1>.png
"""
from __future__ import annotations
import calendar
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sw_trends_monthly as sw

MONTH = 10
FIRST, LAST = 2000, 2024           # search window; the common years are found inside it
CLIM = 10.0                        # colour scale limit, W m-2 per decade (shorter period)


def main():
    sw.START, sw.END = FIRST, LAST                # the loaders read only these years
    data = {}
    for name in sw.NAMES:
        sw.log(f"--- {name} ---")
        da = sw.LOADERS[name](MONTH, "down")
        if da is None:
            sw.log(f"  [WARN] {name}: no data, left out")
            continue
        # a year counts only if the dataset has some valid data in it
        ok = da.notnull().any(("lat", "lon")).values
        data[name] = da.isel(year=np.flatnonzero(ok))
    if not data:
        raise SystemExit("no data loaded")
    common = sorted(set.intersection(*(set(d.year.values.tolist()) for d in data.values())))
    if len(common) < sw.MIN_YEARS:
        raise SystemExit(f"only {len(common)} common years: {common}")
    y0, y1 = common[0], common[-1]
    gaps = sorted(set(range(y0, y1 + 1)) - set(common))
    sw.log(f"\nCommon years: {y0}-{y1} (n = {len(common)})"
           + (f"; missing from at least one dataset: {gaps}" if gaps else ""))

    results = {}
    for name, da in data.items():
        st = sw.ols_stats(da.sel(year=common))
        sig = sw.fdr_mask(st.p.values)
        sw.log(f"  {name}: {100 * sig.sum() / max(np.isfinite(st.p.values).sum(), 1):.0f}% "
               "of points FDR-significant")
        results[name] = st

    # layout as in sw_trends_monthly.py: reanalyses on top (centred), satellites below
    matplotlib.rcParams.update({"font.family": "sans-serif",
                                "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"]})
    levels = np.linspace(-CLIM, CLIM, 21)
    ncol = max(len(r) for r in sw.ROWS)
    W, left, right, gap_x = 20.0, 1.25, 0.35, 0.45
    mw = (W - left - right - (ncol - 1) * gap_x) / ncol
    mh = mw / 2
    tops = [1.55, 1.55 + mh + 1.35]
    H = tops[1] + mh + 1.75
    fig = plt.figure(figsize=(W, H))
    k = 0
    for r, row in enumerate(sw.ROWS):
        shift = (ncol - len(row)) * (mw + gap_x) / 2
        for c, name in enumerate(row):
            x0 = left + shift + c * (mw + gap_x)
            ax = fig.add_axes([x0 / W, 1 - (tops[r] + mh) / H, mw / W, mh / H],
                              projection=ccrs.PlateCarree())
            fig.text((x0 + mw / 2) / W, 1 - (tops[r] - 0.12) / H, name, ha="center",
                     va="bottom", fontsize=15, fontweight="bold")
            if name not in results:
                sw.add_map_features(ax)
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center",
                        va="center", fontsize=12, zorder=6)
            else:
                sw.plot_panel(ax, results[name], c, r, len(sw.ROWS), f"{'abcde'[k]})", levels)
            k += 1
    for r, lbl in enumerate(["Reanalyses", "Satellite products"]):
        fig.text(0.25 / W, 1 - (tops[r] + mh / 2) / H, lbl, fontsize=13, fontweight="bold",
                 rotation=90, va="center", ha="center")
    sm = plt.cm.ScalarMappable(norm=matplotlib.colors.BoundaryNorm(levels, 256, extend="both"),
                               cmap="RdBu_r")
    cb = fig.colorbar(sm, cax=fig.add_axes([0.2, 0.85 / H, 0.6, 0.2 / H]),
                      orientation="horizontal", extend="both", ticks=levels[::4])
    cb.set_label(r"W m$^{-2}$ decade$^{-1}$", fontsize=11)
    cb.ax.tick_params(labelsize=10)
    fig.suptitle(f"{calendar.month_name[MONTH]} decadal OLS trends in downwelling SW radiation, "
                 f"common period {y0}–{y1} (n = {len(common)})\nMarkers: significant after "
                 f"FDR (α = {sw.FDR_ALPHA:.2f}, AR(1)-corrected)",
                 fontsize=15, fontweight="bold", y=1 - 0.15 / H, va="top")
    folder = os.path.join(sw.OUT_DIR, "sw_down")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"SW_down_trends_{MONTH:02d}_{calendar.month_abbr[MONTH]}"
                                f"_common_{y0}-{y1}.png")
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    sw.log(f"Saved: {path}")


if __name__ == "__main__":
    main()
        f"({int(ds.year.min())}-{int(ds.year.max())}) from {len(files)} file(s)")
    return ds


def ols_stats(da):
    """Per-point OLS trend (per decade) and AR(1)-corrected p along 'year'."""
    da = da.reindex(year=np.arange(int(da.year.min()), int(da.year.max()) + 1))
    ok = da.notnull()
    n = ok.sum("year")
    x = da.year.astype(float).where(ok)
    xa, ya = x - x.mean("year"), da - da.mean("year")
    sxx = (xa ** 2).sum("year")
    slope = (xa * ya).sum("year") / sxx
    resid = ya - slope * xa
    ssr = (resid ** 2).sum("year")
    r1 = (resid * resid.shift(year=1)).sum("year") / ssr
    n_eff = np.minimum(n * (1 - r1) / (1 + r1), n)
    se = np.sqrt((ssr / (n_eff - 2) / sxx).where(n_eff > 2))
    p = xr.apply_ufunc(lambda t, df: 2 * stats.t.sf(np.abs(t), df), slope / se, n_eff - 2)
    out = xr.Dataset({"trend": DECADE * slope, "p": p, "n": n})
    return out.where((n >= MIN_YEARS) & (n_eff > 2))


def fdr_mask(p):
    v, sig = np.asarray(p), np.zeros(np.shape(p), bool)
    ok = np.isfinite(v)
    if ok.any():
        sig[ok] = stats.false_discovery_control(v[ok]) < FDR_ALPHA
    return sig


def nice_limit(values):
    v = np.nanpercentile(np.abs(values), CLIM_PCTL)
    if not np.isfinite(v) or v == 0:
        return 1.0
    mag = 10 ** np.floor(np.log10(v))
    return float(min((s * mag for s in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10) if s * mag >= v)))


def draw(ax, st, title, unit, letter, cax):
    """Map, markers and colour bar (the panel title is placed by main() as figure text:
    titles set on the map axes go missing with some cartopy versions)."""
    lim = nice_limit(st.trend.values)
    norm = mcolors.TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
    ax.pcolormesh(st.lon, st.lat, st.trend.values, cmap="RdBu_r", norm=norm, shading="nearest",
                  transform=ccrs.PlateCarree(), rasterized=True, zorder=1)
    sig = fdr_mask(st.p.values)
    keep = np.zeros_like(sig)
    keep[::DOT_EVERY, ::DOT_EVERY] = True
    lon2, lat2 = np.meshgrid(st.lon, st.lat)
    ax.scatter(lon2[sig & keep], lat2[sig & keep], s=DOT_SIZE, facecolor="white",
               edgecolor="black", linewidths=0.4, transform=ccrs.PlateCarree(), zorder=3)
    ax.coastlines(resolution="110m", linewidth=0.5, zorder=4)
    ax.set_global()
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="black", alpha=0.4, linestyle=":")
    gl.top_labels = gl.right_labels = False
    gl.xlocator = mticker.FixedLocator([-120, -60, 0, 60, 120])
    gl.ylocator = mticker.FixedLocator([-60, -30, 0, 30, 60])
    gl.xformatter, gl.yformatter = LongitudeFormatter(), LatitudeFormatter()
    gl.xlabel_style = gl.ylabel_style = {"size": 10}
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="RdBu_r"), cax=cax, extend="both",
                      ticks=np.linspace(-lim, lim, 5))
    cb.set_label(unit, fontsize=12)
    cb.ax.tick_params(labelsize=10)
    pct = 100 * sig.sum() / max(np.isfinite(st.p.values).sum(), 1)
    log(f"  {title}: |trend| {CLIM_PCTL}th pct {lim:g}; {pct:.0f}% of points FDR-significant")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ds = load_ceres([v for v, *_ in VARS])
    y0, y1 = int(ds.year.min()), int(ds.year.max())
    # explicit inch layout (no bbox_inches="tight", which crops cartopy maps)
    W, left, mw, gap_cb, cbw = 13.0, 0.8, 10.4, 0.25, 0.22
    mh = mw / 2
    tops = [1.2 + k * (mh + 0.85) for k in range(len(VARS))]
    H = tops[-1] + mh + 0.75
    fig = plt.figure(figsize=(W, H))
    for k, (var, title, unit, scale) in enumerate(VARS):
        ax = fig.add_axes([left / W, 1 - (tops[k] + mh) / H, mw / W, mh / H],
                          projection=ccrs.PlateCarree())
        cax = fig.add_axes([(left + mw + gap_cb) / W, 1 - (tops[k] + mh - 0.1) / H,
                            cbw / W, (mh - 0.2) / H])
        draw(ax, ols_stats(ds[var] * scale), title, unit, "abc"[k], cax)
        fig.text(left / W, 1 - (tops[k] - 0.12) / H, f"{'abc'[k]})  {title}", ha="left",
                 va="bottom", fontsize=15, fontweight="bold")
    fig.suptitle(f"CERES EBAF Ed4.2.1: {calendar.month_name[MONTH]} trends ({y0}–{y1})\n"
                 f"Markers: significant after FDR (α = {FDR_ALPHA:.2f}, AR(1)-corrected)",
                 fontsize=16, fontweight="bold", y=1 - 0.15 / H, va="top")
    path = os.path.join(OUT_DIR, f"CERES_{calendar.month_name[MONTH]}_trends_{y0}-{y1}.png")
    fig.savefig(path, dpi=250, facecolor="white")
    plt.close(fig)
    log(f"Saved: {path}")


if __name__ == "__main__":
    main()