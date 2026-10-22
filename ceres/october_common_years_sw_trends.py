"""
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