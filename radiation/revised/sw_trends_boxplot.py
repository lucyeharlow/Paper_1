"""
Downwelling shortwave (SW down) trends, in the style of the Southern Africa temperature figure.

PANELS
  (a) Monthly boxplots of grid-cell trends over Southern Africa (10-40E, 35-10S), land only,
      area-weighted (cos latitude), one box per dataset per month: the box is the
      area-weighted interquartile range (25th-75th percentile), the line the area-weighted
      median. Boxes sized and spaced as in the temperature figure.
  (b)-(e) October trend maps over Africa south of the equator (8-52°E), one per dataset; the
      Southern Africa box is outlined. Dots: FDR-significant cells (white, thinned). Oceans white.

DATASETS (loaders from sw_trends_october.py, which must sit in the same folder)
  ERA5, MERRA-2, CLARA-A3, SARAH-3: their shared years within 1983-2024.
  Colours: ERA5 and MERRA-2 as in the temperature figure; new colours for the others.

METHOD (as in the SW and temperature scripts)
  All data on the MERRA-2 0.5 x 0.625 deg grid. OLS trend per cell (W m-2 per decade), lag-1
  autocorrelation of residuals -> n_eff = n (1 - r1) / (1 + r1), capped at n; t-test with
  n_eff - 2 df; Benjamini-Hochberg FDR q < 0.10 per map. Land: cells at least half land
  (Natural Earth 50 m land, sub-grid sampled, as in the temperature script).
  Box: area-weighted quartiles; whiskers at the most extreme cell within 1.5 IQR.

OUTPUTS  <OUT_DIR>/sw_down/
  SW_down_SouthernAfrica_boxplots_Africa_Oct_maps.png
  SW_down_SouthernAfrica_monthly_trends_summary.csv   (area-weighted stats per dataset, month)
"""
from __future__ import annotations
import calendar
import os
import sys
from functools import cache
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Patch, Rectangle
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader
from shapely import box, contains_xy
from shapely.ops import unary_union

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sw_trends_october as sw

# ---------------------------------------------------------------- settings
OUT_DIR = os.path.join(sw.OUT_DIR, "sw_down")
START, END = 1983, 2024
TARGET_MONTH = 10
SA = dict(lon_min=10.0, lon_max=40.0, lat_min=-35.0, lat_max=-10.0)   # boxplot region
AFRICA = dict(lon_min=8.0, lon_max=52.0, lat_min=-37.0, lat_max=0.0)    # map extent: south of the equator
LAND_MIN_FRACTION = 0.5
LAND_SHAPEFILE = None            # None: Natural Earth 50 m land via cartopy; or a local .shp
CLIM = 8.0                       # map colour limit, W m-2 per decade
LEVELS = np.linspace(-CLIM, CLIM, 17)
DOT_EVERY = 4                    # one dot per 4 x 4 grid cells (2 x 2.5 deg)
DOT_SIZE = 2.5
DOT_COLOUR = "white"

ORDER = ["ERA5", "MERRA-2", "CLARA-A3", "SARAH-3"]
OWN_PERIOD = set()               # datasets on their own years (marked *); none at present
COLOURS = {"ERA5": "#FF7C7E", "MERRA-2": "#FEC300",          # as in the temperature figure
           "CLARA-A3": "#4C9A2A", "SARAH-3": "#8C564B"}
MONTHS = np.arange(1, 13)
PC = ccrs.PlateCarree()


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------- loading
def cut(da, region):
    return da.sel(lat=slice(region["lat_min"], region["lat_max"]),
                  lon=slice(region["lon_min"], region["lon_max"]))


def load(name, month):
    """(year, lat, lon) SW down over Africa on the MERRA-2 grid, or None."""
    sw.START, sw.END = START, END
    da = sw.LOADERS[name](month, "down")
    if da is None:
        return None
    da = cut(da, AFRICA)
    ok = da.notnull().any(("lat", "lon")).values            # years with any data
    return da.isel(year=np.flatnonzero(ok))


# ---------------------------------------------------------------- land mask (temperature script)
@cache
def land_geometry():
    path = LAND_SHAPEFILE or shapereader.natural_earth("50m", "physical", "land")
    land = unary_union(list(shapereader.Reader(str(path)).geometries()))
    return land.intersection(box(AFRICA["lon_min"] - 5, AFRICA["lat_min"] - 5,
                                 AFRICA["lon_max"] + 5, AFRICA["lat_max"] + 5))


def land_mask(lat, lon):
    """True where at least LAND_MIN_FRACTION of the cell is land (sub-grid sampled)."""
    dlat, dlon = float(np.median(np.diff(lat))), float(np.median(np.diff(lon)))
    k = int(np.clip(np.ceil(max(dlat, dlon) / 0.1), 1, 10))
    off = (np.arange(k) + 0.5) / k - 0.5
    la = lat[:, None, None, None] + off[None, None, :, None] * dlat
    lo = lon[None, :, None, None] + off[None, None, None, :] * dlon
    la, lo = np.broadcast_arrays(la, lo)
    frac = contains_xy(land_geometry(), lo, la).mean(axis=(2, 3))
    return xr.DataArray(frac >= LAND_MIN_FRACTION, dims=("lat", "lon"),
                        coords={"lat": lat, "lon": lon})


# ---------------------------------------------------------------- area-weighted summaries
def weighted_values(da):
    w = np.cos(np.deg2rad(da.lat)).broadcast_like(da)
    v, w = da.values.ravel(), w.values.ravel()
    ok = np.isfinite(v)
    return v[ok], w[ok]


def wquantile(v, w, q):
    i = np.argsort(v)
    v, w = v[i], w[i]
    return np.interp(q, (np.cumsum(w) - 0.5 * w) / w.sum(), v)


def box_stats(da):
    v, w = weighted_values(da)
    if not v.size:
        return None
    q1, med, q3 = wquantile(v, w, [0.25, 0.5, 0.75])
    lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    return {"q1": q1, "med": med, "q3": q3, "fliers": [],
            "whislo": v[v >= lo].min(), "whishi": v[v <= hi].max()}


def pct_area_sig(st, sig, land):
    w = np.cos(np.deg2rad(st.lat)).broadcast_like(st.trend).where(st.p.notnull() & land)
    return 100 * float((w * sig).sum() / w.sum()) if float(w.sum()) > 0 else np.nan


# ---------------------------------------------------------------- plotting
def span(y):
    return f"{y[0]}–{y[-1]}"


def header(ax, text, pts, **kw):
    ax.annotate(text, (0.5, 1), xycoords="axes fraction", xytext=(0, pts),
                textcoords="offset points", ha="center", va="bottom", **kw)


def plot_map(ax, st, name, years, panel, norm, land):
    ax.pcolormesh(st.lon, st.lat, st.trend, transform=PC, cmap="RdBu_r", norm=norm,
                  shading="nearest", rasterized=True, zorder=1)
    sig = sw.fdr_mask(st.p.values)
    keep = np.zeros_like(sig)
    keep[::DOT_EVERY, ::DOT_EVERY] = True
    pick = sig & keep
    lon2, lat2 = np.meshgrid(st.lon, st.lat)
    ax.scatter(lon2[pick], lat2[pick], s=DOT_SIZE, color=DOT_COLOUR, linewidths=0,
               transform=PC, zorder=1.5)
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="white", zorder=2)
    ax.add_feature(cfeature.LAKES.with_scale("50m"), facecolor="white", zorder=2)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.7, edgecolor="black",
                   zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.4, edgecolor="black",
                   zorder=3)
    ax.add_patch(Rectangle((SA["lon_min"], SA["lat_min"]), SA["lon_max"] - SA["lon_min"],
                           SA["lat_max"] - SA["lat_min"], fill=False, edgecolor="black",
                           linewidth=1.1, linestyle="--", transform=PC, zorder=4))
    ax.set_extent([AFRICA["lon_min"], AFRICA["lon_max"], AFRICA["lat_min"], AFRICA["lat_max"]],
                  crs=PC)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4, linestyle="--")
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"fontsize": 9}
    ax.text(0.03, 0.97, panel, transform=ax.transAxes, fontsize=13, fontweight="bold",
            va="top", zorder=5,
            bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none", alpha=0.7))
    odd = name in OWN_PERIOD
    header(ax, f"({span(years)}){' *' if odd else ''}", 3, fontsize=12,
           fontweight="bold" if odd else "normal")
    header(ax, name, 18, fontsize=14, fontweight="bold")


def make_figure(results, periods, path):
    W = 14.0                     # narrower figure: maps fit the smaller region, boxes sit closer
    left, right, gap = 0.85, 0.25, 0.55
    mw = (W - left - right - (len(ORDER) - 1) * gap) / len(ORDER)
    mh = mw * (AFRICA["lat_max"] - AFRICA["lat_min"]) / (AFRICA["lon_max"] - AFRICA["lon_min"])
    box_top, box_h = 1.35, 2.6
    map_top = box_top + box_h + 2.0
    H = map_top + mh + 1.35
    plt.rcParams.update({"font.size": 12, "xtick.labelsize": 11, "ytick.labelsize": 11})
    fig = plt.figure(figsize=(W, H))
    at = lambda x, top, w, h, **kw: fig.add_axes([x / W, 1 - (top + h) / H, w / W, h / H], **kw)

    # (a) boxplots
    ax = at(left, box_top, W - left - right, box_h)
    line = dict(color="black", linewidth=0.9)
    names = [n for n in ORDER if n in results]
    width = 0.06                                   # as in the temperature figure: touching boxes
    offs = (np.arange(len(names)) - (len(names) - 1) / 2) * width
    for name, off in zip(names, offs):
        boxes = {m: box_stats(results[name][m]["trend_sa"]) for m in MONTHS if m in results[name]}
        have = [m for m in MONTHS if boxes.get(m)]
        if have:
            ax.bxp([boxes[m] for m in have], positions=np.array(have) + off, widths=width,
                   patch_artist=True, showfliers=False, manage_ticks=False, whiskerprops=line,
                   capprops=line, medianprops=line, boxprops=dict(facecolor=COLOURS[name],
                   edgecolor="black", linewidth=0.9, alpha=0.9))
    ax.axhline(0, linestyle="--", linewidth=0.8, color="black")
    ax.set(xlim=(0.55, 12.45), xticks=MONTHS, xticklabels=calendar.month_abbr[1:])
    ax.set_xlabel("Month", labelpad=6)
    ax.set_ylabel("SW down trend, area-weighted\n" r"(W m$^{-2}$ decade$^{-1}$)")
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.text(0.995, 0.96, "(a)", transform=ax.transAxes, fontsize=14, fontweight="bold",
            va="top", ha="right")
    ax.legend(handles=[Patch(facecolor=COLOURS[n], edgecolor="black",
                             label=f"{n} ({span(periods[n])}{'*' if n in OWN_PERIOD else ''})")
                       for n in names],
              loc="lower center", bbox_to_anchor=(0.5, 1.01), frameon=False, fontsize=11,
              ncol=len(names))
    ax.set_title("Southern Africa (10–40°E, 35–10°S), land only", fontsize=13, pad=30)

    # (b)-(e) October maps over sub-Saharan Africa
    norm = BoundaryNorm(LEVELS, ncolors=plt.get_cmap("RdBu_r").N, extend="both")
    fig.text(0.5, 1 - (map_top - 0.8) / H, f"{calendar.month_name[TARGET_MONTH]} trends",
             ha="center", va="bottom", fontsize=16, fontweight="bold")
    for i, (name, panel) in enumerate(zip(ORDER, "bcdefgh")):
        axm = at(left + i * (mw + gap), map_top, mw, mh, projection=PC)
        r = results.get(name, {}).get(TARGET_MONTH)
        if r is None:
            axm.text(0.5, 0.5, "no data", transform=axm.transAxes, ha="center")
            header(axm, name, 18, fontsize=14, fontweight="bold")
            continue
        plot_map(axm, r["st"], name, r["years"], f"({panel})", norm, r["land"])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="RdBu_r"),
                      cax=at(W / 2 - 4.5, map_top + mh + 0.55, 9.0, 0.16),
                      orientation="horizontal", extend="both", ticks=LEVELS[::2])
    cb.set_label(f"{calendar.month_name[TARGET_MONTH]} SW down trend "
                 r"(W m$^{-2}$ decade$^{-1}$); dots: significant after FDR (α = 0.10)",
                 fontsize=12)
    fig.suptitle("Downwelling shortwave radiation trends", fontsize=19, fontweight="bold",
                 y=1 - 0.12 / H, va="top")
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    log(f"Saved: {path}")


# ---------------------------------------------------------------- main
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    land = None
    raw = {n: {} for n in ORDER}
    for m in MONTHS:
        log(f"\n===== {calendar.month_name[m]} =====")
        for name in ORDER:
            log(f"--- {name} ---")
            da = load(name, m)
            if da is not None and da.year.size >= sw.MIN_YEARS:
                raw[name][m] = da
            else:
                log(f"  [WARN] {name} {calendar.month_abbr[m]}: not enough data, left out")
    # shared years per month for the datasets that are not on their own period
    results, periods, rows = {n: {} for n in ORDER}, {}, []
    for m in MONTHS:
        main_sets = [raw[n][m] for n in ORDER if n not in OWN_PERIOD and m in raw[n]]
        shared = sorted(set.intersection(*(set(d.year.values.tolist()) for d in main_sets))) \
            if main_sets else []
        for name in ORDER:
            if m not in raw[name]:
                continue
            da = raw[name][m]
            yrs = sorted(da.year.values.tolist()) if name in OWN_PERIOD else shared
            if len(yrs) < sw.MIN_YEARS:
                continue
            if land is None:
                land = land_mask(da.lat.values, da.lon.values)
            st = sw.ols_stats(da.sel(year=yrs))
            trend_sa = cut(st.trend.where(land), SA)
            results[name][m] = {"st": st, "years": yrs, "land": land, "trend_sa": trend_sa}
            if m == TARGET_MONTH:
                periods[name] = yrs
            v, w = weighted_values(trend_sa)
            q = wquantile(v, w, [0.25, 0.5, 0.75]) if v.size else [np.nan] * 3
            sig = sw.fdr_mask(st.p.values)
            rows.append({"dataset": name, "month": calendar.month_abbr[m], "month_number": m,
                         "years": span(yrs), "n_cells": v.size,
                         "mean": np.average(v, weights=w) if v.size else np.nan,
                         "q25": q[0], "median": q[1], "q75": q[2],
                         "pct_area_sig_map_land": pct_area_sig(st, sig, land)})
    for n in ORDER:
        if n not in periods and results[n]:
            periods[n] = next(iter(results[n].values()))["years"]
    log("\nOctober periods: " + "; ".join(f"{n} {span(p)}" for n, p in periods.items()))
    csv = os.path.join(OUT_DIR, "SW_down_SouthernAfrica_monthly_trends_summary.csv")
    pd.DataFrame(rows).to_csv(csv, index=False)
    log(f"Saved: {csv}")
    make_figure(results, periods,
                os.path.join(OUT_DIR, "SW_down_SouthernAfrica_boxplots_Africa_Oct_maps.png"))


if __name__ == "__main__":
    main()