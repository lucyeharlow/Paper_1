"""
October: detrended correlation between vertical velocity (omega) at 300 hPa and cloud fraction
(CF) at every pressure level (1000-100 hPa), in 5-degree longitude bands from 10 to 40 E
(latitude-pressure cross sections), and the long-term changes where they co-vary.
One figure per dataset: ERA5 and MERRA-2.

USAGE   python omega_cf_correlation_300hPa.py            (both datasets)
        python omega_cf_correlation_300hPa.py era5       (or: merra2)

DATA
  ERA5 (one file per year, scratch/era5/global/...)
    omega  vertical_velocity/global_vertical_velocity_YYYY.nc                 w   (Pa s-1)
    CF     fraction_of_cloud_cover/global_fraction_of_cloud_cover_YYYY.nc     cc  (0-1)
  MERRA-2 (one file per month, scratch/merra_2; only the October files are read)
    omega  MERRA2_*.instM_3d_asm_Np.YYYY10.nc4                                OMEGA (Pa s-1)
    CF     MERRA2_*.tavgM_3d_rad_Np.YYYY10.nc4  (collection M2TMNPRAD)        CLOUD (0-1)
           -> not in the instM_3d_asm_Np files; download with download_merra2_cloud.sh
  Surface pressure (blanks levels below ground and draws the surface line):
    ERA5: SP_FILE if it exists, else the October mean MERRA-2 PS; MERRA-2: its own PS.

METHOD (per band: omega and CF are averaged over the band's longitudes first)
  r      Pearson correlation of the linearly DETRENDED October series, omega(300 hPa, lat) vs
         CF(level, lat).
  sig    SIG_METHOD = "fdr" (default): t-test on the regression slope with n_eff - 3 degrees
         of freedom (n_eff from the lag-1 autocorrelation of the residuals, capped at n), then
         Benjamini-Hochberg FDR q < ALPHA_FDR over each panel. "pointwise": p < P_THRESH, n - 2 df.
  category  from the UNdetrended series: omega regime at 300 hPa per latitude (strengthening /
         weakening subsidence, strengthening / weakening ascent, new subsidence / new ascent,
         as in the omega regime script) x CF trend direction at that level and latitude.

FIGURE  <OUT_DIR>/omega300hPa_cf_crosssections_10to40E_<DATASET>.png
  Row 1   detrended r (RdBu_r, -1 to 1).
  Row 2   CF decreasing. Colour = omega regime (6 colours); saturation = correlation intensity
          (SHADE_BY = "r": |r|, white = 0, full colour = 1). SHADE_BY = "cf_trend" shades by
          |CF trend| per decade instead (full colour = the 95th percentile over all panels).
          Light grey, outlined in dark grey: CF increases there (shown in row 3).
  Row 3   CF increasing, the same.
  All rows: small dots = r significant (fixed pattern on a quarter of the cells; dark grey,
  light grey on very dark colours); brown line = mean surface pressure;
  white below ground; small black mark at 300 hPa on the left = the omega level.
"""
from __future__ import annotations
import glob
import os
import re
import calendar
import sys
from pathlib import Path
import numpy as np
import xarray as xr
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ---------------------------------------------------------------- paths
ERA5_OMEGA_DIR = Path("/soge-home/users/jesu4127/scratch/era5/global/vertical_velocity")
ERA5_CF_DIR = Path("/soge-home/users/jesu4127/scratch/era5/global/fraction_of_cloud_cover")
SP_FILE = Path("/soge-home/users/jesu4127/scratch/era5/global_surface_pressure.nc")
MERRA2_DIR = Path("/soge-home/users/jesu4127/scratch/merra_2")
MERRA2_CLOUD_DIR = Path("/soge-home/users/jesu4127/scratch/merra_2/cloud")
OUT_DIR = Path("/soge-home/users/jesu4127/Paper_1/circulation")

# ---------------------------------------------------------------- settings
START_YEAR, END_YEAR = 1979, 2025
MONTH = 10                        # the monthly script sets this for each month
OMEGA_LEV = 300.0
PLEV_TOP = 100.0                  # CF levels used (and plotted): 1000 .. PLEV_TOP hPa
LAT_MIN, LAT_MAX = -35.0, 10.0
LON_BANDS = [(10.0, 15.0), (15.0, 20.0), (20.0, 25.0), (25.0, 30.0), (30.0, 35.0), (35.0, 40.0)]
SIG_METHOD = "fdr"                # "fdr" or "pointwise"
ALPHA_FDR = 0.10
P_THRESH = 0.05
SHADE_BY = "r"                    # "r" (correlation intensity) or "cf_trend" (trend intensity)
SHADE_PCTL = 95                   # cf_trend: full colour at this percentile of |trend|
CAT_SIG_ONLY = False              # True: colour categories only where r is significant
MIN_PERSIST = 10                  # years of one sign for a "new" regime
DOT_STEP = 2                      # dots on every 2nd level and every 2nd latitude
DOT_STAGGER = True                # shift alternate dotted levels by one latitude (no columns)
DOT_SIZE = 2
DOT_COLOUR = "#333333"
DOT_COLOUR_ON_DARK = "#B4B4B4"    # lighter dots where the background is very dark
DARK_LUMINANCE = 0.35             # background luminance (0-1) below which it counts as dark
OUTLINE_LW = 0.8                  # outline around the grey "other row" regions
OUTLINE_COLOUR = "#4D4D4D"        # dark grey
C_SP_LINE = "#7B3F00"
C_OTHER = "#F2F2F2"               # cells whose CF change belongs to the other row
REGIMES = ["str_sub", "wk_sub", "new_sub", "wk_asc", "str_asc", "new_asc"]
REGIME_LABELS = {"str_sub": "Strengthening subsidence", "wk_sub": "Weakening subsidence",
                 "new_sub": "New subsidence", "wk_asc": "Weakening ascent",
                 "str_asc": "Strengthening ascent", "new_asc": "New ascent"}
REGIME_COLOURS = {"str_sub": "#DC2671", "wk_sub": "#FE6100", "new_sub": "#FFB000",
                  "wk_asc": "#4DB5FA", "str_asc": "#785EF0", "new_asc": "#60D691"}
BAND_COLS = list(plt.cm.tab10.colors[:len(LON_BANDS)])

matplotlib.rcParams.update({"font.family": "sans-serif", "font.size": 13,
                            "axes.labelsize": 13, "xtick.labelsize": 11,
                            "ytick.labelsize": 11})


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------- loading helpers
def to_domain(da):
    """Standard names, -180..180 longitude, ascending latitude, cut to the domain."""
    da = da.rename({k: v for k, v in {"latitude": "lat", "longitude": "lon",
                                      "pressure_level": "lev"}.items() if k in da.dims})
    if float(da.lon.max()) > 180:
        da = da.assign_coords(lon=(da.lon + 180) % 360 - 180).sortby("lon")
    return da.sortby("lat").sel(lat=slice(LAT_MIN, LAT_MAX),
                                lon=slice(LON_BANDS[0][0], LON_BANDS[-1][1]))


def pick_october(da):
    t = "valid_time" if "valid_time" in da.dims else "time"
    idx = np.flatnonzero(da[t].dt.month.values == MONTH)
    if not idx.size:
        raise ValueError(f"no {calendar.month_name[MONTH]} in file")
    return da.isel({t: int(idx[0])}).drop_vars(["number", "expver", t], errors="ignore")


def stack(items, label):
    """items: {year: DataArray}. Concatenate along 'year'."""
    if not items:
        raise RuntimeError(f"no readable files for {label}")
    years = sorted(items)
    log(f"  {label}: {len(years)} years of {calendar.month_abbr[MONTH]} ({years[0]}-{years[-1]})")
    return xr.concat([items[y] for y in years], "year").assign_coords(year=years)


def merra2_october_files(directory, collection):
    out = {}
    for f in glob.glob(str(directory / "**" / f"MERRA2_*.{collection}.*{MONTH:02d}.nc4"),
                       recursive=True):
        m = re.search(rf"\.(\d{{4}}){MONTH:02d}\.nc4$", f)
        if m and START_YEAR <= int(m[1]) <= END_YEAR:
            out[int(m[1])] = f
    return dict(sorted(out.items()))


def merra2_ps_mean(lat, lon):
    """Mean October MERRA-2 surface pressure (hPa) on (lat, lon), or None."""
    files = {**merra2_october_files(MERRA2_DIR, "instM_2d_asm_Nx"),
             **merra2_october_files(MERRA2_DIR, "instM_3d_asm_Np")}
    fields = []
    for y, f in files.items():
        try:
            with xr.open_dataset(f) as ds:
                fields.append(to_domain(ds["PS"].squeeze(drop=True)).load())
        except Exception as e:
            log(f"  [WARN] {os.path.basename(f)}: {e}")
    if not fields:
        return None
    log(f"  surface pressure: MERRA-2 PS, mean of {len(fields)} years")
    return xr.concat(fields, "t").mean("t").interp(lat=lat, lon=lon,
                                                   kwargs={"fill_value": None})


def to_hpa(sp):
    if sp is None:
        return None
    sp = sp.where(np.abs(sp) < 1e10)
    return sp / 100.0 if float(sp.max()) > 2000 else sp


# ---------------------------------------------------------------- dataset loaders
def load_era5():
    """omega (year, lat, lon), cf (year, lev, lat, lon), sp (lat, lon) hPa or None."""
    def yearly(directory, stem, var, lev=None):
        items = {}
        for f in sorted(directory.glob(f"{stem}_*.nc")):
            m = re.search(r"_(\d{4})\.nc$", f.name)
            if not (m and START_YEAR <= int(m[1]) <= END_YEAR):
                continue
            try:
                with xr.open_dataset(f) as ds:
                    da = to_domain(pick_october(ds[var]))
                    if lev is not None:
                        da = da.sel(lev=lev)
                    items[int(m[1])] = da.load()
            except Exception as e:
                log(f"  SKIPPED {f.name}: {type(e).__name__}: {e}")
        return stack(items, f"ERA5 {var}")

    w = yearly(ERA5_OMEGA_DIR, "global_vertical_velocity", "w", lev=OMEGA_LEV)
    w = w.drop_vars("lev", errors="ignore")
    cf = yearly(ERA5_CF_DIR, "global_fraction_of_cloud_cover", "cc")
    cf = cf.where(cf.lev >= PLEV_TOP, drop=True)
    cf = cf.interp(lat=w.lat, lon=w.lon)            # same 0.25 deg grid; aligns coords
    if SP_FILE.exists():
        with xr.open_dataset(SP_FILE) as ds:
            da = to_domain(ds["sp"])
            t = "valid_time" if "valid_time" in da.dims else "time"
            sp = da.isel({t: np.flatnonzero(da[t].dt.month.values == MONTH)}).mean(t)
            sp = sp.interp(lat=w.lat, lon=w.lon).load()
        log("  surface pressure: ERA5")
    else:
        sp = merra2_ps_mean(w.lat, w.lon)
    return w, cf, to_hpa(sp)


def load_merra2():
    om_files = merra2_october_files(MERRA2_DIR, "instM_3d_asm_Np")
    cl_files = {**merra2_october_files(MERRA2_DIR, "tavgM_3d_rad_Np"),
                **merra2_october_files(MERRA2_CLOUD_DIR, "tavgM_3d_rad_Np")}
    if not cl_files:
        raise RuntimeError("no MERRA2_*.tavgM_3d_rad_Np.YYYY10.nc4 files (variable CLOUD) in "
                           f"{MERRA2_DIR} or {MERRA2_CLOUD_DIR}: run download_merra2_cloud.sh")
    w_items, ps_items, cf_items = {}, {}, {}
    for y, f in om_files.items():
        try:
            with xr.open_dataset(f) as ds:
                w_items[y] = to_domain(ds["OMEGA"].squeeze("time", drop=True)
                                       ).sel(lev=OMEGA_LEV).load()
                ps_items[y] = to_domain(ds["PS"].squeeze("time", drop=True)).load()
        except Exception as e:
            log(f"  SKIPPED {os.path.basename(f)}: {type(e).__name__}: {e}")
    for y, f in cl_files.items():
        try:
            with xr.open_dataset(f) as ds:
                da = to_domain(ds["CLOUD"].squeeze("time", drop=True))
                cf_items[y] = da.where(da.lev >= PLEV_TOP, drop=True).load()
        except Exception as e:
            log(f"  SKIPPED {os.path.basename(f)}: {type(e).__name__}: {e}")
    w = stack(w_items, "MERRA-2 OMEGA")
    cf = stack(cf_items, "MERRA-2 CLOUD")
    sp = xr.concat(list(ps_items.values()), "t").mean("t")
    log(f"  surface pressure: MERRA-2 PS, mean of {len(ps_items)} years")
    return w.drop_vars("lev", errors="ignore"), cf, to_hpa(sp)


DATASETS = {"era5": ("ERA5", load_era5), "merra2": ("MERRA-2", load_merra2)}


# ---------------------------------------------------------------- statistics
def _t(years, ndim):
    t = np.asarray(years, float)
    return (t - t.mean()).reshape((-1,) + (1,) * (ndim - 1))


def detrend(a, years):
    """OLS-detrended anomalies along axis 0 (years)."""
    t = _t(years, a.ndim)
    am = a - a.mean(0)
    return am - (t * am).sum(0) / (t ** 2).sum(0) * t


def slope(a, years):
    """OLS trend per year along axis 0."""
    t = _t(years, a.ndim)
    return (t * (a - a.mean(0))).sum(0) / (t ** 2).sum(0)


def corr_and_p(x, y):
    """x (T, N) detrended omega broadcast to y (T, L, N) detrended CF: r and p per (L, N)."""
    x = np.broadcast_to(x[:, None, :], y.shape)
    n = y.shape[0]
    sxx, syy, sxy = (x * x).sum(0), (y * y).sum(0), (x * y).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = sxy / np.sqrt(sxx * syy)
        if SIG_METHOD == "pointwise":
            tt = r * np.sqrt((n - 2) / np.maximum(1 - r ** 2, 1e-30))
            return r, 2 * stats.t.sf(np.abs(tt), n - 2)
        b = sxy / sxx                                       # regression of CF' on omega'
        e = y - b * x
        sse = (e * e).sum(0)
        r1 = (e[1:] * e[:-1]).sum(0) / sse
        n_eff = np.minimum(n * (1 - r1) / (1 + r1), n)
        se = np.sqrt(sse / (n_eff - 3) / sxx)
        p = 2 * stats.t.sf(np.abs(b / se), n_eff - 3)
    return r, np.where(n_eff > 3, p, np.nan)


def significant(p):
    sig, ok = np.zeros(p.shape, bool), np.isfinite(p)
    if not ok.any():
        return sig
    if SIG_METHOD == "pointwise":
        sig[ok] = p[ok] < P_THRESH
    else:
        sig[ok] = stats.false_discovery_control(p[ok]) < ALPHA_FDR
    return sig


def regime_codes(x, slope_):
    """Omega regime per column of x (year, N): index into REGIMES, -1 none (switch logic of
    the omega regime script: first persistent sign change, else mean sign x trend sign)."""
    n, P = x.shape[0], MIN_PERSIST
    fin = np.isfinite(x)
    cum = lambda a: np.concatenate([np.zeros_like(a[:1], float), np.cumsum(a, 0)])
    c_sum, c_fin = cum(np.where(fin, x, 0.0)), cum(fin)
    c_pos, c_neg = cum(fin & (x > 0)), cum(fin & (x < 0))
    ks = np.arange(1, n - P)
    win = lambda c: c[ks + P] - c[ks]
    with np.errstate(invalid="ignore", divide="ignore"):
        pre_n, post_n = c_fin[ks], c_fin[n] - c_fin[ks]
        pre, post = c_sum[ks] / pre_n, (c_sum[n] - c_sum[ks]) / post_n
        ok = ((win(c_fin) == P) & ((win(c_pos) == P) | (win(c_neg) == P)) & (pre_n >= 2)
              & (post_n >= 2) & (pre != 0) & (post != 0) & (np.sign(pre) != np.sign(post)))
        found = ok.any(0)
        post_first = np.take_along_axis(post, ok.argmax(0)[None], 0)[0]
        mean = c_sum[n] / np.maximum(c_fin[n], 1)
    code = np.where(mean > 0, np.where(slope_ > 0, 0, 1), np.where(slope_ < 0, 4, 3))
    code = np.where(found, np.where(post_first > 0, 2, 5), code)
    return np.where(np.isfinite(slope_), code, -1)


def analyse_band(w, cf, sp, years, lo, hi):
    sel = (w.lon >= lo) & (w.lon < hi)
    wb = w.where(sel, drop=True).mean("lon").values                     # (year, lat)
    cfb = cf.where((cf.lon >= lo) & (cf.lon < hi), drop=True).mean("lon").transpose("year", "lev", "lat").values        # (year, lev, lat)
    spb = None if sp is None else sp.where(sel, drop=True).mean("lon").values
    plevs = cf.lev.values.astype(float)
    r, p = corr_and_p(detrend(wb, years), detrend(cfb, years))
    below = (plevs[:, None] > spb[None, :]) if spb is not None else np.zeros(r.shape, bool)
    r = np.where(below, np.nan, r)
    sig = significant(np.where(below, np.nan, p))
    reg = regime_codes(wb, slope(wb, years))                            # (lat,)
    cf_trend = slope(cfb, years) * 10.0                                 # per decade
    cat = np.where((reg[None, :] >= 0) & ~below & np.isfinite(cf_trend),
                   2 * reg[None, :] + np.where(cf_trend > 0, 0, 1), -1)
    return {"label": f"{lo:.0f}°E–{hi:.0f}°E", "r": r, "sig": sig, "cat": cat,
            "cf_trend": np.where(below, np.nan, cf_trend), "below": below, "sp": spb}


# ---------------------------------------------------------------- plotting
def edges(v):
    d = np.diff(v) / 2
    return np.concatenate([[v[0] - d[0]], v[:-1] + d, [v[-1] + d[-1]]])


def shade_cmap(colour):
    return mcolors.LinearSegmentedColormap.from_list("", ["white", colour])


def regime_colours():
    return [REGIME_COLOURS[k] for k in REGIMES]


def mesh(ax, x, y, field, **kw):
    ax.pcolormesh(x, y, field, rasterized=True, shading="flat", edgecolors="face",
                  linewidth=0, antialiased=False, zorder=1, **kw)


def format_section(ax, plevs, sp, show_y):
    ybot = min(1025.0, np.ceil(np.nanmax(sp) / 25.0) * 25.0) if sp is not None else 1000.0
    ax.set_ylim(ybot, PLEV_TOP)
    ax.set_xlim(LAT_MIN, LAT_MAX)
    ax.yaxis.set_major_locator(mticker.FixedLocator(
        [v for v in (100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
         if PLEV_TOP <= v <= ybot]))
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.5, zorder=3)
    ax.set_xlabel("Latitude (°N)")
    if show_y:
        ax.set_ylabel("Pressure (hPa)")
    else:
        ax.tick_params(labelleft=False)
    for s in ax.spines.values():
        s.set_edgecolor("black")
        s.set_linewidth(1.0)


def luminance(rgba):
    return 0.299 * rgba[..., 0] + 0.587 * rgba[..., 1] + 0.114 * rgba[..., 2]


def paint(bg, field, cmap, norm):
    """Put the colours pcolormesh draws for `field` into the background RGBA array bg."""
    ok = np.isfinite(field)
    bg[ok] = cmap(norm(field[ok]))


def mask_outline(ax, lat_e, plev_e, mask, below):
    """Dark-grey line along the cell edges that separate mask cells from coloured cells.
    No line against below-ground cells (the surface pressure line marks that edge) or along
    the panel border (the frame marks it)."""
    m = np.pad(mask, 1, constant_values=False)
    b = np.pad(below, 1, constant_values=True)
    segs = []
    diff = (m[1:, 1:-1] != m[:-1, 1:-1]) & ~b[1:, 1:-1] & ~b[:-1, 1:-1]     # between levels
    for i, j in zip(*np.nonzero(diff)):
        segs.append([(lat_e[j], plev_e[i]), (lat_e[j + 1], plev_e[i])])
    diff = (m[1:-1, 1:] != m[1:-1, :-1]) & ~b[1:-1, 1:] & ~b[1:-1, :-1]     # between latitudes
    for i, j in zip(*np.nonzero(diff)):
        segs.append([(lat_e[j], plev_e[i]), (lat_e[j], plev_e[i + 1])])
    ax.add_collection(LineCollection(segs, colors=OUTLINE_COLOUR, linewidths=OUTLINE_LW,
                                     zorder=4, capstyle="projecting"))


def overlays(ax, lat, plevs, lat_e, plev_e, res, bg, mask=None):
    if res["sp"] is not None:
        ax.plot(lat, res["sp"], color=C_SP_LINE, lw=0.9, zorder=9)     # on top of everything
    # small mark at the omega level, just inside the left axis
    ax.plot([0, 0.045], [OMEGA_LEV, OMEGA_LEV], color="black", lw=3.0, solid_capstyle="butt",
            transform=ax.get_yaxis_transform(), clip_on=False, zorder=8)
    sig = res["sig"] if mask is None else res["sig"] & mask
    # fixed pattern, no randomness: every DOT_STEP-th level and latitude, with alternate
    # dotted levels shifted by one latitude if DOT_STAGGER; dots at the cell centres
    ii, jj = np.meshgrid(np.arange(plevs.size), np.arange(lat.size), indexing="ij")
    shift = (ii // DOT_STEP) % 2 if DOT_STAGGER else 0
    pick = (ii % DOT_STEP == 0) & ((jj + shift) % DOT_STEP == 0)
    li, ci = np.nonzero(sig & pick)
    if not li.size:
        return
    dark = luminance(bg[li, ci]) < DARK_LUMINANCE
    ax.scatter(lat[ci], plevs[li], s=DOT_SIZE, c=np.where(dark, DOT_COLOUR_ON_DARK, DOT_COLOUR),
               linewidths=0, zorder=5)


def make_figure(name, results, lat, plevs, years, path):
    nb = len(results)
    W, left, right, gap = 30.0, 1.6, 0.3, 0.25
    side_w = 3.4
    pw = (W - left - right - nb * gap - side_w - 0.6) / nb
    ph = 4.2
    tops = [1.5, 1.5 + ph + 1.0, 1.5 + 2 * (ph + 1.0)]
    H = tops[2] + ph + 2.0
    fig = plt.figure(figsize=(W, H))
    rnorm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    lat_e, plev_e = edges(lat), edges(plevs)
    cols = regime_colours()
    if SHADE_BY == "r":
        vmax, bar_label, scale = 1.0, "Saturation = Correlation intensity |r|", 1.0
    else:
        allt = np.concatenate([np.abs(r["cf_trend"]).ravel() for r in results])
        vmax = float(np.nanpercentile(allt, SHADE_PCTL)) if np.isfinite(allt).any() else 1.0
        vmax = vmax if vmax > 0 else 1.0
        bar_label, scale = "Saturation = Trend intensity\n(|CF trend|, % per decade)", 100.0
    row_names = ["Detrended r(ω300, CF)", "CF decreasing", "CF increasing"]
    row_parity = [None, 1, 0]                          # cat odd = CF down, even = CF up
    other = mcolors.ListedColormap([C_OTHER])
    for j, res in enumerate(results):
        x0 = left + j * (pw + gap)
        strength = np.abs(res["r"]) if SHADE_BY == "r" else np.abs(res["cf_trend"])
        strength = np.clip(strength / vmax, 0, 1)
        cat = res["cat"]
        keep = (cat >= 0) & np.isfinite(strength)
        if CAT_SIG_ONLY:
            keep &= res["sig"]
        for row, top in enumerate(tops):
            ax = fig.add_axes([x0 / W, 1 - (top + ph) / H, pw / W, ph / H])
            in_row = None
            bg = np.ones(res["r"].shape + (4,))                # what is drawn under the dots
            if row == 0:
                mesh(ax, lat_e, plev_e, res["r"], cmap="RdBu_r", norm=rnorm)
                paint(bg, res["r"], plt.get_cmap("RdBu_r"), rnorm)
                ax.set_title(res["label"], fontsize=15, fontweight="bold",
                             color=BAND_COLS[j], pad=6)
            else:
                in_row = (cat >= 0) & (cat % 2 == row_parity[row])
                not_row = (cat >= 0) & ~in_row
                mesh(ax, lat_e, plev_e, np.where(not_row, 1.0, np.nan), cmap=other, vmin=0, vmax=1)
                mask_outline(ax, lat_e, plev_e, not_row, res["below"])
                for k, colour in enumerate(cols):
                    field = np.where(in_row & keep & (cat // 2 == k), strength, np.nan)
                    if np.isfinite(field).any():
                        mesh(ax, lat_e, plev_e, field, cmap=shade_cmap(colour), vmin=0, vmax=1)
                        paint(bg, field, shade_cmap(colour), mcolors.Normalize(0, 1))
            overlays(ax, lat, plevs, lat_e, plev_e, res, bg, mask=in_row)
            format_section(ax, plevs, res["sp"], show_y=(j == 0))
            if j == 0:
                fig.text(0.45 / W, 1 - (top + ph / 2) / H, row_names[row], rotation=90,
                         ha="center", va="center", fontsize=17, fontweight="bold")
    xr0 = left + nb * (pw + gap) + 0.4
    # domain map (right of row 1)
    axm = fig.add_axes([xr0 / W, 1 - (tops[0] + ph - 0.3) / H, side_w / W, (ph - 0.3) / H],
                       projection=ccrs.PlateCarree())
    axm.set_extent([LON_BANDS[0][0], LON_BANDS[-1][1], LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
    axm.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#d0e8f0", zorder=1)
    axm.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f0ede8", zorder=2)
    axm.coastlines(resolution="50m", linewidth=0.6, zorder=5)
    axm.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.4, edgecolor="grey", zorder=4)
    for j, (lo, hi) in enumerate(LON_BANDS):
        axm.add_patch(mpatches.Rectangle((lo, LAT_MIN), hi - lo, LAT_MAX - LAT_MIN,
                                         facecolor=BAND_COLS[j], alpha=0.35,
                                         transform=ccrs.PlateCarree(), zorder=6))
    gl = axm.gridlines(draw_labels=True, linewidth=0.3, color="grey", alpha=0.5, linestyle=":")
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 9}
    axm.set_title("Domain", fontsize=13, fontweight="bold", pad=4)
    # r colour bar (under the map)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=rnorm, cmap="RdBu_r"),
                      cax=fig.add_axes([xr0 / W, 1 - (tops[0] + ph + 0.55) / H,
                                        side_w / W, 0.3 / H]),
                      orientation="horizontal", ticks=np.linspace(-1, 1, 5))
    cb.set_label("Correlation r", fontsize=14)
    cb.ax.xaxis.set_label_position("top")
    cb.ax.tick_params(labelsize=12)
    # six saturation bars shared by rows 2 and 3
    k = len(cols)
    bw = 2.0 / (k + 0.4 * (k - 1))
    y_top, y_bot = tops[1] + 0.4, tops[2] + ph - 0.4
    for i, colour in enumerate(cols):
        c = fig.colorbar(plt.cm.ScalarMappable(norm=mcolors.Normalize(0, vmax * scale),
                                               cmap=shade_cmap(colour)),
                         cax=fig.add_axes([(xr0 + i * 1.4 * bw) / W, 1 - y_bot / H,
                                           bw / W, (y_bot - y_top) / H]),
                         orientation="vertical")
        c.outline.set_linewidth(0.5)
        if i < k - 1:
            c.set_ticks([])
        else:
            c.ax.tick_params(labelsize=12)
            c.set_label(bar_label, fontsize=14)
    sig_text = (f"BH-FDR q < {ALPHA_FDR} per panel, AR(1)-corrected" if SIG_METHOD == "fdr"
                else f"p < {P_THRESH}")
    handles = [mpatches.Patch(facecolor=c) for c in cols]
    labels = [REGIME_LABELS[r] for r in REGIMES]
    handles += [Line2D([], [], marker="o", ls="none", color=DOT_COLOUR, markeredgewidth=0,
                       markersize=4),
                Line2D([], [], color=C_SP_LINE, lw=1.2),
                Line2D([], [], color="black", lw=3),
                mpatches.Patch(facecolor=C_OTHER, edgecolor=OUTLINE_COLOUR, linewidth=OUTLINE_LW),
                mpatches.Patch(facecolor="white", edgecolor="0.5")]
    labels += [f"r significant ({sig_text})", "Mean surface pressure",
               f"ω level ({OMEGA_LEV:.0f} hPa)", "CF changes the other way (see other row)",
               "Below ground"]
    fig.legend(handles=handles, labels=labels, loc="upper center", ncol=6, fontsize=12,
               frameon=True, title="ω 300 hPa long-term change (colour, rows 2-3)",
               title_fontsize=13, bbox_to_anchor=(0.5, 1 - (tops[2] + ph + 0.7) / H),
               handlelength=1.8, columnspacing=1.5)
    fig.suptitle(f"{name} {calendar.month_name[MONTH]} {years[0]}-{years[-1]} detrended r(ω{OMEGA_LEV:.0f} hPa, "
                 "cloud fraction)", fontsize=22, fontweight="bold", y=1 - 0.4 / H, va="center")
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    log(f"Saved: {path}")


def run(key):
    name, loader = DATASETS[key]
    log(f"===== {name} {calendar.month_name[MONTH]} =====")
    w, cf, sp = loader()
    years = sorted(set(w.year.values.tolist()) & set(cf.year.values.tolist()))
    w, cf = w.sel(year=years), cf.sel(year=years).sortby("lev")
    lat, plevs = w.lat.values, cf.lev.values.astype(float)
    cf = cf.interp(lat=w.lat)                     # same latitudes as omega
    log(f"Common years {years[0]}-{years[-1]} (n = {len(years)}); {lat.size} latitudes; "
        f"{plevs.size} CF levels")
    results = []
    for lo, hi in LON_BANDS:
        res = analyse_band(w, cf, sp, years, lo, hi)
        log(f"  {res['label']}: {100 * res['sig'].sum() / max(np.isfinite(res['r']).sum(), 1):.0f}%"
            " of level-latitude cells significant")
        results.append(res)
    tag = key.upper()
    make_figure(name, results, lat, plevs, years,
                OUT_DIR / f"omega{OMEGA_LEV:.0f}hPa_cf_crosssections_10to40E_{tag}_"
                          f"{MONTH:02d}_{calendar.month_abbr[MONTH]}.png")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    keys = [a.lower().replace("-", "") for a in sys.argv[1:]] or list(DATASETS)
    for key in keys:
        if key not in DATASETS:
            raise SystemExit(f"unknown dataset {key!r}; choose from {list(DATASETS)}")
        try:
            run(key)
        except Exception as e:
            log(f"[ERROR] {key}: {type(e).__name__}: {e}")
    log("Done.")


if __name__ == "__main__":
    main()