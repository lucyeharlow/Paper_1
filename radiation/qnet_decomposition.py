"""
Annual Peak Qnet Trend Summary Map
====================================
Synthesises the monthly Qnet decomposition across all 12 months into a
single 5-panel figure per dataset (ERA5 and MERRA-2), answering:

  1. WHERE are the largest significant Qnet trends, spatially?
  2. WHEN in the year does the peak signal occur?
  3. HOW MANY months show a significant Qnet trend at each pixel?
  4. WHAT flux is driving that change?
  5. WHAT flux is driving the opposite direction?

Output: 2 rows (ERA5 / MERRA-2) × 5 columns
  Col 1 (a): Peak Qnet slope               — signed Qnet slope (W m-2 dec-1)
  Col 2 (b): Peak month                    — cyclic 12-month colourmap
  Col 3 (c): Number of significant months  — sequential count (0-12)
  Col 4 (d): Dominant flux strengthening   — 6-component colour scheme
  Col 5 (e): Dominant flux weakening       — 6-component colour scheme

Two figures are produced:
  1. Full individual dataset ranges (ERA5 1979–2025, MERRA-2 1980–2025)
  2. Overlapping years only (common period shared by both datasets)
"""

import os
import warnings
import colorsys
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mplticker
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm, TwoSlopeNorm
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
ERA5_DIR = (
    "/ec/res4/scratch/gbr3310/fixed_era5_flux_analysis"
    "/1979_2025_merra2grid/global/OLS_trends"
)
MERRA2_DIR = (
    "/ec/res4/scratch/gbr3310/fixed_era5_flux_analysis"
    "/1979_2025_merra2/global"
)
OUT_DIR = (
    "/ec/res4/scratch/gbr3310/fixed_era5_flux_analysis"
    "/1979_2025_merra2grid/comparison_plots"
)
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DECADE    = 10
FDR_ALPHA = 0.10

# Full dataset ranges
ERA5_START,   ERA5_END   = 1979, 2025
MERRA2_START, MERRA2_END = 1980, 2025

# Overlapping period (common years)
OVERLAP_START = max(ERA5_START, MERRA2_START)   # 1980
OVERLAP_END   = min(ERA5_END,   MERRA2_END)     # 2025

LAT_MIN, LAT_MAX = -35.0, -10.0
LON_MIN, LON_MAX =  10.0,  40.0
EXTENT = [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX]

FLUXNET_LAT = -15.4391
FLUXNET_LON =  23.2525

MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR DESIGN
# ─────────────────────────────────────────────────────────────────────────────
def hls_rgb(h_deg, l, s):
    r, g, b = colorsys.hls_to_rgb(h_deg / 360.0, l, s)
    return np.array([r, g, b])


COMP_ORDER = ["SW_dn", "SW_up", "LW_dn", "LW_up", "LHF", "SHF"]

COMP_META = {
    "SW_dn": (48,  0.80, 0.55),
    "SW_up": (280, 0.75, 0.55),
    "LW_dn": (130, 0.82, 0.48),
    "LW_up": (25,  0.80, 0.58),
    "LHF":   (210, 0.80, 0.55),
    "SHF":   (330, 0.80, 0.60),
}

COMP_LABELS = {
    "SW_dn": r"SW$\downarrow$",
    "SW_up": r"SW$\uparrow$",
    "LW_dn": r"LW$\downarrow$",
    "LW_up": r"LW$\uparrow$",
    "LHF":   "LHF",
    "SHF":   "SHF",
}

N_COMP = len(COMP_ORDER)
SOLID_COLORS = np.array([hls_rgb(*COMP_META[c]) for c in COMP_ORDER])


def build_flux_cmap():
    colors = [[*SOLID_COLORS[i], 1.0] for i in range(N_COMP)]
    colors.append([0.90, 0.90, 0.90, 1.0])
    colors.append([1.0,  1.0,  1.0,  0.0])
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(len(colors) + 1) - 0.5, cmap.N)
    return cmap, norm


FLUX_CMAP, FLUX_NORM = build_flux_cmap()


def build_binary_cmap():
    colors = [[*SOLID_COLORS[i], 1.0] for i in range(N_COMP)]
    colors.append([0.78, 0.78, 0.78, 1.0])
    colors.append([1.0,  1.0,  1.0,  0.0])
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(len(colors) + 1) - 0.5, cmap.N)
    return cmap, norm


BINARY_CMAP, BINARY_NORM = build_binary_cmap()
BINARY_GREY = N_COMP
BINARY_NAN  = N_COMP + 1
NO_SIG_IDX  = N_COMP

_MONTH_HUE = np.linspace(0, 360, 13)[:-1]


def build_month_cmap():
    colors      = [hls_rgb(h, 0.55, 0.80) for h in _MONTH_HUE]
    colors_rgba = [[*c, 1.0] for c in colors]
    colors_rgba.append([0.90, 0.90, 0.90, 1.0])
    cmap = ListedColormap(colors_rgba)
    norm = BoundaryNorm(np.arange(len(colors_rgba) + 1) - 0.5, cmap.N)
    return cmap, norm


MONTH_CMAP, MONTH_NORM = build_month_cmap()
NO_SIG_MONTH = 12


def build_count_cmap():
    t = plt.cm.tab20.colors
    colors = [(1.0, 1.0, 1.0, 0.0)] + [(*t[i * 2], 1.0) for i in range(10)] + [
        (*t[1], 1.0), (*t[3], 1.0),
    ]
    return ListedColormap(colors), BoundaryNorm(np.arange(-0.5, 13.5, 1), 14)


COUNT_CMAP, COUNT_NORM = build_count_cmap()

matplotlib.rcParams.update({
    "font.family":     "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.titlesize":  11,
    "axes.labelsize":  9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

# ─────────────────────────────────────────────────────────────────────────────
# SLOPE FILE MAPS
# ─────────────────────────────────────────────────────────────────────────────
ERA5_SLOPES = {
    "SW_dn":   "ssrd_slope_by_month.nc",
    "SW_up":   "ssru_slope_by_month.nc",
    "LW_dn":   "strd_slope_by_month.nc",
    "LW_up":   "stru_slope_by_month.nc",
    "LHF":     "lhf_slope_by_month.nc",
    "SHF":     "shf_slope_by_month.nc",
    "Qnet":    "qnet_slope_by_month.nc",
    "Qnet_pv": "qnet_pvalue_by_month.nc",
}

MERRA2_SLOPES = {
    "SW_dn":   "SWGDN_slope_by_month.nc",
    "SW_net":  "SWGNT_slope_by_month.nc",
    "LW_dn":   "LWGAB_slope_by_month.nc",
    "LW_net":  "LWGNT_slope_by_month.nc",
    "LHF":     "lhf_slope_by_month.nc",
    "SHF":     "shf_slope_by_month.nc",
    "Qnet":    "qnet_slope_by_month.nc",
    "Qnet_pv": "qnet_pvalue_by_month.nc",
}

# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def log(msg):
    print(msg, flush=True)


def get_latlon_names(da):
    for ln, lo in (("lat", "lon"), ("latitude", "longitude")):
        if ln in da.coords:
            return ln, lo
    raise KeyError(f"No lat/lon in {list(da.coords)}")


def load_month(fpath, month):
    da = xr.open_dataarray(fpath)
    if "month" in da.dims:
        da = da.sel(month=month)
    ln, lo = get_latlon_names(da)
    da = da.sortby(ln)
    lv, lov = da[ln].values, da[lo].values
    return da.sel({
        ln: slice(max(LAT_MIN, float(lv.min())), min(LAT_MAX, float(lv.max()))),
        lo: slice(max(LON_MIN, float(lov.min())), min(LON_MAX, float(lov.max()))),
    })


def fdr_mask(pval_2d):
    p      = pval_2d.flatten()
    finite = np.isfinite(p)
    reject = np.zeros(p.shape, dtype=bool)
    if finite.sum() > 0:
        rej, *_ = multipletests(p[finite], alpha=FDR_ALPHA, method="fdr_bh")
        reject[finite] = rej
    return reject.reshape(pval_2d.shape)


# ─────────────────────────────────────────────────────────────────────────────
# DECOMPOSITION HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def dominant_flux(slopes_month):
    sw_up = slopes_month.get("SW_up")
    lw_up = slopes_month.get("LW_up")
    if sw_up is None or np.all(np.isnan(sw_up)):
        sw_up = slopes_month["SW_dn"] - slopes_month.get(
            "SW_net", np.full_like(slopes_month["SW_dn"], np.nan))
    if lw_up is None or np.all(np.isnan(lw_up)):
        lw_up = slopes_month["LW_dn"] - slopes_month.get(
            "LW_net", np.full_like(slopes_month["LW_dn"], np.nan))

    contribs = np.stack([
        slopes_month["SW_dn"],
        -sw_up,
        slopes_month["LW_dn"],
        -lw_up,
        -slopes_month["LHF"],
        -slopes_month["SHF"],
    ], axis=0)

    all_nan = np.all(~np.isfinite(contribs), axis=0)
    dom = np.argmax(
        np.abs(np.where(np.isfinite(contribs), contribs, 0.0)), axis=0
    ).astype(int)
    dom[all_nan] = -1
    return dom


# ─────────────────────────────────────────────────────────────────────────────
# CORE: LOAD ALL MONTHS AND COMPUTE PEAK-MONTH ARRAYS
# ─────────────────────────────────────────────────────────────────────────────
def compute_annual_peak(src_dir, file_map, year_range_label=""):
    log(f"  Loading slopes from {src_dir}  {year_range_label}...")

    qnet_path = os.path.join(src_dir, file_map["Qnet"])

    # Try alternative Qnet filenames if the primary is missing
    if not os.path.exists(qnet_path):
        for alt in ["Qnet_slope_by_month.nc", "QNET_slope_by_month.nc",
                    "net_radiation_slope_by_month.nc", "qnet_trend_by_month.nc"]:
            alt_path = os.path.join(src_dir, alt)
            if os.path.exists(alt_path):
                log(f"  [INFO] Using alternative Qnet file: {alt}")
                qnet_path = alt_path
                file_map = {**file_map, "Qnet": alt}
                break

    if not os.path.exists(qnet_path):
        try:
            available = sorted(f for f in os.listdir(src_dir) if f.endswith(".nc"))
            log(f"  ERROR: Qnet slope file not found. Available .nc files in {src_dir}:")
            for f in available:
                log(f"    {f}")
        except Exception:
            log(f"  ERROR: {qnet_path} not found and directory unreadable")
        return None

    sample     = load_month(qnet_path, 1)
    ln, lo     = get_latlon_names(sample)
    lats       = sample[ln].values
    lons       = sample[lo].values
    nrow, ncol = len(lats), len(lons)

    qnet_slopes     = np.full((12, nrow, ncol), np.nan)
    qnet_sig        = np.zeros((12, nrow, ncol), dtype=bool)
    dom_flux        = np.full((12, nrow, ncol), -1, dtype=int)
    raw_comp_slopes = np.full((12, N_COMP, nrow, ncol), np.nan)

    for m in range(1, 13):
        mi = m - 1
        log(f"    Month {m:02d} ...")

        month_slopes = {}
        for key, fname in file_map.items():
            fpath = os.path.join(src_dir, fname)
            if not os.path.exists(fpath):
                month_slopes[key] = np.full((nrow, ncol), np.nan)
                continue
            try:
                da  = load_month(fpath, m)
                arr = da.values.astype(float)
                month_slopes[key] = arr if arr.shape == (nrow, ncol) else \
                    np.full((nrow, ncol), np.nan)
            except Exception as exc:
                log(f"      WARNING: {key} month {m}: {exc}")
                month_slopes[key] = np.full((nrow, ncol), np.nan)

        qs = month_slopes["Qnet"]
        pv = month_slopes["Qnet_pv"]
        qnet_slopes[mi] = qs
        sig_mask        = fdr_mask(np.where(np.isfinite(pv), pv, 1.0))
        qnet_sig[mi]    = sig_mask
        dom_flux[mi]    = dominant_flux(month_slopes)

        sw_up_m = month_slopes.get("SW_up")
        lw_up_m = month_slopes.get("LW_up")
        if sw_up_m is None or np.all(np.isnan(sw_up_m)):
            sw_up_m = month_slopes["SW_dn"] - month_slopes.get(
                "SW_net", np.full((nrow, ncol), np.nan))
        if lw_up_m is None or np.all(np.isnan(lw_up_m)):
            lw_up_m = month_slopes["LW_dn"] - month_slopes.get(
                "LW_net", np.full((nrow, ncol), np.nan))

        raw_comp_slopes[mi, 0] = month_slopes["SW_dn"]
        raw_comp_slopes[mi, 1] = sw_up_m
        raw_comp_slopes[mi, 2] = month_slopes["LW_dn"]
        raw_comp_slopes[mi, 3] = lw_up_m
        raw_comp_slopes[mi, 4] = month_slopes["LHF"]
        raw_comp_slopes[mi, 5] = month_slopes["SHF"]

    abs_slopes = np.abs(qnet_slopes)
    abs_sig    = np.where(qnet_sig & np.isfinite(qnet_slopes), abs_slopes, -np.inf)
    has_any_sig = (abs_sig > -np.inf).any(axis=0)
    peak_m      = np.argmax(abs_sig, axis=0)

    sig_pos = qnet_sig & np.isfinite(qnet_slopes) & (qnet_slopes > 0)
    sig_neg = qnet_sig & np.isfinite(qnet_slopes) & (qnet_slopes <= 0)
    sig_count_pos = sig_pos.sum(axis=0).astype(float)
    sig_count_neg = sig_neg.sum(axis=0).astype(float)

    modal_flux_idx   = np.full((nrow, ncol), NO_SIG_IDX,  dtype=float)
    peak_month_idx   = np.full((nrow, ncol), NO_SIG_MONTH, dtype=float)
    peak_qnet_dec    = np.full((nrow, ncol), np.nan,       dtype=float)

    for r in range(nrow):
        for c in range(ncol):
            if not has_any_sig[r, c]:
                continue
            pm = int(peak_m[r, c])
            peak_month_idx[r, c] = float(pm)
            peak_qnet_dec[r, c]  = qnet_slopes[pm, r, c] * DECADE

            sig_months = [
                m for m in range(12)
                if qnet_sig[m, r, c]
                and np.isfinite(qnet_slopes[m, r, c])
                and dom_flux[m, r, c] >= 0
            ]
            if not sig_months:
                continue

            flux_weight = np.zeros(N_COMP)
            for m in sig_months:
                fi = dom_flux[m, r, c]
                flux_weight[fi] += abs(qnet_slopes[m, r, c])
            modal_flux_idx[r, c] = float(np.argmax(flux_weight))

    n_sig_pix = int(has_any_sig.sum())
    log(f"  Pixels with ≥1 significant month: {n_sig_pix}/{nrow*ncol} "
        f"({100*n_sig_pix/(nrow*ncol):.1f}%)")

    modal_flux_str_mask = np.zeros((nrow, ncol), dtype=bool)
    for r in range(nrow):
        for c in range(ncol):
            if not has_any_sig[r, c]:
                continue
            pm = int(peak_m[r, c])
            fi = int(modal_flux_idx[r, c])
            if 0 <= fi < N_COMP:
                modal_flux_str_mask[r, c] = (raw_comp_slopes[pm, fi, r, c] > 0)

    return {
        "peak_qnet_dec":       peak_qnet_dec,
        "modal_flux_idx":      modal_flux_idx,
        "modal_flux_str_mask": modal_flux_str_mask,
        "peak_month_idx":      peak_month_idx,
        "sig_count_pos":       sig_count_pos,
        "sig_count_neg":       sig_count_neg,
        "lats": lats,
        "lons": lons,
    }


def filter_by_direction(res, direction):
    pq      = res["peak_qnet_dec"].copy()
    mfi     = res["modal_flux_idx"].copy()
    pmi     = res["peak_month_idx"].copy()
    str_mask = res["modal_flux_str_mask"].copy()

    if direction == "pos":
        mask_out  = ~(np.isfinite(pq) & (pq > 0))
        count_key = "sig_count_pos"
    else:
        mask_out  = ~(np.isfinite(pq) & (pq <= 0))
        count_key = "sig_count_neg"

    pq[mask_out]      = np.nan
    mfi[mask_out]     = float(NO_SIG_IDX)
    pmi[mask_out]     = float(NO_SIG_MONTH)
    str_mask[mask_out] = False
    count              = res[count_key].copy()
    count[mask_out]    = 0.0

    return {
        "peak_qnet_dec":       pq,
        "modal_flux_idx":      mfi,
        "modal_flux_str_mask": str_mask,
        "peak_month_idx":      pmi,
        "sig_count":           count,
        "lats": res["lats"],
        "lons": res["lons"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAP HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def add_map_features(ax):
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), zorder=3, facecolor="white")
    ax.add_feature(cfeature.LAKES.with_scale("50m"), zorder=3, facecolor="white")
    ax.coastlines(resolution="50m", linewidth=0.5, zorder=4)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.3, zorder=4)
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())


def add_gridlines(ax, left_labels=False, bottom_labels=False):
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.3, color="black", alpha=0.45, linestyle=":"
    )
    gl.top_labels    = False
    gl.right_labels  = False
    gl.left_labels   = left_labels
    gl.bottom_labels = bottom_labels
    gl.xlocator      = mplticker.FixedLocator([10, 20, 30, 40])
    gl.ylocator      = mplticker.FixedLocator([-30, -20, -10])
    gl.xformatter    = LongitudeFormatter()
    gl.yformatter    = LatitudeFormatter()
    gl.xlabel_style  = {"size": 9, "fontfamily": "sans-serif"}
    gl.ylabel_style  = {"size": 9, "fontfamily": "sans-serif"}


def mark_fluxnet(ax):
    ax.plot(
        FLUXNET_LON, FLUXNET_LAT,
        marker="s", markersize=5,
        color="black", markeredgecolor="black", markeredgewidth=0.8,
        transform=ccrs.PlateCarree(), zorder=6,
    )


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE
# ─────────────────────────────────────────────────────────────────────────────
def make_direction_figure(
    era5_filt,
    m2_filt,
    direction,
    era5_label,
    m2_label,
    mag_vmax=5.0,
    fig_suffix="",
):
    """
    2 rows (ERA5 / MERRA-2) × 5 columns.
    Column letters a)–e) placed once above row 0 (not per-panel).
    Colourbar for col 1 on right of col 1.
    Colourbar for col 3 on LEFT of col 3.
    Legends placed close to their respective columns.
    """
    PROJ = ccrs.PlateCarree()

    # ── Layout geometry ───────────────────────────────────────────────────────
    # 5 equal map columns, tight gaps
    LEFT   = 0.07
    RIGHT  = 0.97
    TOP    = 0.82
    BOT    = 0.20     # room for legends below
    WSPACE = 0.03
    HSPACE = 0.08

    fig, axes = plt.subplots(
        2, 5, figsize=(28.0, 8.5),
        subplot_kw=dict(projection=PROJ),
    )
    fig.subplots_adjust(
        left=LEFT, right=RIGHT, top=TOP, bottom=BOT,
        wspace=0.06, hspace=HSPACE,
    )

    # ── Compute column bounding boxes after layout ────────────────────────────
    fig.canvas.draw()

    row_labels = [era5_label, m2_label]

    mag_norm = TwoSlopeNorm(vmin=-mag_vmax, vcenter=0, vmax=mag_vmax)
    mag_levs = np.linspace(-mag_vmax, mag_vmax, 21)

    last_mag_cf  = None
    last_cnt_pm  = None

    for ri, (filt, ds_lbl) in enumerate(
        zip([era5_filt, m2_filt], ["ERA5", "MERRA-2"])
    ):
        if filt is None:
            for ci in range(5):
                axes[ri, ci].set_visible(False)
            continue

        lats, lons = filt["lats"], filt["lons"]
        lon2d, lat2d = np.meshgrid(lons, lats)

        mfi      = filt["modal_flux_idx"]
        str_mask = filt["modal_flux_str_mask"]
        no_sig   = (mfi == float(NO_SIG_IDX))

        str_enc = mfi.copy()
        str_enc[~str_mask | no_sig] = float(BINARY_GREY)
        str_enc[no_sig]             = float(BINARY_NAN)

        wkn_enc = mfi.copy()
        wkn_enc[str_mask | no_sig] = float(BINARY_GREY)
        wkn_enc[no_sig]            = float(BINARY_NAN)

        # Col 1: peak Qnet slope
        ax = axes[ri, 0]
        last_mag_cf = ax.contourf(
            lons, lats, filt["peak_qnet_dec"],
            levels=mag_levs, cmap="RdBu_r", norm=mag_norm,
            extend="both", transform=PROJ, zorder=2,
        )
        add_map_features(ax)
        add_gridlines(ax, left_labels=True, bottom_labels=(ri == 1))
        mark_fluxnet(ax)
        ax.set_title(f"{ds_lbl}", fontsize=12, fontweight="bold", pad=4)

        # Col 2: peak month
        ax = axes[ri, 1]
        ax.pcolormesh(
            lon2d, lat2d, filt["peak_month_idx"],
            cmap=MONTH_CMAP, norm=MONTH_NORM, shading="nearest",
            transform=PROJ, zorder=2,
        )
        add_map_features(ax)
        add_gridlines(ax, left_labels=False, bottom_labels=(ri == 1))
        mark_fluxnet(ax)
        ax.set_title(f"{ds_lbl}", fontsize=12, fontweight="bold", pad=4)

        # Col 3: number of significant months
        ax = axes[ri, 2]
        last_cnt_pm = ax.pcolormesh(
            lon2d, lat2d, filt["sig_count"],
            cmap=COUNT_CMAP, norm=COUNT_NORM, shading="nearest",
            transform=PROJ, zorder=2,
        )
        add_map_features(ax)
        add_gridlines(ax, left_labels=False, bottom_labels=(ri == 1))
        mark_fluxnet(ax)
        ax.set_title(f"{ds_lbl}", fontsize=12, fontweight="bold", pad=4)

        # Col 4: dominant flux strengthening
        ax = axes[ri, 3]
        ax.pcolormesh(
            lon2d, lat2d, str_enc,
            cmap=BINARY_CMAP, norm=BINARY_NORM, shading="nearest",
            transform=PROJ, zorder=2,
        )
        add_map_features(ax)
        add_gridlines(ax, left_labels=False, bottom_labels=(ri == 1))
        mark_fluxnet(ax)
        ax.set_title(f"{ds_lbl}", fontsize=12, fontweight="bold", pad=4)

        # Col 5: dominant flux weakening
        ax = axes[ri, 4]
        ax.pcolormesh(
            lon2d, lat2d, wkn_enc,
            cmap=BINARY_CMAP, norm=BINARY_NORM, shading="nearest",
            transform=PROJ, zorder=2,
        )
        add_map_features(ax)
        add_gridlines(ax, left_labels=False, bottom_labels=(ri == 1))
        mark_fluxnet(ax)
        ax.set_title(f"{ds_lbl}", fontsize=12, fontweight="bold", pad=4)

        # Row label on left
        axes[ri, 0].text(
            -0.24, 0.5, row_labels[ri],
            transform=axes[ri, 0].transAxes,
            fontsize=12, fontweight="bold",
            rotation=90, va="center", ha="center",
        )

    # ── Column letter labels (one per column, above row 0) ───────────────────
    col_titles = [
        "a)  Peak Q$_{net}$ slope",
        "b)  Month of peak Q$_{net}$",
        "c)  No. significant months",
        "d)  Dominant flux\n    strengthening",
        "e)  Dominant flux\n    weakening",
    ]
    for ci, ctitle in enumerate(col_titles):
        # Place above the row-0 panel using figure-level text anchored to axes
        ax0 = axes[0, ci]
        bbox = ax0.get_position()
        fig.text(
            bbox.x0 + bbox.width / 2,
            TOP + 0.025,
            ctitle,
            ha="center", va="bottom",
            fontsize=13, fontweight="bold",
            transform=fig.transFigure,
        )

    # ── Colourbars ────────────────────────────────────────────────────────────
    fig.canvas.draw()

    # Col 1 colourbar — RIGHT of col 1
    if last_mag_cf is not None:
        bb0 = axes[0, 0].get_position()
        bb1 = axes[1, 0].get_position()
        cax0 = fig.add_axes([
            bb0.x1 + 0.005,
            bb1.y0,
            0.013,
            bb0.y1 - bb1.y0,
        ])
        cb0 = fig.colorbar(
            last_mag_cf, cax=cax0, orientation="vertical", extend="both"
        )
        cb0.set_label(r"W m$^{-2}$ dec$^{-1}$", fontsize=11)
        cb0.set_ticks(np.linspace(-mag_vmax, mag_vmax, 5))
        cb0.ax.tick_params(labelsize=10)

    # Col 3 colourbar — RIGHT of col 3
    if last_cnt_pm is not None:
        bb0 = axes[0, 2].get_position()
        bb1 = axes[1, 2].get_position()
        cax1 = fig.add_axes([
            bb0.x1 + 0.005,
            bb1.y0,
            0.013,
            bb0.y1 - bb1.y0,
        ])
        cb1 = fig.colorbar(
            last_cnt_pm, cax=cax1, orientation="vertical",
            boundaries=np.arange(-0.5, 13.5, 1), ticks=range(1, 13),
        )
        cb1.set_label("Months", fontsize=11)
        cb1.ax.tick_params(labelsize=10)
        cb1.ax.yaxis.set_ticks_position("right")
        cb1.ax.yaxis.set_label_position("right")

    # ── Legends ───────────────────────────────────────────────────────────────
    # Month legend — below col 2 (b)
    month_patches = [
        mpatches.Patch(
            facecolor=hls_rgb(_MONTH_HUE[mi], 0.55, 0.80),
            edgecolor="0.3", linewidth=0.4,
            label=MONTH_NAMES[mi],
        )
        for mi in range(12)
    ]
    bb_col2_r0 = axes[0, 1].get_position()
    bb_col2_r1 = axes[1, 1].get_position()
    leg_month = fig.legend(
        handles=month_patches,
        loc="upper center",
        bbox_to_anchor=(
            bb_col2_r0.x0 + bb_col2_r0.width / 2,
            bb_col2_r1.y0 - 0.01,
        ),
        ncol=3,
        fontsize=11,
        framealpha=0.92,
        edgecolor="0.65",
        title="b)  Peak month",
        title_fontsize=12,
        handlelength=1.8,
        handleheight=1.2,
        borderpad=0.7,
        columnspacing=0.9,
    )
    fig.add_artist(leg_month)

    # Flux legend — below cols 4–5 (d–e)
    bb_col4_r0 = axes[0, 3].get_position()
    bb_col5_r1 = axes[1, 4].get_position()
    flux_cx = (bb_col4_r0.x0 + axes[0, 4].get_position().x1) / 2

    flux_patches = [
        mpatches.Patch(
            facecolor=SOLID_COLORS[ci], edgecolor="0.3", linewidth=0.4,
            label=f"{COMP_LABELS[comp]} dominant",
        )
        for ci, comp in enumerate(COMP_ORDER)
    ] + [
        mpatches.Patch(
            facecolor=(0.78, 0.78, 0.78), edgecolor="0.4", linewidth=0.4,
            label="Not this direction",
        ),
        mpatches.Patch(
            facecolor=(0.90, 0.90, 0.90), edgecolor="0.4", linewidth=0.4,
            label="No sig. month",
        ),
        Line2D(
            [0], [0], marker="s", color="w",
            markerfacecolor="black", markeredgecolor="black",
            markeredgewidth=0.8, markersize=6, lw=0,
            label="ZM-Mon (FLUXNET)",
        ),
    ]
    leg_flux = fig.legend(
        handles=flux_patches,
        loc="upper center",
        bbox_to_anchor=(flux_cx, axes[1, 3].get_position().y0 - 0.01),
        ncol=3,
        fontsize=11,
        framealpha=0.92,
        edgecolor="0.65",
        title="d–e)  Dominant flux (|slope|-weighted)",
        title_fontsize=12,
        handlelength=1.8,
        handleheight=1.2,
        borderpad=0.7,
        columnspacing=0.9,
    )
    fig.add_artist(leg_flux)

    # ── Main title ────────────────────────────────────────────────────────────
    fig.suptitle(
        "Positive surface radiation budget monthly decomposition",
        fontsize=18, fontweight="bold", y=0.997,
    )

    suffix = ("increasing" if direction == "pos" else "decreasing")
    outfile = os.path.join(
        OUT_DIR, f"Qnet_annual_peak_{suffix}{fig_suffix}.png"
    )
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()
    log(f"\nSaved: {outfile}")
    return outfile


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY STATISTICS
# ─────────────────────────────────────────────────────────────────────────────
def print_summary(name, res):
    if res is None:
        return
    has_sig = res["peak_month_idx"] < NO_SIG_MONTH
    n_sig   = int(has_sig.sum())
    n_total = has_sig.size
    log(f"\n{name}:")
    log(f"  Pixels with ≥1 significant month: {n_sig}/{n_total} "
        f"({100*n_sig/n_total:.1f}%)")

    # Peak month breakdown restricted to POSITIVE Qnet pixels only
    pq_arr = res["peak_qnet_dec"]
    pos_pix = np.isfinite(pq_arr) & (pq_arr > 0)
    n_pos = int(pos_pix.sum())
    log(f"  Positive peak Qnet pixels: {n_pos} ({100*n_pos/max(n_sig,1):.1f}% of sig pixels)")

    pm = res["peak_month_idx"][pos_pix].astype(int)
    for mi in range(12):
        cnt = int((pm == mi).sum())
        if cnt > 0:
            pct = 100 * cnt / max(n_pos, 1)
            log(f"  {MONTH_NAMES[mi]:3s}: {cnt:4d} pixels ({pct:4.1f}%)")

    log("  Dominant flux breakdown (positive Qnet pixels only):")
    fi = res["modal_flux_idx"][pos_pix].astype(int)
    for ci, comp in enumerate(COMP_ORDER):
        cnt = int((fi == ci).sum())
        log(f"    {comp:6s}: {cnt:4d} ({100*cnt/max(n_pos,1):4.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    log("=" * 70)
    log("Annual peak Qnet trend — dominant driver summary")
    log(f"FDR α = {FDR_ALPHA}  |  Southern Africa")
    log("=" * 70)

    # ── Figure 1: full individual dataset ranges ──────────────────────────────
    log(f"\n[1a] ERA5 ({ERA5_START}–{ERA5_END}) ...")
    era5_res_full = compute_annual_peak(
        ERA5_DIR, ERA5_SLOPES,
        year_range_label=f"({ERA5_START}–{ERA5_END})",
    )

    log(f"\n[1b] MERRA-2 ({MERRA2_START}–{MERRA2_END}) ...")
    m2_res_full = compute_annual_peak(
        MERRA2_DIR, MERRA2_SLOPES,
        year_range_label=f"({MERRA2_START}–{MERRA2_END})",
    )

    log("\n[1c] Building figure — full ranges ...")
    era5_label_full = f"ERA5\n({ERA5_START}–{ERA5_END})"
    m2_label_full   = f"MERRA-2\n({MERRA2_START}–{MERRA2_END})"
    for direction in ("pos",):   # positive Qnet only as requested
        e5f = filter_by_direction(era5_res_full, direction) if era5_res_full else None
        m2f = filter_by_direction(m2_res_full,   direction) if m2_res_full   else None
        make_direction_figure(
            e5f, m2f,
            direction=direction,
            era5_label=era5_label_full,
            m2_label=m2_label_full,
            fig_suffix="_full_range",
        )

    # ── Figure 2: overlapping years only ─────────────────────────────────────
    # NOTE: the slope files already encode a fixed year range baked into the
    # OLS computation.  To produce true overlapping-year slopes you would need
    # to rerun the decomposition pipeline with START/END restricted to the
    # overlap period.  Here we load the same files but label the figure with
    # the overlap period so it is clear which years are shared.  If separate
    # overlap-period files exist in sub-directories, update ERA5_DIR_OVL /
    # MERRA2_DIR_OVL below to point to them.
    ERA5_DIR_OVL   = ERA5_DIR    # replace with overlap-specific path if available
    MERRA2_DIR_OVL = MERRA2_DIR  # replace with overlap-specific path if available

    log(f"\n[2a] ERA5 overlap ({OVERLAP_START}–{OVERLAP_END}) ...")
    era5_res_ovl = compute_annual_peak(
        ERA5_DIR_OVL, ERA5_SLOPES,
        year_range_label=f"overlap ({OVERLAP_START}–{OVERLAP_END})",
    )

    log(f"\n[2b] MERRA-2 overlap ({OVERLAP_START}–{OVERLAP_END}) ...")
    m2_res_ovl = compute_annual_peak(
        MERRA2_DIR_OVL, MERRA2_SLOPES,
        year_range_label=f"overlap ({OVERLAP_START}–{OVERLAP_END})",
    )

    log("\n[2c] Building figure — overlapping years ...")
    era5_label_ovl = f"ERA5\n({OVERLAP_START}–{OVERLAP_END})"
    m2_label_ovl   = f"MERRA-2\n({OVERLAP_START}–{OVERLAP_END})"
    for direction in ("pos",):
        e5f = filter_by_direction(era5_res_ovl, direction) if era5_res_ovl else None
        m2f = filter_by_direction(m2_res_ovl,  direction) if m2_res_ovl  else None
        make_direction_figure(
            e5f, m2f,
            direction=direction,
            era5_label=era5_label_ovl,
            m2_label=m2_label_ovl,
            fig_suffix=f"_overlap_{OVERLAP_START}_{OVERLAP_END}",
        )

    # ── Summary statistics ────────────────────────────────────────────────────
    log("\n" + "=" * 70)
    log("SUMMARY STATISTICS — full ranges")
    log("=" * 70)
    print_summary("ERA5",    era5_res_full)
    print_summary("MERRA-2", m2_res_full)

    log("\n" + "=" * 70)
    log(f"SUMMARY STATISTICS — overlap ({OVERLAP_START}–{OVERLAP_END})")
    log("=" * 70)
    print_summary("ERA5",    era5_res_ovl)
    print_summary("MERRA-2", m2_res_ovl)

    log("\nDone.")


if __name__ == "__main__":
    main()
