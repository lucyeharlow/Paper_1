"""
Global October decadal trend comparison: ERA5 vs MERRA-2 vs CLARA-A3
=====================================================================
3 rows × 5 columns:
  Row 0 : ERA5   (1979–2025)
  Row 1 : MERRA-2 (1980–2025)
  Row 2 : CLARA-A3 / CM SAF (1979–2020)  — only Qnet (SRB), SW↓ (SIS), LW↓ (SDL)

CLARA-A3 files (0.25° global):
  SISmm{YEAR}1001...GL.nc  → SIS (surface incoming shortwave, W m-2)
  SDLmm{YEAR}1001...GL.nc  → SDL (surface downwelling LW,     W m-2)
  SRBmm{YEAR}1001...GL.nc  → SRB (surface net radiation,      W m-2)

CLARA is bilinearly regridded to the MERRA-2 0.5°×0.625° grid and OLS
trends (with lag-1 autocorrelation effective-N correction) are computed
pixel-wise over 1979–2020.
"""

from __future__ import annotations

import glob
import os
import warnings
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mplticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter
from scipy.stats import t as tdist
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ERA5_DIR   = "/ec/res4/scratch/gbr3310/fixed_era5_flux_analysis/1979_2025_merra2grid/global/OLS_trends"
MERRA2_DIR = "/ec/res4/scratch/gbr3310/fixed_era5_flux_analysis/1979_2025_merra2/global"
CLARA_DIR  = "/ec/res4/scratch/gbr3310/fixed_era5_flux_analysis/copernicus_eumetsat"
OUT_DIR    = "/ec/res4/scratch/gbr3310/fixed_era5_flux_analysis/1979_2025_merra2grid/comparison_plots"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MONTH       = 10
FDR_ALPHA   = 0.10
DECADE      = 10
CLIM        = 8.0
LEVELS      = np.linspace(-CLIM, CLIM, 21)
CLARA_START = 1979
CLARA_END   = 2020

# MERRA-2 global grid (target for CLARA regridding)
M2_LAT = np.arange(-90.0,  90.01, 0.5)
M2_LON = np.arange(-180.0, 180.0, 0.625)

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------
COLUMNS = [
    {
        "title"          : r"Net Surface Radiation $Q_{net}$",
        "col_letter"     : "a",
        "era5_slope"     : "qnet_slope_by_month.nc",
        "era5_pval"      : "qnet_pvalue_by_month.nc",
        "m2_slope"       : "qnet_slope_by_month.nc",
        "m2_pval"        : "qnet_pvalue_by_month.nc",
        "clara_pfx"      : "SRB",
        "clara_subtitle" : "Net radiation balance (SRB)\nnot equivalent to ERA5/MERRA-2 Qnet",
    },
    {
        "title"          : r"Downward SW Radiation $S_\downarrow$",
        "col_letter"     : "b",
        "era5_slope"     : "ssrd_slope_by_month.nc",
        "era5_pval"      : "ssrd_pvalue_by_month.nc",
        "m2_slope"       : "SWGDN_slope_by_month.nc",
        "m2_pval"        : "SWGDN_pvalue_by_month.nc",
        "clara_pfx"      : "SIS",
        "clara_subtitle" : None,
    },
    {
        "title"          : r"Downward LW Radiation $L_\downarrow$",
        "col_letter"     : "c",
        "era5_slope"     : "strd_slope_by_month.nc",
        "era5_pval"      : "strd_pvalue_by_month.nc",
        "m2_slope"       : "LWGAB_slope_by_month.nc",
        "m2_pval"        : "LWGAB_pvalue_by_month.nc",
        "clara_pfx"      : "SDL",
        "clara_subtitle" : None,
    },
    {
        "title"          : r"Latent Heat Flux $H_L$",
        "col_letter"     : "d",
        "era5_slope"     : "lhf_slope_by_month.nc",
        "era5_pval"      : "lhf_pvalue_by_month.nc",
        "m2_slope"       : "lhf_slope_by_month.nc",
        "m2_pval"        : "lhf_pvalue_by_month.nc",
        "clara_pfx"      : None,   # blank for CLARA row
        "clara_subtitle" : None,
    },
    {
        "title"          : r"Sensible Heat Flux $H_S$",
        "col_letter"     : "e",
        "era5_slope"     : "shf_slope_by_month.nc",
        "era5_pval"      : "shf_pvalue_by_month.nc",
        "m2_slope"       : "shf_slope_by_month.nc",
        "m2_pval"        : "shf_pvalue_by_month.nc",
        "clara_pfx"      : None,   # blank for CLARA row
        "clara_subtitle" : None,
    },
]

ROW_LABELS = [
    "ERA5\n(1979–2025)",
    "MERRA-2\n(1980–2025)",
    "CLARA-A3\n(CM SAF)\n1979–2020",
]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def get_latlon_names(da):
    for ln, lo in (("lat", "lon"), ("latitude", "longitude")):
        if ln in da.coords:
            return ln, lo
    raise KeyError(f"No lat/lon coords in {list(da.coords)}")


def load_month(fpath, month=MONTH):
    da = xr.open_dataarray(fpath)
    if "month" in da.dims:
        da = da.sel(month=month)
    ln, lo = get_latlon_names(da)
    da = da.sortby(ln)
    lov = da[lo].values
    if lov.max() > 180.5:
        da = da.assign_coords({lo: ((da[lo] + 180) % 360) - 180})
        da = da.sortby(lo)
    return da


def fdr_mask(pval_array):
    p      = np.asarray(pval_array).flatten()
    finite = np.isfinite(p)
    reject = np.zeros(p.shape, dtype=bool)
    if finite.sum() > 0:
        rej, *_ = multipletests(p[finite], alpha=FDR_ALPHA, method="fdr_bh")
        reject[finite] = rej
    return reject.reshape(pval_array.shape)


def add_map_features(ax):
    ax.add_feature(cfeature.OCEAN.with_scale("110m"),
                   zorder=1, facecolor="white")
    ax.add_feature(cfeature.LAKES.with_scale("110m"),
                   zorder=1, facecolor="white")
    ax.coastlines(resolution="110m", linewidth=0.4)
    ax.set_global()


def add_gridlines(ax, left_labels=False, bottom_labels=False):
    gl = ax.gridlines(draw_labels=True, linewidth=0.3,
                      color="black", alpha=0.45, linestyle=":")
    gl.top_labels    = False
    gl.right_labels  = False
    gl.left_labels   = left_labels
    gl.bottom_labels = bottom_labels
    gl.xlocator      = mplticker.FixedLocator([-120, -60, 0, 60, 120])
    gl.ylocator      = mplticker.FixedLocator([-60, -30, 0, 30, 60])
    gl.xformatter    = LongitudeFormatter()
    gl.yformatter    = LatitudeFormatter()
    gl.xlabel_style  = {"size": 7, "color": "black",
                        "rotation": 45, "fontfamily": "sans-serif"}
    gl.ylabel_style  = {"size": 7, "color": "black",
                        "fontfamily": "sans-serif"}


def plot_panel(ax, slope_da, pval_da, col_idx, row_idx, nrows):
    ln, lo = get_latlon_names(slope_da)
    lons   = slope_da[lo].values
    lats   = slope_da[ln].values
    sv     = slope_da.values * DECADE
    pv     = pval_da.values
    sig    = fdr_mask(pv)

    ax.contourf(lons, lats, sv,
                levels=LEVELS, colors="#CCCCCC",
                antialiased=False, transform=ccrs.PlateCarree())
    cf = ax.contourf(lons, lats, np.where(sig, sv, np.nan),
                     levels=LEVELS, cmap="RdBu_r", extend="both",
                     antialiased=False, transform=ccrs.PlateCarree())
    add_map_features(ax)
    add_gridlines(ax,
                  left_labels=(col_idx == 0),
                  bottom_labels=(row_idx == nrows - 1))
    return cf


# ---------------------------------------------------------------------------
# OLS trend with lag-1 autocorrelation correction
# ---------------------------------------------------------------------------

def lag1_autocorr(ts: np.ndarray) -> float:
    ts = np.asarray(ts, dtype=float)
    ok = np.isfinite(ts)
    if ok.sum() < 3:
        return np.nan
    t0, t1 = ts[:-1], ts[1:]
    m = np.isfinite(t0) & np.isfinite(t1)
    return float(np.corrcoef(t1[m], t0[m])[0, 1]) if m.sum() >= 3 else np.nan


def ols_pixel(ts: np.ndarray, years: np.ndarray):
    y = np.asarray(ts, dtype=float)
    x = np.asarray(years, dtype=float)
    ok = np.isfinite(y) & np.isfinite(x)
    n  = ok.sum()
    if n < 8:
        return np.nan, np.nan
    y, x = y[ok], x[ok]
    x = x - x.mean()
    slope, intercept = np.polyfit(x, y, 1)
    resid  = y - (slope * x + intercept)
    dof    = max(n - 2, 1)
    sigma2 = np.sum(resid ** 2) / dof
    sxx    = np.sum(x ** 2)
    if sxx == 0:
        return np.nan, np.nan
    se   = np.sqrt(sigma2 / sxx)
    rho1 = lag1_autocorr(resid)
    if np.isnan(rho1):
        neff = float(n)
    else:
        denom = 1.0 + rho1
        neff  = float(n) * (1.0 - rho1) / denom if denom > 1e-6 else float(n)
        neff  = max(neff, 3.0)
    se_adj = se * np.sqrt(float(n) / neff)
    df     = max(int(np.floor(neff - 2)), 1)
    tstat  = slope / se_adj if se_adj > 0 else np.nan
    pval   = 2.0 * tdist.sf(abs(tstat), df=df)
    return float(slope), float(pval)


def compute_ols_field(da: xr.DataArray, years: np.ndarray):
    """
    Vectorised OLS over the 'year' dimension of a (year, lat, lon) DataArray.
    Returns (slope_da, pval_da) on the same lat/lon grid.
    """
    years_f = years.astype(float)

    def _ols(ts):
        return ols_pixel(ts, years_f)

    slope, pval = xr.apply_ufunc(
        _ols, da,
        input_core_dims=[["year"]],
        output_core_dims=[[], []],
        vectorize=True,
        output_dtypes=[float, float],
    )
    return slope.compute(), pval.compute()


# ---------------------------------------------------------------------------
# CLARA-A3 loading, regridding and trend computation
# ---------------------------------------------------------------------------

def _load_clara_year(clara_dir: str, prefix: str, year: int):
    """Load one annual CLARA file, return DataArray or None."""
    pattern = os.path.join(clara_dir, f"{prefix}mm{year}1001*.nc")
    files   = sorted(glob.glob(pattern))
    if not files:
        return None
    with xr.open_dataset(files[0]) as ds:
        # Variable name may match prefix exactly or be lowercase
        varname = prefix if prefix in ds else prefix.lower()
        if varname not in ds:
            log(f"    [WARN] '{prefix}' not found in {files[0]}")
            return None
        da = ds[varname].squeeze().load()
        fv = ds[varname].attrs.get("_FillValue", 1e15)
        da = da.where(np.abs(da) < 1e14)
    return da


def load_clara_trends(clara_dir: str, prefix: str) -> tuple[xr.DataArray, xr.DataArray] | None:
    """
    Load all annual CLARA files for `prefix`, regrid to MERRA-2 grid,
    compute pixel-wise OLS trend with lag-1 correction.
    Returns (slope_da, pval_da) or None if no files found.
    """
    log(f"  Loading CLARA {prefix} ...")
    slices, valid_years = [], []
    for yr in range(CLARA_START, CLARA_END + 1):
        da = _load_clara_year(clara_dir, prefix, yr)
        if da is not None:
            slices.append(da)
            valid_years.append(yr)

    if not slices:
        log(f"  [WARN] No CLARA files found for {prefix}")
        return None

    log(f"    {prefix}: {len(valid_years)} years ({valid_years[0]}–{valid_years[-1]})")

    # Stack into (year, lat, lon)
    stacked = xr.concat(slices, dim="year")
    stacked["year"] = np.array(valid_years, dtype=int)

    # Normalise CLARA lons to –180..180 if 0..360
    lon_name = "lon" if "lon" in stacked.coords else "longitude"
    lat_name = "lat" if "lat" in stacked.coords else "latitude"
    if stacked[lon_name].values.max() > 180.5:
        stacked = stacked.assign_coords(
            {lon_name: ((stacked[lon_name] + 180) % 360) - 180}
        )
        stacked = stacked.sortby(lon_name)
    stacked = stacked.sortby(lat_name)

    # Bilinear regrid to MERRA-2 grid
    log(f"    Regridding {prefix} to MERRA-2 grid ...")
    stacked_rg = stacked.interp(
        {lat_name: M2_LAT, lon_name: M2_LON},
        method="linear",
        kwargs={"bounds_error": False, "fill_value": None},
    )
    # Rename coords to lat/lon if needed
    rename = {}
    if lat_name != "lat": rename[lat_name] = "lat"
    if lon_name != "lon": rename[lon_name] = "lon"
    if rename:
        stacked_rg = stacked_rg.rename(rename)

    log(f"    Computing OLS trends for {prefix} ...")
    years_arr = np.array(valid_years, dtype=int)
    slope, pval = compute_ols_field(stacked_rg, years_arr)

    # Attach lat/lon coords so plot_panel can find them
    slope = slope.assign_coords(lat=M2_LAT, lon=M2_LON)
    pval  = pval.assign_coords(lat=M2_LAT, lon=M2_LON)

    log(f"    {prefix} mean slope: {float(slope.mean()):.4f} W m-2 yr-1")
    return slope, pval


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def make_figure():
    matplotlib.rcParams.update({
        "font.family":     "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.titlesize":  12,
    })

    # Pre-compute all CLARA trends so we only load each prefix once
    clara_prefixes = {
        col["clara_pfx"]
        for col in COLUMNS
        if col["clara_pfx"] is not None
    }
    clara_cache: dict[str, tuple | None] = {}
    for pfx in sorted(clara_prefixes):
        clara_cache[pfx] = load_clara_trends(CLARA_DIR, pfx)

    ncols = len(COLUMNS)
    nrows = 3   # ERA5 | MERRA-2 | CLARA-A3
    PROJ  = ccrs.PlateCarree()

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.5 * ncols, 3.2 * nrows),
        subplot_kw=dict(projection=PROJ),
    )
    fig.subplots_adjust(
        left=0.07, right=0.98,
        top=0.88, bottom=0.12,
        wspace=0.05, hspace=0.22,
    )

    last_cf = None

    for col_idx, col in enumerate(COLUMNS):

        # ── Rows 0 & 1: ERA5 and MERRA-2 ────────────────────────────────
        for row_idx, (src_dir, sk, pk) in enumerate([
            (ERA5_DIR,   col["era5_slope"], col["era5_pval"]),
            (MERRA2_DIR, col["m2_slope"],   col["m2_pval"]),
        ]):
            ax = axes[row_idx, col_idx]
            sp = os.path.join(src_dir, sk)
            pp = os.path.join(src_dir, pk)

            if not os.path.exists(sp) or not os.path.exists(pp):
                ax.set_visible(False)
                log(f"  MISSING: {sp}  or  {pp}")
                continue

            log(f"  Loading [{ROW_LABELS[row_idx].splitlines()[0]}] {sk} ...")
            slope_da = load_month(sp)
            pval_da  = load_month(pp)
            cf = plot_panel(ax, slope_da, pval_da, col_idx, row_idx, nrows)
            last_cf = cf

            if row_idx == 0:
                ax.set_title(
                    f"{col['col_letter']})  {col['title']}",
                    fontsize=8, fontweight="bold", pad=14,
                )

        # ── Row 2: CLARA-A3 ──────────────────────────────────────────────
        ax_c  = axes[2, col_idx]
        pfx   = col["clara_pfx"]

        if pfx is None or clara_cache.get(pfx) is None:
            # Blank panel with a "no data" note
            ax_c.set_visible(False)
            continue

        slope_c, pval_c = clara_cache[pfx]
        cf = plot_panel(ax_c, slope_c, pval_c, col_idx, 2, nrows)
        last_cf = cf

        # Subtitle for CLARA Qnet panel noting it's not equivalent
        subtitle = col.get("clara_subtitle")
        if subtitle:
            ax_c.set_title(
                subtitle,
                fontsize=7, fontweight="bold", fontstyle="italic",
                pad=6, color="#333333",
            )

    # ── Row labels ────────────────────────────────────────────────────────
    for ri, lbl in enumerate(ROW_LABELS):
        axes[ri, 0].text(
            -0.07, 0.5, lbl,
            transform=axes[ri, 0].transAxes,
            fontsize=8, fontweight="bold",
            rotation=90, va="center", ha="center",
        )

    # ── Shared colourbar ──────────────────────────────────────────────────
    if last_cf is not None:
        cbar_ax = fig.add_axes([0.10, 0.02, 0.80, 0.020])
        cbar = fig.colorbar(last_cf, cax=cbar_ax,
                            orientation="horizontal", extend="both")
        cbar.set_label(r"W m$^{-2}$ dec$^{-1}$", fontsize=11)
        cbar.ax.tick_params(labelsize=10)

    # ── Suptitle ──────────────────────────────────────────────────────────
    fig.suptitle(
        "Global October rate of decadal change\n"
        "ERA5 (1979–2025)  |  MERRA-2 (1980–2025)  |  "
        "CLARA-A3 CM SAF (1979–2020)  |  Grey: fails FDR correction (α = 0.10)",
        fontsize=10, fontweight="bold", y=0.975,
    )

    outfile = os.path.join(
        OUT_DIR,
        "ERA5_vs_MERRA2_vs_CLARA_global_October_trends.png",
    )
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()
    log(f"\nSaved: {outfile}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    log("=" * 70)
    log("ERA5 vs MERRA-2 vs CLARA-A3 — Global October decadal trends")
    log("=" * 70)
    make_figure()
    log("\nDone.")
