"""
October SW↓ decadal trend comparison:
ERA5 (1983–2024) | MERRA-2 (1983–2024) | CLARA-A3 (1983–2024) | SARAH-3 (1983–2024)
"""

from __future__ import annotations
import glob, os, warnings
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

ERA5_RAW   = "/ec/res4/scratch/gbr3310/fixed_era5_flux_analysis/1979_2025/global_surface_solar_radiation_downwards.nc"
MERRA2_DIR = "/ec/res4/hpcperm/gbr3310/merra2"
CLARA_DIR  = "/ec/res4/scratch/gbr3310/fixed_era5_flux_analysis/copernicus_eumetsat"
SARAH_DIR  = "/ec/res4/scratch/gbr3310/fixed_era5_flux_analysis/saraha3/ORD66840"
OUT_DIR    = "/ec/res4/scratch/gbr3310/fixed_era5_flux_analysis/1979_2025_merra2grid/comparison_plots"
os.makedirs(OUT_DIR, exist_ok=True)

MONTH       = 10
FDR_ALPHA   = 0.10
DECADE      = 10
CLIM        = 8.0
LEVELS      = np.linspace(-CLIM, CLIM, 21)

ERA5_START,   ERA5_END   = 1983, 2024
MERRA2_START, MERRA2_END = 1983, 2024
CLARA_START,  CLARA_END  = 1983, 2024
SARAH_START,  SARAH_END  = 1983, 2024

M2_LAT = np.arange(-90.0,  90.01, 0.5)
M2_LON = np.arange(-180.0, 180.0, 0.625)
SECS_PER_MONTH = 86400.0

def log(msg): print(msg, flush=True)

def get_latlon_names(da):
    for ln, lo in (("lat", "lon"), ("latitude", "longitude")):
        if ln in da.coords: return ln, lo
    raise KeyError(f"No lat/lon coords in {list(da.coords)}")

def fdr_mask(pval_array):
    p = np.asarray(pval_array).flatten()
    finite = np.isfinite(p)
    reject = np.zeros(p.shape, dtype=bool)
    if finite.sum() > 0:
        rej, *_ = multipletests(p[finite], alpha=FDR_ALPHA, method="fdr_bh")
        reject[finite] = rej
    return reject.reshape(pval_array.shape)

def add_map_features(ax):
    ax.add_feature(cfeature.OCEAN.with_scale("110m"), zorder=1, facecolor="white")
    ax.add_feature(cfeature.LAKES.with_scale("110m"), zorder=1, facecolor="white")
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
    gl.xlabel_style  = {"size": 9, "color": "black",
                        "rotation": 45, "fontfamily": "sans-serif"}
    gl.ylabel_style  = {"size": 9, "color": "black",
                        "fontfamily": "sans-serif"}

def plot_panel(ax, slope_da, pval_da, col_idx, row_idx,
               nrows, ncols, label, subtitle=None):
    ln, lo = get_latlon_names(slope_da)
    lons = slope_da[lo].values
    lats = slope_da[ln].values
    sv   = slope_da.values * DECADE
    pv   = pval_da.values
    sig  = fdr_mask(pv)
    ax.contourf(lons, lats, sv, levels=LEVELS, colors="#CCCCCC",
                antialiased=False, transform=ccrs.PlateCarree())
    cf = ax.contourf(lons, lats, np.where(sig, sv, np.nan),
                     levels=LEVELS, cmap="RdBu_r", extend="both",
                     antialiased=False, transform=ccrs.PlateCarree())
    add_map_features(ax)
    add_gridlines(ax, left_labels=(col_idx == 0),
                  bottom_labels=(row_idx == nrows - 1))
    ax.text(0.02, 0.97, label, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top", ha="left",
            bbox=dict(facecolor="white", edgecolor="none",
                      alpha=0.7, boxstyle="round,pad=0.15"))
    if subtitle:
        ax.set_title(subtitle, fontsize=12, fontstyle="normal",
                     pad=4, color="#333333")
    return cf

def lag1_autocorr(ts):
    ts = np.asarray(ts, dtype=float)
    ok = np.isfinite(ts)
    if ok.sum() < 3: return np.nan
    t0, t1 = ts[:-1], ts[1:]
    m = np.isfinite(t0) & np.isfinite(t1)
    return float(np.corrcoef(t1[m], t0[m])[0, 1]) if m.sum() >= 3 else np.nan

def ols_pixel(ts, years):
    y = np.asarray(ts, dtype=float); x = np.asarray(years, dtype=float)
    ok = np.isfinite(y) & np.isfinite(x); n = ok.sum()
    if n < 8: return np.nan, np.nan
    y, x = y[ok], x[ok]; x = x - x.mean()
    slope, intercept = np.polyfit(x, y, 1)
    resid  = y - (slope * x + intercept)
    dof    = max(n - 2, 1)
    sigma2 = np.sum(resid**2) / dof; sxx = np.sum(x**2)
    if sxx == 0: return np.nan, np.nan
    se   = np.sqrt(sigma2 / sxx)
    rho1 = lag1_autocorr(resid)
    if np.isnan(rho1): neff = float(n)
    else:
        denom = 1.0 + rho1
        neff  = float(n) * (1.0 - rho1) / denom if denom > 1e-6 else float(n)
        neff  = max(neff, 3.0)
    se_adj = se * np.sqrt(float(n) / neff)
    df     = max(int(np.floor(neff - 2)), 1)
    tstat  = slope / se_adj if se_adj > 0 else np.nan
    pval   = 2.0 * tdist.sf(abs(tstat), df=df)
    return float(slope), float(pval)

def compute_ols_field(da, years):
    years_f = years.astype(float)
    slope, pval = xr.apply_ufunc(
        ols_pixel, da, input_core_dims=[["year"]],
        output_core_dims=[[], []], vectorize=True,
        output_dtypes=[float, float], kwargs={"years": years_f})
    return slope.compute(), pval.compute()

def regrid_to_m2(da, lat_name, lon_name):
    da_rg = da.interp({lat_name: M2_LAT, lon_name: M2_LON},
                      method="linear",
                      kwargs={"bounds_error": False, "fill_value": None})
    rename = {}
    if lat_name != "lat": rename[lat_name] = "lat"
    if lon_name != "lon": rename[lon_name] = "lon"
    if rename: da_rg = da_rg.rename(rename)
    return da_rg

def load_era5_trends(fpath):
    log("Loading ERA5 ssrd ...")
    if not os.path.exists(fpath): log(f"  [ERROR] not found: {fpath}"); return None
    ds = xr.open_dataset(fpath); da = ds["ssrd"]
    time_coord = "valid_time" if "valid_time" in da.coords else "time"
    times = da[time_coord]
    mask  = (times.dt.month == MONTH) & (times.dt.year >= ERA5_START) & (times.dt.year <= ERA5_END)
    da_oct = da.sel({time_coord: mask})
    valid_years = da_oct[time_coord].dt.year.values
    log(f"  ERA5: {len(valid_years)} October months ({valid_years.min()}–{valid_years.max()})")
    da_oct = da_oct / SECS_PER_MONTH
    da_oct = da_oct.assign_coords({time_coord: valid_years}).rename({time_coord: "year"})
    lat_name = "latitude" if "latitude" in da_oct.coords else "lat"
    lon_name = "longitude" if "longitude" in da_oct.coords else "lon"
    da_oct = da_oct.sortby(lat_name)
    if da_oct[lon_name].values.max() > 180.5:
        da_oct = da_oct.assign_coords({lon_name: ((da_oct[lon_name]+180)%360)-180})
        da_oct = da_oct.sortby(lon_name)
    log("  Regridding ERA5 ...")
    da_rg = regrid_to_m2(da_oct, lat_name, lon_name)
    log("  Computing OLS for ERA5 ...")
    slope, pval = compute_ols_field(da_rg, valid_years)
    slope = slope.assign_coords(lat=M2_LAT, lon=M2_LON)
    pval  = pval.assign_coords(lat=M2_LAT, lon=M2_LON)
    ds.close(); return slope, pval

def load_merra2_trends(merra2_dir):
    log("Loading MERRA-2 SWGDN ...")
    pattern   = os.path.join(merra2_dir, "MERRA2_*.tavgM_2d_rad_Nx.*.nc4")
    all_files = sorted(glob.glob(pattern))
    all_files = [f for f in all_files if not any(f.endswith(f".nc4.{i}") for i in range(1,10))]
    if not all_files: log(f"  [ERROR] No MERRA-2 files in {merra2_dir}"); return None
    oct_files, valid_years = [], []
    for f in all_files:
        stem = os.path.basename(f).replace(".nc4",""); yyyymm = stem.split(".")[-1]
        if len(yyyymm) != 6: continue
        yr, mo = int(yyyymm[:4]), int(yyyymm[4:])
        if mo == MONTH and MERRA2_START <= yr <= MERRA2_END:
            oct_files.append(f); valid_years.append(yr)
    if not oct_files: log("  [ERROR] No October MERRA-2 files found"); return None
    valid_years = np.array(valid_years, dtype=int)
    log(f"  MERRA-2: {len(oct_files)} October files ({valid_years.min()}–{valid_years.max()})")
    slices = []
    for f in oct_files:
        try:
            with xr.open_dataset(f) as ds:
                da = ds["SWGDN"].squeeze(drop=True).load()
                da = da.where(da < 1e14)
        except Exception as e: log(f"  [WARN] {f}: {e}"); continue
        slices.append(da)
    if not slices: log("  [ERROR] No MERRA-2 slices loaded"); return None
    stacked = xr.concat(slices, dim="year"); stacked["year"] = valid_years
    stacked = stacked.sortby("lat").sortby("lon")
    log("  Computing OLS for MERRA-2 ...")
    slope, pval = compute_ols_field(stacked, valid_years)
    slope = slope.assign_coords(lat=stacked.lat.values, lon=stacked.lon.values)
    pval  = pval.assign_coords(lat=stacked.lat.values, lon=stacked.lon.values)
    return slope, pval

def load_clara_trends(clara_dir):
    log("Loading CLARA-A3 SIS ...")
    slices, valid_years = [], []
    for yr in range(CLARA_START, CLARA_END+1):
        pattern = os.path.join(clara_dir, f"SISmm{yr}1001*.nc")
        files   = sorted(glob.glob(pattern))
        if not files: log(f"  [WARN] No CLARA file for {yr}"); continue
        with xr.open_dataset(files[0]) as ds:
            varname = "SIS" if "SIS" in ds else "sis"
            if varname not in ds: continue
            da = ds[varname].squeeze().load(); da = da.where(da > -900)
        slices.append(da); valid_years.append(yr)
    if not slices: log("  [ERROR] No CLARA files loaded"); return None
    log(f"  CLARA: {len(valid_years)} years ({valid_years[0]}–{valid_years[-1]})")
    stacked = xr.concat(slices, dim="year"); stacked["year"] = np.array(valid_years, dtype=int)
    lon_name = "lon" if "lon" in stacked.coords else "longitude"
    lat_name = "lat" if "lat" in stacked.coords else "latitude"
    if stacked[lon_name].values.max() > 180.5:
        stacked = stacked.assign_coords({lon_name: ((stacked[lon_name]+180)%360)-180})
        stacked = stacked.sortby(lon_name)
    stacked = stacked.sortby(lat_name)
    log("  Regridding CLARA ...")
    stacked_rg = regrid_to_m2(stacked, lat_name, lon_name)
    log("  Computing OLS for CLARA ...")
    slope, pval = compute_ols_field(stacked_rg, np.array(valid_years, dtype=int))
    slope = slope.assign_coords(lat=M2_LAT, lon=M2_LON)
    pval  = pval.assign_coords(lat=M2_LAT, lon=M2_LON)
    return slope, pval

def load_sarah_trends(sarah_dir):
    log("Loading SARAH-3 SIS ...")
    slices, valid_years = [], []
    for yr in range(SARAH_START, SARAH_END+1):
        pattern = os.path.join(sarah_dir, f"SISmm{yr}{MONTH:02d}01*.nc")
        files   = sorted(glob.glob(pattern))
        if not files: log(f"  [WARN] No SARAH file for {yr}"); continue
        try:
            with xr.open_dataset(files[0]) as ds:
                varname = "SIS" if "SIS" in ds else "sis"
                if varname not in ds: continue
                da = ds[varname].squeeze().load(); da = da.where(da > -900)
        except Exception as e: log(f"  [WARN] {files[0]}: {e}"); continue
        slices.append(da); valid_years.append(yr)
    if not slices: log("  [ERROR] No SARAH files loaded"); return None
    log(f"  SARAH-3: {len(valid_years)} years ({valid_years[0]}–{valid_years[-1]})")
    stacked = xr.concat(slices, dim="year"); stacked["year"] = np.array(valid_years, dtype=int)
    lon_name = "lon" if "lon" in stacked.coords else "longitude"
    lat_name = "lat" if "lat" in stacked.coords else "latitude"
    if stacked[lon_name].values.max() > 180.5:
        stacked = stacked.assign_coords({lon_name: ((stacked[lon_name]+180)%360)-180})
        stacked = stacked.sortby(lon_name)
    stacked = stacked.sortby(lat_name)
    log("  Regridding SARAH-3 ...")
    stacked_rg = stacked.interp({lat_name: M2_LAT, lon_name: M2_LON},
                                method="linear",
                                kwargs={"bounds_error": False, "fill_value": np.nan})
    rename = {}
    if lat_name != "lat": rename[lat_name] = "lat"
    if lon_name != "lon": rename[lon_name] = "lon"
    if rename: stacked_rg = stacked_rg.rename(rename)
    log("  Computing OLS for SARAH-3 ...")
    slope, pval = compute_ols_field(stacked_rg, np.array(valid_years, dtype=int))
    slope = slope.assign_coords(lat=M2_LAT, lon=M2_LON)
    pval  = pval.assign_coords(lat=M2_LAT, lon=M2_LON)
    return slope, pval

def make_figure():
    matplotlib.rcParams.update({
        "font.family":     "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.titlesize":  15,
    })
    PROJ  = ccrs.PlateCarree()
    nrows, ncols = 2, 2

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6.5*ncols, 3.8*nrows),
        subplot_kw=dict(projection=PROJ),
    )
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.87,
        bottom=0.14,
        wspace=0.02,
        hspace=0.12,
    )

    last_cf = None

    # Row 0 col 0: ERA5
    log("\n--- ERA5 ---")
    era5_result = load_era5_trends(ERA5_RAW)
    ax = axes[0, 0]
    if era5_result is not None:
        slope_da, pval_da = era5_result
        cf = plot_panel(ax, slope_da, pval_da, 0, 0, nrows, ncols, "a)")
        last_cf = cf
        ax.set_title("ERA5", fontsize=15, fontweight="bold", pad=6)
    else:
        ax.set_visible(False)

    # Row 0 col 1: MERRA-2
    log("\n--- MERRA-2 ---")
    merra2_result = load_merra2_trends(MERRA2_DIR)
    ax = axes[0, 1]
    if merra2_result is not None:
        slope_da, pval_da = merra2_result
        cf = plot_panel(ax, slope_da, pval_da, 1, 0, nrows, ncols, "b)")
        last_cf = cf
        ax.set_title("MERRA-2", fontsize=15, fontweight="bold", pad=6)
    else:
        ax.set_visible(False)

    # Row 1 col 0: CLARA-A3
    log("\n--- CLARA-A3 ---")
    clara_result = load_clara_trends(CLARA_DIR)
    ax = axes[1, 0]
    if clara_result is not None:
        slope_c, pval_c = clara_result
        cf = plot_panel(ax, slope_c, pval_c, 0, 1, nrows, ncols, "c)")
        last_cf = cf
        ax.set_title("CLARA-A3", fontsize=15, fontweight="bold", pad=10)
    else:
        ax.set_visible(False)

    # Row 1 col 1: SARAH-3
    log("\n--- SARAH-3 ---")
    sarah_result = load_sarah_trends(SARAH_DIR)
    ax = axes[1, 1]
    if sarah_result is not None:
        slope_s, pval_s = sarah_result
        cf = plot_panel(ax, slope_s, pval_s, 1, 1, nrows, ncols, "d)",
                        subtitle="Domain limited to ±65° lat/lon (Meteosat SEVIRI)")
        last_cf = cf
        ax.set_title("SARAH-3", fontsize=15, fontweight="bold", pad=10)
    else:
        ax.set_visible(False)

    # Row labels — placed dynamically using axes bounding boxes
    fig.canvas.draw()   # force layout so bbox is populated
    for ri, lbl in enumerate(["Reanalyses", "Satellite CDRs"]):
        ax_left = axes[ri, 0]
        bbox    = ax_left.get_position()          # in figure coords
        y_ctr   = bbox.y0 + bbox.height / 2.0    # vertical centre of map
        fig.text(0.01, y_ctr, lbl,
                 fontsize=12, fontweight="bold",
                 rotation=90, va="center", ha="center")

    # Colourbar — moved down to give lat labels room
    if last_cf is not None:
        cbar_ax = fig.add_axes([0.10, 0.05, 0.80, 0.022])
        cbar = fig.colorbar(last_cf, cax=cbar_ax,
                            orientation="horizontal", extend="both")
        cbar.set_label(r"W m$^{-2}$ dec$^{-1}$", fontsize=11)
        cbar.ax.tick_params(labelsize=10)

    fig.suptitle(
        "October decadal OLS trends in downwelling SW radiation (1983-2024)\n"
        "Grey: fails FDR significance (α = 0.10)  ",
        fontsize=15, fontweight="bold", y=0.975,
    )

    outfile = os.path.join(OUT_DIR,
                           "ERA5_MERRA2_CLARA_SARAH_October_SW_trends.png")
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()
    log(f"\nSaved: {outfile}")

if __name__ == "__main__":
    log("="*70)
    log("October SW↓ trends: ERA5 | MERRA-2 | CLARA-A3 | SARAH-3")
    log(f"Common period: {ERA5_START}–{ERA5_END}")
    log("="*70)
    make_figure()
    log("\nDone.")
