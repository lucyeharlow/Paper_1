"""
plot_cf_validation_overlay.py
==============================
Standalone version of the cross-dataset overlay panel from the
CLARA-A3 / MODIS comparison figure.

Three panels (one per lat band), each showing anomaly timeseries for:
  - CLARA-A3
  - MODIS Terra
  - MODIS Aqua
  - Terra+Aqua combined
  - CERES EBAF Ed4.2.1 cloud area fraction (day + night), from
    data/ceres/CERES_EBAF_Ed4.2.1_Subset_*.nc (cldarea_total_daynight_mon, %, 1 x 1 deg).
    Note: the CERES cloud fraction is retrieved from MODIS on the same Terra and Aqua
    satellites, so it is not independent of the MODIS series.

Common period: 2002–2024. Land only. 10°E–38°E.
Output: single figure at 300 dpi.

Statistics printed to stdout (bands 2 and 3 only, 2002–2024):
  - Per-dataset OLS trend (per decade) and p-value
  - Cross-dataset mean trend and combined p-value (Fisher's method; the datasets are not
    independent - CERES and MODIS share the same satellites - so treat it as indicative)
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
import matplotlib.ticker as mticker
from scipy import stats
from scipy.stats import t as tdist, chi2

try:
    from pyhdf.SD import SD, SDC
except ImportError:
    raise ImportError("pyhdf not found — run: {insert} install pyhdf")

warnings.filterwarnings("ignore")

# Paths 
clara_dir = Path("/soge-home/users/jesu4127/data/clara_a3")
terra_dir = Path("/soge-home/users/jesu4127/data/modis/MOD08_M3")
aqua_dir  = Path("/soge-home/users/jesu4127/data/modis/MYD08_M3")
ceres_glob = "/soge-home/users/jesu4127/data/ceres/CERES_EBAF_Ed4.2.1_Subset_*.nc"
ceres_var  = "cldarea_total_daynight_mon"
lsm_file   = None     # optional land-sea mask file (variable lsm/land/...); None -> cartopy
out_dir  = Path("/soge-home/users/jesu4127/Paper_1/ceres/plots")
out_dir.mkdir(parents=True, exist_ok=True)

common_y1, common_y2 = 2002, 2024
clara_y1,  clara_y2  = 1982, 2024
terra_y1,  terra_y2  = 2000, 2024
aqua_y1,   aqua_y2   = 2002, 2024
ceres_y1,  ceres_y2  = 2000, 2025       # CERES EBAF starts March 2000

lon_w, lon_e= 10.0, 38.0
lat_s, lat_n = -35.0, 10.0

bands = [
    ("10°N–0°",    0.0,  10.0),
    ("0°–25°S",  -25.0,   0.0),
    ("25°S–35°S",-35.0, -25.0),
]
# Indices for the two southern bands we report stats on
stats_bands_indicies = [1, 2]   # 0°–25°S and 25°S–35°S

modis_var     = "Cloud_Fraction_Mean_Mean"
leap_years     = {2000, 2004, 2008, 2012, 2016, 2020, 2024}
oct_doy_noleap = "274"
oct_doy_leap   = "275"

cfc_patterns= [
    "clara_a3_thematic_cloud_properties_m10_CFCmm{year}1001000000319AVPOS01GL.nc",
    "clara_a3_interim_cloud_properties_m10_CFCmm{year}1001000000319AVPOSI1GL.nc"
]
cfc_var_candidates = ["CFC", "cf_total", "cloud_fraction", "cfc", "CFC_TOTAL"]

col = {
    "clara":    "#D81B60",
    "terra":    "#004D40",
    "aqua":     "#1E88E5",
    "combined": "#FFC107",
    "ceres":    "#5E35B1",
}
ds_labels= {
    "clara":    "CLARA-A3",
    "terra":    "MODIS Terra",
    "aqua":     "MODIS Aqua",
    "combined": "Terra+Aqua combined",
    "ceres":    "CERES EBAF",
}
keys_all = ["clara", "terra", "aqua", "combined", "ceres"]


# Only including land cloud cover changes
def build_land_mask(lats, lons):
    if lsm_file is not None and Path(lsm_file).is_file():
        ds = xr.open_dataset(lsm_file)
        for v in ("lsm", "land", "LAND", "LSM", "land_mask"):
            if v in ds:
                lsm = ds[v].values.squeeze()
                ds.close()
                return lsm > 0.5
        ds.close()
    print("  Building land mask from cartopy ...")
    from matplotlib.path import Path as MplPath
    import cartopy.io.shapereader as shpreader
    geoms = list(shpreader.Reader(
        shpreader.natural_earth(resolution="110m",
                                category="physical", name="land")
    ).geometries())
    lo2d, la2d = np.meshgrid(lons, lats)
    pts = np.column_stack([lo2d.ravel(), la2d.ravel()])
    inside = np.zeros(len(pts), dtype=bool)
    for g in geoms:
        if g.geom_type == "Polygon":
            inside |= MplPath(np.array(g.exterior.coords)).contains_points(pts)
        elif g.geom_type == "MultiPolygon":
            for p in g.geoms:
                inside |= MplPath(np.array(p.exterior.coords)).contains_points(pts)
    return inside.reshape(lo2d.shape)


# Loaders
def _to_fraction(arr):
    if np.nanmedian(arr[np.isfinite(arr)]) > 1.0:
        return arr / 100.0
    return arr


def load_clara(y1=clara_y1, y2=clara_y2):
    print(f"Loading CLARA-A3 ({y1}–{y2}) ...")
    slabs, yrs_ok = [], []
    lats_r = lons_r = cfc_var = None
    for yr in range(y1, y2 + 1):
        fpath = None
        for pat in cfc_patterns:
            p = clara_dir / pat.format(year=yr)
            if p.exists(): fpath = p; break
        if fpath is None:
            if lats_r is not None:
                slabs.append(np.full((len(lats_r), len(lons_r)), np.nan))
                yrs_ok.append(yr)
            continue
        ds = xr.open_dataset(fpath)
        if cfc_var is None:
            for n in cfc_var_candidates:
                if n in ds: cfc_var = n; break
            if cfc_var is None:
                skip = {"time", "time_bnds", "lat", "lon", "record_status"}
                cfc_var = next(n for n in ds.data_vars if n not in skip)
        lats = ds["lat"].values if "lat" in ds else ds["latitude"].values
        lons = ds["lon"].values if "lon" in ds else ds["longitude"].values
        if lats_r is None: lats_r, lons_r = lats, lons
        if "record_status" in ds:
            if int(np.array(ds["record_status"].values).flat[0]) != 0:
                ds.close()
                slabs.append(np.full((len(lats_r), len(lons_r)), np.nan))
                yrs_ok.append(yr); continue
        data = ds[cfc_var].values.squeeze().astype(float)
        fv = ds[cfc_var].attrs.get("_FillValue",
             ds[cfc_var].attrs.get("missing_value", None))
        if fv is not None: data = np.where(data == fv, np.nan, data)
        data = np.where(np.abs(data) > 1e10, np.nan, data)
        ds.close()
        slabs.append(data); yrs_ok.append(yr)
    arr = _to_fraction(np.stack(slabs, axis=0))
    return np.array(yrs_ok), arr, lats_r, lons_r


def _find_modis(base_dir, year, prefix):
    doy = oct_doy_leap if year in leap_years else oct_doy_noleap
    for d in [doy] + [str(int(doy) + o).zfill(3) for o in [-1, 1]]:
        m = glob.glob(os.path.join(str(base_dir), str(year), d,
                                   f"{prefix}.A*.hdf"))
        if m: return m[0]
    return None


def _load_modis(base_dir, prefix, y1, y2):
    slabs, yrs_ok = [], []
    lats_r = lons_r = None
    for yr in range(y1, y2 + 1):
        fpath = _find_modis(base_dir, yr, prefix)
        if fpath is None:
            if lats_r is not None:
                slabs.append(np.full((len(lats_r), len(lons_r)), np.nan))
                yrs_ok.append(yr)
            continue
        hdf  = SD(fpath, SDC.READ)
        ds_h = hdf.select(modis_var)
        data = ds_h[:].astype(np.float32)
        att  = ds_h.attributes()
        fill = att.get("_FillValue", -9999)
        scl  = att.get("scale_factor", 1.0)
        off  = att.get("add_offset", 0.0)
        data = np.where(data == fill, np.nan, data.astype(float))
        data = data * scl + off
        hdf.end()
        nlat, nlon = data.shape
        if lats_r is None:
            lats_r = np.linspace(90 - 0.5, -90 + 0.5, nlat)
            lons_r = np.linspace(-180 + 0.5, 180 - 0.5, nlon)
        slabs.append(data); yrs_ok.append(yr)
    return np.array(yrs_ok), np.stack(slabs, axis=0), lats_r, lons_r


def load_terra(y1=terra_y1, y2=terra_y2):
    print(f"Loading MODIS Terra ({y1}–{y2}) ...")
    return _load_modis(terra_dir, "MOD08_M3", y1, y2)


def load_aqua(y1=aqua_y1, y2=aqua_y2):
    print(f"Loading MODIS Aqua ({y1}–{y2}) ...")
    return _load_modis(aqua_dir, "MYD08_M3", y1, y2)


def load_ceres(y1=ceres_y1, y2=ceres_y2, month=10):
    """CERES EBAF October cloud area fraction: years, (year, lat, lon) as a 0-1 fraction,
    lats, lons (-180..180), read from all CERES subset files."""
    print(f"Loading CERES EBAF ({y1}–{y2}) ...")
    files = sorted(glob.glob(ceres_glob))
    if not files:
        raise FileNotFoundError(f"no files match {ceres_glob}")
    parts = []
    for f in files:
        with xr.open_dataset(f) as ds:
            da = ds[ceres_var]
            parts.append(da.isel(time=np.flatnonzero(da.time.dt.month.values == month)).load())
    da = xr.concat(parts, "time").sortby("time")
    yrs = da.time.dt.year.values
    da = da.isel(time=np.unique(yrs, return_index=True)[1])
    da = da.assign_coords(time=da.time.dt.year.values).rename(time="year")
    da = da.sel(year=slice(y1, y2))
    if float(da.lon.max()) > 180:
        da = da.assign_coords(lon=(da.lon + 180) % 360 - 180).sortby("lon")
    da = da.where(da >= 0)                                  # _FillValue -999
    arr = _to_fraction(da.values.astype(float))
    return da.year.values, arr, da.lat.values, da.lon.values


# Regrid to common resolution
def regrid(data3d, src_lats, src_lons, tgt_lats, tgt_lons):
    if (len(src_lats) == len(tgt_lats) and
            np.allclose(src_lats, tgt_lats, atol=0.1) and
            len(src_lons) == len(tgt_lons) and
            np.allclose(src_lons, tgt_lons, atol=0.1)):
        return data3d
    # nearest neighbour (same as before, vectorised)
    ij = np.abs(np.asarray(src_lats)[None, :] - np.asarray(tgt_lats)[:, None]).argmin(1)
    ii = np.abs(np.asarray(src_lons)[None, :] - np.asarray(tgt_lons)[:, None]).argmin(1)
    return data3d[:, ij][:, :, ii]


def combine_terra_aqua(t_data, t_yrs, a_data, a_yrs, ref_yrs):
    nlat, nlon = t_data.shape[1], t_data.shape[2]
    out = np.full((len(ref_yrs), nlat, nlon), np.nan)
    t_map = {y: i for i, y in enumerate(t_yrs)}
    a_map = {y: i for i, y in enumerate(a_yrs)}
    for ri, yr in enumerate(ref_yrs):
        if yr in t_map and yr in a_map:
            out[ri] = np.nanmean(
                np.stack([t_data[t_map[yr]], a_data[a_map[yr]]], axis=0), axis=0)
        elif yr in t_map:
            out[ri] = t_data[t_map[yr]]
        elif yr in a_map:
            out[ri] = a_data[a_map[yr]]
    return out


# Compute regional mean timeseries for a given lat band and land mask

def regional_mean_ts(data3d, lats, lons, land_mask, lat_s, lat_n):
    lat_m = (lats >= lat_s) & (lats <= lat_n)
    lon_m = (lons >= lon_w) & (lons <= lon_e)
    sub   = data3d[:, lat_m, :][:, :, lon_m]
    lm    = land_mask[lat_m, :][:, lon_m]
    wts   = np.cos(np.deg2rad(lats[lat_m]))[:, np.newaxis]
    ts    = np.full(data3d.shape[0], np.nan)
    for t in range(data3d.shape[0]):
        slab  = sub[t]
        valid = np.isfinite(slab) & lm
        if valid.sum() == 0: continue
        w = np.where(valid, wts, np.nan)
        ts[t] = np.nansum(slab * w) / np.nansum(w)
    return ts


# Compute OLS trend with autocorrelation-adjusted p-value 
def lag1(ts):
    ok = np.isfinite(ts)
    if ok.sum() < 3: return 0.0
    a, b = ts[:-1], ts[1:]
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() >= 3 else 0.0


def ols_ts(years, ts):
    ok = np.isfinite(ts)
    n  = ok.sum()
    if n < 6: return np.nan, np.nan, np.nan
    x, y = years[ok].astype(float), ts[ok]
    slope, intercept, r, _, _ = stats.linregress(x, y)
    resid  = y - (slope * x + intercept)
    rho    = lag1(resid)
    neff   = max(n * (1 - rho) / max(1 + rho, 1e-6), 3.0)
    se     = np.sqrt(np.sum(resid**2) / max(n - 2, 1) /
                     max(np.sum((x - x.mean())**2), 1e-12))
    se_adj = se * np.sqrt(n / neff)
    t_adj  = slope / se_adj if se_adj > 0 else np.nan
    p_adj  = float(2 * tdist.sf(abs(t_adj), df=max(int(neff - 2), 1)))
    return slope * 10, p_adj, r**2


def sig_stars(p):
    if np.isnan(p):  return "n/a"
    if p < 0.001:    return "***"
    if p < 0.01:     return "**"
    if p < 0.05:     return "*"
    return "ns"


def fishers_combined_p(pvals):
    """
    Fisher's method: combine k independent p-values into one.
    Statistic = -2 * sum(ln(p_i)) ~ chi2(2k).
    Only uses finite, valid (0 < p <= 1) values.
    Returns combined p-value, or nan if fewer than 2 valid p-values.
    """
    valid = [p for p in pvals if np.isfinite(p) and 0 < p <= 1]
    if len(valid) < 2:
        return np.nan
    chi_stat = -2.0 * np.sum(np.log(valid))
    return float(chi2.sf(chi_stat, df=2 * len(valid)))


# Statisical summary for bands 2 and 3 (2002–2024) (for paper)
def print_band_trends(datasets, common_yrs, land_mask, c_lats, c_lons):
    """
    For bands 2 (0°–25°S) and 3 (25°S–35°S), over the common period
    2002–2024, print:
      - OLS trend per dataset (CF per decade) + autocorrelation-adjusted p
      - Cross-dataset mean trend
      - Combined p-value via Fisher's method
    """
    keys = keys_all
    SEP  = "=" * 68
    SEP2 = "-" * 55

    print(f"\n{SEP}")
    print("  CF Trend Summary — Bands 2 & 3, 2002–2024 (land, 10°E–38°E)")
    print("  OLS trend with autocorrelation-adjusted p (Newey-West lag-1)")
    print(f"  Combined p: Fisher's method across all {len(keys)} datasets "
          "(not independent: CERES and MODIS share satellites)")
    print(SEP)

    for bi in stats_bands_indicies:
        band_label, lat_s, lat_n = bands[bi]
        print(f"\n  Band {bi+1}: {band_label}  "
              f"(lat {lat_s}°–{lat_n}°, lon {lon_w}°–{lon_e}°E, land only)")
        print(f"  {'Dataset':<22}  {'Trend (CF/dec)':>15}  "
              f"{'p-value':>9}  {'Sig':>5}  {'R²':>7}")
        print(f"  {SEP2}")

        trends, pvals = [], []
        for key in keys:
            _, data3d = datasets[key]
            ts = regional_mean_ts(data3d, c_lats, c_lons,
                                  land_mask, lat_s, lat_n)
            s, p, r2 = ols_ts(common_yrs, ts)
            trends.append(s)
            pvals.append(p)
            s_str  = f"{s:+.5f}" if np.isfinite(s)  else "    N/A"
            p_str  = f"{p:.4f}"  if np.isfinite(p)  else "    N/A"
            r2_str = f"{r2:.3f}" if np.isfinite(r2) else "  N/A"
            print(f"  {ds_labels[key]:<22}  {s_str:>15}  "
                  f"{p_str:>9}  {sig_stars(p):>5}  {r2_str:>7}")

        # Cross-dataset summary
        valid_trends = [t for t in trends if np.isfinite(t)]
        mean_trend   = float(np.mean(valid_trends)) if valid_trends else np.nan
        combined_p   = fishers_combined_p(pvals)

        print(f"  {SEP2}")
        mean_str = f"{mean_trend:+.5f}" if np.isfinite(mean_trend) else "N/A"
        cp_str   = f"{combined_p:.4f}"  if np.isfinite(combined_p) else "N/A"
        print(f"  {'Cross-dataset mean':<22}  {mean_str:>15}  "
              f"{cp_str:>9}  {sig_stars(combined_p):>5}")
        print(f"  (mean of {len(valid_trends)} datasets; "
              f"combined p via Fisher's method, k={len([p for p in pvals if np.isfinite(p) and 0 < p <= 1])})")

    print(f"\n{SEP}\n")


# Plot two files
def make_combined_figure(left_path: Path, right_path: Path, out_path: Path) -> None:
    """
    Load two existing PNG files, resize both to the same height (preserving
    aspect ratio), then place them side by side in a single figure at 200 dpi.
    """
    from PIL import Image as PILImage

    img_l = PILImage.open(left_path).convert("RGB")
    img_r = PILImage.open(right_path).convert("RGB")

    # Scale both to the taller image's height
    target_h = max(img_l.height, img_r.height)

    def _resize(img, h):
        w = int(img.width * h / img.height)
        return img.resize((w, h), PILImage.LANCZOS)

    img_l = _resize(img_l, target_h)
    img_r = _resize(img_r, target_h)

    dpi   = 200
    w_l   = img_l.width  / dpi
    w_r   = img_r.width  / dpi
    h     = target_h     / dpi
    gap   = 0.25
    fig_w = w_l + w_r + gap
    fig_h = h + 0.3

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)

    ax_l = fig.add_axes([0.0,
                         0.15 / fig_h,
                         w_l / fig_w,
                         h / fig_h])
    ax_l.imshow(np.array(img_l))
    ax_l.axis("off")
    ax_l.text(0.01, 1.01, "a)", transform=ax_l.transAxes,
              fontsize=16, fontweight="bold", va="bottom")

    ax_r = fig.add_axes([(w_l + gap) / fig_w,
                         0.15 / fig_h,
                         w_r / fig_w,
                         h / fig_h])
    ax_r.imshow(np.array(img_r))
    ax_r.axis("off")
    ax_r.text(0.01, 1.01, "b)", transform=ax_r.transAxes,
              fontsize=16, fontweight="bold", va="bottom")

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Combined figure saved → {out_path}")

# Main loading and plotting 
def main():
    # Load data
    c_yrs, c_data, c_lats, c_lons = load_clara()
    t_yrs, t_data, t_lats, t_lons = load_terra()
    a_yrs, a_data, a_lats, a_lons = load_aqua()
    e_yrs, e_data, e_lats, e_lons = load_ceres()

    print("Regridding onto CLARA grid ...")
    t_data_rg = regrid(t_data, t_lats, t_lons, c_lats, c_lons)
    a_data_rg = regrid(a_data, a_lats, a_lons, c_lats, c_lons)
    e_data_rg = regrid(e_data, e_lats, e_lons, c_lats, c_lons)

    print("Building land mask ...")
    land_mask = build_land_mask(c_lats, c_lons)

    # Align to common years
    common_yrs = np.arange(common_y1, common_y2 + 1)

    def _align(src_yrs, src_data):
        out = np.full((len(common_yrs), src_data.shape[1],
                       src_data.shape[2]), np.nan)
        m = {y: i for i, y in enumerate(src_yrs)}
        for ri, yr in enumerate(common_yrs):
            if yr in m: out[ri] = src_data[m[yr]]
        return out

    datasets = {
        "clara":    (common_yrs, _align(c_yrs, c_data)),
        "terra":    (common_yrs, _align(t_yrs, t_data_rg)),
        "aqua":     (common_yrs, _align(a_yrs, a_data_rg)),
        "combined": (common_yrs, combine_terra_aqua(
            t_data_rg, t_yrs, a_data_rg, a_yrs, common_yrs)),
        "ceres":    (common_yrs, _align(e_yrs, e_data_rg)),
    }

    keys = keys_all

    # Pre-compute anomalies
    anom_store = {}
    for key in keys:
        yrs, data3d = datasets[key]
        for bi, (_, lat_s, lat_n) in enumerate(bands):
            ts   = regional_mean_ts(data3d, c_lats, c_lons,
                                    land_mask, lat_s, lat_n)
            anom = ts - np.nanmean(ts)
            anom_store[(key, bi)] = anom

    # ── Also compute CLARA-only anomalies over the full CLARA period ─────────
    c_anom_full = {}
    for bi, (_, lat_s, lat_n) in enumerate(bands):
        ts = regional_mean_ts(c_data, c_lats, c_lons,
                              land_mask, lat_s, lat_n)
        c_anom_full[bi] = (c_yrs, ts - np.nanmean(ts))

    # Shared y limits for cross-validation panels
    all_vals = np.concatenate([
        anom_store[(k, bi)][np.isfinite(anom_store[(k, bi)])]
        for k in keys for bi in range(len(bands))
    ])
    ymax_cv = np.nanmax(np.abs(all_vals)) * 1.20
    shared_ylim_cv = (-ymax_cv, ymax_cv)

    # Shared y limits for CLARA-only panels
    all_clara = np.concatenate([
        c_anom_full[bi][1][np.isfinite(c_anom_full[bi][1])]
        for bi in range(len(bands))
    ])
    ymax_cl = np.nanmax(np.abs(all_clara)) * 1.20
    shared_ylim_cl = (-ymax_cl, ymax_cl)

    n_bands = len(bands)

    # ── Combined figure: left = CLARA bar panels, right = cross-val lines ────
    # GridSpec: n_bands rows × 2 cols, left col narrower
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(22, 5 * n_bands), facecolor="white")
    gs  = gridspec.GridSpec(
        n_bands, 2,
        figure=fig,
        width_ratios=[1, 1.8],
        wspace=0.28,
        hspace=0.42,
        top=0.94, bottom=0.06,
        left=0.06, right=0.98,
    )

    for bi, (band_label, lat_s, lat_n) in enumerate(bands):

        # CLARA_A3 anomaly timeseries bar panel (left)
        ax_l = fig.add_subplot(gs[bi, 0])
        yrs_cl, anom_cl = c_anom_full[bi]
        ok_cl = np.isfinite(anom_cl)
        bar_cols = ["#B71C1C" if (np.isfinite(a) and a > 0)
                    else "#0D47A1" for a in anom_cl]
        ax_l.bar(yrs_cl, np.where(ok_cl, anom_cl, np.nan),
                 color=bar_cols, width=0.7, alpha=0.85)
        ax_l.axhline(0, color="k", lw=0.8)

        if ok_cl.sum() >= 6:
            s_cl, p_cl, r2_cl = ols_ts(yrs_cl, anom_cl)
            if np.isfinite(s_cl):
                x_cl  = yrs_cl[ok_cl].astype(float)
                yf_cl = (s_cl / 10 * x_cl +
                         (np.nanmean(anom_cl) - s_cl / 10 * np.mean(x_cl)))
                ax_l.plot(yrs_cl[ok_cl], yf_cl,
                          color=col["clara"], lw=2.2)
                sig_cl = ("**" if p_cl < 0.01 else
                           ("*" if p_cl < 0.05 else ""))
                ax_l.text(
                    0.03, 0.05,
                    f"Trend: {s_cl:+.4f} dec⁻¹{sig_cl}  "
                    f"p={p_cl:.3f}  r²={r2_cl:.3f}",
                    transform=ax_l.transAxes, fontsize=7.5,
                    color=col["clara"], va="bottom",
                    bbox=dict(fc="white", alpha=0.6, ec="none", pad=1),
                )

        for sp in ax_l.spines.values():
            sp.set_edgecolor(col["clara"]); sp.set_linewidth(1.8)
        ax_l.set_ylim(shared_ylim_cl)
        ax_l.set_ylabel("Anomaly (CF)", fontsize=9)
        if bi == n_bands - 1:
            ax_l.set_xlabel("Year", fontsize=9)
        ax_l.xaxis.set_major_locator(mticker.MultipleLocator(10))
        ax_l.tick_params(labelsize=8)
        ax_l.grid(axis="y", alpha=0.3, lw=0.4)
        ax_l.axvline(2020, ls="--", lw=1.0, color="grey", alpha=0.65)
        if bi == 0:
            ax_l.set_title(
                f"a)  CLARA-A3 ({clara_y1}–{clara_y2})  |  Band: {band_label}",
                fontsize=12, fontweight="bold", pad=5,
            )
        else:
            ax_l.set_title(f"Band: {band_label}",
                           fontsize=12, fontweight="bold", loc="left", pad=5)
        

        # Cross validation anomaly timeseries line panel (right)
        ax_r = fig.add_subplot(gs[bi, 1])

        for key in keys:
            anom = anom_store[(key, bi)]
            ok   = np.isfinite(anom)
            ax_r.plot(common_yrs, anom,
                      color=col[key], lw=2.0,
                      marker="o", ms=4, alpha=0.85,
                      label=ds_labels[key])
            if ok.sum() >= 6:
                s, p, r2 = ols_ts(common_yrs, anom)
                if np.isfinite(s):
                    x  = common_yrs[ok].astype(float)
                    yf = (s / 10 * x +
                          (np.nanmean(anom) - s / 10 * np.mean(x)))
                    ax_r.plot(common_yrs[ok], yf,
                              color=col[key], lw=1.5, ls="--", alpha=0.65)

        ax_r.axhline(0, color="k", lw=0.9)
        ax_r.set_ylim(shared_ylim_cv)
        ax_r.set_xlim(common_yrs[0] - 0.5, common_yrs[-1] + 0.5)

        y_txt = 0.97
        for key in keys:
            s, p, r2 = ols_ts(common_yrs, anom_store[(key, bi)])
            if np.isfinite(s):
                stars = sig_stars(p)
                label = ds_labels[key].replace("\n", " ")
                ax_r.text(0.01, y_txt,
                          f"{label}: {s:+.4f} dec⁻¹  "
                          f"p={p:.3f}  r²={r2:.3f}  {stars}",
                          transform=ax_r.transAxes, fontsize=7.2,
                          color=col[key], va="top",
                          bbox=dict(fc="white", alpha=0.0, ec="none"))
                y_txt -= 0.085

        ax_r.set_ylabel("Anomaly (CF)", fontsize=9)
        if bi == n_bands - 1:
            ax_r.set_xlabel("Year", fontsize=9)
        ax_r.xaxis.set_major_locator(mticker.MultipleLocator(5))
        ax_r.xaxis.set_minor_locator(mticker.MultipleLocator(1))
        ax_r.tick_params(labelsize=9)
        ax_r.grid(axis="y", alpha=0.35, lw=0.5)
        ax_r.grid(axis="x", alpha=0.20, lw=0.4)
        for sp in ax_r.spines.values():
            sp.set_linewidth(1.2)

        if bi == 0:
            ax_r.legend(fontsize=10, loc="upper right",
                        framealpha=0.85, edgecolor="grey")
            ax_r.set_title(
                f"b)  Cross validation (2002–{common_y2})  |  Band: {band_label}",
                fontsize=12, fontweight="bold", pad=5,
            )
        else:
            ax_r.set_title(f"Band: {band_label}",
                           fontsize=12, fontweight="bold", loc="left", pad=5)

    combined_path = out_dir / "cf_combined_panel.png"
    fig.savefig(combined_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved → {combined_path}")

    # Print statistical summary for bands 2 and 3 (2002–2024) (for paper)
    print_band_trends(datasets, common_yrs, land_mask, c_lats, c_lons)

    print("Done.")


# Run
if __name__ == "__main__":
    main()