"""
October vertical velocity (omega) trends and regime change: ERA5 vs MERRA-2 vs JRA-3Q.

Domain 35 S - 20 N, 10 W - 60 E (Africa / Indian Ocean sector); dotted reference line at 10 S.
omega > 0 is subsidence (descent), omega < 0 is ascent (Pa s-1).

DATA (raw files on the current server; only the needed months, levels and domain are read)
  ERA5     scratch/era5/global/vertical_velocity/global_vertical_velocity_YYYY.nc
           w(valid_time, pressure_level, latitude, longitude), monthly means, 0.25 deg
  MERRA-2  scratch/merra_2/MERRA2_<ver>.instM_3d_asm_Np.YYYYMM.nc4, OMEGA(time, lev, lat, lon),
           0.5 x 0.625 deg; where two versions exist for a month (e.g. _400 and the
           reprocessed _401) the highest version is used
  JRA-3Q   /soge-home/data/analysis/jra-q3/anl_p125_vvel/anl_p125_vvel.YYYYMMDDHH
           GRIB, one file per 6-hourly analysis, regular 1.25 deg grid (read with cfgrib);
           a month is the mean of all its 6-hourly analyses and is used only if all are
           there; longitudes are converted from 0-360 to -180-180 before subsetting
  Unreadable or missing files are skipped and reported.
  SPEED: reading the raw files is the slow part (JRA-3Q above all). Each dataset-month, once
  put on the common grid, is saved to <out_dir>/cache (a few hundred kB), so later runs and the
  monthly / assumption-check scripts reuse it. Files are read in parallel with one worker per
  CPU given by SLURM (--cpus-per-task); a progress line is printed per dataset-year.

COMMON GRID AND COMPLETENESS (no NaNs)
  All three datasets are box-averaged to one regular 1.25 x 1.25 deg grid: every source
  point is averaged into the box that contains it (no interpolation, so fine-scale noise is
  averaged rather than sampled). Pressure levels are taken exactly (all datasets carry them).
  Only years present in all three datasets are used. A box/level is analysed only if all
  three datasets have it in every year; MERRA-2 leaves levels below ground undefined, so this
  also removes below-ground levels (ERA5 and JRA-3Q extrapolate there) consistently.

TRENDS AND SIGNIFICANCE (same conventions as the temperature scripts; vectorised)
  OLS slope, reported per decade (Pa s-1 decade-1); lag-1 autocorrelation r1 of the residuals gives
  n_eff = n (1 - r1) / (1 + r1), capped at n (Santer et al. 2000); two-sided t-test with
  n_eff - 2 degrees of freedom. Significance: Benjamini-Hochberg FDR, q < alpha_fdr = 0.10
  (Wilks 2016), applied per map (dataset x level).

REGIMES (significant boxes only; unchanged logic, vectorised)
  1. Persistent sign change: the first year k with min_persist (10) consecutive years of one
     sign, whose pre-k and post-k means have opposite signs -> new subsidence (post > 0) or
     new ascent (post < 0). Pettitt test (p < 0.05) reported as a check on these boxes.
  2. Otherwise long-term mean and OLS slope sign:
     mean > 0, slope > 0 strengthening subsidence  | mean > 0, slope < 0 weakening subsidence
     mean < 0, slope < 0 strengthening ascent      | mean < 0, slope > 0 weakening ascent

STATISTICS PRINTED PER LEVEL (unchanged)
  [1] Congo Basin weakening ascent (10-32 E, 10 S-20 N)  [2] significant positive trends south
  of 10 S  [3] significant ocean boxes  [4] regime breakdown of significant ocean boxes.
  Ocean = boxes with < 50 % land (Natural Earth 50 m land polygons).

FIGURE per level  (2 rows x 3 cols)
  Row 1 signed trend; one colour scale for all levels and datasets (98th percentile of |slope|
        over all tested boxes).
  Row 2 regime of every tested box, shaded by |trend| on that same scale: white at zero
        trend, full regime colour at or beyond the colour-bar limit.
  Black dots mark FDR-significant boxes; white areas have no data (below ground).
"""
from __future__ import annotations
import calendar
import multiprocessing
import os
import re
import time
from functools import cache
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader
from shapely import box, contains_xy
from shapely.ops import unary_union

# ---------------------------------------------------------------- paths
era5_dir = Path("/soge-home/users/jesu4127/scratch/era5/global/vertical_velocity")
merra2_dir = Path("/soge-home/users/jesu4127/scratch/merra_2")
jra3q_dir = Path("/soge-home/data/analysis/jra-q3/anl_p125_vvel")
out_dir = Path("/soge-home/users/jesu4127/Paper_1/vertical_velocity/fixed")

# ---------------------------------------------------------------- settings
years = range(1980, 2026)
months = [10]
levels = [100.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0, 600.0, 700.0, 850.0, 925.0, 1000.0]
min_persist = 10
alpha_fdr = 0.10
lat_s, lat_n, lon_w, lon_e = -35.0, 20.0, -10.0, 60.0
grid_step = 1.25
ref_lat = -10.0
congo = (10.0, 32.0, -10.0, 20.0)                  # W, E, S, N
s_africa_lat_n = -10.0
land_min_fraction = 0.5
land_shapefile = None                              # or a local ne_50m_land.shp
overwrite = True                                   # False: keep figures that already exist
cache_dir = None                                   # None: <out_dir>/cache (delete it if you
                                                   # change the domain or grid_step)
n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))   # parallel file reading
names = ["ERA5", "MERRA-2", "JRA-3Q"]

regime_keys = ["str_sub", "wk_sub", "new_sub", "wk_asc", "str_asc", "new_asc"]
regime_colours = {"str_sub": "#DC2671", "wk_sub": "#FE6100", "new_sub": "#FFB000",
                  "wk_asc": "#4DB5FA", "str_asc": "#785EF0", "new_asc": "#60D691"}
regime_labels = {"str_sub": "Strengthening subsidence", "wk_sub": "Weakening subsidence",
                 "new_sub": "New subsidence", "wk_asc": "Weakening ascent",
                 "str_asc": "Strengthening ascent", "new_asc": "New ascent"}
grey = "#CCCCCC"
diverging = plt.get_cmap("RdBu_r")
pc = ccrs.PlateCarree()
grid_lat = np.arange(lat_s, lat_n + 1e-6, grid_step)
grid_lon = np.arange(lon_w, lon_e + 1e-6, grid_step)
half = grid_step / 2


# ---------------------------------------------------------------- loading
def box_mean(da):
    """Average every source point into the 1.25-degree box containing it (NaN propagates)."""
    da = da.where(np.abs(da) < 1e10)                      # fill values (1e15, 9.999e20) -> NaN
    for dim, centres in (("lat", grid_lat), ("lon", grid_lon)):
        edges = np.append(centres - half, centres[-1] + half)
        da = da.groupby_bins(dim, edges, labels=centres).mean(skipna=False)
        da = da.rename({f"{dim}_bins": dim}).assign_coords({dim: centres})
    return da.transpose(..., "lat", "lon").astype("float32")


def by_month(parts):
    """{month: DataArray or None} -> DataArray with a month dim; missing months are NaN."""
    ok = [p for p in parts.values() if p is not None]
    if not ok:
        raise FileNotFoundError("no readable month")
    return xr.concat([(p if p is not None else xr.full_like(ok[0], np.nan)).expand_dims(month=[mo])
                      for mo, p in parts.items()], "month")


# Each reader returns one month on the common grid: (lev, lat, lon), or raises if unavailable.
def read_era5(year, mo):
    with xr.open_dataset(era5_dir / f"global_vertical_velocity_{year}.nc") as ds:
        da = ds["w"].rename(valid_time="time", latitude="lat", longitude="lon",
                            pressure_level="lev")
        t = np.flatnonzero(da.time.dt.month.values == mo)
        if not t.size:
            raise ValueError("month not in file")
        da = da.isel(time=int(t[0])).sel(lev=levels, lat=slice(lat_n + half, lat_s - half),
                                         lon=slice(lon_w - half, lon_e + half)).load()
    return box_mean(da.drop_vars(["number", "expver", "time"], errors="ignore"))


@cache
def merra2_files():
    found = {}
    for p in merra2_dir.glob("MERRA2_*.instM_3d_asm_Np.*.nc4"):
        m = re.match(r"MERRA2_(\d{3})\.instM_3d_asm_Np\.(\d{4})(\d{2})\.nc4$", p.name)
        if m and (key := (int(m[2]), int(m[3]))) and (key not in found or int(m[1]) > found[key][0]):
            found[key] = (int(m[1]), p)
    return {k: v[1] for k, v in found.items()}


def read_merra2(year, mo):
    p = merra2_files().get((year, mo))
    if p is None:
        raise FileNotFoundError("no file")
    with xr.open_dataset(p) as ds:
        da = ds["OMEGA"].isel(time=0, drop=True).sel(
            lev=levels, lat=slice(lat_s - half, lat_n + half),
            lon=slice(lon_w - half, lon_e + half)).load()
    return box_mean(da)


@cache
def jra3q_files():
    """{(year, month): [paths]} for the 6-hourly GRIB files anl_p125_vvel.YYYYMMDDHH."""
    found = {}
    for p in jra3q_dir.glob("anl_p125_vvel.[0-9]*"):
        if m := re.fullmatch(r"anl_p125_vvel\.(\d{4})(\d{2})\d{4}", p.name):
            found.setdefault((int(m[1]), int(m[2])), []).append(p)
    return found


def read_jra3q(year, mo):
    """Monthly mean of the 6-hourly GRIB analyses (already on a 1.25 deg grid); complete months only."""
    paths = sorted(jra3q_files().get((year, mo), []))
    expected = calendar.monthrange(year, mo)[1] * 4
    if len(paths) < expected:
        raise ValueError(f"only {len(paths)} of {expected} 6-hourly files")
    total = None
    for p in paths:
        with xr.open_dataset(p, engine="cfgrib", backend_kwargs={
                "indexpath": "", "filter_by_keys": {"typeOfLevel": "isobaricInhPa"}}) as ds:
            da = ds["w" if "w" in ds else list(ds.data_vars)[0]]
            da = da.rename(latitude="lat", longitude="lon", isobaricInhPa="lev")
            lat_ok = (da.lat >= lat_s - half) & (da.lat <= lat_n + half)
            da = da.sel(lev=levels).isel(lat=np.flatnonzero(lat_ok.values)).load()
        da = da.drop_vars(["time", "step", "valid_time", "number"], errors="ignore")
        da = da.assign_coords(lon=(da.lon + 180) % 360 - 180).sortby("lon")
        da = da.sel(lon=slice(lon_w - half, lon_e + half)).astype("float64")
        total = da if total is None else total + da
    return box_mean(total / len(paths))


readers = {"ERA5": read_era5, "MERRA-2": read_merra2, "JRA-3Q": read_jra3q}


def cached_month(name, year, mo):
    """One dataset-month on the common grid, read from the cache if it is there."""
    path = (cache_dir or out_dir / "cache") / f"{name}_{year}{mo:02d}.nc"
    if path.exists():
        try:
            with xr.open_dataarray(path) as d:
                if set(levels) <= set(d.lev.values.tolist()):
                    return d.sel(lev=levels).load()
        except Exception:
            pass                                          # damaged cache file: read again
    da = readers[name](year, mo).rename("omega")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    da.to_netcdf(tmp)
    tmp.replace(path)
    return da


def _task(job):
    try:
        return job, cached_month(*job), None
    except Exception as e:
        return job, None, f"{type(e).__name__}: {e}"


def read_all_months():
    """{(name, year, month): DataArray or None}, read in parallel, with progress lines."""
    jobs = [(n, y, mo) for y in years for n in names for mo in months]
    merra2_files(), jra3q_files()                         # list the files once, before forking
    todo = {(n, y): len(months) for n, y, _ in jobs}
    out, t0 = {}, time.time()
    print(f"Reading {len(jobs)} dataset-months with {n_workers} worker(s); cache: "
          f"{cache_dir or out_dir / 'cache'}", flush=True)
    if n_workers > 1:
        pool = multiprocessing.get_context("fork").Pool(n_workers)
        results = pool.imap_unordered(_task, jobs)
    else:
        pool, results = None, map(_task, jobs)
    for (n, y, mo), da, err in results:
        out[(n, y, mo)] = da
        if err:
            print(f"  {n} {y}-{mo:02d} missing/unreadable: {err}", flush=True)
        todo[(n, y)] -= 1
        if not todo[(n, y)]:
            print(f"  {n} {y} done  [{sum(v == 0 for v in todo.values())}/{len(todo)}, "
                  f"{time.time() - t0:.0f} s]", flush=True)
    if pool:
        pool.close()
        pool.join()
    return out


def load_all():
    """All three datasets on the common grid: common years, complete boxes only (no NaNs)."""
    got, data = read_all_months(), {}
    for name in names:
        parts = []
        for y in years:
            try:
                parts.append(by_month({mo: got[(name, y, mo)] for mo in months})
                             .expand_dims(year=[y]))
            except FileNotFoundError:
                print(f"  SKIPPED {name} {y}: no readable month", flush=True)
        if not parts:
            raise RuntimeError(f"No {name} data loaded.")
        data[name] = xr.concat(parts, "year").transpose("year", "month", "lev", "lat", "lon")
    common = sorted(set.intersection(*(set(d.year.values.tolist()) for d in data.values())))
    if len(common) < 8:
        raise RuntimeError(f"Too few common years: {len(common)}")
    data = {n: d.sel(year=common) for n, d in data.items()}
    # a month missing from any dataset in a year is removed from all three for that year
    missing = sum(d.isnull().all(("lev", "lat", "lon")) for d in data.values()) > 0
    data = {n: d.where(~missing) for n, d in data.items()}
    # no NaNs: a box/level is kept only if every dataset has it in every remaining year
    complete = xr.concat([d.notnull() | missing for d in data.values()], "ds").all(("ds", "year"))
    data = {n: d.where(complete) for n, d in data.items()}
    print(f"Common years {common[0]}-{common[-1]} (n = {len(common)}); "
          f"{int(missing.sum())} month-year(s) missing in at least one dataset; complete boxes "
          f"per level: {complete.isel(month=0).sum(('lat', 'lon')).values.tolist()}", flush=True)
    return data


# ---------------------------------------------------------------- statistics (vectorised)
def ols_stats(da):
    """OLS slope per decade with AR(1)-corrected p, along 'year', for every other index."""
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
    se = np.sqrt(ssr / (n_eff - 2) / sxx)
    p = xr.apply_ufunc(lambda t, df: 2 * stats.t.sf(np.abs(t), df), slope / se, n_eff - 2)
    out = xr.Dataset({"slope": 10 * slope, "p": p, "n": n, "r1": r1, "n_eff": n_eff})
    return out.where((n >= 8) & (n_eff > 2))


def _bh_map(p):
    sig, ok = np.zeros(p.shape, bool), np.isfinite(p)
    if ok.any():
        sig[ok] = stats.false_discovery_control(p[ok]) < alpha_fdr
    return sig


def fdr(p):
    """Benjamini-Hochberg separately on every (lat, lon) map."""
    return xr.apply_ufunc(_bh_map, p, input_core_dims=[["lat", "lon"]],
                          output_core_dims=[["lat", "lon"]], vectorize=True)


def _regime_map(x, slope):
    """x (year, lat, lon), slope (lat, lon) -> regime code (index into regime_keys), -1 none."""
    n, P = x.shape[0], min_persist
    fin = np.isfinite(x)
    cum = lambda a: np.concatenate([np.zeros_like(a[:1], float), np.cumsum(a, 0)])
    c_sum, c_fin = cum(np.where(fin, x, 0.0)), cum(fin)
    c_pos, c_neg = cum(fin & (x > 0)), cum(fin & (x < 0))
    ks = np.arange(1, n - P)                              # as range(1, n - min_persist)
    win = lambda c: c[ks + P] - c[ks]
    with np.errstate(invalid="ignore", divide="ignore"):
        pre_n, post_n = c_fin[ks], c_fin[n] - c_fin[ks]
        pre, post = c_sum[ks] / pre_n, (c_sum[n] - c_sum[ks]) / post_n
        ok = ((win(c_fin) == P) & ((win(c_pos) == P) | (win(c_neg) == P)) & (pre_n >= 2)
              & (post_n >= 2) & (pre != 0) & (post != 0) & (np.sign(pre) != np.sign(post)))
        found = ok.any(0)
        post_first = np.take_along_axis(post, ok.argmax(0)[None], 0)[0]
        mean = np.where(fin.any(0), c_sum[n] / np.maximum(c_fin[n], 1), np.nan)
    code = np.where(mean > 0, np.where(slope > 0, 0, 1), np.where(slope < 0, 4, 3))
    code = np.where(found, np.where(post_first > 0, 2, 5), code)
    return np.where(np.isfinite(slope) & (fin.sum(0) >= 8), code, -1)


def regimes(da, slope):
    return xr.apply_ufunc(_regime_map, da, slope, input_core_dims=[["year", "lat", "lon"],
                          ["lat", "lon"]], output_core_dims=[["lat", "lon"]], vectorize=True)


def _pettitt_map(x):
    """Pettitt change-point test, p < 0.05, for every box of x (year, lat, lon) at once."""
    n = np.isfinite(x).sum(0)
    S = np.nan_to_num(np.sign(x[None] - x[:, None]))     # [i, j] = sign(x_j - x_i)
    R = np.cumsum(S, 0)                                  # sum over i <= t
    U = np.einsum("tj...,tj->t...", R, np.triu(np.ones((x.shape[0],) * 2), 1))
    K = np.abs(U).max(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.clip(2 * np.exp(-6 * K ** 2 / (n ** 3 + n ** 2)), 0, 1)
    return p < 0.05


def pettitt(da):
    return xr.apply_ufunc(_pettitt_map, da, input_core_dims=[["year", "lat", "lon"]],
                          output_core_dims=[["lat", "lon"]], vectorize=True)


def analyse(data):
    """Trends, FDR significance, regimes and Pettitt flags for every dataset/month/level."""
    out = {}
    for name, da in data.items():
        st = ols_stats(da)
        st["sig"] = fdr(st.p) & st.p.notnull()
        st["regime"] = regimes(da, st.slope)
        st["pettitt"] = pettitt(da)
        out[name] = st
    return out


# ---------------------------------------------------------------- land / ocean
@cache
def land_geometry():
    path = land_shapefile or shapereader.natural_earth("50m", "physical", "land")
    land = unary_union(list(shapereader.Reader(str(path)).geometries()))
    return land.intersection(box(lon_w - 10, lat_s - 10, lon_e + 10, lat_n + 10))


@cache
def ocean_mask():
    """True where a common-grid box is < land_min_fraction land (10 x 10 points per box)."""
    off = (np.arange(10) + 0.5) / 10 - 0.5
    lat = grid_lat[:, None, None, None] + off[None, None, :, None] * grid_step
    lon = grid_lon[None, :, None, None] + off[None, None, None, :] * grid_step
    lat, lon = np.broadcast_arrays(lat, lon)
    return contains_xy(land_geometry(), lon, lat).mean(axis=(2, 3)) < land_min_fraction


# ---------------------------------------------------------------- printed statistics
def print_level_stats(lev, res):
    lat2d, lon2d = np.meshgrid(grid_lat, grid_lon, indexing="ij")
    sa = lat2d < s_africa_lat_n
    cb = (lat2d >= congo[2]) & (lat2d <= congo[3]) & (lon2d >= congo[0]) & (lon2d <= congo[1])
    ocean = ocean_mask()
    get = lambda n, v: res[n][v].sel(lev=lev).isel(month=0).values
    sep = "=" * 68
    print(f"\n{sep}\n  Statistics at {lev:.0f} hPa\n{sep}")
    print(f"\n  [1] Congo Basin weakening ascent (lat {congo[2]}-{congo[3]}, lon {congo[0]}-"
          f"{congo[1]} E); % of all {cb.sum()} Congo-box boxes")
    print(f"      {'Dataset':<10}{'N sig wk_asc':>14}{'% Congo':>10}{'Mean slope (Pa s-1 dec-1)':>28}")
    for n in names:
        a = get(n, "sig") & cb & (get(n, "regime") == regime_keys.index("wk_asc"))
        mt = f"{np.nanmean(get(n, 'slope')[a]):.4e}" if a.any() else "N/A"
        print(f"      {n:<10}{int(a.sum()):>14,d}{100 * a.sum() / cb.sum():>9.1f}%{mt:>28}")
    print(f"\n  [2] Significant positive trends south of {s_africa_lat_n}°; % of all {sa.sum()}")
    for n in names:
        a = get(n, "sig") & sa & (get(n, "slope") > 0)
        print(f"      {n:<10}{int(a.sum()):>14,d}{100 * a.sum() / sa.sum():>9.1f}%")
    print(f"\n  [3] Significant ocean boxes; % of all {ocean.sum()} ocean boxes")
    for n in names:
        a = get(n, "sig") & ocean
        print(f"      {n:<10}{int(a.sum()):>14,d}{100 * a.sum() / max(ocean.sum(), 1):>9.1f}%")
    print("\n  [4] Regime breakdown of significant ocean boxes")
    print(f"      {'Regime':<26}" + "".join(f"{n:>10}" for n in names))
    for k, key in enumerate(regime_keys):
        print(f"      {regime_labels[key]:<26}" + "".join(
            f"{int((get(n, 'sig') & ocean & (get(n, 'regime') == k)).sum()):>10,d}" for n in names))
    for n in names:
        new = get(n, "regime")
        for key in ("new_sub", "new_asc"):
            a = new == regime_keys.index(key)
            pt = 100 * (a & get(n, "pettitt")).sum() / a.sum() if a.any() else np.nan
            print(f"  [{n:7s}] {key}: {int(a.sum())} boxes, Pettitt p < 0.05 in {pt:.1f}%")


# ---------------------------------------------------------------- plotting
def add_geo(ax, labels=(True, True)):
    ax.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor="none", zorder=1)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=1.1, edgecolor="black",
                   zorder=5)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.9, edgecolor="black",
                   zorder=5)
    ax.set_extent([lon_w, lon_e, lat_s, lat_n], crs=pc)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="grey", alpha=0.5, linestyle="--",
                      xlocs=mticker.MultipleLocator(10), ylocs=mticker.MultipleLocator(10))
    gl.top_labels = gl.right_labels = False
    gl.left_labels, gl.bottom_labels = labels
    gl.xlabel_style = gl.ylabel_style = {"size": 10}
    ax.plot([lon_w, lon_e], [ref_lat, ref_lat], color="black", linewidth=1.0, linestyle=":",
            transform=pc, zorder=8)


def field_plot(ax, field, cmap, norm):
    """A field drawn as 1.25-degree boxes."""
    ax.pcolormesh(grid_lon, grid_lat, field, cmap=cmap, norm=norm, shading="nearest",
                  transform=pc, zorder=2, rasterized=True)


def stipple(ax, s):
    """Black dots on FDR-significant boxes, sized to the box size on this axes."""
    sig = np.asarray(s.sig.values, bool)
    if not sig.any():
        return
    box_pts = ax.get_position().width * ax.figure.get_figwidth() * 72 / grid_lon.size
    lon2, lat2 = np.meshgrid(grid_lon, grid_lat)
    ax.scatter(lon2[sig], lat2[sig], s=(0.3 * box_pts) ** 2, color="black", linewidths=0,
               transform=pc, zorder=4)


def shade_cmap(colour):
    """White -> regime colour: the shade shows the trend magnitude."""
    return mcolors.LinearSegmentedColormap.from_list("", ["white", colour])


def draw_regime(ax, s, vlim):
    """Regime colour of every tested box, shaded by |trend| / vlim (white at zero trend)."""
    strength = np.clip(np.abs(s.slope.values) / vlim, 0, 1)
    code = np.where(s.regime >= 0, s.regime, -1)
    for k, key in enumerate(regime_keys):
        field = np.where(code == k, strength, np.nan)
        if np.isfinite(field).any():
            field_plot(ax, field, shade_cmap(regime_colours[key]),
                       mcolors.Normalize(0, 1))
    stipple(ax, s)


def draw_trend(ax, s, norm):
    """Trend of every tested box; black dots where FDR-significant; no data stays white."""
    field_plot(ax, s.slope.values, diverging, norm)
    stipple(ax, s)


def regime_legend():
    return ([mpatches.Patch(facecolor=regime_colours[k], label=regime_labels[k]) for k in regime_keys]
            + [Line2D([], [], marker="o", linestyle="none", markersize=8, markerfacecolor="black",
                      markeredgecolor="black", label=f"FDR-significant (BH q < {alpha_fdr})"),
               mpatches.Patch(facecolor="white", edgecolor="0.5", label="No data")])


def global_vlim(res, percentile=98.0):
    """One colour limit for all maps: the 98th percentile of |slope| over all tested boxes."""
    mags = np.concatenate([np.abs(r.slope.values).ravel() for r in res.values()])
    mags = mags[np.isfinite(mags)]
    return max(float(np.percentile(mags, percentile)), 1e-15) if mags.size else 1e-9


def colourbar(fig, cax, norm, vlim, label, orientation="horizontal"):
    e = int(np.floor(np.log10(vlim)))                      # label ticks as multiples of 10^e
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=diverging), cax=cax,
                      orientation=orientation, ticks=np.linspace(-vlim, vlim, 7), extend="both")
    cb.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x / 10 ** e:.2g}"))
    cb.ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x / 10 ** e:.2g}"))
    cb.set_label(label.format(e=e), fontsize=14)
    cb.ax.tick_params(labelsize=12)


def trend_colourbar(fig, cax, norm, vlim, orientation="horizontal"):
    colourbar(fig, cax, norm, vlim, "Trend (10$^{{{e}}}$ Pa s$^{{-1}}$ decade$^{{-1}}$)", orientation)


def intensity_colourbars(fig, x0, y0, width, height, vlim):
    """One thin bar per regime, white -> its colour on the 0..vlim |trend| scale: exactly the
    shading used in the classification maps. Ticks and label on the last bar only."""
    e = int(np.floor(np.log10(vlim)))
    k = len(regime_keys)
    w = width / (k + 0.4 * (k - 1))                      # bar width; gaps are 0.4 of a bar
    for i, key in enumerate(regime_keys):
        cax = fig.add_axes([x0 + i * 1.4 * w, y0, w, height])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=mcolors.Normalize(0, vlim),
                                                cmap=shade_cmap(regime_colours[key])),
                          cax=cax, orientation="vertical", extend="max",
                          ticks=np.linspace(0, vlim, 4) if i == k - 1 else [])
        cb.outline.set_linewidth(0.6)
        if i == k - 1:
            cb.ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _: f"{x / 10 ** e:.2g}"))
            cb.ax.tick_params(labelsize=12)
            cb.set_label(f"|trend| (10$^{{{e}}}$ Pa s$^{{-1}}$ decade$^{{-1}}$)", fontsize=15)


def make_figure(lev, res, vlim, yrs, out_path):
    """Rows: trend, regime (shaded by |trend|); laid out in inches so nothing is cropped."""
    norm = mcolors.TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)
    W, left, right, gap = 24.0, 1.3, 2.9, 0.8
    mw = (W - left - right - 2 * gap) / 3
    mh = mw * (lat_n - lat_s) / (lon_e - lon_w)
    tops = [1.8, 1.8 + mh + 0.7]                          # inches from the top of the figure
    H = tops[1] + mh + 1.7                                # room for the legend underneath
    fig = plt.figure(figsize=(W, H))
    row_names = ["Trend", "Trend classification"]
    for col, n in enumerate(names):
        s = res[n].sel(lev=lev).isel(month=0)
        for row, top in enumerate(tops):
            ax = fig.add_axes([(left + col * (mw + gap)) / W, 1 - (top + mh) / H, mw / W, mh / H],
                              projection=pc)
            add_geo(ax)
            if row == 0:
                draw_trend(ax, s, norm)
                # dataset name centred above its column (figure coordinates, not the axes title)
                fig.text((left + col * (mw + gap) + mw / 2) / W, 1 - (top - 0.35) / H, n,
                         ha="center", va="bottom", fontsize=26, fontweight="bold")
            else:
                draw_regime(ax, s, vlim)
            if col == 0:
                ax.text(-0.12, 0.5, row_names[row], transform=ax.transAxes, rotation=90,
                        ha="center", va="center", fontsize=15, fontweight="bold")
    trend_colourbar(fig, fig.add_axes([(W - right + 0.35) / W, 1 - (tops[0] + mh - 0.3) / H,
                                       0.22 / W, (mh - 0.6) / H]), norm, vlim, "vertical")
    intensity_colourbars(fig, (W - right + 0.35) / W, 1 - (tops[1] + mh - 0.3) / H,
                         1.05 / W, (mh - 0.6) / H, vlim)
    fig.legend(handles=regime_legend(), loc="center", ncol=8, fontsize=15, frameon=True,
               bbox_to_anchor=(0.5, 1 - (tops[1] + mh + 0.95) / H), handlelength=1.8,
               handleheight=1.4, borderpad=0.7, columnspacing=1.0)
    fig.suptitle(f"{calendar.month_name[months[0]]} vertical velocity decadal trends at {lev:.0f} hPa ({yrs})",
                 fontsize=20, fontweight="bold", y=1 - 0.4 / H, va="center")
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}", flush=True)


# ---------------------------------------------------------------- main
def main():
    out = out_dir / "october" / "regime_change"
    out.mkdir(parents=True, exist_ok=True)
    data = load_all()
    print("\nTrends, FDR, regimes and Pettitt for all levels ...", flush=True)
    res = analyse(data)
    vlim = global_vlim(res)
    yrs = f"{int(data['ERA5'].year.min())}-{int(data['ERA5'].year.max())}"
    print(f"Global colour limit (98th percentile |slope|, all tested boxes): {vlim:.4e} Pa s-1 decade-1")
    for lev in levels:
        print_level_stats(lev, res)
        path = out / f"regime_change_oct_{lev:06.1f}hPa.png"
        if overwrite or not path.exists():
            make_figure(lev, res, vlim, yrs, path)
    print("\nAll levels complete.", flush=True)


if __name__ == "__main__":
    main()