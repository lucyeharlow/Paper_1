#!/usr/bin/env python3
"""
full_reconstruct_era5_merra2_chirts.py

Reconstruction of the published figure, extended with CHIRTS, MLOST, and JRA-3Q.

DATASETS
--------
ERA5    /soge-home/users/jesu4127/scratch/era5/global_2m_temperature.nc   (t2m, K)
MERRA2  /soge-home/users/jesu4127/scratch/merra_2/MERRA2_*instM_2d_asm_Nx.YYYYMM.nc4  (T2M, K)
JRA-3Q  /soge-home/users/jesu4127/scratch/jra_3q/915830.tmp2m-hgt-an-gauss.jra3q.anl_surf.0_0_0.tmp2m-hgt-an-gauss.*.nc
CHIRTS  /soge-home/users/jesu4127/data/chirts/{Tmax,Tmin}/{Tmax,Tmin}.YYYY.nc
        Tmean = (Tmax + Tmin) / 2
MLOST   /soge-home/users/jesu4127/data/mlost/gridded/NOAAGlobalTemp_*.nc
    Monthly temperature anomalies in degrees C

YEAR HANDLING
-------------
ERA5, MERRA2, JRA-3Q, MLOST, and CRUTEM use their maximal shared period.
CHIRTS is restricted to 1983-2016. October regional series use
cosine-latitude area weighting.

SPATIAL DOMAIN
--------------
10-40 E, 35-10 S

STATISTICS
----------
OLS only (exploratory pass, ahead of Mann-Kendall / Theil-Sen). No
interpolation between grids in this pass.

PANELS
------
(a) Monthly grid-cell OLS trend distributions (boxplots), one box per
    dataset per month, in colour.
(b)/(c)/(d)/(e)/(f)/(g) October OLS trend maps for ERA5 / MERRA2 / JRA-3Q /
    CHIRTS / MLOST / CRUTEM.
Overlaid area-weighted October regional mean series for all six datasets.

OUTPUTS
-------
southern_africa_temperature_trends_jra_3q_era5_merra2_chirts_noaa_one_timeseries.png
era5_october_trend_C_per_decade.npy
merra2_october_trend_C_per_decade.npy
jra3q_october_trend_C_per_decade.npy
chirts_october_trend_C_per_decade.npy
mlost_october_trend_C_per_decade.npy
crutem_october_trend_C_per_decade.npy
monthly_trends_summary.csv
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

try:
    from osgeo import ogr

    OGR_AVAILABLE = True
except ImportError:
    OGR_AVAILABLE = False

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    # Optional: point this at a local Natural Earth cache when working offline.
    CARTOPY_DATA_DIR = None
    if CARTOPY_DATA_DIR is not None:
        import cartopy
        cartopy.config["data_dir"] = str(CARTOPY_DATA_DIR)

    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False


# ============================================================================
# CONFIGURATION
# ============================================================================

ERA5_FILE = Path("/soge-home/users/jesu4127/scratch/era5/global_2m_temperature.nc")
ERA5_VAR = "t2m"

MERRA2_DIR = Path("/soge-home/users/jesu4127/scratch/merra_2")
MERRA2_PATTERN = "MERRA2_*instM_2d_asm_Nx.*.nc4"
MERRA2_VAR = "T2M"

CHIRTS_TMAX_DIR = Path("/soge-home/users/jesu4127/data/chirts/Tmax")
CHIRTS_TMIN_DIR = Path("/soge-home/users/jesu4127/data/chirts/Tmin")
MLOST_FILE = Path("/soge-home/users/jesu4127/data/mlost/gridded/NOAAGlobalTemp_v6.1.0_gridded_s185001_e202607_c20260807T143156.nc")
MLOST_VAR = "anom"
MLOST_LABEL = "NOAA Global Temp v6.1.0"
MLOST_SHORT_LABEL = "NOAA Global Temp"
CRUTEM_FILE = Path("/soge-home/users/jesu4127/data/crutem/CRUTEM.5.1.0.0.anomalies.nc")
CRUTEM_VAR = "tas"
CRUTEM_LABEL = "CRUTEM5"
CRUTEM_MAX_NAN_FRACTION = 0.50

JRA_3Q_DIR = Path("/soge-home/users/jesu4127/scratch/jra_3q/")
JRA_3Q_PATTERN = "915830.tmp2m-hgt-an-gauss.jra3q.anl_surf.0_0_0.tmp2m-hgt-an-gauss.*.nc"
JRA_3Q_VAR = "tmp2m-hgt-an-gauss"


OUT_DIR = Path("/soge-home/users/jesu4127/p1_plots/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LON_MIN, LON_MAX = 10.0, 40.0
LAT_MIN, LAT_MAX = -35.0, -10.0

TARGET_MONTH = 10
TARGET_MONTH_NAME = "October"
LATE_START_YEAR = 2000

TIME_CHUNK = 50
MIN_YEARS = 20
TARGET_START_YEAR = 1983
TARGET_END_YEAR = 2016
SHARED_YEARS = list(range(TARGET_START_YEAR, TARGET_END_YEAR + 1))
CLIM_START_YEAR = 1991
CLIM_END_YEAR = 2020

FIGSIZE = (18, 12)
FIG_DPI = 300

# Requested dataset colours: ERA5, MERRA2, JRA-3Q, CHIRTS, MLOST, CRUTEM5.
COLORS = {
    "ERA5": "#FF7C7E",
    "MERRA2": "#FEC300",
    "CHIRTS": "#9E3F71",
    "MLOST": "#31BCBA",
    "CRUTEM": "#2F5597",
    "JRA_3Q": "#016ff2",
}

TREND_LEVELS = np.linspace(-0.6, 0.6, 25)

LOCAL_COASTLINE_FILE = Path("/usr/share/magics/10m/ne_10m_land.shp")
LOCAL_BORDER_FILE = Path("/usr/share/magics/10m/ne_10m_admin_0_boundary_lines_land.shp")


# ============================================================================
# GENERIC HELPERS
# ============================================================================

def find_coord(ds, candidates):
    """Return the first coordinate/dim/variable name found in ds from candidates."""
    for name in candidates:
        if name in ds.coords or name in ds.dims or name in ds.variables:
            return name
    raise KeyError(f"Could not find coordinate from {candidates}. Available: {list(ds.coords)}")


def normalise_longitudes(ds, lon_name):
    """Convert 0-360 longitudes to -180..180 and sort."""
    lon = ds[lon_name].values.astype(float)
    if np.nanmax(lon) > 180.0:
        ds = ds.assign_coords({lon_name: ((lon + 180.0) % 360.0) - 180.0})
        ds = ds.sortby(lon_name)
    return ds


def flatten_valid(data_array):
    """Return finite values as a 1-D NumPy array."""
    values = np.asarray(data_array.values, dtype=float).ravel()
    return values[np.isfinite(values)]


def standardise_grid(data, lat_name, lon_name):
    """Rename to lat/lon, sort lat ascending, clip to the study region."""
    data = data.transpose(data.dims[0], lat_name, lon_name)
    if float(data[lat_name].values[0]) > float(data[lat_name].values[-1]):
        data = data.sortby(lat_name)
    data = data.sel({lat_name: slice(LAT_MIN, LAT_MAX), lon_name: slice(LON_MIN, LON_MAX)})
    return data.rename({lat_name: "lat", lon_name: "lon"})


def years_of(data):
    return set(np.unique(data.time.dt.year.values.astype(int)))


def restrict_to_years(data, years):
    years = np.asarray(sorted(years))
    return data.where(data.time.dt.year.isin(years), drop=True)


def calculate_start_years(data, month=None):
    """Return the first valid year at each grid point for an optional month."""
    monthly = data.resample(time="1MS").mean("time", skipna=True)
    if month is not None:
        monthly = monthly.where(monthly.time.dt.month == month, drop=True)
    years = xr.DataArray(monthly.time.dt.year.values, dims="time", coords={"time": monthly.time})
    return years.where(monthly.notnull()).min("time", skipna=True)


def calculate_monthly_anomalies(data):
    """Subtract each dataset's monthly climatology from its monthly values."""
    baseline = data.where(
        data.time.dt.year.isin(range(CLIM_START_YEAR, CLIM_END_YEAR + 1)),
        drop=True,
    )
    if baseline.sizes.get("time", 0) == 0:
        raise RuntimeError(f"No data available for the {CLIM_START_YEAR}-{CLIM_END_YEAR} climatology.")

    available_years = years_of(baseline)
    if max(available_years) < CLIM_END_YEAR:
        print(f"Warning: using available {min(available_years)}-{max(available_years)} data "
              f"for the {CLIM_START_YEAR}-{CLIM_END_YEAR} climatology.")

    climatology = baseline.groupby("time.month").mean("time", skipna=True)
    return data.groupby("time.month") - climatology


def plot_local_boundaries(ax):
    """Draw local Natural Earth coastlines and country borders on Matplotlib axes."""
    if not OGR_AVAILABLE:
        return

    for path, linewidth in ((LOCAL_COASTLINE_FILE, 0.8), (LOCAL_BORDER_FILE, 0.7)):
        if not path.exists():
            continue
        datasource = ogr.Open(str(path))
        if datasource is None:
            continue
        layer = datasource.GetLayer()

        def draw_geometry(geometry):
            if geometry.GetGeometryCount() == 0:
                points = geometry.GetPoints()
                if len(points) > 1:
                    coordinates = np.asarray(points, dtype=float)
                    plot_kwargs = {"transform": ccrs.PlateCarree()} if CARTOPY_AVAILABLE else {}
                    ax.plot(coordinates[:, 0], coordinates[:, 1], color="black",
                        linewidth=linewidth, zorder=5, **plot_kwargs)
                return
            for index in range(geometry.GetGeometryCount()):
                draw_geometry(geometry.GetGeometryRef(index))

        for feature in layer:
            geometry = feature.GetGeometryRef()
            if geometry is not None:
                draw_geometry(geometry)
        datasource = None


# ============================================================================
# LOAD ERA5
# ============================================================================

def load_era5():
    print("\n" + "=" * 78 + "\nLOADING ERA5\n" + "=" * 78)
    print(f"File: {ERA5_FILE}")

    if not ERA5_FILE.exists():
        raise FileNotFoundError(f"ERA5 file not found:\n{ERA5_FILE}")

    ds = xr.open_dataset(ERA5_FILE, decode_times=True)
    try:
        if ERA5_VAR not in ds:
            raise KeyError(f"ERA5 variable '{ERA5_VAR}' not found. Variables: {list(ds.data_vars)}")

        lat_name = find_coord(ds, ["latitude", "lat", "Latitude", "LAT", "y"])
        lon_name = find_coord(ds, ["longitude", "lon", "Longitude", "LON", "x"])
        ds = normalise_longitudes(ds, lon_name)

        data = ds[ERA5_VAR].rename({"valid_time": "time"} if "valid_time" in ds[ERA5_VAR].dims else {})
        if "valid_time" in data.dims:
            data = data.rename({"valid_time": "time"})
        data = standardise_grid(data, lat_name, lon_name)
        data = data - 273.15  # K -> degC

        years = np.unique(data.time.dt.year.values.astype(int))
        print(f"Coverage: {years.min()}-{years.max()} ({len(years)} years)")
        print(f"Grid: {data.sizes['lat']} x {data.sizes['lon']}")
        return data.load()
    finally:
        ds.close()


# ============================================================================
# LOAD MERRA2
# ============================================================================

def discover_merra2_files():
    """Return {year: {month: filepath}}."""
    regex = re.compile(r"instM_2d_asm_Nx\.(\d{4})(\d{2})\.nc4$")
    files: dict[int, dict[int, Path]] = {}
    matches = list(MERRA2_DIR.glob(MERRA2_PATTERN))
    if not matches:
        raise FileNotFoundError(f"No MERRA2 files matching {MERRA2_PATTERN} found in {MERRA2_DIR}")

    for path in matches:
        match = regex.search(path.name)
        if match is None:
            continue
        year, month = int(match.group(1)), int(match.group(2))
        files.setdefault(year, {})[month] = path
    return files


def load_merra2():
    print("\n" + "=" * 78 + "\nLOADING MERRA2\n" + "=" * 78)
    print(f"Directory: {MERRA2_DIR}")

    files = discover_merra2_files()
    complete_years = [y for y in sorted(files) if all(m in files[y] for m in range(1, 13))]
    if not complete_years:
        raise RuntimeError("No complete MERRA2 years found.")
    print(f"Complete coverage: {complete_years[0]}-{complete_years[-1]} ({len(complete_years)} years)")

    paths = [files[y][m] for y in complete_years for m in range(1, 13)]
    print(f"Files to open: {len(paths)}")

    ds = xr.open_mfdataset(
        paths, combine="nested", concat_dim="time", decode_times=True,
        chunks={"time": TIME_CHUNK}, data_vars="minimal", coords="minimal",
        compat="override", combine_attrs="override",
    )

    if MERRA2_VAR not in ds:
        ds.close()
        raise KeyError(f"MERRA2 variable '{MERRA2_VAR}' not found. Variables: {list(ds.data_vars)}")

    lat_name = find_coord(ds, ["lat", "latitude", "Latitude", "LAT", "y"])
    lon_name = find_coord(ds, ["lon", "longitude", "Longitude", "LON", "x"])
    ds = normalise_longitudes(ds, lon_name)

    data = ds[MERRA2_VAR]
    data = standardise_grid(data, lat_name, lon_name)
    data = data.where(np.abs(data) < 1e4)  # drop fill values
    data = data - 273.15  # K -> degC

    years = np.unique(data.time.dt.year.values.astype(int))
    print(f"Coverage: {years.min()}-{years.max()} ({len(years)} years)")
    print(f"Grid: {data.sizes['lat']} x {data.sizes['lon']}")
    return data, ds


# ============================================================================
# LOAD JRA-3Q
# ============================================================================

def discover_jra3q_files():
    """Return {year: {month: filepath}}."""
    regex = re.compile(r"\.tmp2m-hgt-an-gauss\.(\d{4})(\d{2})\d{2}00_\d{10}\.nc$")
    files: dict[int, dict[int, Path]] = {}
    matches = list(JRA_3Q_DIR.glob(JRA_3Q_PATTERN))
    if not matches:
        raise FileNotFoundError(f"No JRA-3Q files matching {JRA_3Q_PATTERN} found in {JRA_3Q_DIR}")

    for path in matches:
        match = regex.search(path.name)
        if match is None:
            continue
        year, month = int(match.group(1)), int(match.group(2))
        files.setdefault(year, {})[month] = path
    return files


def load_jra3q():
    print("\n" + "=" * 78 + "\nLOADING JRA-3Q\n" + "=" * 78)
    print(f"Directory: {JRA_3Q_DIR}")

    files = discover_jra3q_files()
    complete_years = [y for y in sorted(files) if all(m in files[y] for m in range(1, 13))]
    if not complete_years:
        raise RuntimeError("No complete JRA-3Q years found.")
    print(f"Complete coverage: {complete_years[0]}-{complete_years[-1]} ({len(complete_years)} years)")

    paths = [files[y][m] for y in complete_years for m in range(1, 13)]
    print(f"Files to open: {len(paths)}")

    ds = xr.open_mfdataset(
        paths, combine="nested", concat_dim="time", decode_times=True,
        chunks={"time": TIME_CHUNK}, data_vars="minimal", coords="minimal",
        compat="override", combine_attrs="override",
    )

    if JRA_3Q_VAR not in ds:
        ds.close()
        raise KeyError(f"JRA-3Q variable '{JRA_3Q_VAR}' not found. Variables: {list(ds.data_vars)}")

    lat_name = find_coord(ds, ["lat", "latitude", "Latitude", "LAT", "y"])
    lon_name = find_coord(ds, ["lon", "longitude", "Longitude", "LON", "x"])
    ds = normalise_longitudes(ds, lon_name)

    data = ds[JRA_3Q_VAR]
    data = standardise_grid(data, lat_name, lon_name)
    data = data.where(np.abs(data) < 1e4)  # drop fill values
    data = data - 273.15  # K -> degC

    years = np.unique(data.time.dt.year.values.astype(int))
    print(f"Coverage: {years.min()}-{years.max()} ({len(years)} years)")
    print(f"Grid: {data.sizes['lat']} x {data.sizes['lon']}")
    return data, ds


# ============================================================================
# LOAD CHIRTS
# ============================================================================

def load_chirts():
    print("\n" + "=" * 78 + "\nLOADING CHIRTS\n" + "=" * 78)

    max_regex = re.compile(r"Tmax\.(\d{4})\.nc$")
    min_regex = re.compile(r"Tmin\.(\d{4})\.nc$")

    tmax_files = {int(m.group(1)): p for p in CHIRTS_TMAX_DIR.glob("Tmax.*.nc")
                  if (m := max_regex.search(p.name))}
    tmin_files = {int(m.group(1)): p for p in CHIRTS_TMIN_DIR.glob("Tmin.*.nc")
                  if (m := min_regex.search(p.name))}

    common_years = sorted(set(tmax_files) & set(tmin_files))
    if not common_years:
        raise RuntimeError("No common CHIRTS Tmax/Tmin years found.")
    print(f"Coverage: {common_years[0]}-{common_years[-1]} ({len(common_years)} years)")

    max_paths = [tmax_files[y] for y in common_years]
    min_paths = [tmin_files[y] for y in common_years]
    chunks = {"time": TIME_CHUNK}

    print("Opening Tmax...")
    ds_max = xr.open_mfdataset(max_paths, combine="by_coords", decode_times=True, chunks=chunks,
                                data_vars="minimal", coords="minimal", compat="override",
                                combine_attrs="override")
    print("Opening Tmin...")
    ds_min = xr.open_mfdataset(min_paths, combine="by_coords", decode_times=True, chunks=chunks,
                                data_vars="minimal", coords="minimal", compat="override",
                                combine_attrs="override")

    lat_name = find_coord(ds_max, ["lat", "latitude", "Latitude", "LAT", "y"])
    lon_name = find_coord(ds_max, ["lon", "longitude", "Longitude", "LON", "x"])
    ds_max = normalise_longitudes(ds_max, lon_name)
    ds_min = normalise_longitudes(ds_min, lon_name)

    tmax = ds_max["Tmax"].transpose("time", lat_name, lon_name)
    tmin = ds_min["Tmin"].transpose("time", lat_name, lon_name)
    tmax, tmin = xr.align(tmax, tmin, join="inner")

    if float(tmax[lat_name].values[0]) > float(tmax[lat_name].values[-1]):
        tmax = tmax.sortby(lat_name)
        tmin = tmin.sortby(lat_name)

    tmax = tmax.where(np.abs(tmax) < 1e4)
    tmin = tmin.where(np.abs(tmin) < 1e4)

    print("Calculating daily Tmean...")
    tmean = (tmax + tmin) / 2.0
    tmean = standardise_grid(tmean, lat_name, lon_name)

    print(f"Grid: {tmean.sizes['lat']} x {tmean.sizes['lon']}")
    print(f"Days: {tmean.sizes['time']}")
    return tmean, ds_max, ds_min


def load_mlost():
    print("\n" + "=" * 78 + f"\nLOADING {MLOST_LABEL}\n" + "=" * 78)
    print(f"File: {MLOST_FILE}")

    if not MLOST_FILE.exists():
        raise FileNotFoundError(f"MLOST file not found:\n{MLOST_FILE}")

    ds = xr.open_dataset(MLOST_FILE, decode_times=True)
    if MLOST_VAR not in ds:
        ds.close()
        raise KeyError(f"MLOST variable '{MLOST_VAR}' not found. Variables: {list(ds.data_vars)}")

    lat_name = find_coord(ds, ["lat", "latitude", "Latitude", "LAT", "y"])
    lon_name = find_coord(ds, ["lon", "longitude", "Longitude", "LON", "x"])
    ds = normalise_longitudes(ds, lon_name)
    data = ds[MLOST_VAR].squeeze(drop=True)
    data = standardise_grid(data, lat_name, lon_name)
    data = data.where(np.abs(data) < 1e3)

    years = np.unique(data.time.dt.year.values.astype(int))
    print(f"Coverage: {years.min()}-{years.max()} ({len(years)} years)")
    print(f"Grid: {data.sizes['lat']} x {data.sizes['lon']}")
    return data.load(), ds


def load_crutem():
    print("\n" + "=" * 78 + f"\nLOADING {CRUTEM_LABEL}\n" + "=" * 78)
    print(f"File: {CRUTEM_FILE}")

    if not CRUTEM_FILE.exists():
        raise FileNotFoundError(f"CRUTEM file not found:\n{CRUTEM_FILE}")

    ds = xr.open_dataset(CRUTEM_FILE, decode_times=True)
    try:
        if CRUTEM_VAR not in ds:
            raise KeyError(f"CRUTEM variable '{CRUTEM_VAR}' not found. Variables: {list(ds.data_vars)}")

        lat_name = find_coord(ds, ["latitude", "lat", "Latitude", "LAT", "y"])
        lon_name = find_coord(ds, ["longitude", "lon", "Longitude", "LON", "x"])
        ds = normalise_longitudes(ds, lon_name)
        data = ds[CRUTEM_VAR].squeeze(drop=True)
        data = standardise_grid(data, lat_name, lon_name)
        data = data.where(np.abs(data) < 1e3).load()
        nan_fraction = data.isnull().mean(dim="time")
        valid_gridpoints = nan_fraction <= CRUTEM_MAX_NAN_FRACTION
        data = data.where(valid_gridpoints)
        total_gridpoints = nan_fraction.size
        retained_gridpoints = int(valid_gridpoints.sum())
        print(f"Grid points retained after NaN filter: {retained_gridpoints}/{total_gridpoints} "
              f"(threshold: {CRUTEM_MAX_NAN_FRACTION:.0%})")

        years = np.unique(data.time.dt.year.values.astype(int))
        print(f"Coverage: {years.min()}-{max(years)} ({len(years)} years)")
        print(f"Grid: {data.sizes['lat']} x {data.sizes['lon']}")
        return data
    finally:
        ds.close()


# ============================================================================
# YEAR RANGES: ERA5/MERRA2/JRA-3Q/MLOST/CRUTEM shared years; CHIRTS fixed 1983-2016
# ============================================================================

def resolve_year_ranges(era5, merra2, jra3q, chirts, mlost, crutem):
    era5_years, merra_years, jra3q_years = years_of(era5), years_of(merra2), years_of(jra3q)
    chirts_years, mlost_years, crutem_years = years_of(chirts), years_of(mlost), years_of(crutem)
    shared_common = sorted(era5_years & merra_years & jra3q_years & mlost_years & crutem_years)
    chirts_years = sorted(set(chirts_years) & set(SHARED_YEARS))
    mlost_years = sorted(mlost_years)

    if len(shared_common) < MIN_YEARS:
        raise RuntimeError(f"Only {len(shared_common)} common ERA5/MERRA2/JRA-3Q/MLOST/CRUTEM years.")
    if len(chirts_years) < MIN_YEARS:
        raise RuntimeError(f"Only {len(chirts_years)} CHIRTS years in {TARGET_START_YEAR}-{TARGET_END_YEAR}.")
    if len(mlost_years) < MIN_YEARS:
        raise RuntimeError(f"Only {len(mlost_years)} MLOST years available.")
    if len(crutem_years) < MIN_YEARS:
        raise RuntimeError(f"Only {len(crutem_years)} CRUTEM years available.")

    print("\n" + "=" * 78 + "\nYEAR RANGES\n" + "=" * 78)
    print(f"ERA5:               {min(era5_years)}-{max(era5_years)}")
    print(f"MERRA2:             {min(merra_years)}-{max(merra_years)}")
    print(f"JRA-3Q:             {min(jra3q_years)}-{max(jra3q_years)}")
    print(f"CHIRTS (used):      {chirts_years[0]}-{chirts_years[-1]} ({len(chirts_years)} years)")
    print(f"{MLOST_LABEL} (used): {mlost_years[0]}-{mlost_years[-1]} ({len(mlost_years)} years)")
    print(f"{CRUTEM_LABEL} (used): {shared_common[0]}-{shared_common[-1]} ({len(shared_common)} years)")
    print(f"Shared ERA5/MERRA2/JRA-3Q/{MLOST_LABEL}/{CRUTEM_LABEL} years: "
          f"{shared_common[0]}-{shared_common[-1]} ({len(shared_common)} years)")

    era5 = restrict_to_years(era5, shared_common)
    merra2 = restrict_to_years(merra2, shared_common)
    jra3q = restrict_to_years(jra3q, shared_common)
    chirts = restrict_to_years(chirts, chirts_years)
    mlost = restrict_to_years(mlost, shared_common)
    crutem = restrict_to_years(crutem, shared_common)

    # ERA5, MERRA2, JRA-3Q, and CRUTEM all share the same restricted year set
    # here (shared_common); MLOST keeps its own fuller record.
    return (era5, merra2, jra3q, chirts, mlost, crutem,
            np.array(shared_common), np.array(shared_common),
            np.array(chirts_years), np.array(mlost_years))


# ============================================================================
# TREND CALCULATIONS
# ============================================================================

def _ols_slope_per_decade(annual_or_monthly, time_dim="time"):
    """Fit OLS slope per grid cell along time and convert to degC/decade."""
    years = annual_or_monthly[time_dim].dt.year.astype("float64")
    data = annual_or_monthly.assign_coords(year=(time_dim, years.data)).swap_dims({time_dim: "year"})
    data = data.drop_vars(time_dim, errors="ignore")
    fit = data.polyfit(dim="year", deg=1, skipna=True)
    return (fit["polyfit_coefficients"].sel(degree=1) * 10.0).compute()


def calculate_monthly_trends(data):
    """{month: grid-cell OLS slope, degC/decade}, computed independently per calendar month."""
    monthly = data.resample(time="1MS").mean("time", skipna=True)
    trends = {}
    for month in range(1, 13):
        print(f"    {pd.Timestamp(2000, month, 1):%b}")
        series = monthly.where(monthly.time.dt.month == month, drop=True)
        trends[month] = _ols_slope_per_decade(series)
    return trends


def calculate_october_trend(data):
    """Grid-cell OLS trend of October mean temperature, degC/decade."""
    october = data.where(data.time.dt.month == TARGET_MONTH, drop=True)
    october_annual = october.resample(time="1YS").mean("time", skipna=True)
    return _ols_slope_per_decade(october_annual)


def calculate_october_area_weighted_series(data):
    """Return October area-weighted regional means for each year."""
    october = data.where(data.time.dt.month == TARGET_MONTH, drop=True)
    october_annual = october.resample(time="1YS").mean("time", skipna=True)
    weights = np.cos(np.deg2rad(october_annual.lat))
    return october_annual.weighted(weights).mean(dim=("lat", "lon"), skipna=True).compute()


# ============================================================================
# SUMMARY TABLE
# ============================================================================

def make_summary(era5_monthly, merra_monthly, jra3q_monthly, chirts_monthly,
                 mlost_monthly, crutem_monthly):
    rows = []
    for month in range(1, 13):
        month_name = pd.Timestamp(2000, month, 1).strftime("%b")
        datasets = {"ERA5": era5_monthly[month], "MERRA2": merra_monthly[month],
            "JRA_3Q": jra3q_monthly[month],
            "CHIRTS": chirts_monthly[month], "MLOST": mlost_monthly[month],
            "CRUTEM": crutem_monthly[month]}
        for name, data in datasets.items():
            values = flatten_valid(data)
            rows.append({
                "dataset": name, "month": month_name, "month_number": month,
                "n_gridcells": values.size,
                "mean_C_per_decade": np.mean(values),
                "q25_C_per_decade": np.percentile(values, 25),
                "median_C_per_decade": np.median(values),
                "q75_C_per_decade": np.percentile(values, 75),
                "min_C_per_decade": np.min(values),
                "max_C_per_decade": np.max(values),
            })
    return pd.DataFrame(rows)


# ============================================================================
# PLOTTING
# ============================================================================

def _format_lon(x, _pos=None):
    if x < 0:
        return f"{abs(x):.0f}\u00b0W"
    if x > 0:
        return f"{x:.0f}\u00b0E"
    return "0\u00b0"


def _format_lat(y, _pos=None):
    if y < 0:
        return f"{abs(y):.0f}\u00b0S"
    if y > 0:
        return f"{y:.0f}\u00b0N"
    return "0\u00b0"


def plot_map_panel(ax, trend, dataset_name, year_range, panel_label, use_cell_grid=False, start_years=None):
    """
        Cartopy supplies the geographic coastlines and country borders when available.

        Titling: the dataset name is drawn as a bold header above the panel
        (e.g. "ERA5"), with the year range shown as a smaller subtitle
        immediately above the axes (e.g. "(1983-2016)").
    """
    kwargs = dict(levels=TREND_LEVELS, cmap="RdBu_r", extend="both")

    plot_data = trend

    if CARTOPY_AVAILABLE:
        if use_cell_grid:
            mesh = ax.pcolormesh(plot_data.lon.values, plot_data.lat.values, plot_data.values,
                                 transform=ccrs.PlateCarree(), cmap="RdBu_r",
                                 vmin=TREND_LEVELS.min(), vmax=TREND_LEVELS.max(), shading="nearest")
        else:
            mesh = ax.contourf(plot_data.lon.values, plot_data.lat.values, plot_data.values,
                                transform=ccrs.PlateCarree(), **kwargs)
        ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=1.0,
                   edgecolor="black", zorder=5)
        ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.9,
                   edgecolor="black", linestyle="-", zorder=5)
        ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4, linestyle="--")
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"fontsize": 8}
        gl.ylabel_style = {"fontsize": 8}
    else:
        # No cartopy on this system -> plain axes, but still styled to read like
        # a geographic map (degree-formatted ticks, dashed graticule).
        if use_cell_grid:
            mesh = ax.pcolormesh(plot_data.lon.values, plot_data.lat.values, plot_data.values,
                                 cmap="RdBu_r", vmin=TREND_LEVELS.min(),
                                 vmax=TREND_LEVELS.max(), shading="nearest")
        else:
            mesh = ax.contourf(plot_data.lon.values, plot_data.lat.values, plot_data.values, **kwargs)
        ax.set_xlim(LON_MIN, LON_MAX)
        ax.set_ylim(LAT_MIN, LAT_MAX)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_format_lon))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_format_lat))
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.35, linewidth=0.4, linestyle="--")
        ax.set_aspect("equal")
        plot_local_boundaries(ax)

    if CARTOPY_AVAILABLE:
        plot_local_boundaries(ax)

    if start_years is not None:
        late_start = np.isfinite(start_years.values) & (start_years.values > LATE_START_YEAR)
        if np.any(late_start):
            lon_grid, lat_grid = np.meshgrid(start_years.lon.values, start_years.lat.values)
            scatter_kwargs = {"transform": ccrs.PlateCarree()} if CARTOPY_AVAILABLE else {}
            ax.scatter(lon_grid[late_start], lat_grid[late_start], marker="*", s=24,
                       color="black", edgecolors="white", linewidths=0.35,
                       zorder=7, **scatter_kwargs)

    # Year-range subtitle sits close to the axes (standard title position).
    ax.set_title(f"({year_range})", fontsize=9, pad=6)
    # Dataset name header sits above that, bold and slightly larger, so each
    # panel reads "<Dataset Name>" above "(<years>)" above the map itself.
    ax.text(0.5, 1.16, dataset_name, transform=ax.transAxes, fontsize=12,
            fontweight="bold", ha="center", va="bottom")
    ax.text(0.02, 0.96, panel_label, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top",
            bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none", alpha=0.7))
    return mesh


def plot_monthly_boxplot(ax, era5_monthly, merra_monthly, jra3q_monthly,
                          chirts_monthly, mlost_monthly, crutem_monthly,
                          crutem_start_years, era5_merra_years, jra3q_years,
                          chirts_years, mlost_years):
    datasets = {
        "ERA5": (era5_monthly, -0.15),
        "MERRA2": (merra_monthly, -0.09),
        "JRA_3Q": (jra3q_monthly, -0.03),
        "CHIRTS": (chirts_monthly, 0.03),
        "MLOST": (mlost_monthly, 0.09),
        "CRUTEM": (crutem_monthly, 0.15),
    }
    centres = np.arange(1, 13, dtype=float)

    for name, (monthly, offset) in datasets.items():
        values = [flatten_valid(monthly[m]) for m in range(1, 13)]
        bp = ax.boxplot(values, positions=centres + offset, widths=0.06,
                         patch_artist=True, showfliers=False, manage_ticks=False)
        for box in bp["boxes"]:
            box.set_facecolor(COLORS[name])
            box.set_edgecolor("black")
            box.set_linewidth(0.9)
            box.set_alpha(0.9)
        for key in ("whiskers", "caps", "medians"):
            for artist in bp[key]:
                artist.set_color("black")
                artist.set_linewidth(0.9)

    for month, start_years in crutem_start_years.items():
        if np.any(np.isfinite(start_years.values) & (start_years.values > LATE_START_YEAR)):
            median_y = bp["medians"][month - 1].get_ydata().mean()
            ax.plot(centres[month - 1] + 0.15, median_y, marker="*", markersize=7,
                    markerfacecolor="black", markeredgecolor="white", markeredgewidth=0.4,
                    linestyle="None", zorder=6)

    ax.axhline(0, linestyle="--", linewidth=0.8, color="black")
    ax.set_xlim(0.6, 12.8)
    ax.set_xticks(centres)
    ax.set_xticklabels([pd.Timestamp(2000, m, 1).strftime("%b") for m in range(1, 13)])
    ax.set_xlabel("Month", labelpad=8, fontsize=10)
    ax.set_ylabel(r"Tmean trend ($^\circ$C decade$^{-1}$)", fontsize=10)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    # Panel label sits top-right so it never collides with the legend (top-left).
    ax.text(0.99, 0.97, "(a)", transform=ax.transAxes, fontsize=12, fontweight="bold",
            va="top", ha="right")

    era5_merra_label = f"{era5_merra_years[0]}-{era5_merra_years[-1]}"
    jra3q_label = f"{jra3q_years[0]}-{jra3q_years[-1]}"
    chirts_label = f"{chirts_years[0]}-{chirts_years[-1]}"

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORS["ERA5"], edgecolor="black",
                   label=f"ERA5 ({era5_merra_label})"),
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORS["MERRA2"], edgecolor="black",
                       label=f"MERRA2 ({era5_merra_label})"),
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORS["JRA_3Q"], edgecolor="black",
                       label=f"JRA-3Q ({jra3q_label})"),
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORS["CHIRTS"], edgecolor="black",
                   label=f"CHIRTS ({chirts_label})"),
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORS["MLOST"], edgecolor="black",
               label=f"{MLOST_LABEL} ({mlost_years[0]}-{mlost_years[-1]})"),
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORS["CRUTEM"], edgecolor="black",
               label=f"{CRUTEM_LABEL} ({era5_merra_years[0]}-{era5_merra_years[-1]})"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.04),
              frameon=False, fontsize=9, ncol=6, borderaxespad=0.0)


def make_full_figure(era5_monthly, merra_monthly, jra3q_monthly, chirts_monthly, mlost_monthly,
                  crutem_monthly, crutem_start_years, crutem_oct_start_years,
                  era5_oct, merra_oct, jra3q_oct, chirts_oct,
                  mlost_oct, crutem_oct, era5_annual, merra_annual, jra3q_annual, chirts_annual,
                  mlost_annual, crutem_annual,
                      era5_merra_years, jra3q_years, chirts_years, mlost_years):
    """
    Layout:

        row 0: (a) monthly boxplot, full width
        row 1: (b) (c) (d) (e) (f) (g) October trend maps, one per dataset
        row 2: overlaid area-weighted October series for all datasets

    """
    print("\n" + "=" * 78 + "\nCREATING FULL FIGURE\n" + "=" * 78)

    fig = plt.figure(figsize=FIGSIZE, dpi=FIG_DPI)
    # Extra height on row 1 (and hspace) makes room for the dataset-name
    # header text drawn above each October-trend map panel.
    gs = fig.add_gridspec(3, 6, height_ratios=[1.0, 1.25, 0.8], hspace=0.85, wspace=0.28)

    ax_box = fig.add_subplot(gs[0, :])

    if CARTOPY_AVAILABLE:
        projection = ccrs.PlateCarree()
        ax_era5 = fig.add_subplot(gs[1, 0], projection=projection)
        ax_merra = fig.add_subplot(gs[1, 1], projection=projection)
        ax_jra3q = fig.add_subplot(gs[1, 2], projection=projection)
        ax_chirts = fig.add_subplot(gs[1, 3], projection=projection)
        ax_mlost = fig.add_subplot(gs[1, 4], projection=projection)
        ax_crutem = fig.add_subplot(gs[1, 5], projection=projection)
    else:
        ax_era5 = fig.add_subplot(gs[1, 0])
        ax_merra = fig.add_subplot(gs[1, 1])
        ax_jra3q = fig.add_subplot(gs[1, 2])
        ax_chirts = fig.add_subplot(gs[1, 3])
        ax_mlost = fig.add_subplot(gs[1, 4])
        ax_crutem = fig.add_subplot(gs[1, 5])

    ax_series = fig.add_subplot(gs[2, :])

    plot_monthly_boxplot(ax_box, era5_monthly, merra_monthly, jra3q_monthly, chirts_monthly, mlost_monthly,
                         crutem_monthly, crutem_start_years,
                          era5_merra_years, jra3q_years, chirts_years, mlost_years)

    era5_merra_label = f"{era5_merra_years[0]}-{era5_merra_years[-1]}"
    jra3q_label = f"{jra3q_years[0]}-{jra3q_years[-1]}"
    chirts_label = f"{chirts_years[0]}-{chirts_years[-1]}"
    mlost_label = f"{mlost_years[0]}-{mlost_years[-1]}"

    # Dataset name and year range are now passed separately so plot_map_panel
    # can draw the dataset name as a bold header above each panel, with the
    # year range as a smaller subtitle beneath it.
    mesh = plot_map_panel(ax_era5, era5_oct, "ERA5", era5_merra_label, "(b)")
    plot_map_panel(ax_merra, merra_oct, "MERRA-2", era5_merra_label, "(c)")
    plot_map_panel(ax_jra3q, jra3q_oct, "JRA-3Q", jra3q_label, "(d)")
    plot_map_panel(ax_chirts, chirts_oct, "CHIRTS", chirts_label, "(e)")
    plot_map_panel(ax_mlost, mlost_oct, MLOST_SHORT_LABEL, mlost_label, "(f)",
                   use_cell_grid=True)
    plot_map_panel(ax_crutem, crutem_oct, CRUTEM_LABEL, era5_merra_label, "(g)",
                   use_cell_grid=True, start_years=crutem_oct_start_years)
    fig.text(0.5, 0.66, "October trends", ha="center", va="center",
             fontsize=14, fontweight="bold")

    for ax in (ax_merra, ax_jra3q, ax_chirts, ax_mlost, ax_crutem):
        if not CARTOPY_AVAILABLE:
            ax.set_ylabel("")

    cbar = fig.colorbar(mesh, ax=[ax_era5, ax_merra, ax_jra3q, ax_chirts, ax_mlost, ax_crutem],
                         orientation="horizontal", fraction=0.06, pad=0.1, aspect=45,
                         shrink=0.9, extend="both")
    cbar.set_label(r"October Tmean trend ($^\circ$C decade$^{-1}$)")

    plotted_series = (era5_annual, merra_annual, jra3q_annual, chirts_annual,
                      mlost_annual, crutem_annual)
    years = np.concatenate([
        series.time.dt.year.values.astype(int) for series in plotted_series
    ])
    series_values = np.concatenate([
        np.asarray(series.values, dtype=float).ravel() for series in plotted_series
    ])
    series_values = series_values[np.isfinite(series_values)]
    series_min = np.min(series_values)
    series_max = np.max(series_values)
    series_margin = max((series_max - series_min) * 0.05, 0.5)
    shared_series_ylim = (series_min - series_margin, series_max + series_margin)

    for series, name, color in [
        (era5_annual, "ERA5", COLORS["ERA5"]),
        (merra_annual, "MERRA2", COLORS["MERRA2"]),
        (jra3q_annual, "JRA-3Q", COLORS["JRA_3Q"]),
        (chirts_annual, "CHIRTS", COLORS["CHIRTS"]),
        (mlost_annual, MLOST_LABEL, COLORS["MLOST"]),
        (crutem_annual, CRUTEM_LABEL, COLORS["CRUTEM"]),
    ]:
        ax_series.plot(series.time.dt.year.values, series.values, color=color,
                       linewidth=2, label=name)

    ax_series.set_title("October area-weighted anomalies", fontsize=12, pad=10)
    ax_series.set_xlim(years.min(), years.max())
    ax_series.set_ylim(shared_series_ylim)
    ax_series.set_xlabel("Year")
    ax_series.set_ylabel(r"October anomaly ($^\circ$C)", labelpad=8)
    ax_series.grid(alpha=0.25)
    ax_series.tick_params(labelsize=8)
    ax_series.text(0.01, 0.97, "(h)", transform=ax_series.transAxes,
                   fontsize=12, fontweight="bold", va="top")
    ax_series.legend(loc="upper left", frameon=False, ncol=6)

    fig.suptitle("Southern Africa temperature trends", fontsize=17, fontweight="bold", y=0.97)

    output = OUT_DIR / "southern_africa_temperature_trends_jra_3q_era5_merra2_chirts_noaa_one_timeseries.png"
    fig.savefig(output, dpi=FIG_DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved figure:\n  {output}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 78 + "\nERA5 / MERRA2 / JRA-3Q / CHIRTS / MLOST / CRUTEM5 RECONSTRUCTION\n" + "=" * 78)
    print("Trend method: OLS")
    print(f"Region: {LON_MIN}-{LON_MAX} E, {abs(LAT_MIN)}-{abs(LAT_MAX)} S")
    print(f"Spatial map month: {TARGET_MONTH_NAME}")
    print(f"Year handling: ERA5/MERRA2/JRA-3Q/{MLOST_LABEL}/{CRUTEM_LABEL} shared years; "
          f"CHIRTS restricted to {TARGET_START_YEAR}-{TARGET_END_YEAR}.")

    era5 = load_era5()
    merra2, merra2_ds = load_merra2()
    jra3q, jra3q_ds = load_jra3q()
    chirts, chirts_max_ds, chirts_min_ds = load_chirts()
    mlost, mlost_ds = load_mlost()
    crutem = load_crutem()

    try:
        print("\n" + "=" * 78 + "\nCALCULATING MONTHLY ANOMALIES\n" + "=" * 78)
        era5 = calculate_monthly_anomalies(era5)
        merra2 = calculate_monthly_anomalies(merra2)
        jra3q = calculate_monthly_anomalies(jra3q)
        chirts = calculate_monthly_anomalies(chirts)
        mlost = calculate_monthly_anomalies(mlost)
        crutem = calculate_monthly_anomalies(crutem)

        (era5, merra2, jra3q, chirts, mlost, crutem, era5_merra_years,
         jra3q_years, chirts_years, mlost_years) = resolve_year_ranges(
            era5, merra2, jra3q, chirts, mlost, crutem
        )
        crutem_start_years = {
            month: calculate_start_years(crutem, month=month)
            for month in range(1, 13)
        }
        crutem_oct_start_years = crutem_start_years[TARGET_MONTH]

        print("\n" + "=" * 78 + "\nMONTHLY GRID-CELL TRENDS\n" + "=" * 78)
        print("\nERA5:")
        era5_monthly = calculate_monthly_trends(era5)
        print("\nMERRA2:")
        merra_monthly = calculate_monthly_trends(merra2)
        print("\nJRA-3Q:")
        jra3q_monthly = calculate_monthly_trends(jra3q)
        print("\nCHIRTS:")
        chirts_monthly = calculate_monthly_trends(chirts)
        print("\nMLOST:")
        mlost_monthly = calculate_monthly_trends(mlost)
        print(f"\n{CRUTEM_LABEL}:")
        crutem_monthly = calculate_monthly_trends(crutem)

        print("\n" + "=" * 78 + "\nOCTOBER SPATIAL TRENDS\n" + "=" * 78)
        print("\nERA5:")
        era5_oct = calculate_october_trend(era5)
        print("\nMERRA2:")
        merra_oct = calculate_october_trend(merra2)
        print("\nJRA-3Q:")
        jra3q_oct = calculate_october_trend(jra3q)
        print("\nCHIRTS:")
        chirts_oct = calculate_october_trend(chirts)
        print("\nMLOST:")
        mlost_oct = calculate_october_trend(mlost)
        print(f"\n{CRUTEM_LABEL}:")
        crutem_oct = calculate_october_trend(crutem)

        print("\n" + "=" * 78 + "\nAREA-WEIGHTED OCTOBER SERIES\n" + "=" * 78)
        era5_annual = calculate_october_area_weighted_series(era5)
        merra_annual = calculate_october_area_weighted_series(merra2)
        jra3q_annual = calculate_october_area_weighted_series(jra3q)
        chirts_annual = calculate_october_area_weighted_series(chirts)
        mlost_annual = calculate_october_area_weighted_series(mlost)
        crutem_annual = calculate_october_area_weighted_series(crutem)

        print("\n" + "=" * 78 + "\nOCTOBER DIAGNOSTICS\n" + "=" * 78)
        for name, data in [("ERA5", era5_oct), ("MERRA2", merra_oct), ("JRA-3Q", jra3q_oct),
                   ("CHIRTS", chirts_oct), ("MLOST", mlost_oct),
                   (CRUTEM_LABEL, crutem_oct)]:
            values = flatten_valid(data)
            print(f"\n{name}")
            print(f"  valid pixels: {values.size:,}")
            print(f"  minimum: {np.min(values):+.4f}")
            print(f"  mean:    {np.mean(values):+.4f}")
            print(f"  median:  {np.median(values):+.4f}")
            print(f"  maximum: {np.max(values):+.4f}")

        summary = make_summary(era5_monthly, merra_monthly, jra3q_monthly, chirts_monthly, mlost_monthly,
                       crutem_monthly)
        summary_path = OUT_DIR / "monthly_trends_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"\nSaved summary:\n  {summary_path}")

        np.save(OUT_DIR / "era5_october_trend_C_per_decade.npy", era5_oct.values)
        np.save(OUT_DIR / "merra2_october_trend_C_per_decade.npy", merra_oct.values)
        np.save(OUT_DIR / "jra3q_october_trend_C_per_decade.npy", jra3q_oct.values)
        np.save(OUT_DIR / "chirts_october_trend_C_per_decade.npy", chirts_oct.values)
        np.save(OUT_DIR / "mlost_october_trend_C_per_decade.npy", mlost_oct.values)
        np.save(OUT_DIR / "crutem_october_trend_C_per_decade.npy", crutem_oct.values)

        make_full_figure(era5_monthly, merra_monthly, jra3q_monthly, chirts_monthly, mlost_monthly,
              crutem_monthly, crutem_start_years, crutem_oct_start_years,
              era5_oct, merra_oct, jra3q_oct, chirts_oct, mlost_oct, crutem_oct,
              era5_annual, merra_annual, jra3q_annual, chirts_annual, mlost_annual, crutem_annual,
                          era5_merra_years, jra3q_years, chirts_years, mlost_years)

        print("\n" + "=" * 78 + "\nDONE\n" + "=" * 78)
        print(f"ERA5/MERRA2/JRA-3Q period: {era5_merra_years[0]}-{era5_merra_years[-1]}")
        print(f"CHIRTS period: {TARGET_START_YEAR}-{TARGET_END_YEAR}")

    finally:
        merra2_ds.close()
        jra3q_ds.close()
        chirts_max_ds.close()
        chirts_min_ds.close()
        mlost_ds.close()


if __name__ == "__main__":
    main()