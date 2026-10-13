
"""
DATASETS
--------
ERA5    /soge-home/users/jesu4127/scratch/era5/global_2m_temperature.nc   (t2m, K)
MERRA2  /soge-home/users/jesu4127/scratch/merra_2/MERRA2_*instM_2d_asm_Nx.YYYYMM.nc4  (T2M, K)
JRA-3Q  /soge-home/users/jesu4127/scratch/jra_3q/915830.tmp2m-hgt-an-gauss.jra3q.anl_surf.0_0_0.tmp2m-hgt-an-gauss.*.nc
CHIRTS  /soge-home/users/jesu4127/data/chirts/{Tmax,Tmin}/{Tmax,Tmin}.YYYY.nc
        Tmean = (Tmax + Tmin) / 2
NOAA (previously called MLOST)   /soge-home/users/jesu4127/data/mlost/gridded/NOAAGlobalTemp_*.nc
    Monthly temperature anomalies in degrees C
CRUTEM /soge-home/users/jesu4127/data/crutem/CRUTEM.5.1.0.0.anomalies.nc
    Monthly temperature anomalies in degrees C

YEAR HANDLING
-------------
ERA5, MERRA2, JRA-3Q, NOAA, and CRUTEM use their maximal shared period.
CHIRTS is restricted to 1983-2016. October regional series use
cosine-latitude area weighting.

SPATIAL DOMAIN - SOUTHERN AFRICA
--------------
10-40 E, 35-10 S

STATISTICS
----------
All data plotted on their native grids. Grid-cell OLS trends are calculated independently for each dataset and each month.
Statistics used: Ordinary Least Squares (OLS) linear regression, slope in degC/decade

PANELS
------
(a) Monthly grid-cell OLS trend distributions (boxplots), one box per
    dataset per month, in colour.
(b)/(c)/(d)/(e)/(f)/(g) October OLS trend maps for ERA5 / MERRA2 / JRA-3Q /
    CHIRTS / NOAA / CRUTEM.
(h) Overlaid area-weighted October regional mean series for all six datasets.

OUTPUTS
-------
southern_africa_temperature_trends_jra_3q_era5_merra2_chirts_noaa_one_timeseries.png
era5_october_trend_C_per_decade.npy
merra2_october_trend_C_per_decade.npy
jra3q_october_trend_C_per_decade.npy
chirts_october_trend_C_per_decade.npy
noaa_october_trend_C_per_decade.npy
crutem_october_trend_C_per_decade.npy
monthly_trends_summary.csv
"""
# Packages required
from __future__ import annotations
import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature

### Full paths to datasets ##

# Reanalyses: Era5, MERRA2, JRA-3Q #
era5_file = Path("/soge-home/users/jesu4127/scratch/era5/global_2m_temperature.nc")
era5_variable = "t2m"

merra2_directory = Path("/soge-home/users/jesu4127/scratch/merra_2")
merra2_file_pattern = "MERRA2_*instM_2d_asm_Nx.*.nc4"
merra2_variable = "T2M"

jra_3q_directory = Path("/soge-home/users/jesu4127/scratch/jra_3q/")
jra_3q_file_pattern = "915830.tmp2m-hgt-an-gauss.jra3q.anl_surf.0_0_0.tmp2m-hgt-an-gauss.*.nc"
jra_3q_variable = "tmp2m-hgt-an-gauss"

# Observational datasets: CHIRTS, NOAA, CRUTEM #
chirts_tmax_directory = Path("/soge-home/users/jesu4127/data/chirts/Tmax")
chirts_tmin_directory = Path("/soge-home/users/jesu4127/data/chirts/Tmin")

noaa_file = Path("/soge-home/users/jesu4127/data/mlost/gridded/NOAAGlobalTemp_v6.1.0_gridded_s185001_e202607_c20260807T143156.nc")
noaa_variable = "anom"
noaa_label = "NOAA Global Temp v6.1.0"
noaa_short_label = "NOAA Global Temp"

crutem_file= Path("/soge-home/users/jesu4127/data/crutem/CRUTEM.5.1.0.0.anomalies.nc")
crutem_variable= "tas"
crutem_label = "CRUTEM5"
# Necessary for later filtering of grid points with too many NaNs in CRUTEM5.
crutem_max_nan_fraction= 0.50

## Output direction- full path #
out_dir = Path("/soge-home/users/jesu4127/Paper_1/section1_radiation/plots")
out_dir.mkdir(parents=True, exist_ok=True)

# Southern Africa domain #
lon_min, lon_max = 10.0, 40.0
lat_min, lat_max = -35.0, -10.0

target_month = 10
target_month_name = "October"

# Option included to check that no trend calculations are performed on datasets that start too late (e.g. after 2000).
late_start_year = 2000

# Minimum number of years required for a dataset to be included in the analysis
min_years = 20

# Years for shared CHIRTS dataset (CHIRTS is fixed to 1983-2016)
target_start_year= 1983
target_end_year= 2016
shared_year = list(range(target_start_year, target_end_year + 1))

# Dates for calculating the climatology (used for monthly anomalies calculations)
clim_start_year = 1991
clim_end_year = 2020

# Figure size information
figsize = (18, 12)
fig_dpi= 300

# Dataset colours for plotting
colours = {
    "ERA5": "#FF7C7E",
    "MERRA2": "#FEC300",
    "CHIRTS": "#9E3F71",
    "NOAA": "#31BCBA",
    "CRUTEM": "#2F5597",
    "JRA_3Q": "#016ff2",
}

# For colourbar levels in trend maps
trend_levels = np.linspace(-0.6, 0.6, 25)

## Helper functions ## 
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
    data = data.sel({lat_name: slice(lat_min, lat_max), lon_name: slice(lon_min, lon_max)})
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
        data.time.dt.year.isin(range(clim_start_year, clim_end_year + 1)),
        drop=True,)
    if baseline.sizes.get("time", 0) == 0:
        raise RuntimeError(f"No data available for the {clim_start_year}-{clim_end_year} climatology.")
    available_years = years_of(baseline)
    if max(available_years) < clim_end_year:
        print(f"Warning: using available {min(available_years)}-{max(available_years)} data "
              f"for the {clim_start_year}-{clim_end_year} climatology.")
    climatology = baseline.groupby("time.month").mean("time", skipna=True)
    return data.groupby("time.month") - climatology


# Loading datasets

# ERA5
def load_era5():
    if not era5_file.exists():
        raise FileNotFoundError(f"ERA5 file not found:\n{era5_file}")
    ds = xr.open_dataset(era5_file, decode_times=True)
    try:
        if era5_variable not in ds:
            raise KeyError(f"ERA5 variable '{era5_variable}' not found. Variables: {list(ds.data_vars)}")

        lat_name = find_coord(ds, ["latitude", "lat", "Latitude", "LAT", "y"])
        lon_name = find_coord(ds, ["longitude", "lon", "Longitude", "LON", "x"])
        ds = normalise_longitudes(ds, lon_name)

        data = ds[era5_variable].rename({"valid_time": "time"} if "valid_time" in ds[era5_variable].dims else {})
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

def discover_merra2_files():
    """Return {year: {month: filepath}}."""
    regex = re.compile(r"instM_2d_asm_Nx\.(\d{4})(\d{2})\.nc4$")
    files: dict[int, dict[int, Path]] = {}
    matches = list(merra2_directory.glob(merra2_file_pattern))
    if not matches:
        raise FileNotFoundError(f"No MERRA2 files matching {merra2_file_pattern} found in {merra2_directory}")

    for path in matches:
        match = regex.search(path.name)
        if match is None:
            continue
        year, month = int(match.group(1)), int(match.group(2))
        files.setdefault(year, {})[month] = path
    return files

# MERRA-2

def load_merra2():
    files = discover_merra2_files()
    complete_years = [y for y in sorted(files) if all(m in files[y] for m in range(1, 13))]
    if not complete_years:
        raise RuntimeError("No complete MERRA2 years found.")
    print(f"Complete coverage: {complete_years[0]}-{complete_years[-1]} ({len(complete_years)} years)")

    paths = [files[y][m] for y in complete_years for m in range(1, 13)]
    print(f"Files to open: {len(paths)}")

    ds = xr.open_mfdataset(
        paths, combine="nested", concat_dim="time", decode_times=True,
        chunks={}, data_vars="minimal", coords="minimal",
        compat="override", combine_attrs="override",
    )

    if merra2_variable not in ds:
        ds.close()
        raise KeyError(f"MERRA2 variable '{merra2_variable}' not found. Variables: {list(ds.data_vars)}")

    lat_name = find_coord(ds, ["lat", "latitude", "Latitude", "LAT", "y"])
    lon_name = find_coord(ds, ["lon", "longitude", "Longitude", "LON", "x"])
    ds = normalise_longitudes(ds, lon_name)

    data = ds[merra2_variable]
    data = standardise_grid(data, lat_name, lon_name)
    data = data.where(np.abs(data) < 1e4)  # drop fill values
    data = data - 273.15  # K -> degC

    years = np.unique(data.time.dt.year.values.astype(int))
    print(f"Coverage: {years.min()}-{years.max()} ({len(years)} years)")
    print(f"Grid: {data.sizes['lat']} x {data.sizes['lon']}")
    return data, ds

# JRA-3Q
def discover_jra3q_files():
    """Return {year: {month: filepath}}."""
    regex = re.compile(r"\.tmp2m-hgt-an-gauss\.(\d{4})(\d{2})\d{2}00_\d{10}\.nc$")
    files: dict[int, dict[int, Path]] = {}
    matches = list(jra_3q_directory.glob(jra_3q_file_pattern))
    if not matches:
        raise FileNotFoundError(f"No JRA-3Q files matching {jra_3q_file_pattern} found in {jra_3q_directory}")

    for path in matches:
        match = regex.search(path.name)
        if match is None:
            continue
        year, month = int(match.group(1)), int(match.group(2))
        files.setdefault(year, {})[month] = path
    return files


def load_jra3q():
    files = discover_jra3q_files()
    complete_years = [y for y in sorted(files) if all(m in files[y] for m in range(1, 13))]
    if not complete_years:
        raise RuntimeError("No complete JRA-3Q years found.")
    print(f"Complete coverage: {complete_years[0]}-{complete_years[-1]} ({len(complete_years)} years)")

    paths = [files[y][m] for y in complete_years for m in range(1, 13)]
    print(f"Files to open: {len(paths)}")

    ds = xr.open_mfdataset(
        paths, combine="nested", concat_dim="time", decode_times=True,
        chunks={}, data_vars="minimal", coords="minimal",
        compat="override", combine_attrs="override",
    )

    if jra_3q_variable not in ds:
        ds.close()
        raise KeyError(f"JRA-3Q variable '{jra_3q_variable}' not found. Variables: {list(ds.data_vars)}")

    lat_name = find_coord(ds, ["lat", "latitude", "Latitude", "LAT", "y"])
    lon_name = find_coord(ds, ["lon", "longitude", "Longitude", "LON", "x"])
    ds = normalise_longitudes(ds, lon_name)

    data = ds[jra_3q_variable]
    data = standardise_grid(data, lat_name, lon_name)
    data = data.where(np.abs(data) < 1e4)  # drop fill values
    data = data - 273.15  # K -> degC

    years = np.unique(data.time.dt.year.values.astype(int))
    print(f"Coverage: {years.min()}-{years.max()} ({len(years)} years)")
    print(f"Grid: {data.sizes['lat']} x {data.sizes['lon']}")
    return data, ds

# CHIRTS
def load_chirts():
    max_regex = re.compile(r"Tmax\.(\d{4})\.nc$")
    min_regex = re.compile(r"Tmin\.(\d{4})\.nc$")

    tmax_files = {int(m.group(1)): p for p in chirts_tmax_directory.glob("Tmax.*.nc")
                  if (m := max_regex.search(p.name))}
    tmin_files = {int(m.group(1)): p for p in chirts_tmin_directory.glob("Tmin.*.nc")
                  if (m := min_regex.search(p.name))}

    common_years = sorted(set(tmax_files) & set(tmin_files))
    if not common_years:
        raise RuntimeError("No common CHIRTS Tmax/Tmin years found.")
    print(f"Coverage: {common_years[0]}-{common_years[-1]} ({len(common_years)} years)")

    max_paths = [tmax_files[y] for y in common_years]
    min_paths = [tmin_files[y] for y in common_years]

    print("Opening Tmax...")
    ds_max = xr.open_mfdataset(max_paths, combine="by_coords", decode_times=True, chunks={},
                                data_vars="minimal", coords="minimal", compat="override",
                                combine_attrs="override")
    print("Opening Tmin...")
    ds_min = xr.open_mfdataset(min_paths, combine="by_coords", decode_times=True, chunks={},
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


def load_noaa():
    if not noaa_file.exists():
        raise FileNotFoundError(f"NOAA file not found:\n{noaa_file}")

    ds = xr.open_dataset(noaa_file, decode_times=True)
    if noaa_variable not in ds:
        ds.close()
        raise KeyError(f"NOAA variable '{noaa_variable}' not found. Variables: {list(ds.data_vars)}")

    lat_name = find_coord(ds, ["lat", "latitude", "Latitude", "LAT", "y"])
    lon_name = find_coord(ds, ["lon", "longitude", "Longitude", "LON", "x"])
    ds = normalise_longitudes(ds, lon_name)
    data = ds[noaa_variable].squeeze(drop=True)
    data = standardise_grid(data, lat_name, lon_name)
    data = data.where(np.abs(data) < 1e3)

    years = np.unique(data.time.dt.year.values.astype(int))
    print(f"Coverage: {years.min()}-{years.max()} ({len(years)} years)")
    print(f"Grid: {data.sizes['lat']} x {data.sizes['lon']}")
    return data.load(), ds


def load_crutem():
    ds = xr.open_dataset(crutem_file, decode_times=True)
    try:
        if crutem_variable not in ds:
            raise KeyError(f"CRUTEM variable '{crutem_variable}' not found. Variables: {list(ds.data_vars)}")

        lat_name = find_coord(ds, ["latitude", "lat", "Latitude", "LAT", "y"])
        lon_name = find_coord(ds, ["longitude", "lon", "Longitude", "LON", "x"])
        ds = normalise_longitudes(ds, lon_name)
        data = ds[crutem_variable].squeeze(drop=True)
        data = standardise_grid(data, lat_name, lon_name)
        data = data.where(np.abs(data) < 1e3).load()
        nan_fraction = data.isnull().mean(dim="time")
        valid_gridpoints = nan_fraction <= crutem_max_nan_fraction
        data = data.where(valid_gridpoints)
        total_gridpoints = nan_fraction.size
        retained_gridpoints = int(valid_gridpoints.sum())
        print(f"Grid points retained after NaN filter: {retained_gridpoints}/{total_gridpoints} "
              f"(threshold: {crutem_max_nan_fraction:.0%})")

        years = np.unique(data.time.dt.year.values.astype(int))
        print(f"Coverage: {years.min()}-{max(years)} ({len(years)} years)")
        print(f"Grid: {data.sizes['lat']} x {data.sizes['lon']}")
        return data
    finally:
        ds.close()


# Year range handling across datasets
def resolve_year_ranges(era5, merra2, jra3q, chirts, noaa, crutem):
    era5_years, merra_years, jra3q_years = years_of(era5), years_of(merra2), years_of(jra3q)
    chirts_years, noaa_years, crutem_years = years_of(chirts), years_of(noaa), years_of(crutem)
    shared_common = sorted(era5_years & merra_years & jra3q_years & noaa_years & crutem_years)
    chirts_years = sorted(set(chirts_years) & set(shared_year))
    noaa_years = sorted(noaa_years)

    if len(shared_common) < min_years:
        raise RuntimeError(f"Only {len(shared_common)} common ERA5/MERRA2/JRA-3Q/MLOST/CRUTEM years.")
    if len(chirts_years) < min_years:
        raise RuntimeError(f"Only {len(chirts_years)} CHIRTS years in {target_start_year}-{target_end_year}.")
    if len(noaa_years) < min_years:
        raise RuntimeError(f"Only {len(noaa_years)} NOAA years available.")
    if len(crutem_years) < min_years:
        raise RuntimeError(f"Only {len(crutem_years)} CRUTEM years available.")

    print("\n" + "=" * 78 + "\nYEAR RANGES\n" + "=" * 78)
    print(f"ERA5:               {min(era5_years)}-{max(era5_years)}")
    print(f"MERRA2:             {min(merra_years)}-{max(merra_years)}")
    print(f"JRA-3Q:             {min(jra3q_years)}-{max(jra3q_years)}")
    print(f"CHIRTS (used):      {chirts_years[0]}-{chirts_years[-1]} ({len(chirts_years)} years)")
    print(f"{noaa_label} (used): {noaa_years[0]}-{noaa_years[-1]} ({len(noaa_years)} years)")
    print(f"{crutem_label} (used): {shared_common[0]}-{shared_common[-1]} ({len(shared_common)} years)")
    print(f"Shared ERA5/MERRA2/JRA-3Q/{noaa_label}/{crutem_label} years: "
          f"{shared_common[0]}-{shared_common[-1]} ({len(shared_common)} years)")

    era5 = restrict_to_years(era5, shared_common)
    merra2 = restrict_to_years(merra2, shared_common)
    jra3q = restrict_to_years(jra3q, shared_common)
    chirts = restrict_to_years(chirts, chirts_years)
    noaa = restrict_to_years(noaa, noaa_years)
    crutem = restrict_to_years(crutem, shared_common)

    # ERA5, MERRA2, JRA-3Q, and CRUTEM all share the same restricted year set here (shared_common); NOAA keeps its own fuller record.
    return (era5, merra2, jra3q, chirts, noaa, crutem,
            np.array(shared_common), np.array(shared_common),
            np.array(chirts_years), np.array(noaa_years))


# Trend calculations
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
    october = data.where(data.time.dt.month == target_month, drop=True)
    october_annual = october.resample(time="1YS").mean("time", skipna=True)
    return _ols_slope_per_decade(october_annual)


def calculate_october_area_weighted_series(data):
    """Return October area-weighted regional means for each year."""
    october = data.where(data.time.dt.month == target_month, drop=True)
    october_annual = october.resample(time="1YS").mean("time", skipna=True)
    weights = np.cos(np.deg2rad(october_annual.lat))
    return october_annual.weighted(weights).mean(dim=("lat", "lon"), skipna=True).compute()


# Summary table for monthly trends 
def make_summary(era5_monthly, merra_monthly, jra3q_monthly, chirts_monthly,
                 noaa_monthly, crutem_monthly):
    rows = []
    for month in range(1, 13):
        month_name = pd.Timestamp(2000, month, 1).strftime("%b")
        datasets = {"ERA5": era5_monthly[month], "MERRA2": merra_monthly[month],
            "JRA_3Q": jra3q_monthly[month],
            "CHIRTS": chirts_monthly[month], "NOAA": noaa_monthly[month],
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


# Plotting

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
    kwargs = dict(levels=trend_levels, cmap="RdBu_r", extend="both")

    plot_data = trend

    mesh = ax.pcolormesh(plot_data.lon.values, plot_data.lat.values, plot_data.values,
    transform=ccrs.PlateCarree(), cmap="RdBu_r",
    vmin=trend_levels.min(), vmax=trend_levels.max(), shading="nearest")
   
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=1.0,
                edgecolor="black", zorder=5)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.9,
                edgecolor="black", linestyle="-", zorder=5)
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"fontsize": 8}
    gl.ylabel_style = {"fontsize": 8}
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.35, linewidth=0.4, linestyle="--")
    ax.set_aspect("equal")
  
    if start_years is not None:
        late_start = np.isfinite(start_years.values) & (start_years.values > late_start_year)
        if np.any(late_start):
            lon_grid, lat_grid = np.meshgrid(start_years.lon.values, start_years.lat.values)
            scatter_kwargs = {"transform": ccrs.PlateCarree()}
            ax.scatter(lon_grid[late_start], lat_grid[late_start], marker="*", s=24,
                       color="black", edgecolors="white", linewidths=0.35,
                       zorder=7, **scatter_kwargs)
    ax.set_title(f"({year_range})", fontsize=9, pad=6)
    ax.text(0.5, 1.16, dataset_name, transform=ax.transAxes, fontsize=12,
            fontweight="bold", ha="center", va="bottom")
    ax.text(0.02, 0.96, panel_label, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top",
            bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none", alpha=0.7))
    return mesh


def plot_monthly_boxplot(ax, era5_monthly, merra_monthly, jra3q_monthly,
                          chirts_monthly, noaa_monthly, crutem_monthly,
                          crutem_start_years, era5_merra_years, jra3q_years,
                          chirts_years, noaa_years):
    datasets = {
        "ERA5": (era5_monthly, -0.15),
        "MERRA2": (merra_monthly, -0.09),
        "JRA_3Q": (jra3q_monthly, -0.03),
        "CHIRTS": (chirts_monthly, 0.03),
        "NOAA": (noaa_monthly, 0.09),
        "CRUTEM": (crutem_monthly, 0.15),
    }
    centres = np.arange(1, 13, dtype=float)

    for name, (monthly, offset) in datasets.items():
        values = [flatten_valid(monthly[m]) for m in range(1, 13)]
        bp = ax.boxplot(values, positions=centres + offset, widths=0.06,
                         patch_artist=True, showfliers=False, manage_ticks=False)
        for box in bp["boxes"]:
            box.set_facecolor(colours[name])
            box.set_edgecolor("black")
            box.set_linewidth(0.9)
            box.set_alpha(0.9)
        for key in ("whiskers", "caps", "medians"):
            for artist in bp[key]:
                artist.set_color("black")
                artist.set_linewidth(0.9)

    for month, start_years in crutem_start_years.items():
        if np.any(np.isfinite(start_years.values) & (start_years.values > late_start_year)):
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
    ax.text(0.99, 0.97, "(a)", transform=ax.transAxes, fontsize=12, fontweight="bold",
            va="top", ha="right")

    era5_merra_label = f"{era5_merra_years[0]}-{era5_merra_years[-1]}"
    jra3q_label = f"{jra3q_years[0]}-{jra3q_years[-1]}"
    chirts_label = f"{chirts_years[0]}-{chirts_years[-1]}"

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colours["ERA5"], edgecolor="black",
                   label=f"ERA5 ({era5_merra_label})"),
        plt.Rectangle((0, 0), 1, 1, facecolor=colours["MERRA2"], edgecolor="black",
                       label=f"MERRA2 ({era5_merra_label})"),
        plt.Rectangle((0, 0), 1, 1, facecolor=colours["JRA_3Q"], edgecolor="black",
                       label=f"JRA-3Q ({jra3q_label})"),
        plt.Rectangle((0, 0), 1, 1, facecolor=colours["CHIRTS"], edgecolor="black",
                   label=f"CHIRTS ({chirts_label})"),
        plt.Rectangle((0, 0), 1, 1, facecolor=colours["NOAA"], edgecolor="black",
               label=f"{noaa_label} ({noaa_years[0]}-{noaa_years[-1]})"),
        plt.Rectangle((0, 0), 1, 1, facecolor=colours["CRUTEM"], edgecolor="black",
               label=f"{crutem_label} ({era5_merra_years[0]}-{era5_merra_years[-1]})"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.04),
              frameon=False, fontsize=9, ncol=6, borderaxespad=0.0)


def make_full_figure(era5_monthly, merra_monthly, jra3q_monthly, chirts_monthly, noaa_monthly,
                  crutem_monthly, crutem_start_years, crutem_oct_start_years,
                  era5_oct, merra_oct, jra3q_oct, chirts_oct,
                  noaa_oct, crutem_oct, era5_annual, merra_annual, jra3q_annual, chirts_annual,
                  noaa_annual, crutem_annual,
                      era5_merra_years, jra3q_years, chirts_years, noaa_years):
    """
    Layout:

        row 0: (a) monthly boxplot, full width
        row 1: (b) (c) (d) (e) (f) (g) October trend maps, one per dataset
        row 2: overlaid area-weighted October series for all datasets

    """
    print("\n" + "=" * 78 + "\nCREATING FULL FIGURE\n" + "=" * 78)

    fig = plt.figure(figsize=figsize, dpi=fig_dpi)
   
    gs = fig.add_gridspec(3, 6, height_ratios=[1.0, 1.25, 0.8], hspace=0.85, wspace=0.28)

    ax_box = fig.add_subplot(gs[0, :])


    projection = ccrs.PlateCarree()
    ax_era5 = fig.add_subplot(gs[1, 0], projection=projection)
    ax_merra = fig.add_subplot(gs[1, 1], projection=projection)
    ax_jra3q = fig.add_subplot(gs[1, 2], projection=projection)
    ax_chirts = fig.add_subplot(gs[1, 3], projection=projection)
    ax_noaa = fig.add_subplot(gs[1, 4], projection=projection)
    ax_crutem = fig.add_subplot(gs[1, 5], projection=projection)
  
    ax_series = fig.add_subplot(gs[2, :])

    plot_monthly_boxplot(ax_box, era5_monthly, merra_monthly, jra3q_monthly, chirts_monthly, noaa_monthly,
    crutem_monthly, crutem_start_years, era5_merra_years, jra3q_years, chirts_years, noaa_years)

    era5_merra_label = f"{era5_merra_years[0]}-{era5_merra_years[-1]}"
    jra3q_label = f"{jra3q_years[0]}-{jra3q_years[-1]}"
    chirts_label = f"{chirts_years[0]}-{chirts_years[-1]}"
    noaa_label = f"{noaa_years[0]}-{noaa_years[-1]}"

    mesh = plot_map_panel(ax_era5, era5_oct, "ERA5", era5_merra_label, "(b)")
    plot_map_panel(ax_merra, merra_oct, "MERRA-2", era5_merra_label, "(c)")
    plot_map_panel(ax_jra3q, jra3q_oct, "JRA-3Q", jra3q_label, "(d)")
    plot_map_panel(ax_chirts, chirts_oct, "CHIRTS", chirts_label, "(e)")
    plot_map_panel(ax_noaa, noaa_oct, noaa_short_label, noaa_label, "(f)",
                   use_cell_grid=True)
    plot_map_panel(ax_crutem, crutem_oct, crutem_label, era5_merra_label, "(g)",
                   use_cell_grid=True, start_years=crutem_oct_start_years)
    fig.text(0.5, 0.66, "October trends", ha="center", va="center",
             fontsize=14, fontweight="bold")


    cbar = fig.colorbar(mesh, ax=[ax_era5, ax_merra, ax_jra3q, ax_chirts, ax_noaa, ax_crutem],
                         orientation="horizontal", fraction=0.06, pad=0.1, aspect=45,
                         shrink=0.9, extend="both")
    cbar.set_label(r"October Tmean trend ($^\circ$C decade$^{-1}$)")

    plotted_series = (era5_annual, merra_annual, jra3q_annual, chirts_annual,
                      noaa_annual, crutem_annual)
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
        (era5_annual, "ERA5", colours["ERA5"]),
        (merra_annual, "MERRA2", colours["MERRA2"]),
        (jra3q_annual, "JRA-3Q", colours["JRA_3Q"]),
        (chirts_annual, "CHIRTS", colours["CHIRTS"]),
        (noaa_annual, noaa_short_label, colours["NOAA"]),
        (crutem_annual, crutem_label, colours["CRUTEM"]),
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

    output = out_dir / "southern_africa_temperature_trends_jra_3q_era5_merra2_chirts_noaa_one_timeseries.png"
    fig.savefig(output, dpi=fig_dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved figure:\n  {output}")


# Main execution
def main():
    era5 = load_era5()
    merra2, merra2_ds = load_merra2()
    jra3q, jra3q_ds = load_jra3q()
    chirts, chirts_max_ds, chirts_min_ds = load_chirts()
    noaa, noaa_ds = load_noaa()
    crutem = load_crutem()

    try:
        era5 = calculate_monthly_anomalies(era5)
        merra2 = calculate_monthly_anomalies(merra2)
        jra3q = calculate_monthly_anomalies(jra3q)
        chirts = calculate_monthly_anomalies(chirts)
        noaa = calculate_monthly_anomalies(noaa)
        crutem = calculate_monthly_anomalies(crutem)

        (era5, merra2, jra3q, chirts, noaa, crutem, era5_merra_years,
         jra3q_years, chirts_years, noaa_years) = resolve_year_ranges(
            era5, merra2, jra3q, chirts, noaa, crutem
        )
        crutem_start_years = {
            month: calculate_start_years(crutem, month=month)
            for month in range(1, 13)
        }
        crutem_oct_start_years = crutem_start_years[target_month]

        print("\nEra5:")
        era5_monthly = calculate_monthly_trends(era5)
        print("\nMERRA2:")
        merra_monthly = calculate_monthly_trends(merra2)
        print("\nJRA-3Q:")
        jra3q_monthly = calculate_monthly_trends(jra3q)
        print("\nCHIRTS:")
        chirts_monthly = calculate_monthly_trends(chirts)
        print("\nNOAA:")
        noaa_monthly = calculate_monthly_trends(noaa)
        print(f"\n{crutem_label}:")
        crutem_monthly = calculate_monthly_trends(crutem)

        print("\n" + "=" * 78 + "\nOCTOBER SPATIAL TRENDS\n" + "=" * 78)
        print("\nEra5:")
        era5_oct = calculate_october_trend(era5)
        print("\nMERRA2:")
        merra_oct = calculate_october_trend(merra2)
        print("\nJRA-3Q:")
        jra3q_oct = calculate_october_trend(jra3q)
        print("\nCHIRTS:")
        chirts_oct = calculate_october_trend(chirts)
        print("\nNOAA:")
        noaa_oct = calculate_october_trend(noaa)
        print(f"\n{crutem_label}:")
        crutem_oct = calculate_october_trend(crutem)

        print("\n" + "=" * 78 + "\nAREA-WEIGHTED OCTOBER SERIES\n" + "=" * 78)
        era5_annual = calculate_october_area_weighted_series(era5)
        merra_annual = calculate_october_area_weighted_series(merra2)
        jra3q_annual = calculate_october_area_weighted_series(jra3q)
        chirts_annual = calculate_october_area_weighted_series(chirts)
        noaa_annual = calculate_october_area_weighted_series(noaa)
        crutem_annual = calculate_october_area_weighted_series(crutem)

        print("\n" + "=" * 78 + "\nOCTOBER DIAGNOSTICS\n" + "=" * 78)
        for name, data in [("ERA5", era5_oct), ("MERRA2", merra_oct), ("JRA-3Q", jra3q_oct),
                   ("CHIRTS", chirts_oct), ("NOAA", noaa_oct),
                   (crutem_label, crutem_oct)]:
            values = flatten_valid(data)
            print(f"\n{name}")
            print(f"  valid pixels: {values.size:,}")
            print(f"  minimum: {np.min(values):+.4f}")
            print(f"  mean:    {np.mean(values):+.4f}")
            print(f"  median:  {np.median(values):+.4f}")
            print(f"  maximum: {np.max(values):+.4f}")

        summary = make_summary(era5_monthly, merra_monthly, jra3q_monthly, chirts_monthly, noaa_monthly,
                       crutem_monthly)
        summary_path = out_dir/ "monthly_trends_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"\nSaved summary:\n  {summary_path}")

        np.save(out_dir / "era5_october_trend_C_per_decade.npy", era5_oct.values)
        np.save(out_dir / "merra2_october_trend_C_per_decade.npy", merra_oct.values)
        np.save(out_dir / "jra3q_october_trend_C_per_decade.npy", jra3q_oct.values)
        np.save(out_dir / "chirts_october_trend_C_per_decade.npy", chirts_oct.values)
        np.save(out_dir / "noaa_october_trend_C_per_decade.npy", noaa_oct.values)
        np.save(out_dir / "crutem_october_trend_C_per_decade.npy", crutem_oct.values)

        make_full_figure(era5_monthly, merra_monthly, jra3q_monthly, chirts_monthly, noaa_monthly,
              crutem_monthly, crutem_start_years, crutem_oct_start_years,
              era5_oct, merra_oct, jra3q_oct, chirts_oct, noaa_oct, crutem_oct,
              era5_annual, merra_annual, jra3q_annual, chirts_annual, noaa_annual, crutem_annual,
                          era5_merra_years, jra3q_years, chirts_years, noaa_years)

    finally:
        merra2_ds.close()
        jra3q_ds.close()
        chirts_max_ds.close()
        chirts_min_ds.close()
        noaa_ds.close()


if __name__ == "__main__":
    main()