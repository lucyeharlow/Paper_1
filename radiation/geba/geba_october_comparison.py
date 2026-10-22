#!/usr/bin/env python3
"""
geba_station_ssrd_vs_gridded_ols.py

For each GEBA GLOBAL-radiation station passing the post-1980 October filter
(more than 15 October records after 1980), plot the station's October
downwelling shortwave flux (raw values, W m-2) against the October flux
extracted at the same location from five datasets: ERA5, MERRA-2, CLARA-A3,
SARAH-3 and CERES-EBAF (2000 onwards). One legend for the station and the
datasets sits below the panels. Dataset colours match the Southern Africa SW boxplot figure. The map shows the
stations, with a legend naming the station for each coloured dot; each panel
title is drawn in its station's colour.

LOADERS
-------
The gridded-dataset loaders reuse the same helper pattern as the
full_reconstruct_era5_merra2_chirts.py script: find_coord() to locate the
lat/lon names, normalise_longitudes() to force -180..180, and
standardise_grid() to rename to lat/lon, sort latitude ascending, and clip
to a buffered region around the stations. Each loader returns one continuous
October-only xr.DataArray(time, lat, lon); station values are then pulled
out with a simple .sel(..., method="nearest").

VALUES
------
Raw monthly-mean fluxes (no anomalies), so differences in level between the
station and each dataset (biases) are visible. No trend lines are drawn.
Every panel uses the same year and flux ranges (from all stations and
datasets), so the panels can be compared directly.

OUTPUT
------
    geba_station_ssrd_vs_datasets_ols.png
"""

import glob
import re
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import hsv_to_rgb
from matplotlib.lines import Line2D

# ============================================================================
# CONFIGURATION
# ============================================================================

GEBA_DIR = Path("/soge-home/users/jesu4127/data/geba")
GEBA_METADATA_FILE = GEBA_DIR / "GEBA_2026-09-03_15-37-10_metadata.csv"
GEBA_MONTHLY_FILE = GEBA_DIR / "GEBA_2026-09-03_15-37-10_monthly_data.csv"

LATITUDE_LIMIT = -3.0
LON_MIN, LON_MAX = 10.0, 40.0
TARGET_TYPE = "GLOBAL"
TARGET_MONTH = 10  # October
MIN_YEAR = 1980  # station records must be strictly after this year
MIN_ENTRIES = 15  # stations must have MORE than this many post-1980 October entries

# Buffered region used when clipping the gridded datasets (must comfortably
# contain every qualifying station).
REGION_LAT_MIN, REGION_LAT_MAX = -36.0, -2.0
REGION_LON_MIN, REGION_LON_MAX = 9.0, 41.0

ERA5_SSRD_FILE = Path("/soge-home/users/jesu4127/scratch/era5/global_surface_solar_radiation_downwards.nc")
ERA5_SSRD_VAR = "ssrd"  # J m-2, daily-accumulated monthly mean -> convert to W m-2 by /86400

MERRA2_DIR = Path("/soge-home/users/jesu4127/scratch/merra_2")
MERRA2_RAD_PATTERN = "MERRA2_*.tavgM_2d_rad_Nx.*.nc4"
MERRA2_SWGDN_VAR = "SWGDN"  # W m-2

CLARA_DIR = Path("/soge-home/users/jesu4127/data/clara_a3")
CLARA_SIS_REGEX = re.compile(r"SISmm(\d{4})1001")
CLARA_SIS_VAR = "SIS"  # W m-2

SARAH3_DIR = Path("/soge-home/users/jesu4127/data/sarah_3")
SARAH3_SIS_REGEX = re.compile(r"SISmm(\d{4})(\d{2})")
SARAH3_SIS_VAR = "SIS"  # W m-2

CERES_GLOB = "/soge-home/users/jesu4127/data/ceres/CERES_EBAF_Ed4.2.1_Subset_*.nc"
CERES_VAR = "sfc_sw_down_all_mon"  # W m-2, all-sky surface SW down

OUTPUT_FIGURE = Path("/soge-home/users/jesu4127/Paper_1/radiation/geba/geba_station_ssrd_vs_datasets_ols.png")

# Same colours as the Southern Africa SW boxplot figure (sw_down_africa_trends.py)
DATASET_COLOURS = {
	"ERA5": "#FF7C7E",
	"MERRA-2": "#FEC300",
	"CLARA-A3": "#4C9A2A",
	"SARAH-3": "#8C564B",
	"CERES-EBAF": "#7E57C2",
}
STATION_COLOUR = "black"


# ============================================================================
# GENERIC HELPERS (same pattern as full_reconstruct_era5_merra2_chirts.py)
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


def standardise_grid(data, lat_name, lon_name):
	"""Rename to lat/lon, sort lat ascending, clip to the buffered study region."""
	data = data.transpose(data.dims[0], lat_name, lon_name)
	if float(data[lat_name].values[0]) > float(data[lat_name].values[-1]):
		data = data.sortby(lat_name)
	data = data.sel({lat_name: slice(REGION_LAT_MIN, REGION_LAT_MAX),
					  lon_name: slice(REGION_LON_MIN, REGION_LON_MAX)})
	return data.rename({lat_name: "lat", lon_name: "lon"})


# ============================================================================
# GEBA STATION SELECTION (post-1980, >15 October entries)
# ============================================================================

def load_geba_stations():
	metadata = pd.read_csv(GEBA_METADATA_FILE, sep=";", low_memory=False)
	monthly = pd.read_csv(GEBA_MONTHLY_FILE, sep=";", low_memory=False)
	for frame in (metadata, monthly):
		frame["tskey"] = pd.to_numeric(frame["tskey"], errors="coerce")
	metadata["sgxlon"] = pd.to_numeric(metadata["sgxlon"], errors="coerce")
	metadata["sgylat"] = pd.to_numeric(metadata["sgylat"], errors="coerce")
	monthly["year"] = pd.to_numeric(monthly["year"], errors="coerce")
	monthly["month"] = pd.to_numeric(monthly["month"], errors="coerce")
	monthly["converted_flux_avg"] = pd.to_numeric(monthly["converted_flux_avg"], errors="coerce")

	stations = metadata.loc[
		(metadata["ebcode"] == TARGET_TYPE)
		& metadata["sgylat"].notna()
		& metadata["sgxlon"].notna()
		& (metadata["sgylat"] < LATITUDE_LIMIT)
		& (metadata["sgxlon"] >= LON_MIN)
		& (metadata["sgxlon"] <= LON_MAX)
	].copy()

	october = monthly.loc[
		monthly["tskey"].isin(stations["tskey"])
		& monthly["converted_flux_avg"].notna()
		& (monthly["month"] == TARGET_MONTH)
		& (monthly["year"] > MIN_YEAR)
	].copy()

	entries_per_station = october.groupby("tskey")["converted_flux_avg"].count()
	qualifying_ids = entries_per_station[entries_per_station > MIN_ENTRIES].index
	stations = stations.loc[stations["tskey"].isin(qualifying_ids)].copy()
	stations = stations.sort_values("sgname").reset_index(drop=True)
	stations["colour"] = [
		"#%02x%02x%02x" % tuple((255 * hsv_to_rgb(
			(index / max(len(stations), 1), 0.72, 0.82)
		)).astype(int))
		for index in range(len(stations))
	]

	station_series = {}
	for _, station in stations.iterrows():
		series = october.loc[october["tskey"] == station["tskey"]].set_index("year")["converted_flux_avg"]
		station_series[int(station["tskey"])] = series.groupby(level=0).mean().to_dict()

	return stations, station_series


# ============================================================================
# LOAD ERA5 SSRD (October only)
# ============================================================================

def load_era5_ssrd_october():
	print("Loading ERA5 ssrd...")
	if not ERA5_SSRD_FILE.exists():
		print(f"  ERA5 file not found, skipping: {ERA5_SSRD_FILE}")
		return None

	ds = xr.open_dataset(ERA5_SSRD_FILE, decode_times=True)
	try:
		if ERA5_SSRD_VAR not in ds:
			print(f"  ERA5 variable '{ERA5_SSRD_VAR}' not found, skipping.")
			return None

		lat_name = find_coord(ds, ["latitude", "lat", "Latitude", "LAT", "y"])
		lon_name = find_coord(ds, ["longitude", "lon", "Longitude", "LON", "x"])
		ds = normalise_longitudes(ds, lon_name)

		data = ds[ERA5_SSRD_VAR]
		if "valid_time" in data.dims:
			data = data.rename({"valid_time": "time"})
		data = standardise_grid(data, lat_name, lon_name)
		data = data / 86400.0  # J m-2 (daily-accum mean) -> W m-2

		october = data.where(data.time.dt.month == TARGET_MONTH, drop=True).load()
		years = np.unique(october.time.dt.year.values.astype(int))
		print(f"  Coverage: {years.min()}-{years.max()} ({len(years)} Octobers)")
		return october
	finally:
		ds.close()


# ============================================================================
# LOAD MERRA2 SWGDN (October files only)
# ============================================================================

def discover_merra2_october_files():
	regex = re.compile(r"tavgM_2d_rad_Nx\.(\d{4})(\d{2})\.nc4$")
	files = {}
	for path in MERRA2_DIR.glob(MERRA2_RAD_PATTERN):
		match = regex.search(path.name)
		if match is None:
			continue
		year, month = int(match.group(1)), int(match.group(2))
		if month == TARGET_MONTH:
			files[year] = path
	return files


def load_merra2_swgdn_october():
	print("Loading MERRA2 SWGDN (October files)...")
	files = discover_merra2_october_files()
	if not files:
		print(f"  No MERRA2 October radiation files found in {MERRA2_DIR}, skipping.")
		return None

	paths = [files[year] for year in sorted(files)]
	print(f"  Found {len(paths)} October files ({min(files)}-{max(files)})")

	ds = xr.open_mfdataset(
		paths, combine="nested", concat_dim="time", decode_times=True,
		data_vars="minimal", coords="minimal", compat="override", combine_attrs="override",
	)
	try:
		if MERRA2_SWGDN_VAR not in ds:
			print(f"  MERRA2 variable '{MERRA2_SWGDN_VAR}' not found, skipping.")
			return None

		lat_name = find_coord(ds, ["lat", "latitude", "Latitude", "LAT", "y"])
		lon_name = find_coord(ds, ["lon", "longitude", "Longitude", "LON", "x"])
		ds = normalise_longitudes(ds, lon_name)

		data = ds[MERRA2_SWGDN_VAR]
		data = standardise_grid(data, lat_name, lon_name)
		data = data.where(np.abs(data) < 1e4)  # drop fill values
		return data.load()
	finally:
		ds.close()


# ============================================================================
# LOAD CLARA-A3 SIS (already October-only files, thematic + interim)
# ============================================================================

def discover_clara_october_files():
	files = {}
	for path in CLARA_DIR.glob("*surface_radiation_budget_m10_SISmm*.nc"):
		match = CLARA_SIS_REGEX.search(path.name)
		if match is None:
			continue
		files[int(match.group(1))] = path
	return files


def load_clara_sis_october():
	print("Loading CLARA-A3 SIS (October files)...")
	files = discover_clara_october_files()
	if not files:
		print(f"  No CLARA-A3 October SIS files found in {CLARA_DIR}, skipping.")
		return None

	paths = [files[year] for year in sorted(files)]
	print(f"  Found {len(paths)} October files ({min(files)}-{max(files)})")

	ds = xr.open_mfdataset(
		paths, combine="nested", concat_dim="time", decode_times=True,
		data_vars="minimal", coords="minimal", compat="override", combine_attrs="override",
	)
	try:
		if CLARA_SIS_VAR not in ds:
			print(f"  CLARA-A3 variable '{CLARA_SIS_VAR}' not found, skipping.")
			return None

		lat_name = find_coord(ds, ["lat", "latitude", "Latitude", "LAT", "y"])
		lon_name = find_coord(ds, ["lon", "longitude", "Longitude", "LON", "x"])
		ds = normalise_longitudes(ds, lon_name)

		data = ds[CLARA_SIS_VAR]
		data = standardise_grid(data, lat_name, lon_name)
		data = data.where(data > -9998)
		return data.load()
	finally:
		ds.close()


# ============================================================================
# LOAD SARAH-3 SIS (monthly files, filtered to October by filename)
# ============================================================================

def discover_sarah3_october_files():
	files = {}
	for path in SARAH3_DIR.glob("SISmm*.nc"):
		match = SARAH3_SIS_REGEX.search(path.name)
		if match is None:
			continue
		year, month = int(match.group(1)), int(match.group(2))
		if month == TARGET_MONTH:
			files[year] = path
	return files


def load_sarah3_sis_october():
	print("Loading SARAH-3 SIS (October files)...")
	files = discover_sarah3_october_files()
	if not files:
		print(f"  No SARAH-3 October SIS files found in {SARAH3_DIR}, skipping.")
		return None

	paths = [files[year] for year in sorted(files)]
	print(f"  Found {len(paths)} October files ({min(files)}-{max(files)})")

	ds = xr.open_mfdataset(
		paths, combine="nested", concat_dim="time", decode_times=True,
		data_vars="minimal", coords="minimal", compat="override", combine_attrs="override",
	)
	try:
		if SARAH3_SIS_VAR not in ds:
			print(f"  SARAH-3 variable '{SARAH3_SIS_VAR}' not found, skipping.")
			return None

		lat_name = find_coord(ds, ["lat", "latitude", "Latitude", "LAT", "y"])
		lon_name = find_coord(ds, ["lon", "longitude", "Longitude", "LON", "x"])
		ds = normalise_longitudes(ds, lon_name)

		data = ds[SARAH3_SIS_VAR]
		data = standardise_grid(data, lat_name, lon_name)
		# mask_and_scale (default True) already turns _FillValue into NaN.
		return data.load()
	finally:
		ds.close()


# ============================================================================
# LOAD CERES-EBAF SW DOWN (October only)
# ============================================================================

def load_ceres_october():
	print("Loading CERES-EBAF sfc_sw_down_all_mon...")
	files = sorted(glob.glob(CERES_GLOB))
	if not files:
		print(f"  No CERES files match {CERES_GLOB}, skipping.")
		return None
	parts = []
	for path in files:
		with xr.open_dataset(path, decode_times=True) as ds:
			if CERES_VAR not in ds:
				print(f"  {CERES_VAR} not in {path}, skipped.")
				continue
			lat_name = find_coord(ds, ["lat", "latitude", "Latitude", "LAT", "y"])
			lon_name = find_coord(ds, ["lon", "longitude", "Longitude", "LON", "x"])
			ds = normalise_longitudes(ds, lon_name)
			data = ds[CERES_VAR]
			data = data.isel(time=np.flatnonzero(data.time.dt.month.values == TARGET_MONTH))
			parts.append(standardise_grid(data, lat_name, lon_name).load())
	if not parts:
		return None
	data = xr.concat(parts, "time").sortby("time")
	data = data.isel(time=np.unique(data.time.dt.year.values, return_index=True)[1])  # files may overlap
	data = data.where(data > -900)
	years = data.time.dt.year.values.astype(int)
	print(f"  Coverage: {years.min()}-{years.max()} ({len(years)} Octobers)")
	return data


# ============================================================================
# POINT EXTRACTION
# ============================================================================

def extract_station_series(data, station_lat, station_lon):
	"""Nearest-gridpoint {year: value} series from a continuous October-only DataArray."""
	if data is None:
		return {}
	point = data.sel(lat=station_lat, lon=station_lon, method="nearest")
	years = point.time.dt.year.values.astype(int)
	values = point.values
	return {int(year): float(value) for year, value in zip(years, values) if np.isfinite(value)}


# ============================================================================
# PLOTTING
# ============================================================================

def plot_map(ax, stations):
	ax.set_extent([LON_MIN - 3, LON_MAX + 3, -35, LATITUDE_LIMIT + 2], crs=ccrs.PlateCarree())
	ax.add_feature(cfeature.LAND, facecolor="#eeeeee")
	ax.add_feature(cfeature.OCEAN, facecolor="#dcecf2")
	ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
	ax.add_feature(cfeature.BORDERS, linewidth=0.45, alpha=0.7)
	ax.set_xticks([10, 20, 30, 40], crs=ccrs.PlateCarree())
	ax.set_yticks([-30, -20, -10, 0], crs=ccrs.PlateCarree())
	ax.tick_params(labelsize=8)
	for _, station in stations.iterrows():
		ax.scatter(station["sgxlon"], station["sgylat"], s=60,
				   color=station["colour"], edgecolors="black", linewidths=0.5,
				   transform=ccrs.PlateCarree(), zorder=4,
				   label=f"{int(station['tskey'])} {station['sgname']}")
	ax.set_title("Station locations", fontsize=12)
	ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), frameon=False, fontsize=10,
			  ncol=1, handletextpad=0.3, borderaxespad=0.0, title="Stations", title_fontsize=11)


def plot_series(axis, years_values, colour, linewidth=1.3, marker=None, zorder=3):
	"""Plot a {year: value} series."""
	if not years_values:
		return
	years_sorted = sorted(years_values)
	axis.plot(years_sorted, [years_values[y] for y in years_sorted], color=colour,
			  linewidth=linewidth, marker=marker, markersize=3.5, alpha=0.9, zorder=zorder)


def make_figure(stations, station_series, dataset_series_by_name):
	if stations.empty:
		print("No qualifying stations - nothing to plot.")
		return

	n_stations = len(stations)
	n_cols = 2
	n_rows = int(np.ceil(n_stations / n_cols))

	figure = plt.figure(figsize=(11 * n_cols + 7, 4.2 * n_rows + 1.4))
	grid = figure.add_gridspec(n_rows, n_cols + 1, width_ratios=[2.6] * n_cols + [1.3],
								hspace=0.4, wspace=0.16,
								left=0.04, right=0.99, top=1 - 0.9 / (4.2 * n_rows + 1.4),
								bottom=1.1 / (4.2 * n_rows + 1.4))

	# one x and y range for every panel, from all stations and datasets, so panels compare directly
	all_years, all_values = [], []
	for series in [station_series] + list(dataset_series_by_name.values()):
		for tskey in stations["tskey"].astype(int):
			d = series.get(tskey, {})
			all_years += list(d)
			all_values += [v for v in d.values() if np.isfinite(v)]
	xlim = (min(all_years) - 1, max(all_years) + 1) if all_years else None
	ylim = ((np.floor(min(all_values) / 25) * 25, np.ceil(max(all_values) / 25) * 25)
			if all_values else None)
	print(f"Shared axes: years {xlim}, flux {ylim} W m-2")

	for index, (_, station) in enumerate(stations.iterrows()):
		row, col = divmod(index, n_cols)
		axis = figure.add_subplot(grid[row, col])
		tskey = int(station["tskey"])

		station_dict = station_series.get(tskey, {})
		plot_series(axis, station_dict, STATION_COLOUR, linewidth=2.0, marker="o", zorder=5)
		for dataset_name, per_station_series in dataset_series_by_name.items():
			plot_series(axis, per_station_series.get(tskey, {}), DATASET_COLOURS[dataset_name])

		axis.set_title(f"\u25CF {tskey} {station['sgname']}", fontsize=12, color=station["colour"],
					   fontweight="bold")
		axis.set_ylabel("Oct SW down (W m$^{-2}$)", fontsize=11)
		axis.tick_params(labelsize=10)
		axis.grid(alpha=0.25)
		if xlim:
			axis.set_xlim(*xlim)
		if ylim:
			axis.set_ylim(*ylim)

	map_axis = figure.add_subplot(grid[:, -1], projection=ccrs.PlateCarree())
	plot_map(map_axis, stations)

	handles = [Line2D([], [], color=STATION_COLOUR, linewidth=2.0, marker="o", markersize=4,
					  label="GEBA station")]
	handles += [Line2D([], [], color=c, linewidth=2.0, label=n) for n, c in DATASET_COLOURS.items()]
	figure.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
				  fontsize=12, bbox_to_anchor=(0.43, 0.0))

	figure.suptitle(
		"GEBA station October SW down vs ERA5 / MERRA-2 / CLARA-A3 / SARAH-3 / CERES-EBAF",
		fontsize=15, fontweight="bold"
	)
	OUTPUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
	figure.savefig(OUTPUT_FIGURE, dpi=200, bbox_inches="tight", pad_inches=0.15)
	plt.close(figure)


# ============================================================================
# MAIN
# ============================================================================

def main():
	stations, station_series = load_geba_stations()
	print(f"Qualifying GEBA stations (GLOBAL, post-{MIN_YEAR}, >{MIN_ENTRIES} October entries): {len(stations)}")
	if stations.empty:
		return

	era5_data = load_era5_ssrd_october()
	merra2_data = load_merra2_swgdn_october()
	clara_data = load_clara_sis_october()
	sarah3_data = load_sarah3_sis_october()
	ceres_data = load_ceres_october()

	gridded = {"ERA5": era5_data, "MERRA-2": merra2_data, "CLARA-A3": clara_data,
			   "SARAH-3": sarah3_data, "CERES-EBAF": ceres_data}
	dataset_series_by_name = {name: {} for name in gridded}
	for _, station in stations.iterrows():
		tskey = int(station["tskey"])
		for name, data in gridded.items():
			dataset_series_by_name[name][tskey] = extract_station_series(
				data, station["sgylat"], station["sgxlon"])

	make_figure(stations, station_series, dataset_series_by_name)
	print(f"Figure: {OUTPUT_FIGURE}")


if __name__ == "__main__":
	main()