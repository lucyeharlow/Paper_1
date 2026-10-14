"""
Southern Africa (10-40E, 35-10S) 2 m temperature trends from six datasets.

DATASETS
  ERA5    scratch/era5/global_2m_temperature.nc                      (t2m, K)
  MERRA2  scratch/merra_2/MERRA2_*instM_2d_asm_Nx.YYYYMM.nc4         (T2M, K)
  JRA-3Q  scratch/jra_3q/915830.tmp2m-hgt-an-gauss...*.nc            (tmp2m-hgt-an-gauss, K)
  CHIRTS  data/chirts/{Tmax,Tmin}/{Tmax,Tmin}.YYYY.nc                Tmean = (Tmax + Tmin) / 2
  NOAA    data/mlost/gridded/NOAAGlobalTemp_*.nc                     (anom, degC anomalies)
  CRUTEM  data/crutem/CRUTEM.5.1.0.0.anomalies.nc                    (tas, degC anomalies)

METHOD
  Every dataset is cut to the domain and reduced to monthly means once, in memory.
  Unreadable MERRA2 / JRA-3Q / CHIRTS files are skipped (reported) and left as missing months.
  Anomalies are taken against each dataset's own 1991-2020 monthly climatology.
  ERA5, MERRA2, JRA-3Q, NOAA and CRUTEM use their maximal shared period; CHIRTS 1983-2016.
  NO MISSING VALUES: a cell's trend for a month is only computed if that month has a value
  in every year of the period; otherwise the cell is left out (never filled with 0).
  Regional series average only over those complete cells.
  Grid-cell OLS trends (degC/decade) per dataset and calendar month, on native grids.
  The October regional series use cosine-latitude area weighting.

PANELS
  (a) monthly trend distributions (boxplots)   (b)-(g) October trend maps
  (h) area-weighted October anomaly series.
"""
from __future__ import annotations
import calendar
import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature

scratch = Path("/soge-home/users/jesu4127/scratch")
data_dir = Path("/soge-home/users/jesu4127/data")
out_dir = Path("/soge-home/users/jesu4127/Paper_1/temperature/plots")
out_dir.mkdir(parents=True, exist_ok=True)

era5_file = scratch / "era5/global_2m_temperature.nc"
merra2_files = (scratch / "merra_2", "MERRA2_*instM_2d_asm_Nx.*.nc4",
                r"instM_2d_asm_Nx\.(\d{4})(\d{2})\.nc4$")
jra3q_files = (scratch / "jra_3q",
               "915830.tmp2m-hgt-an-gauss.jra3q.anl_surf.0_0_0.tmp2m-hgt-an-gauss.*.nc",
               r"\.tmp2m-hgt-an-gauss\.(\d{4})(\d{2})\d{2}00_\d{10}\.nc$")
chirts_dir = data_dir / "chirts"
noaa_file = data_dir / ("mlost/gridded/NOAAGlobalTemp_v6.1.0_gridded_"
                        "s185001_e202607_c20260807T143156.nc")
crutem_file = data_dir / "crutem/CRUTEM.5.1.0.0.anomalies.nc"

lon_min, lon_max, lat_min, lat_max = 10.0, 40.0, -35.0, -10.0
target_month = 10
min_years = 20
chirts_years = set(range(1983, 2017))
clim_years = range(1991, 2021)
trend_levels = np.linspace(-0.6, 0.6, 25)

labels = {"ERA5": "ERA5", "MERRA2": "MERRA-2", "JRA-3Q": "JRA-3Q", "CHIRTS": "CHIRTS",
          "NOAA": "NOAA Global Temp v6.1.0", "CRUTEM": "CRUTEM5"}
colours = {"ERA5": "#FF7C7E", "MERRA2": "#FEC300", "JRA-3Q": "#016ff2",
           "CHIRTS": "#9E3F71", "NOAA": "#31BCBA", "CRUTEM": "#2F5597"}
months = np.arange(1, 13)
pc = ccrs.PlateCarree()


# loading and prep functions
def coord(ds, *names):
    for n in names:
        if n in ds.variables:
            return n
    raise KeyError(f"None of {names} found. Available: {list(ds.variables)}")


def prep(ds):
    """Rename to time/lat/lon, put lon on -180..180, cut to the domain, sort ascending."""
    rename = {coord(ds, "latitude", "lat", "Latitude", "LAT", "y"): "lat",
              coord(ds, "longitude", "lon", "Longitude", "LON", "x"): "lon"}
    if "valid_time" in ds.variables:
        rename["valid_time"] = "time"
    ds = ds.rename({k: v for k, v in rename.items() if k != v})
    if ds.lon.max() > 180:
        ds = ds.assign_coords(lon=(ds.lon + 180) % 360 - 180)
    ds = ds.sel(lat=ds.lat[(ds.lat >= lat_min) & (ds.lat <= lat_max)],
                lon=ds.lon[(ds.lon >= lon_min) & (ds.lon <= lon_max)])
    return ds.sortby(["lat", "lon"])


def mask(da, fill=1e4):
    return da.where(abs(da) < fill)


def to_monthly(da, kelvin=False, fill=1e4):
    """Mask fill values, convert K -> degC, average to monthly and load into memory."""
    extra = [d for d in da.dims if d not in ("time", "lat", "lon")]   # e.g. NOAA's z
    da = mask(da.squeeze(extra, drop=True).transpose("time", "lat", "lon"), fill)
    return (da - 273.15 if kelvin else da).resample(time="1MS").mean().load()


def complete_years(directory, glob, regex):
    found = {}
    for p in directory.glob(glob):
        if m := re.search(regex, p.name):
            found.setdefault(int(m[1]), {})[int(m[2])] = p
    years = [y for y in sorted(found) if len(found[y]) == 12]
    if not years:
        raise RuntimeError(f"No complete years in {directory}")
    print(f"  {directory.name}: {years[0]}-{years[-1]} ({len(years)} complete years)")
    return [found[y][m] for y in years for m in months]


def read_file(path, var, kelvin=True):
    """One file -> domain-only monthly means in memory."""
    with xr.open_dataset(path) as ds:
        return to_monthly(prep(ds)[var], kelvin=kelvin)


def load_skipping_bad(items, read, label):
    """Read each item separately; skip (and report) any unreadable file, keep going."""
    parts, bad = [], []
    for item in items:
        try:
            parts.append(read(item))
        except Exception as e:
            bad.append(item)
            print(f"  SKIPPED {label} {getattr(item, 'name', item)}: {type(e).__name__}: {e}")
    if not parts:
        raise RuntimeError(f"No readable {label} files.")
    if bad:
        print(f"  {label}: skipped {len(bad)} bad file(s); those months are left missing.")
    return xr.concat(parts, "time", coords="minimal", compat="override",
                     combine_attrs="override").sortby("time")


def load_chirts():
    paths = {}
    for v in ("Tmax", "Tmin"):
        paths[v] = {int(m[1]): p for p in (chirts_dir / v).glob(f"{v}.*.nc")
                    if (m := re.search(rf"{v}\.(\d{{4}})\.nc$", p.name))}
    years = sorted(set(paths["Tmax"]) & set(paths["Tmin"]))
    if not years:
        raise RuntimeError("No common CHIRTS Tmax/Tmin years found.")

    def read_year(year):
        with xr.open_dataset(paths["Tmax"][year]) as a, xr.open_dataset(paths["Tmin"][year]) as b:
            tmax, tmin = xr.align(mask(prep(a)["Tmax"]), mask(prep(b)["Tmin"]), join="inner")
            return to_monthly((tmax + tmin) / 2)
    return load_skipping_bad(years, read_year, "CHIRTS year")


def load_all():
    print("Loading (domain only, monthly, into memory)...")
    with xr.open_dataset(era5_file) as ds:
        era5 = to_monthly(prep(ds)["t2m"], kelvin=True)
    merra2 = load_skipping_bad(complete_years(*merra2_files),
                               lambda p: read_file(p, "T2M"), "MERRA2")
    jra3q = load_skipping_bad(complete_years(*jra3q_files),
                              lambda p: read_file(p, "tmp2m-hgt-an-gauss"), "JRA-3Q")
    chirts = load_chirts()
    with xr.open_dataset(noaa_file) as ds:
        noaa = to_monthly(prep(ds)["anom"], fill=1e3)
    with xr.open_dataset(crutem_file) as ds:
        crutem = to_monthly(prep(ds)["tas"], fill=1e3)
    return {"ERA5": era5, "MERRA2": merra2, "JRA-3Q": jra3q,
            "CHIRTS": chirts, "NOAA": noaa, "CRUTEM": crutem}


# Analysis
def years_of(da):
    return set(da.time.dt.year.values.tolist())


def select_years(da, years):
    return da.sel(time=da.time.dt.year.isin(sorted(years)))


def anomalies(name, da):
    base = select_years(da, clim_years)
    if base.time.size == 0:
        raise RuntimeError(f"{name}: no data in {clim_years[0]}-{clim_years[-1]}.")
    if max(years_of(base)) < clim_years[-1]:
        print(f"  {name}: climatology uses {min(years_of(base))}-{max(years_of(base))}")
    return (da.groupby("time.month") - base.groupby("time.month").mean()).drop_vars("month")


def ols_trend(da):
    """Per-cell OLS slope (degC/decade); cells with any missing year are left out."""
    da = complete(da)
    xa, ya = da.time.dt.year - da.time.dt.year.mean(), da - da.mean("time")
    return 10 * (xa * ya).sum("time", skipna=False) / (xa ** 2).sum()


def october_series(da):
    """Area-weighted mean over cells with an October value in every year (no NaNs)."""
    oct_ = complete(da.sel(time=da.time.dt.month == target_month))
    return oct_.weighted(np.cos(np.deg2rad(oct_.lat))).mean(("lat", "lon"))


def complete(da):
    """Keep only cells with a value at every time step."""
    return da.where(da.notnull().all("time"))


def describe(da):
    """Summary of the valid trend values; NaN if there are none (no crash)."""
    v = valid(da)
    q = np.percentile(v, [0, 25, 50, 75, 100]) if v.size else [np.nan] * 5
    return {"n_gridcells": v.size, "mean_C_per_decade": v.mean() if v.size else np.nan,
            "min_C_per_decade": q[0], "q25_C_per_decade": q[1], "median_C_per_decade": q[2],
            "q75_C_per_decade": q[3], "max_C_per_decade": q[4]}


def valid(da):
    v = np.asarray(da, dtype=float).ravel()
    return v[np.isfinite(v)]


# Plotting
def span(years):
    return f"{years[0]}-{years[-1]}"


def plot_map(ax, trend, name, years, panel, norm):
    mesh = ax.pcolormesh(trend.lon, trend.lat, trend, transform=pc, cmap="RdBu_r",
                         norm=norm, shading="nearest")
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=1.0, edgecolor="black", zorder=5)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.9, edgecolor="black", zorder=5)
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=pc)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4, linestyle="--")
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"fontsize": 8}
    ax.set_title(f"({span(years)})", fontsize=9, pad=6)
    ax.text(0.5, 1.16, labels[name], transform=ax.transAxes, fontsize=12,
            fontweight="bold", ha="center", va="bottom")
    ax.text(0.02, 0.96, panel, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top",
            bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none", alpha=0.7))
    return mesh


def make_figure(trends, periods, series):
    print("\nCreating figure...")
    fig = plt.figure(figsize=(18, 12), dpi=300)
    gs = fig.add_gridspec(3, 6, height_ratios=[1.0, 1.25, 0.8], hspace=0.85, wspace=0.28)

    # (a) monthly boxplots
    ax = fig.add_subplot(gs[0, :])
    line = dict(color="black", linewidth=0.9)
    bps = {}
    for (name, t), off in zip(trends.items(), np.linspace(-0.15, 0.15, 6)):
        bps[name] = ax.boxplot([valid(t.sel(month=m)) for m in months], positions=months + off,
                               widths=0.06, patch_artist=True, showfliers=False,
                               manage_ticks=False, whiskerprops=line, capprops=line,
                               medianprops=line, boxprops=dict(facecolor=colours[name],
                               edgecolor="black", linewidth=0.9, alpha=0.9))
    ax.axhline(0, linestyle="--", linewidth=0.8, color="black")
    ax.set(xlim=(0.6, 12.8), xticks=months, xticklabels=calendar.month_abbr[1:])
    ax.set_xlabel("Month", labelpad=8, fontsize=10)
    ax.set_ylabel(r"Tmean trend ($^\circ$C decade$^{-1}$)", fontsize=10)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.text(0.99, 0.97, "(a)", transform=ax.transAxes, fontsize=12, fontweight="bold",
            va="top", ha="right")
    ax.legend(handles=[Patch(facecolor=colours[n], edgecolor="black",
                             label=f"{labels[n]} ({span(periods[n])})") for n in trends],
              loc="lower center", bbox_to_anchor=(0.5, 1.04), frameon=False, fontsize=9, ncol=6)

    # (b)-(g) October trend maps
    norm = BoundaryNorm(trend_levels, ncolors=plt.get_cmap("RdBu_r").N, extend="both")
    map_axes = [fig.add_subplot(gs[1, i], projection=pc) for i in range(6)]
    for ax_map, name, panel in zip(map_axes, trends, "bcdefg"):
        mesh = plot_map(ax_map, trends[name].sel(month=target_month), name, periods[name],
                        f"({panel})", norm)
    fig.text(0.5, 0.66, "October trends", ha="center", va="center", fontsize=14, fontweight="bold")
    cbar = fig.colorbar(mesh, ax=map_axes, orientation="horizontal", fraction=0.06,
                        pad=0.1, aspect=45, shrink=0.9)
    cbar.set_label(r"October Tmean trend ($^\circ$C decade$^{-1}$)")

    # (h) area-weighted October series
    ax = fig.add_subplot(gs[2, :])
    for name, s in series.items():
        if s.notnull().any():
            ax.plot(s.time.dt.year, s, color=colours[name], linewidth=2, label=labels[name])
    all_years = [y for p in periods.values() for y in p]
    ax.set_xlim(min(all_years), max(all_years))
    ax.set_title("October area-weighted anomalies", fontsize=12, pad=10)
    ax.set_xlabel("Year")
    ax.set_ylabel(r"October anomaly ($^\circ$C)", labelpad=8)
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=8)
    ax.text(0.01, 0.97, "(h)", transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")
    ax.legend(loc="lower right", frameon=False, ncol=6)

    fig.suptitle("Southern Africa temperature trends", fontsize=17, fontweight="bold", y=0.97)
    out = out_dir / "southern_africa_temperature_trends_jra_3q_era5_merra2_chirts_noaa_one_timeseries.png"
    fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out}")


# Main execution
def main():
    data = load_all()

    print("\nAnomalies...")
    data = {name: anomalies(name, da) for name, da in data.items()}

    # Year ranges: shared period for all but CHIRTS (fixed 1983-2016)
    shared = set.intersection(*(years_of(da) for n, da in data.items() if n != "CHIRTS"))
    periods = {n: sorted(shared) for n in data}
    periods["CHIRTS"] = sorted(years_of(data["CHIRTS"]) & chirts_years)
    print("\nYEAR RANGES")
    for name, yrs in periods.items():
        if len(yrs) < min_years:
            raise RuntimeError(f"{name}: only {len(yrs)} years available.")
        print(f"  {labels[name]:<26}{span(yrs)} ({len(yrs)} years)")
    data = {n: select_years(da, periods[n]) for n, da in data.items()}


    print("\nTrends...")
    trends = {n: da.groupby("time.month").map(ols_trend) for n, da in data.items()}
    series = {n: october_series(da) for n, da in data.items()}

    print("\nOCTOBER DIAGNOSTICS (degC/decade)")
    for name, t in trends.items():
        d = describe(t.sel(month=target_month))
        if not d["n_gridcells"]:
            print(f"  {labels[name]:<26}no complete cells"); continue
        print(f"  {labels[name]:<26}complete cells={d['n_gridcells']:>7,}  "
              f"min {d['min_C_per_decade']:+.4f}  mean {d['mean_C_per_decade']:+.4f}  "
              f"median {d['median_C_per_decade']:+.4f}  max {d['max_C_per_decade']:+.4f}")
        np.save(out_dir / f"{name.lower().replace('-', '')}_october_trend_C_per_decade.npy",
                t.sel(month=target_month).values)

    rows = []
    for m in months:
        for name, t in trends.items():
            rows.append({"dataset": name, "month": calendar.month_abbr[m], "month_number": m,
                         **describe(t.sel(month=m))})
    pd.DataFrame(rows).to_csv(out_dir / "monthly_trends_summary.csv", index=False)
    print(f"Saved summary and .npy files to {out_dir}")

    make_figure(trends, periods, series)


if __name__ == "__main__":
    main()