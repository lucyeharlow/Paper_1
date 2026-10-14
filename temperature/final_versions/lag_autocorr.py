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
  LAND ONLY: every dataset is masked to the same land surface, so land-only (CHIRTS, CRUTEM)
  and land+ocean (ERA5, MERRA2, JRA-3Q, NOAA) products are compared like for like. A cell
  is kept if >= 50 % of its area is land (Natural Earth 50 m land polygons, estimated
  from a sub-grid of points in each cell).
  NO MISSING VALUES: a cell's trend for a month is only computed if that month has a value
  in every year of the period; otherwise the cell is left out (never filled with 0).
  Regional series average only over those complete cells.
  Grid-cell OLS trends (degC/decade) per dataset and calendar month, on native grids.
  Boxplots and summary statistics are area-weighted by cos(latitude), so each box
  describes how trends are distributed over the area of the domain, not over grid cells.
  Significance: lag-1 autocorrelation (r1) of the OLS residuals gives an effective sample
  size n_eff = n (1 - r1) / (1 + r1), capped at n (Santer et al. 2000, JGR 105, 7337).
  The slope's standard error uses n_eff - 2 degrees of freedom, and a two-sided t-test
  gives p. The slope itself is plain OLS; only its uncertainty changes.
  Field significance: on each map (dataset x month) the p-values are adjusted with the
  Benjamini-Hochberg false discovery rate at alpha_FDR = 0.10, the recommended choice for
  spatially correlated fields (= 2 x a 0.05 global level; Wilks 2016, BAMS 97, 2263).
  The regional series is a single test and uses its p-value directly.
  The October regional series use cosine-latitude area weighting.

OUTPUTS (suffix _ar1)
  Figure: dots mark FDR-significant cells; each map states its median n_eff and n.
  Legend (h): regional October trend, p and n_eff per dataset.
  CSV: median n, r1, n_eff and % of area significant per dataset and month.
  .npy: October trend, n_eff and p per dataset.

PANELS
  (a) monthly trend distributions (boxplots)   (b)-(g) October trend maps
  (h) area-weighted October anomaly series.
"""
from __future__ import annotations
import calendar
import re
from functools import cache
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader
from shapely import box, contains_xy
from shapely.ops import unary_union

# ---------------------------------------------------------------- settings
scratch = Path("/soge-home/users/jesu4127/scratch")
data_dir = Path("/soge-home/users/jesu4127/data")
out_dir = Path("/soge-home/users/jesu4127/Paper_1/temperature/final_versions/plots")
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
land_min_fraction = 0.5          # keep cells that are at least half land
land_shapefile = None            # None: Natural Earth 50 m land via cartopy (downloaded once);
                                 # or a path to a local ne_50m_land.shp on an offline node

labels = {"ERA5": "ERA5", "MERRA2": "MERRA-2", "JRA-3Q": "JRA-3Q", "CHIRTS": "CHIRTS",
          "NOAA": "NOAA Global Temp v6.1.0", "CRUTEM": "CRUTEM5"}
colours = {"ERA5": "#FF7C7E", "MERRA2": "#FEC300", "JRA-3Q": "#016ff2",
           "CHIRTS": "#9E3F71", "NOAA": "#31BCBA", "CRUTEM": "#2F5597"}
months = np.arange(1, 13)
pc = ccrs.PlateCarree()


# ---------------------------------------------------------------- loading
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

@cache
def land_geometry():
    """Natural Earth land polygons, clipped to (a margin around) the domain."""
    path = land_shapefile or shapereader.natural_earth("50m", "physical", "land")
    land = unary_union(list(shapereader.Reader(str(path)).geometries()))
    return land.intersection(box(lon_min - 10, lat_min - 10, lon_max + 10, lat_max + 10))


def land_fraction(da):
    """Fraction of each grid cell that is land, from a k x k sub-grid of points per cell."""
    step = lambda x: float(np.median(np.abs(np.diff(x)))) if x.size > 1 else 0.0
    dlat, dlon = step(da.lat.values), step(da.lon.values)
    k = int(np.clip(np.ceil(max(dlat, dlon) / 0.1), 1, 10))   # ~0.1 deg sampling, 1-100 points
    off = (np.arange(k) + 0.5) / k - 0.5
    lat = da.lat.values[:, None, None, None] + off[None, None, :, None] * dlat
    lon = da.lon.values[None, :, None, None] + off[None, None, None, :] * dlon
    lat, lon = np.broadcast_arrays(lat, lon)
    frac = contains_xy(land_geometry(), lon, lat).mean(axis=(2, 3))
    return xr.DataArray(frac, dims=("lat", "lon"), coords={"lat": da.lat, "lon": da.lon})


def land_only(data):
    """Mask every dataset to cells that are at least land_min_fraction land."""
    print(f"\nLAND MASK (cells >= {land_min_fraction:.0%} land)")
    out = {}
    for name, da in data.items():
        land = land_fraction(da) >= land_min_fraction
        print(f"  {labels[name]:<26}{int(land.sum()):>7,} of {land.size:>7,} cells are land")
        out[name] = da.where(land)
    return out


# ---------------------------------------------------------------- analysis
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


alpha_fdr = 0.10      # Benjamini-Hochberg FDR level for the maps (Wilks 2016)


def ols_stats(da):
    """Per-cell OLS slope (degC/decade) with AR(1)-corrected significance.

    Returns trend, n, r1 (lag-1 autocorrelation of residuals), n_eff and p.
    Years are put on a gap-free axis first, so lag 1 always means the previous year.
    """
    da = complete(da)                       # no NaNs: drop cells with any missing year
    years = da.time.dt.year.values
    da = da.assign_coords(time=years).reindex(time=np.arange(years.min(), years.max() + 1))
    ok = da.notnull()
    n = ok.sum("time")
    x = da.time.astype(float).where(ok)
    xa, ya = x - x.mean("time"), da - da.mean("time")
    sxx = (xa ** 2).sum("time")
    slope = (xa * ya).sum("time") / sxx
    resid = ya - slope * xa
    ssr = (resid ** 2).sum("time")
    r1 = (resid * resid.shift(time=1)).sum("time") / ssr
    n_eff = np.minimum(n * (1 - r1) / (1 + r1), n)
    se = np.sqrt(ssr / (n_eff - 2) / sxx)
    p = xr.apply_ufunc(lambda t, df: 2 * stats.t.sf(np.abs(t), df), slope / se, n_eff - 2)
    out = xr.Dataset({"trend": 10 * slope, "n": n, "r1": r1, "n_eff": n_eff, "p": p})
    return out.where((n >= 3) & (n_eff > 2))


def october_series(da):
    """Area-weighted mean over cells with an October value in every year (no NaNs)."""
    oct_ = complete(da.sel(time=da.time.dt.month == target_month))
    return oct_.weighted(np.cos(np.deg2rad(oct_.lat))).mean(("lat", "lon"))


def complete(da):
    """Keep only cells with a value at every time step."""
    return da.where(da.notnull().all("time"))


def area_weights(da):
    """cos(latitude) weights, proportional to grid-cell area on a regular lat-lon grid."""
    return np.cos(np.deg2rad(da.lat)).broadcast_like(da)


def weighted_values(da):
    """Finite values of a lat-lon field and their area weights."""
    v, w = da.values.ravel(), area_weights(da).values.ravel()
    ok = np.isfinite(v)
    return v[ok], w[ok]


def wquantile(v, w, q):
    """Area-weighted quantiles: each cell counts in proportion to its area.

    Each value sits at the midpoint of its share of the cumulative weight (Hazen
    plotting positions); with equal weights this is the ordinary sample quantile.
    """
    i = np.argsort(v)
    v, w = v[i], w[i]
    return np.interp(q, (np.cumsum(w) - 0.5 * w) / w.sum(), v)


def box_stats(da):
    """Area-weighted box: quartiles weighted; whiskers at the most extreme cell within 1.5 IQR."""
    v, w = weighted_values(da)
    if not v.size:
        return None
    q1, med, q3 = wquantile(v, w, [0.25, 0.5, 0.75])
    lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    return {"q1": q1, "med": med, "q3": q3, "fliers": [],
            "whislo": v[v >= lo].min(), "whishi": v[v <= hi].max()}


def describe(da):
    """Area-weighted summary of the trend values; NaN if there are none (no crash)."""
    v, w = weighted_values(da)
    if not v.size:
        return {"n_gridcells": 0, **{f"{k}_C_per_decade": np.nan for k in
                ("mean", "min", "q25", "median", "q75", "max")}}
    q = wquantile(v, w, [0.25, 0.5, 0.75])
    return {"n_gridcells": v.size, "mean_C_per_decade": np.average(v, weights=w),
            "min_C_per_decade": v.min(), "q25_C_per_decade": q[0], "median_C_per_decade": q[1],
            "q75_C_per_decade": q[2], "max_C_per_decade": v.max()}


def fdr_significant(p):
    """Benjamini-Hochberg across one map: True where the adjusted p <= alpha_fdr."""
    v, out = p.values, np.zeros(p.shape, bool)
    ok = np.isfinite(v)
    if ok.any():
        out[ok] = stats.false_discovery_control(v[ok]) <= alpha_fdr
    return p.copy(data=out)


def add_fdr(r):
    """Add a boolean 'sig' field: FDR-significant, applied separately to each month's map."""
    return r.assign(sig=xr.concat([fdr_significant(r.p.sel(month=m)) for m in r.month], "month"))


def pct_area_significant(st):
    """Percentage of the (complete land) area that is FDR-significant; NaN if no cells."""
    w = area_weights(st.p).where(st.p.notnull())
    return 100 * float((w * st.sig).sum() / w.sum()) if w.sum() > 0 else np.nan

def regional(st):
    """'trend, p, n_eff/n' text for a regional series, or a note if there is none."""
    if np.isnan(float(st.trend)):
        return "no complete cells"
    p = float(st.p)
    return (f"{float(st.trend):+.2f} °C/dec, {'p < 0.001' if p < 0.001 else f'p = {p:.3f}'}, "
            f"n$_{{eff}}$ = {float(st.n_eff):.0f}/{int(st.n)}")



# ---------------------------------------------------------------- plotting
def span(years):
    return f"{years[0]}-{years[-1]}"


def plot_map(ax, st, name, years, panel, norm):
    trend = st.trend
    mesh = ax.pcolormesh(trend.lon, trend.lat, trend, transform=pc, cmap="RdBu_r",
                         norm=norm, shading="nearest")
    sig = st.sig.values
    if sig.any():   # dots on significant cells, thinned so fine grids stay readable
        step = max(1, int(np.ceil(np.sqrt(sig.size / 400))))
        lon2, lat2 = np.meshgrid(trend.lon, trend.lat)
        keep = np.zeros_like(sig)
        keep[::step, ::step] = True
        keep &= sig
        ax.scatter(lon2[keep], lat2[keep], s=1.5, color="black", linewidths=0, zorder=6,
                   transform=pc)
    note = (f"n$_{{eff}}$ ≈ {float(st.n_eff.median()):.0f} (n = {float(st.n.median()):.0f})\n"
            f"{pct_area_significant(st):.0f}% of area sig." if st.p.notnull().any()
            else "no complete cells")
    ax.text(0.02, 0.03, note, transform=ax.transAxes, fontsize=7, va="bottom", zorder=8,
            bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none", alpha=0.8))
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


def make_figure(results, periods, series, series_stats):
    trends = {n: r.trend for n, r in results.items()}
    print("\nCreating figure...")
    fig = plt.figure(figsize=(18, 12), dpi=300)
    gs = fig.add_gridspec(3, 6, height_ratios=[1.0, 1.25, 0.8], hspace=0.85, wspace=0.28)

    # (a) monthly boxplots
    ax = fig.add_subplot(gs[0, :])
    line = dict(color="black", linewidth=0.9)
    for (name, t), off in zip(trends.items(), np.linspace(-0.15, 0.15, 6)):
        boxes = {m: box_stats(t.sel(month=m)) for m in months}      # area-weighted
        have = [m for m in months if boxes[m]]
        if have:
            ax.bxp([boxes[m] for m in have], positions=np.array(have) + off, widths=0.06,
                   patch_artist=True, showfliers=False, manage_ticks=False, whiskerprops=line,
                   capprops=line, medianprops=line, boxprops=dict(facecolor=colours[name],
                   edgecolor="black", linewidth=0.9, alpha=0.9))
    ax.axhline(0, linestyle="--", linewidth=0.8, color="black")
    ax.set(xlim=(0.6, 12.8), xticks=months, xticklabels=calendar.month_abbr[1:])
    ax.set_xlabel("Month", labelpad=8, fontsize=10)
    ax.set_ylabel("Tmean trend, area-weighted\n" r"($^\circ$C decade$^{-1}$)", fontsize=10)
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
        mesh = plot_map(ax_map, results[name].sel(month=target_month), name, periods[name],
                        f"({panel})", norm)
    fig.text(0.5, 0.66, "October trends", ha="center", va="center", fontsize=14, fontweight="bold")
    cbar = fig.colorbar(mesh, ax=map_axes, orientation="horizontal", fraction=0.06,
                        pad=0.1, aspect=45, shrink=0.9)
    cbar.set_label(r"October Tmean trend ($^\circ$C decade$^{-1}$)")

    # (h) area-weighted October series
    ax = fig.add_subplot(gs[2, :])
    for name, s in series.items():
        if s.notnull().any():
            ax.plot(s.time.dt.year, s, color=colours[name], linewidth=2,
                    label=f"{labels[name]}: {regional(series_stats[name])}")
    all_years = [y for p in periods.values() for y in p]
    ax.set_xlim(min(all_years), max(all_years))
    ax.set_title("October area-weighted anomalies", fontsize=12, pad=10)
    ax.set_xlabel("Year")
    ax.set_ylabel(r"October anomaly ($^\circ$C)", labelpad=8)
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=8)
    ax.text(0.01, 0.97, "(h)", transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), frameon=False, ncol=3, fontsize=8)

    fig.suptitle("Southern Africa land temperature trends (dots: significant after FDR, α$_{FDR}$ = 0.10, AR(1)-corrected)",
                 fontsize=17, fontweight="bold", y=0.97)
    out = out_dir / "southern_africa_temperature_trends_jra_3q_era5_merra2_chirts_noaa_one_timeseries_ar1.png"
    fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out}")


# ---------------------------------------------------------------- main
def main():
    data = land_only(load_all())

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


    print("\nTrends with AR(1)-corrected significance...")
    results = {n: add_fdr(da.groupby("time.month").map(ols_stats)) for n, da in data.items()}
    series = {n: october_series(da) for n, da in data.items()}
    series_stats = {n: ols_stats(s) for n, s in series.items()}

    print(f"\nOCTOBER DIAGNOSTICS (degC/decade; map significance: BH-FDR at {alpha_fdr})")
    print(f"  {'dataset':<26}{'cells':>7} {'mean':>8} {'median':>8}   {'n':>3} {'r1':>6} "
          f"{'n_eff':>6} {'%area sig':>10}   regional trend / p / n_eff")
    for name, r in results.items():
        oct_ = r.sel(month=target_month)
        d = describe(oct_.trend)
        if not d["n_gridcells"]:
            print(f"  {labels[name]:<26}no complete cells"); continue
        print(f"  {labels[name]:<26}{d['n_gridcells']:>7,} {d['mean_C_per_decade']:+8.4f} "
              f"{d['median_C_per_decade']:+8.4f}   {float(oct_.n.median()):>3.0f} "
              f"{float(oct_.r1.median()):+6.2f} {float(oct_.n_eff.median()):>6.1f} "
              f"{pct_area_significant(oct_):>9.0f}%   "
              f"{regional(series_stats[name]).replace('$_{eff}$', '_eff')}")
        key = name.lower().replace("-", "")
        for var, suffix in (("trend", "trend_C_per_decade"), ("n_eff", "n_eff"), ("p", "p_value")):
            np.save(out_dir / f"{key}_october_{suffix}_ar1.npy", oct_[var].values)

    rows = []
    for m in months:
        for name, r in results.items():
            rm = r.sel(month=m)
            rows.append({"dataset": name, "month": calendar.month_abbr[m], "month_number": m,
                         **describe(rm.trend),
                         "median_n_years": float(rm.n.median()),
                         "median_r1": float(rm.r1.median()),
                         "median_n_eff": float(rm.n_eff.median()),
                         "min_n_eff": float(rm.n_eff.min()),
                         "pct_area_significant_fdr": pct_area_significant(rm)})
    pd.DataFrame(rows).to_csv(out_dir / "monthly_trends_summary_ar1.csv", index=False)
    print(f"Saved summary and .npy files to {out_dir}")

    make_figure(results, periods, series, series_stats)


if __name__ == "__main__":
    main()