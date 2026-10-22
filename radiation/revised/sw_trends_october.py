"""
October downwelling SW decadal trend comparison:
ERA5 | MERRA-2 | CLARA-A3 | SARAH-3, common period 1983-2024.
(Set MONTHS / QUANTITIES below to make other months, or absorbed SW = surface net SW, again.)

DATA (current server)
  ERA5      scratch/era5/global_surface_solar_radiation_downwards.nc   ssrd  (SW down)
            scratch/era5/global_surface_net_solar_radiation.nc          ssr   (net = down - up)
            monthly means of daily accumulations (J m-2 per day) -> W m-2 by dividing by 86400 s
  MERRA-2   scratch/merra_2/MERRA2_<ver>.tavgM_2d_rad_Nx.YYYYMM.nc4: SWGDN (down), SWGNT
            (net); where two versions exist for a month the highest is used
  CLARA-A3  data/clara_a3/*SISmmYYYYMM01*.nc (SIS, down), *SNSmmYYYYMM01*.nc (SNS, net):
            thematic CDR (TCDR, ...AVPOS01GL) to 2020, interim CDR (ICDR, ...AVPOSI1GL) from
            2021; the TCDR file is used where both exist
  SARAH-3   data/sarah_3/**/*SISmmYYYYMM01*.nc, SIS (down) only; Meteosat disc (about +-65 deg).
            SARAH-3 has no surface albedo / upward SW, so it has no absorbed-SW panel.
  Missing or unreadable months are reported and left out; a panel with no data says so.

METHOD
  All datasets bilinearly interpolated to the MERRA-2 0.5 x 0.625 deg grid.
  OLS trend per grid point (W m-2 per decade), vectorised. Significance as in the temperature
  and vertical velocity scripts: lag-1 autocorrelation r1 of the residuals gives
  n_eff = n (1 - r1) / (1 + r1), capped at n (Santer et al. 2000); two-sided t-test with
  n_eff - 2 degrees of freedom; Benjamini-Hochberg FDR q < FDR_ALPHA per map (Wilks 2016).
  A grid point needs at least MIN_YEARS years.

FIGURE  <OUT_DIR>/sw_down/SW_down_trends_10_Oct.png
  All trends are plotted; small mid-grey dots (no outline) mark FDR-significant points,
  thinned to every DOT_EVERY-th grid point so they stay readable on the global maps.
  Oceans and lakes are masked white.
"""
from __future__ import annotations
import calendar
import glob
import os
import re
import warnings
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mplticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter
from scipy import stats
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- paths
ERA5_DOWN = "/soge-home/users/jesu4127/scratch/era5/global_surface_solar_radiation_downwards.nc"
ERA5_NET = "/soge-home/users/jesu4127/scratch/era5/global_surface_net_solar_radiation.nc"
MERRA2_DIR = "/soge-home/users/jesu4127/scratch/merra_2"
CLARA_DIR = "/soge-home/users/jesu4127/data/clara_a3"
SARAH_DIR = "/soge-home/users/jesu4127/data/sarah_3"
OUT_DIR = "/soge-home/users/jesu4127/Paper_1/radiation/revised/plots"

# ---------------------------------------------------------------- settings
MONTHS = [10]
MONTH = 10                        # default month for the assumption-check script
QUANTITIES = ["down"]
FDR_ALPHA = 0.10
DECADE = 10
CLIM = {"down": 8.0, "absorbed": 8.0}
START, END = 1983, 2024
MIN_YEARS = 8
DOT_EVERY = 4                     # marker every 4th grid point (about 2 x 2.5 deg)
DOT_SIZE = 3                      # marker area (points^2)
DOT_COLOUR = "#747474"            # one mid grey (halfway between #333333 and #B4B4B4)
SECS_PER_DAY = 86400.0
M2_LAT = np.arange(-90.0, 90.01, 0.5)
M2_LON = np.arange(-180.0, 180.0, 0.625)
NAMES = ["ERA5", "MERRA-2", "CLARA-A3", "SARAH-3"]
LABEL = {"down": "downwelling", "absorbed": "absorbed (surface net = down − up)"}
SOURCE = {   # dataset -> quantity -> variable name (None: not available)
    "ERA5": {"down": "ssrd", "absorbed": "ssr"},
    "MERRA-2": {"down": "SWGDN", "absorbed": "SWGNT"},
    "CLARA-A3": {"down": "SIS", "absorbed": "SNS"},
    "SARAH-3": {"down": "SIS", "absorbed": None},
}
NOT_AVAILABLE = {("SARAH-3", "absorbed"): "not available\n(SARAH-3 has no surface albedo / upward SW)"}


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------- loading
def to_m2_grid(da):
    """Rename to lat/lon, put lon on -180..180, sort, interpolate onto the MERRA-2 grid."""
    da = da.rename({k: v for k, v in (("latitude", "lat"), ("longitude", "lon")) if k in da.dims})
    if float(da.lon.max()) > 180.5:
        da = da.assign_coords(lon=(da.lon + 180) % 360 - 180)
    da = da.sortby("lat").sortby("lon")
    return da.interp(lat=M2_LAT, lon=M2_LON, method="linear")


def load_era5(month=MONTH, quantity="down"):
    """(year, lat, lon) ERA5 SW in W m-2 on the MERRA-2 grid, for one calendar month."""
    path = ERA5_DOWN if quantity == "down" else ERA5_NET
    if not os.path.exists(path):
        log(f"  [ERROR] not found: {path}")
        return None
    with xr.open_dataset(path) as ds:
        da = ds[SOURCE["ERA5"][quantity]]
        t = "valid_time" if "valid_time" in da.dims else "time"
        sel = (da[t].dt.month == month) & (da[t].dt.year >= START) & (da[t].dt.year <= END)
        da = da.isel({t: np.flatnonzero(sel.values)}).load()
    years = da[t].dt.year.values
    da = (da / SECS_PER_DAY).assign_coords({t: years}).rename({t: "year"})
    da = da.drop_vars(["number", "expver"], errors="ignore")
    log(f"  ERA5: {len(years)} years ({years.min()}-{years.max()})")
    return to_m2_grid(da)


def stack_years(files, varname, label, fill_ok):
    """{year: path} -> (year, lat, lon) of one variable, skipping unreadable files."""
    slices, years, missing = [], [], []
    for yr in range(START, END + 1):
        f = files.get(yr)
        if f is None:
            missing.append(yr)
            continue
        try:
            with xr.open_dataset(f) as ds:
                v = varname if varname in ds else varname.lower()
                da = ds[v].squeeze(drop=True).load()
            slices.append(da.where(fill_ok(da)))
            years.append(yr)
        except Exception as e:
            log(f"  [WARN] {label} {os.path.basename(f)}: {type(e).__name__}: {e}")
    if missing:
        log(f"  [WARN] no {label} file for {len(missing)} year(s): {missing}")
    if not slices:
        log(f"  [ERROR] no {label} data loaded")
        return None
    log(f"  {label}: {len(years)} years ({years[0]}-{years[-1]})")
    return xr.concat(slices, "year").assign_coords(year=np.array(years))


_merra2_index = {}


def merra2_files(month):
    """{year: path} for one month, keeping the highest version where a month appears twice."""
    if not _merra2_index:
        for f in glob.glob(os.path.join(MERRA2_DIR, "MERRA2_*.tavgM_2d_rad_Nx.*.nc4")):
            m = re.match(r"MERRA2_(\d{3})\.tavgM_2d_rad_Nx\.(\d{4})(\d{2})\.nc4$",
                         os.path.basename(f))
            if m:
                key = (int(m[2]), int(m[3]))
                if key not in _merra2_index or int(m[1]) > _merra2_index[key][0]:
                    _merra2_index[key] = (int(m[1]), f)
    return {y: v[1] for (y, mo), v in _merra2_index.items() if mo == month}


def load_merra2(month=MONTH, quantity="down"):
    da = stack_years(merra2_files(month), SOURCE["MERRA-2"][quantity], "MERRA-2",
                     lambda d: np.abs(d) < 1e14)
    return None if da is None else to_m2_grid(da)


def cmsaf_files(directory, recursive, code, month):
    """{year: path} for <code>mmYYYYMM01*.nc; the thematic CDR is preferred over the interim."""
    pattern = os.path.join(directory, "**" if recursive else "", f"*{code}mm*{month:02d}01*.nc")
    found = {}
    for f in sorted(glob.glob(pattern, recursive=recursive)):
        m = re.search(rf"{code}mm(\d{{4}}){month:02d}01", os.path.basename(f))
        if not m:
            continue
        yr, interim = int(m[1]), "AVPOSI" in os.path.basename(f) or "interim" in f
        if yr not in found or (found[yr][0] and not interim):
            found[yr] = (interim, f)
    return {y: v[1] for y, v in found.items()}


def load_clara(month=MONTH, quantity="down"):
    var = SOURCE["CLARA-A3"][quantity]
    da = stack_years(cmsaf_files(CLARA_DIR, False, var, month), var, "CLARA-A3",
                     lambda d: d > -900)
    return None if da is None else to_m2_grid(da)


def load_sarah(month=MONTH, quantity="down"):
    var = SOURCE["SARAH-3"][quantity]
    if var is None:
        return None
    da = stack_years(cmsaf_files(SARAH_DIR, True, var, month), var, "SARAH-3",
                     lambda d: d > -900)
    return None if da is None else to_m2_grid(da)


LOADERS = {"ERA5": load_era5, "MERRA-2": load_merra2, "CLARA-A3": load_clara,
           "SARAH-3": load_sarah}


# ---------------------------------------------------------------- statistics (vectorised)
def ols_stats(da):
    """Per-point OLS trend (per decade) and AR(1)-corrected p along 'year'."""
    da = da.reindex(year=np.arange(int(da.year.min()), int(da.year.max()) + 1))  # gaps -> NaN
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
    se = np.sqrt((ssr / (n_eff - 2) / sxx).where(n_eff > 2))
    p = xr.apply_ufunc(lambda t, df: 2 * stats.t.sf(np.abs(t), df), slope / se, n_eff - 2)
    out = xr.Dataset({"trend": DECADE * slope, "p": p, "n": n, "r1": r1, "n_eff": n_eff})
    return out.where((n >= MIN_YEARS) & (n_eff > 2))


def fdr_mask(p):
    """Benjamini-Hochberg over one map: True where the adjusted p < FDR_ALPHA."""
    v, sig = np.asarray(p), np.zeros(np.shape(p), bool)
    ok = np.isfinite(v)
    if ok.any():
        sig[ok] = stats.false_discovery_control(v[ok]) < FDR_ALPHA
    return sig


# ---------------------------------------------------------------- plotting
def add_map_features(ax):
    ax.add_feature(cfeature.OCEAN.with_scale("110m"), zorder=2, facecolor="white")
    ax.add_feature(cfeature.LAKES.with_scale("110m"), zorder=2, facecolor="white")
    ax.coastlines(resolution="110m", linewidth=0.4, zorder=3)
    ax.set_global()


def add_gridlines(ax, left_labels=False, bottom_labels=False):
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="black", alpha=0.45, linestyle=":")
    gl.top_labels = gl.right_labels = False
    gl.left_labels, gl.bottom_labels = left_labels, bottom_labels
    gl.xlocator = mplticker.FixedLocator([-120, -60, 0, 60, 120])
    gl.ylocator = mplticker.FixedLocator([-60, -30, 0, 30, 60])
    gl.xformatter, gl.yformatter = LongitudeFormatter(), LatitudeFormatter()
    gl.xlabel_style = {"size": 9, "color": "black", "rotation": 45}
    gl.ylabel_style = {"size": 9, "color": "black"}


def plot_panel(ax, st, col, row, nrows, label, levels):
    cf = ax.contourf(st.lon, st.lat, st.trend.values, levels=levels, cmap="RdBu_r",
                     extend="both", antialiased=False, transform=ccrs.PlateCarree(), zorder=1)
    sig = fdr_mask(st.p.values)
    keep = np.zeros_like(sig)
    keep[::DOT_EVERY, ::DOT_EVERY] = True
    pick = sig & keep
    lon2, lat2 = np.meshgrid(st.lon, st.lat)
    # under the ocean mask, so land only
    ax.scatter(lon2[pick], lat2[pick], s=DOT_SIZE, color=DOT_COLOUR, linewidths=0,
               transform=ccrs.PlateCarree(), zorder=1.5)
    add_map_features(ax)
    add_gridlines(ax, left_labels=(col == 0), bottom_labels=(row == nrows - 1))
    ax.text(0.02, 0.97, label, transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="top", ha="left", zorder=5,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, boxstyle="round,pad=0.15"))
    return cf


def make_figure(results, month, quantity, out_path):
    """2 x 2 global maps laid out in inches (no bbox_inches='tight', which crops cartopy maps)."""
    matplotlib.rcParams.update({"font.family": "sans-serif",
                                "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"]})
    levels = np.linspace(-CLIM[quantity], CLIM[quantity], 21)
    W, left, right, gap_x = 14.0, 1.25, 0.35, 0.45
    mw = (W - left - right - gap_x) / 2
    mh = mw / 2                                          # global PlateCarree: 360 x 180 deg
    tops = [1.55, 1.55 + mh + 1.35]                      # inches from the top of the figure
    H = tops[1] + mh + 1.75
    fig = plt.figure(figsize=(W, H))
    for k, name in enumerate(NAMES):
        r, c = divmod(k, 2)
        x0 = left + c * (mw + gap_x)
        ax = fig.add_axes([x0 / W, 1 - (tops[r] + mh) / H, mw / W, mh / H],
                          projection=ccrs.PlateCarree())
        fig.text((x0 + mw / 2) / W, 1 - (tops[r] - 0.12) / H, name, ha="center",
                 va="bottom", fontsize=15, fontweight="bold")
        if results.get(name) is None:
            add_map_features(ax)
            ax.text(0.5, 0.5, NOT_AVAILABLE.get((name, quantity), "no data for this month"),
                    transform=ax.transAxes, ha="center", va="center", fontsize=12, zorder=6,
                    bbox=dict(facecolor="white", edgecolor="0.6", boxstyle="round,pad=0.4"))
            continue
        plot_panel(ax, results[name], c, r, 2, f"{'abcd'[k]})", levels)
    for r, lbl in enumerate(["Reanalyses", "Satellite CDRs"]):
        fig.text(0.25 / W, 1 - (tops[r] + mh / 2) / H, lbl, fontsize=13, fontweight="bold",
                 rotation=90, va="center", ha="center")
    sm = plt.cm.ScalarMappable(norm=matplotlib.colors.BoundaryNorm(levels, 256, extend="both"),
                               cmap="RdBu_r")
    cb = fig.colorbar(sm, cax=fig.add_axes([0.12, 0.85 / H, 0.76, 0.2 / H]),
                      orientation="horizontal", extend="both", ticks=levels[::3])
    cb.set_label(r"W m$^{-2}$ decade$^{-1}$", fontsize=11)
    cb.ax.tick_params(labelsize=10)
    fig.suptitle(f"{calendar.month_name[month]} decadal OLS trends in {LABEL[quantity]} SW "
                 f"radiation ({START}-{END})\nMarkers: significant after FDR "
                 f"(α = {FDR_ALPHA:.2f}, AR(1)-corrected)",
                 fontsize=15, fontweight="bold", y=1 - 0.15 / H, va="top")
    fig.savefig(out_path, dpi=300, facecolor="white")
    plt.close(fig)
    log(f"  Saved: {out_path}")


def main():
    log("=" * 70)
    log(f"SW trends, every month: {' | '.join(NAMES)}, {START}-{END}")
    log("=" * 70)
    for quantity in QUANTITIES:
        folder = os.path.join(OUT_DIR, f"sw_{quantity}")
        os.makedirs(folder, exist_ok=True)
        for month in MONTHS:
            log(f"\n===== SW {quantity}, {calendar.month_name[month]} =====")
            results = {}
            for name in NAMES:
                if SOURCE[name][quantity] is None:
                    continue
                log(f"--- {name} ---")
                data = LOADERS[name](month, quantity)
                if data is None:
                    continue
                st = ols_stats(data)
                sig = fdr_mask(st.p.values)
                log(f"  median n_eff {float(st.n_eff.median()):.0f} of n "
                    f"{float(st.n.median()):.0f}; {100 * sig.sum() / max(np.isfinite(st.p.values).sum(), 1):.0f}"
                    "% of points FDR-significant")
                results[name] = st
            make_figure(results, month, quantity, os.path.join(
                folder, f"SW_{quantity}_trends_{month:02d}_{calendar.month_abbr[month]}.png"))
    log("\nDone.")


if __name__ == "__main__":
    main()