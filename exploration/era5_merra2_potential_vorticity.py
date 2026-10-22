"""
ERA5 and MERRA-2 potential vorticity (PV) over sub-Saharan Africa: the seasonal evolution of the
trend and of the climatology, one figure per pressure level.

DATA
  ERA5     scratch/era5/global/potential_vorticity/global_potential_vorticity_YYYY.nc
           monthly means, pv(valid_time, pressure_level, latitude, longitude), 0.25 deg
  MERRA-2  scratch/merra_2/MERRA2_<ver>.instM_3d_asm_Np.YYYYMM.nc4, EPV(time, lev, lat, lon)
           (Ertel's PV), monthly means, 0.5 x 0.625 deg; where two versions exist for a month
           (e.g. _400 and the reprocessed _401) the highest version is used
  Both in K m2 kg-1 s-1; shown in PVU (1 PVU = 1e-6 K m2 kg-1 s-1). Fill values (MERRA-2
  1e15) are masked. Each dataset stays on its own grid (no regridding).

METHOD
  Only the chosen pressure levels and the domain are read. Unreadable files are skipped and
  reported (those months are left missing). Both datasets use the same years: those present
  in both. For every dataset, month, level and grid point at once (vectorised):
    OLS trend (PVU/decade); lag-1 autocorrelation r1 of the residuals gives
    n_eff = n (1 - r1) / (1 + r1), capped at n (Santer et al. 2000); two-sided t-test with
    n_eff - 2 degrees of freedom. Dots: significant after Benjamini-Hochberg FDR at
    alpha_FDR = 0.10 (Wilks 2016), applied to each map (dataset x month) separately.
  Climatology: mean PV for each month over clim_years.

SIGN OF PV
  PV takes the sign of the Coriolis parameter: positive north of the equator, negative south.
  South of the equator a NEGATIVE trend means PV of larger magnitude. On the climatology
  maps the dashed / solid black contours mark -2 / +2 PVU (the dynamical tropopause).
  The script prints the mean climatological PV south of 10 S for both datasets, as a check
  that they use the same sign convention.

FIGURES  pv_era5_merra2_monthly_evolution_<level>hPa.png, one per level
  Rows 1-2: ERA5 and MERRA-2 trend maps, Jan -> Dec.
  Rows 3-4: ERA5 and MERRA-2 monthly climatology, Jan -> Dec.
  One colour scale shared by all 24 trend maps, and one shared by all 24 climatology maps
  (+-98th percentile of |value| over both datasets and all months). Set
  same_scale_all_levels = True to share them across levels too (PV at 200 hPa is about 10x
  larger than at 500 hPa, so this flattens 500 hPa).
"""
from __future__ import annotations
import calendar
import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import numpy as np
import xarray as xr
from scipy import stats
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ---------------------------------------------------------------- settings
era5_dir = Path("/soge-home/users/jesu4127/scratch/era5/global/potential_vorticity")
merra2_dir = Path("/soge-home/users/jesu4127/scratch/merra_2")
out_dir = Path("/soge-home/users/jesu4127/Paper_1/pv/plots")

levels = [200, 300, 500]                    # hPa, one figure each
years = range(1980, 2026)                   # trend period (years in both datasets are used)
clim_years = range(1991, 2021)              # climatology period
lon_min, lon_max, lat_min, lat_max = -20.0, 55.0, -35.0, 15.0     # sub-Saharan Africa
alpha_fdr = 0.10
show_significance = True
same_scale_all_levels = False
names = ["ERA5", "MERRA-2"]
pc = ccrs.PlateCarree()
months = np.arange(1, 13)


# ---------------------------------------------------------------- loading
def tidy(da):
    """PVU, fill values masked, ascending latitude, (year, month, level, lat, lon) order."""
    da = da.where(np.abs(da) < 1e10) * 1e6
    return da.sortby("lat").astype("float32")


def read_era5_year(path):
    """All months of the chosen levels and domain from one yearly file."""
    with xr.open_dataset(path) as ds:
        da = ds["pv"].rename(valid_time="time", latitude="lat", longitude="lon",
                             pressure_level="level")
        da = da.sel(level=levels, lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
        year = int(da.time.dt.year[0])
        da = da.load()                                              # reads only this subset
    da = da.drop_vars(["number", "expver"], errors="ignore")
    da = da.assign_coords(time=da.time.dt.month.values).rename(time="month")
    return tidy(da.reindex(month=months)).expand_dims(year=[year])   # missing months -> NaN


def load_era5():
    parts = []
    for f in sorted(era5_dir.glob("global_potential_vorticity_*.nc")):
        if int(f.stem.split("_")[-1]) not in years:
            continue
        try:
            parts.append(read_era5_year(f))
        except Exception as e:
            print(f"  SKIPPED ERA5 {f.name}: {type(e).__name__}: {e}", flush=True)
    return xr.concat(parts, "year")


def merra2_files():
    """{(year, month): path}, keeping the highest version where a month appears twice."""
    found = {}
    for p in merra2_dir.glob("MERRA2_*.instM_3d_asm_Np.*.nc4"):
        m = re.match(r"MERRA2_(\d{3})\.instM_3d_asm_Np\.(\d{4})(\d{2})\.nc4$", p.name)
        if m and (key := (int(m[2]), int(m[3]))) and (key not in found or int(m[1]) > found[key][0]):
            found[key] = (int(m[1]), p)
    return {k: v[1] for k, v in found.items()}


def read_merra2_month(path):
    with xr.open_dataset(path) as ds:
        da = ds["EPV"].isel(time=0, drop=True).rename(lev="level")
        return da.sel(level=levels, lat=slice(lat_min, lat_max),
                      lon=slice(lon_min, lon_max)).load()


def load_merra2():
    files, parts = merra2_files(), []
    for y in years:
        months_here = {}
        for mo in months:
            p = files.get((y, mo))
            if p is None:
                continue
            try:
                months_here[mo] = read_merra2_month(p)
            except Exception as e:
                print(f"  SKIPPED MERRA-2 {p.name}: {type(e).__name__}: {e}", flush=True)
        if months_here:
            da = xr.concat(list(months_here.values()), "month").assign_coords(
                month=list(months_here))
            parts.append(tidy(da.reindex(month=months)).expand_dims(year=[y]))
    if not parts:
        raise RuntimeError(f"No MERRA-2 EPV files found in {merra2_dir}")
    return xr.concat(parts, "year")


def load():
    print("Loading ERA5 ...", flush=True)
    era5 = load_era5()
    print("Loading MERRA-2 ...", flush=True)
    merra2 = load_merra2()
    common = sorted(set(era5.year.values.tolist()) & set(merra2.year.values.tolist()))
    data = {"ERA5": era5.sel(year=common), "MERRA-2": merra2.sel(year=common)}
    for n, d in data.items():
        missing = int(d.isnull().all(("level", "lat", "lon")).sum())
        print(f"  {n:8s} {common[0]}-{common[-1]} ({len(common)} years), {d.sizes['lat']} x "
              f"{d.sizes['lon']} grid, {missing} missing month(s)", flush=True)
    return data


# ---------------------------------------------------------------- statistics (vectorised)
def ols_stats(da):
    """OLS trend per decade with AR(1)-corrected p, along 'year', for every other index."""
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
    se = np.sqrt(ssr / (n_eff - 2) / sxx)
    p = xr.apply_ufunc(lambda t, df: 2 * stats.t.sf(np.abs(t), df), slope / se, n_eff - 2)
    return xr.Dataset({"trend": 10 * slope, "p": p}).where((n >= 3) & (n_eff > 2))


def fdr(p):
    """Benjamini-Hochberg over one map: True where the adjusted p <= alpha_fdr."""
    v, sig = p.values, np.zeros(p.shape, bool)
    ok = np.isfinite(v)
    if ok.any():
        sig[ok] = stats.false_discovery_control(v[ok]) <= alpha_fdr
    return sig


def symmetric_norm(fields, cmap):
    """One diverging colour scale for a set of maps: +-98th percentile of |value|."""
    vals = np.concatenate([np.abs(np.asarray(f)).ravel() for f in fields])
    lim = float(np.nanpercentile(vals, 98)) or 1e-3
    bounds = MaxNLocator(nbins=14, symmetric=True).tick_values(-lim, lim)
    return BoundaryNorm(bounds, plt.get_cmap(cmap).N, extend="both")


# ---------------------------------------------------------------- plotting
def draw_map(ax, field, norm, cmap, col, last_row):
    mesh = ax.pcolormesh(field.lon, field.lat, field, transform=pc, cmap=cmap, norm=norm,
                         shading="nearest")
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.5, zorder=5)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.25, zorder=5)
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=pc)
    gl = ax.gridlines(draw_labels=True, linewidth=0.2, alpha=0.4, linestyle="--",
                      xlocs=[-20, 0, 20, 40], ylocs=[-30, -15, 0, 15])
    gl.top_labels = gl.right_labels = False
    gl.left_labels, gl.bottom_labels = col == 0, last_row
    gl.xlabel_style = gl.ylabel_style = {"fontsize": 7}
    return mesh


def dots(ax, s):
    sig = fdr(s.p)
    step = max(1, int(np.ceil(np.sqrt(sig.size / 500))))          # thin dots on fine grids
    lon2, lat2 = np.meshgrid(s.lon, s.lat)
    keep = np.zeros_like(sig)
    keep[::step, ::step] = True
    keep &= sig
    ax.scatter(lon2[keep], lat2[keep], s=0.6, color="black", linewidths=0, transform=pc,
               zorder=4)


def plot_level(lev, st, clim, trend_norm, clim_norm, yrs):
    # explicit layout (no bbox_inches="tight", which can crop wide cartopy figures)
    fig, axes = plt.subplots(4, 12, figsize=(32, 8.8), subplot_kw={"projection": pc})
    fig.subplots_adjust(left=0.05, right=0.93, top=0.86, bottom=0.04, wspace=0.06, hspace=0.22)
    for j, mm in enumerate(months):
        for i, n in enumerate(names):
            s = st[n].sel(level=lev, month=mm)
            t_mesh = draw_map(axes[i, j], s.trend, trend_norm, "RdBu_r", j, False)
            if show_significance:
                dots(axes[i, j], s)
            c = clim[n].sel(level=lev, month=mm)
            ax = axes[2 + i, j]
            c_mesh = draw_map(ax, c, clim_norm, "PuOr_r", j, i == 1)
            if np.nanmax(np.abs(c)) >= 2:                          # dynamical tropopause
                ax.contour(c.lon, c.lat, c, levels=[-2, 2], colors="black", linewidths=0.8,
                           linestyles=["--", "-"], transform=pc, zorder=6)
    fig.canvas.draw()                       # cartopy fixes each map's final box at draw time
    for j, mm in enumerate(months):         # month labels above the trend and climatology blocks
        for i in (0, 2):
            b = axes[i, j].get_position()
            fig.text((b.x0 + b.x1) / 2, b.y1 + 0.008, calendar.month_name[mm],
                     ha="center", va="bottom", fontsize=12, fontweight="bold")
    for i in range(4):                      # row labels
        b = axes[i, 0].get_position()
        kind = "Trend" if i < 2 else "Climatology"
        fig.text(0.008, (b.y0 + b.y1) / 2, f"{names[i % 2]}\n{kind}", rotation=90, ha="left",
                 va="center", fontsize=13, fontweight="bold")
    for rows, mesh, unit in (((0, 1), t_mesh, "PV trend (PVU decade$^{-1}$)"),
                             ((2, 3), c_mesh, "PV (PVU)")):
        top, bottom = axes[rows[0], -1].get_position(), axes[rows[1], -1].get_position()
        cax = fig.add_axes([top.x1 + 0.008, bottom.y0, 0.007, top.y1 - bottom.y0])
        cb = fig.colorbar(mesh, cax=cax, orientation="vertical")
        cb.set_label(unit, fontsize=11)
        cb.ax.tick_params(labelsize=9)
    sig_text = (f"; dots: FDR-significant (α$_{{FDR}}$ = {alpha_fdr:.2f}, AR(1)-corrected p)"
                if show_significance else "")
    fig.suptitle(f"ERA5 and MERRA-2 {lev} hPa potential vorticity over sub-Saharan Africa: "
                 f"monthly trend {yrs}{sig_text}\nand monthly climatology "
                 f"{clim_years[0]}-{clim_years[-1]} (contours ±2 PVU; PV < 0 south of the "
                 "equator)", fontsize=15, fontweight="bold", y=0.985)
    out = out_dir / f"pv_era5_merra2_monthly_evolution_{lev}hPa.png"
    fig.savefig(out, dpi=250, facecolor="white")
    plt.close(fig)
    print(f"Saved {out}", flush=True)


def main():
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load()
    yrs = f"{int(data['ERA5'].year.min())}-{int(data['ERA5'].year.max())}"
    st = {n: ols_stats(d) for n, d in data.items()}         # every month, level and point
    clim = {n: d.sel(year=d.year.isin(list(clim_years))).mean("year") for n, d in data.items()}
    print("\nSign check: mean climatological PV south of 10 S (PVU), should agree in sign")
    for lev in levels:
        vals = {n: float(c.sel(level=lev).where(c.lat < -10).mean()) for n, c in clim.items()}
        print(f"  {lev} hPa: " + ", ".join(f"{n} {v:+.2f}" for n, v in vals.items()))
        if np.sign(vals["ERA5"]) != np.sign(vals["MERRA-2"]):
            print("  WARNING: the two datasets disagree in sign; check the PV convention.")
    for lev in levels:
        scope = levels if same_scale_all_levels else [lev]
        trend_norm = symmetric_norm([st[n].trend.sel(level=scope) for n in names], "RdBu_r")
        clim_norm = symmetric_norm([clim[n].sel(level=scope) for n in names], "PuOr_r")
        plot_level(lev, st, clim, trend_norm, clim_norm, yrs)


if __name__ == "__main__":
    main()