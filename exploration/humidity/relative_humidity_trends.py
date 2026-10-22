
from __future__ import annotations
import calendar
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
src_dir = Path("/soge-home/users/jesu4127/scratch/era5/global/relative_humidity")
out_dir = Path("/soge-home/users/jesu4127/Paper_1/exploration/humidity/plots")
out_dir.mkdir(parents=True, exist_ok=True)

levels = [200, 300, 500, 700, 850]                    # hPa, one figure each
years = range(1980, 2026)                   # trend period
clim_years = range(1991, 2021)              # climatology period
lon_min, lon_max, lat_min, lat_max = -20.0, 55.0, -35.0, 15.0     # sub-Saharan Africa
alpha_fdr = 0.10
show_significance = True
same_scale_all_levels = False
pc = ccrs.PlateCarree()
months = np.arange(1, 13)


# ---------------------------------------------------------------- loading
def read_year(path):
    """All months of the chosen levels and domain from one yearly file (PVU)."""
    with xr.open_dataset(path) as ds:
        da = ds["r"].rename(valid_time="time", latitude="lat", longitude="lon")
        da = da.sel(pressure_level=levels, lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
        year = int(da.time.dt.year[0])
        da = (da * 1e6).astype("float32").load()                   # reads only this subset
    da = da.drop_vars(["number", "expver"], errors="ignore")
    da = da.assign_coords(time=da.time.dt.month.values).rename(time="month")
    return da.reindex(month=months).expand_dims(year=[year])      # missing months -> NaN


def load():
    files = sorted(src_dir.glob("global_relative_humidity_*.nc"))
    files = [f for f in files if int(f.stem.split("_")[-1]) in years]
    parts = []
    for f in files:
        try:
            parts.append(read_year(f))
        except Exception as e:
            print(f"  SKIPPED {f.name}: {type(e).__name__}: {e}")
    pv = xr.concat(parts, "year").sortby("lat")
    print(f"Loaded {pv.sizes['year']} years ({int(pv.year.min())}-{int(pv.year.max())}), "
          f"levels {levels} hPa, {pv.sizes['lat']} x {pv.sizes['lon']} grid, 12 months")
    return pv


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
    lim = float(np.nanpercentile(np.abs(np.asarray(fields)), 98)) or 1e-3
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


def plot_level(lev, st, clim, trend_norm, clim_norm):
    # explicit layout (no bbox_inches="tight", which can crop wide cartopy figures)
    fig, axes = plt.subplots(2, 12, figsize=(32, 6.2), subplot_kw={"projection": pc})
    fig.subplots_adjust(left=0.045, right=0.93, top=0.80, bottom=0.07, wspace=0.06, hspace=0.18)
    for j, mm in enumerate(months):
        s = st.sel(pressure_level=lev, month=mm)
        ax = axes[0, j]
        t_mesh = draw_map(ax, s.trend, trend_norm, "RdBu_r", j, False)
        if show_significance:
            sig = fdr(s.p)
            step = max(1, int(np.ceil(np.sqrt(sig.size / 500))))      # thin dots
            lon2, lat2 = np.meshgrid(s.lon, s.lat)
            keep = np.zeros_like(sig); keep[::step, ::step] = True; keep &= sig
            ax.scatter(lon2[keep], lat2[keep], s=0.6, color="black", linewidths=0,
                       transform=pc, zorder=4)
        c = clim.sel(pressure_level=lev, month=mm)
        ax = axes[1, j]
        c_mesh = draw_map(ax, c, clim_norm, "PuOr_r", j, True)
        if np.nanmax(np.abs(c)) >= 2:                              # dynamical tropopause
            ax.contour(c.lon, c.lat, c, levels=[-2, 2], colors="black", linewidths=0.8,
                       linestyles=["--", "-"], transform=pc, zorder=6)
    yrs = st.attrs["years"]
    fig.canvas.draw()                       # cartopy fixes each map's final box at draw time
    for j, mm in enumerate(months):         # month labels above every column, both rows
        for i in range(2):
            b = axes[i, j].get_position()
            fig.text((b.x0 + b.x1) / 2, b.y1 + 0.012, calendar.month_name[mm],
                     ha="center", va="bottom", fontsize=12, fontweight="bold")
    for i, (label, mesh, unit) in enumerate((("Trend", t_mesh, "RH decade$^{-1}$"),
                                             ("Climatology", c_mesh, "RH "))):
        first, last = axes[i, 0].get_position(), axes[i, -1].get_position()
        fig.text(0.008, (first.y0 + first.y1) / 2, label, rotation=90, ha="left",
                 va="center", fontsize=13, fontweight="bold")
        cax = fig.add_axes([last.x1 + 0.008, last.y0, 0.007, last.height])
        cb = fig.colorbar(mesh, cax=cax, orientation="vertical")
        cb.set_label(unit, fontsize=10)
        cb.ax.tick_params(labelsize=8)
    dots = (f"; dots: FDR-significant (α$_{{FDR}}$ = {alpha_fdr:.2f}, AR(1)-corrected p)"
            if show_significance else "")
    fig.suptitle(f"ERA5 {lev} hPa relative humidity over sub-Saharan Africa: monthly trend "
                 f"{yrs}{dots}\nand monthly climatology {clim_years[0]}-{clim_years[-1]} ",
                 fontweight="bold", y=0.985)
    out = out_dir / f"era5_rh_monthly_evolution_{lev}hPa.png"
    fig.savefig(out, dpi=250, facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    pv = load()
    st = ols_stats(pv)                           # every month, level and point at once
    st.attrs["years"] = f"{int(pv.year.min())}-{int(pv.year.max())}"
    clim = pv.sel(year=pv.year.isin(list(clim_years))).mean("year")
    for lev in levels:
        scope = levels if same_scale_all_levels else [lev]
        trend_norm = symmetric_norm(st.trend.sel(pressure_level=scope), "RdBu_r")
        clim_norm = symmetric_norm(clim.sel(pressure_level=scope), "PuOr_r")
        plot_level(lev, st, clim, trend_norm, clim_norm)


if __name__ == "__main__":
    main()