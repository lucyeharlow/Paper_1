"""
ERA5 October potential vorticity (PV): meridional gradient over southern Africa,
5-35 S, in 5-degree longitude bands from 0 to 40 E. One figure per pressure level.

DATA
  scratch/era5/global/potential_vorticity/global_potential_vorticity_YYYY.nc
  monthly means, pv(valid_time, pressure_level, latitude, longitude), K m2 kg-1 s-1.
  Shown in PVU (1 PVU = 1e-6 K m2 kg-1 s-1).

METHOD (all vectorised over years, levels, bands and latitudes)
  October PV is read for the chosen levels over 37.5-2.5 S, 0-40 E (a 2.5-degree margin so
  the gradient at 5 S and 35 S uses centred differences), then averaged in longitude within
  each band [w, w + 5).
  Profile:   dPV/dy at every latitude, y = northward distance (R = 6371 km), centred
             differences, PVU per 1000 km.
  Bulk:      one number per year and band: the least-squares slope of band-mean PV against
             y over 5-35 S (uses every latitude, so it is less noisy than an end-point
             difference), PVU per 1000 km.
  Trends:    OLS per decade; lag-1 autocorrelation r1 of the residuals gives
             n_eff = n (1 - r1) / (1 + r1), capped at n (Santer et al. 2000); two-sided t-test
             with n_eff - 2 degrees of freedom. Profile trends: filled markers are significant
             after Benjamini-Hochberg FDR (alpha_FDR = 0.10) across the latitudes of a band.
  Climatology: October mean over clim_years.

SIGN
  South of the equator PV is negative and grows in magnitude towards the pole, so dPV/dy
  (northward) is normally POSITIVE: PV becomes more negative going south. A positive trend
  in the gradient means a sharper meridional PV contrast (e.g. a stronger PV gradient /
  waveguide along the subtropical jet); a negative trend means a weaker one.

FIGURES  era5_pv_meridional_gradient_Oct_<level>hPa.png  (+ CSV of the bulk-gradient series)
  Left: map of the bands (their colours frame the matching subfigures) over October
  climatological PV contours. Right: one subfigure per band:
  (left panel) climatological dPV/dy profile (black, bottom axis) and its trend (band colour,
  top axis); (right panel) bulk gradient time series with its OLS trend.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ---------------------------------------------------------------- settings
src_dir = Path("/soge-home/users/jesu4127/scratch/era5/global/potential_vorticity")
out_dir = Path("/soge-home/users/jesu4127/Paper_1/pv/plots")
out_dir.mkdir(parents=True, exist_ok=True)

levels = [200, 300, 500]                    # hPa, one figure each
month = 10
years = range(1980, 2026)
clim_years = range(1991, 2021)
lat_s, lat_n = -35.0, -5.0                  # gradient range
band_edges = np.arange(0, 45, 5)            # 0, 5, ..., 40 E -> 8 bands
margin = 2.5                                # extra latitude read for centred differences
alpha_fdr = 0.10
earth_radius_1000km = 6.371
colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
           "#9467bd", "#8c564b", "#e377c2", "#17becf"]
pc = ccrs.PlateCarree()
bands = [f"{w}-{w + 5}°E" for w in band_edges[:-1]]


# ---------------------------------------------------------------- loading
def read_year(path):
    """October, chosen levels, 0-40 E and 37.5-2.5 S from one yearly file (PVU)."""
    with xr.open_dataset(path) as ds:
        da = ds["pv"].rename(valid_time="time", latitude="lat", longitude="lon")
        da = da.sel(time=da.time.dt.month == month, pressure_level=levels,
                    lat=slice(lat_n + margin, lat_s - margin),
                    lon=slice(band_edges[0], band_edges[-1]))
        if da.time.size != 1:
            raise ValueError("October missing")
        year = int(da.time.dt.year[0])
        da = (da.isel(time=0, drop=True) * 1e6).load()
    return da.drop_vars(["number", "expver"], errors="ignore").expand_dims(year=[year])


def load():
    files = [f for f in sorted(src_dir.glob("global_potential_vorticity_*.nc"))
             if int(f.stem.split("_")[-1]) in years]
    parts = []
    for f in files:
        try:
            parts.append(read_year(f))
        except Exception as e:
            print(f"  SKIPPED {f.name}: {type(e).__name__}: {e}")
    pv = xr.concat(parts, "year").sortby("lat")
    print(f"Loaded October {int(pv.year.min())}-{int(pv.year.max())} ({pv.sizes['year']} years), "
          f"levels {levels} hPa, {pv.sizes['lat']} x {pv.sizes['lon']} grid")
    return pv


# ---------------------------------------------------------------- gradients (vectorised)
def band_means(pv):
    """Longitude mean within each 5-degree band [w, w + 5): dims (..., band, lat)."""
    b = pv.groupby_bins("lon", band_edges, right=False).mean("lon")
    return b.rename(lon_bins="band").assign_coords(band=bands)


def gradients(b):
    """dPV/dy profile (PVU per 1000 km) and the bulk least-squares gradient over 5-35 S."""
    y = earth_radius_1000km * np.deg2rad(b.lat)                    # northward distance
    profile = b.differentiate("lat") / (earth_radius_1000km * np.pi / 180)
    profile = profile.sel(lat=slice(lat_s, lat_n))
    inner = b.sel(lat=slice(lat_s, lat_n))
    yi = y.sel(lat=slice(lat_s, lat_n))
    ya = yi - yi.mean()
    bulk = (ya * (inner - inner.mean("lat"))).sum("lat") / (ya ** 2).sum()
    return profile, bulk


# ---------------------------------------------------------------- statistics (vectorised)
def ols_stats(da):
    """OLS trend per decade with AR(1)-corrected p, along 'year', for every other index."""
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
    out = xr.Dataset({"trend": 10 * slope, "slope": slope,
                      "intercept": da.mean("year") - slope * x.mean("year"),
                      "n": n, "n_eff": n_eff, "p": p})
    return out.where((n >= 3) & (n_eff > 2))


def fdr(p):
    v, sig = np.asarray(p, float), np.zeros(np.shape(p), bool)
    ok = np.isfinite(v)
    if ok.any():
        sig[ok] = stats.false_discovery_control(v[ok]) <= alpha_fdr
    return sig


# ---------------------------------------------------------------- plotting
def band_map(fig, clim_lev):
    ax = fig.add_subplot(projection=pc)
    cs = ax.contour(clim_lev.lon, clim_lev.lat, clim_lev, levels=12, colors="0.45",
                    linewidths=0.6, transform=pc, zorder=2)
    ax.clabel(cs, fontsize=6, fmt="%.1f")
    for (w, name), c in zip(zip(band_edges[:-1], bands), colours):
        ax.add_patch(Rectangle((w, lat_s), 5, lat_n - lat_s, facecolor=c, alpha=0.35,
                               edgecolor=c, linewidth=2, transform=pc, zorder=3))
        ax.text(w + 2.5, lat_n + 0.8, name.replace("°E", ""), color=c, fontsize=7,
                fontweight="bold", ha="center", va="bottom", rotation=90, transform=pc)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.8, zorder=4)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.4, zorder=4)
    ax.set_extent([-5, 45, -38, 0], crs=pc)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4, linestyle="--")
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"fontsize": 8}
    ax.set_title("5° longitude bands, 5-35°S\n(grey: October climatological PV, PVU)",
                 fontsize=10, fontweight="bold")


def band_panel(sf, name, c, prof_clim, prof_st, series, bulk_st, xlims):
    sf.set_facecolor(matplotlib.colors.to_rgba(c, 0.06))
    sf.set_edgecolor(c)
    sf.set_linewidth(3)
    a1, a2 = sf.subplots(1, 2, gridspec_kw={"width_ratios": [1, 1.6], "wspace": 0.38})
    sf.subplots_adjust(left=0.11, right=0.97, bottom=0.2, top=0.74)
    # profile: climatology (bottom axis) and its trend (top axis)
    a1.plot(prof_clim, prof_clim.lat, color="black", linewidth=1.5)
    a1.axvline(0, color="black", linewidth=0.5)
    a1.set(ylim=(lat_s, lat_n), xlim=xlims[0])
    a1.set_xlabel("clim. dPV/dy\n(PVU / 1000 km)", fontsize=7)
    a1.set_ylabel("Latitude", fontsize=7)
    a1.tick_params(labelsize=6)
    top = a1.twiny()
    sig = fdr(prof_st.p)
    top.plot(prof_st.trend, prof_st.lat, color=c, linewidth=1.3)
    top.plot(prof_st.trend.values[sig], prof_st.lat.values[sig], "o", ms=2.5, color=c)
    top.axvline(0, color=c, linewidth=0.5, linestyle=":")
    top.set_xlim(xlims[1])
    top.set_xlabel("trend (PVU / 1000 km / decade)", fontsize=6.5, color=c)
    top.tick_params(labelsize=6, colors=c)
    # bulk-gradient series and trend
    p = float(bulk_st.p)
    a2.plot(series.year, series, "o-", ms=2.5, linewidth=1, color=c)
    a2.plot(series.year, float(bulk_st.intercept) + float(bulk_st.slope) * series.year, "--",
            color="black", linewidth=1)
    a2.set_ylim(xlims[2])
    a2.set_ylabel("bulk dPV/dy, 5-35°S\n(PVU / 1000 km)", fontsize=7)
    a2.tick_params(labelsize=6)
    a2.grid(alpha=0.25)
    a2.text(0.02, 0.97, f"trend {float(bulk_st.trend):+.4f} PVU/1000 km/dec\n"
            f"{'p < 0.001' if p < 0.001 else f'p = {p:.3f}'}, "
            f"n$_{{eff}}$ = {float(bulk_st.n_eff):.0f}/{int(bulk_st.n)}",
            transform=a2.transAxes, fontsize=7, va="top")
    sf.suptitle(name, fontsize=11, fontweight="bold", color=c, y=0.985)


def padded(values, pad=0.08, symmetric=False):
    v = np.asarray(values, float)
    lo, hi = np.nanmin(v), np.nanmax(v)
    if symmetric:
        hi = max(abs(lo), abs(hi)); lo = -hi
    d = (hi - lo) or abs(hi) or 1e-3
    return lo - pad * d, hi + pad * d


def plot_level(lev, clim_map, prof_clim, prof_st, bulk, bulk_st):
    fig = plt.figure(figsize=(26, 14.5))
    head, body = fig.subfigures(2, 1, height_ratios=[0.07, 1])      # title row above everything
    left, right = body.subfigures(1, 2, width_ratios=[1.25, 3.2], wspace=0.01)
    band_map(left, clim_map.sel(pressure_level=lev))
    grid = right.subfigures(4, 2, hspace=0.04, wspace=0.03).T.ravel()   # bands down columns
    pc_l, ps_l = prof_clim.sel(pressure_level=lev), prof_st.sel(pressure_level=lev)
    bk_l, bs_l = bulk.sel(pressure_level=lev), bulk_st.sel(pressure_level=lev)
    xlims = (padded(pc_l), padded(ps_l.trend, symmetric=True), padded(bk_l))   # shared scales
    for sf, name, c in zip(grid, bands, colours):
        band_panel(sf, name, c, pc_l.sel(band=name), ps_l.sel(band=name),
                   bk_l.sel(band=name), bs_l.sel(band=name), xlims)
    yrs = f"{int(bulk.year.min())}-{int(bulk.year.max())}"
    head.suptitle(f"ERA5 October {lev} hPa PV meridional gradient over southern Africa, 5-35°S, "
                  f"{yrs}\nclimatology {clim_years[0]}-{clim_years[-1]}; dPV/dy > 0: PV more "
                  "negative to the south; filled markers: FDR-significant trend "
                  f"(α$_{{FDR}}$ = {alpha_fdr:.2f}, AR(1)-corrected p)",
                  fontsize=14, fontweight="bold", y=0.75)
    out = out_dir / f"era5_pv_meridional_gradient_Oct_{lev}hPa.png"
    fig.savefig(out, dpi=250, facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    pv = load()
    profile, bulk = gradients(band_means(pv))          # (year, level, band, lat) / (year, level, band)
    prof_st, bulk_st = ols_stats(profile), ols_stats(bulk)
    in_clim = pv.year.isin(list(clim_years))
    prof_clim = profile.sel(year=in_clim).mean("year")
    clim_map = pv.sel(year=in_clim).mean("year")
    for lev in levels:
        print(f"\n{lev} hPa bulk dPV/dy 5-35 S (PVU per 1000 km)")
        for name in bands:
            s = bulk_st.sel(pressure_level=lev, band=name)
            print(f"  {name:<9} mean {float(bulk.sel(pressure_level=lev, band=name).mean()):+.3f}"
                  f"   trend {float(s.trend):+.4f}/decade   p = {float(s.p):.3f}   "
                  f"n_eff = {float(s.n_eff):.1f} of {int(s.n)}")
        plot_level(lev, clim_map, prof_clim, prof_st, bulk, bulk_st)
    table = bulk.to_dataframe("bulk_dPVdy_PVU_per_1000km").unstack(["pressure_level", "band"])
    table.to_csv(out_dir / "era5_pv_meridional_gradient_Oct_bulk_series.csv")


if __name__ == "__main__":
    main()