"""
ERA5 potential vorticity (PV) trends over sub-Saharan Africa.

DATA
  scratch/era5/global/potential_vorticity/global_potential_vorticity_YYYY.nc
  monthly means, pv(valid_time, pressure_level, latitude, longitude), K m2 kg-1 s-1.
  Shown in PVU (1 PVU = 1e-6 K m2 kg-1 s-1).

METHOD
  From each yearly file only the chosen months, pressure levels and domain are read, so the
  1.8 GB global files are never loaded whole. Unreadable files are skipped and reported.
  The chosen months are averaged per year (October by default), then for every grid point
  and level at once (vectorised):
    OLS trend (PVU/decade); lag-1 autocorrelation r1 of the residuals gives
    n_eff = n (1 - r1) / (1 + r1), capped at n (Santer et al. 2000); two-sided t-test with
    n_eff - 2 degrees of freedom. Map dots: significant after Benjamini-Hochberg FDR at
    alpha_FDR = 0.10 (Wilks 2016), applied separately to each level's map.
  Time series: cos-latitude area-weighted mean over each box in `boxes`, shown as anomalies
  from the box's own mean (given in the legend) with its trend line.

SIGN OF PV
  PV takes the sign of the Coriolis parameter: positive north of the equator, negative south.
  In the south a NEGATIVE trend means PV of larger magnitude (e.g. more stratospheric air
  / stronger cyclonic anomalies aloft). The time series are therefore split at the equator;
  a mean across it would add positive and negative PV and largely cancel.

OUTPUT
  era5_pv_trends_<months>.png   one row per level: trend map | box-mean time series
  era5_pv_box_series_<months>.csv
"""
from __future__ import annotations
import calendar
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
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

levels = [200, 300, 500]                    # hPa
months = [10]                               # [10] October; [9, 10, 11] SON; range(1, 13) annual
years = range(1980, 2026)                   # analysis period (files outside it are ignored)
lon_min, lon_max, lat_min, lat_max = -20.0, 55.0, -90.0, 0.0     # sub-Saharan Africa
#boxes = {"SSA south of the equator": (-20.0, 55.0, -35.0, 0.0),   # (W, E, S, N)
 #        "SSA north of the equator": (-20.0, 55.0, 0.0, 15.0)}
boxes = {"SSA south of the equator": (-20.0, 55.0, -35.0, 0.0),}  # (W, E, S, N)
box_colours = ["#2F5597", "#C55A11"]
alpha_fdr = 0.10
pc = ccrs.PlateCarree()
tag = "".join(calendar.month_abbr[mm][0] for mm in months) if len(months) > 1 \
      else calendar.month_abbr[months[0]]


# ---------------------------------------------------------------- loading
def read_year(path):
    """Selected months, levels and domain of one yearly file -> one mean field (PVU)."""
    with xr.open_dataset(path) as ds:
        da = ds["pv"].rename(valid_time="time", latitude="lat", longitude="lon")
        da = da.sel(time=da.time.dt.month.isin(months), pressure_level=levels,
                    lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
        if da.time.size != len(months):
            raise ValueError(f"only {da.time.size} of {len(months)} months present")
        year = int(da.time.dt.year[0])
        out = (da.mean("time") * 1e6).load()               # reads only this subset
    return out.drop_vars(["number", "expver"], errors="ignore").expand_dims(year=[year])


def load():
    files = sorted(src_dir.glob("global_potential_vorticity_*.nc"))
    files = [f for f in files if int(f.stem.split("_")[-1]) in years]
    parts = []
    for f in files:
        try:
            parts.append(read_year(f))
        except Exception as e:
            print(f"  SKIPPED {f.name}: {type(e).__name__}: {e}")
    pv = xr.concat(parts, "year").sortby("lat")
    print(f"Loaded {pv.sizes['year']} years ({int(pv.year.min())}-{int(pv.year.max())}), "
          f"levels {levels} hPa, {pv.sizes['lat']} x {pv.sizes['lon']} grid, months {months}")
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
    out = xr.Dataset({"trend": 10 * slope, "intercept": da.mean("year") - slope * x.mean("year"),
                      "slope": slope, "n": n, "r1": r1, "n_eff": n_eff, "p": p})
    return out.where((n >= 3) & (n_eff > 2))


def fdr(p):
    """Benjamini-Hochberg over one map: True where the adjusted p <= alpha_fdr."""
    v, sig = p.values, np.zeros(p.shape, bool)
    ok = np.isfinite(v)
    if ok.any():
        sig[ok] = stats.false_discovery_control(v[ok]) <= alpha_fdr
    return p.copy(data=sig)


def box_mean(da, box):
    w, e, s, n = box
    sub = da.sel(lon=slice(w, e), lat=slice(s, n))
    return sub.weighted(np.cos(np.deg2rad(sub.lat))).mean(("lat", "lon"))


# ---------------------------------------------------------------- plotting
def plot(pv, st, series, series_st):
    fig = plt.figure(figsize=(16, 4.6 * len(levels)))
    gs = fig.add_gridspec(len(levels), 2, width_ratios=[1.0, 1.25], wspace=0.12, hspace=0.35)
    for i, lev in enumerate(levels):
        s = st.sel(pressure_level=lev)
        lim = float(np.nanpercentile(np.abs(s.trend), 98)) or 1e-3
        bounds = MaxNLocator(nbins=12, symmetric=True).tick_values(-lim, lim)
        norm = BoundaryNorm(bounds, plt.get_cmap("RdBu_r").N, extend="both")

        ax = fig.add_subplot(gs[i, 0], projection=pc)
        mesh = ax.pcolormesh(s.lon, s.lat, s.trend, transform=pc, cmap="RdBu_r", norm=norm,
                             shading="nearest")
        sig = fdr(s.p).values
        step = max(1, int(np.ceil(np.sqrt(sig.size / 900))))           # thin dots
        lon2, lat2 = np.meshgrid(s.lon, s.lat)
        keep = np.zeros_like(sig); keep[::step, ::step] = True; keep &= sig
        ax.scatter(lon2[keep], lat2[keep], s=1.2, color="black", linewidths=0, transform=pc,
                   zorder=4)
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.7, zorder=5)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.4, zorder=5)
        for (name, (w, e, so, no)), c in zip(boxes.items(), box_colours):
            ax.add_patch(Rectangle((w, so), e - w, no - so, fill=False, edgecolor=c,
                                   linewidth=1.4, transform=pc, zorder=6))
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=pc)
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4, linestyle="--")
        gl.top_labels = gl.right_labels = False
        gl.xlabel_style = gl.ylabel_style = {"fontsize": 8}
        ax.set_title(f"({'abcdefgh'[2 * i]}) {lev} hPa PV trend, {tag}", fontsize=11,
                     fontweight="bold", loc="left")
        ax.text(0.01, 0.02, f"{100 * sig.mean():.0f}% of grid points FDR-significant",
                transform=ax.transAxes, fontsize=7.5, zorder=7,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5))
        cb = fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.08, fraction=0.05,
                          aspect=40, shrink=0.8)
        cb.set_label("PVU decade$^{-1}$", fontsize=9)
        cb.ax.tick_params(labelsize=7)

        ax = fig.add_subplot(gs[i, 1])
        for (name, ser), c in zip(series.items(), box_colours):
            y = ser.sel(pressure_level=lev)
            r = series_st[name].sel(pressure_level=lev)
            pval = float(r.p)
            mean = float(y.mean())                      # anomalies, so both boxes share an axis
            ax.plot(y.year, y - mean, "o-", ms=3, linewidth=1.2, color=c,
                    label=f"{name} (mean {mean:+.2f} PVU): {float(r.trend):+.3g} PVU/dec, "
                          f"{'p < 0.001' if pval < 0.001 else f'p = {pval:.3f}'}, "
                          f"n$_{{eff}}$ = {float(r.n_eff):.0f}/{int(r.n)}")
            ax.plot(y.year, float(r.intercept) + float(r.slope) * y.year - mean, "--", color=c,
                    linewidth=1)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_ylabel(f"{lev} hPa PV anomaly, {tag} (PVU)", fontsize=9)
        ax.set_title(f"({'abcdefgh'[2 * i + 1]}) Box-mean {lev} hPa PV anomaly (area-weighted)",
                     fontsize=11, fontweight="bold", loc="left")
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    yrs = pv.year.values
    fig.suptitle(f"ERA5 potential vorticity trends over sub-Saharan Africa, {tag}, "
                 f"{yrs.min()}-{yrs.max()}\n(OLS; dots: significant after FDR, α$_{{FDR}}$ = "
                 f"{alpha_fdr:.2f}, AR(1)-corrected p; PV < 0 south of the equator)",
                 fontsize=13, fontweight="bold", y=0.93)
    out = out_dir / f"era5_pv_trends_{tag}.png"
    fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    pv = load()
    st = ols_stats(pv)                                       # all levels, all points at once
    series = {name: box_mean(pv, b) for name, b in boxes.items()}
    series_st = {name: ols_stats(s) for name, s in series.items()}
    for lev in levels:
        print(f"\n{lev} hPa")
        for name in boxes:
            r = series_st[name].sel(pressure_level=lev)
            print(f"  {name:<26}{float(r.trend):+.4f} PVU/decade  p = {float(r.p):.3f}  "
                  f"r1 = {float(r.r1):+.2f}  n_eff = {float(r.n_eff):.1f} of {int(r.n)}")
    pd.DataFrame({f"{name} {lev} hPa (PVU)": s.sel(pressure_level=lev).to_series()
                  for name, s in series.items() for lev in levels}).to_csv(
        out_dir / f"era5_pv_box_series_{tag}.csv")
    plot(pv, st, series, series_st)


if __name__ == "__main__":
    main()