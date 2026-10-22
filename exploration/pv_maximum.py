"""
ERA5 October potential vorticity (PV): strength and position of the meridional PV-gradient
maximum over southern Africa, in 5-degree longitude bands from 0 to 40 E. One figure per level.

For each October t, band and level:
    G_max(t)   = max over y of dPV/dy          (strength,  PVU per 1000 km)
    phi_max(t) = argmax over y of dPV/dy       (position,  degrees latitude)
searched over 5-35 S.

DATA
  scratch/era5/global/potential_vorticity/global_potential_vorticity_YYYY.nc
  monthly means, pv(valid_time, pressure_level, latitude, longitude), K m2 kg-1 s-1 -> PVU.

METHOD (vectorised over years, levels, bands and latitudes)
  1. October PV, chosen levels, 0-40 E, 5-35 S plus a margin; longitude mean in each band.
  2. Optional meridional running mean over smooth_deg degrees (default 1 degree), so a single
     noisy grid row cannot set the maximum.
  3. dPV/dy by centred differences, y = northward distance (R = 6371 km), PVU per 1000 km.
  4. The grid maximum over 5-35 S, refined by fitting a parabola through it and its two
     neighbours: sub-grid latitude phi_max and peak value G_max.
  5. Maxima on the 5 S or 35 S boundary are not true peaks (the real one lies outside the
     search range). They are shown as open markers and excluded from the trends.
  6. Trends: OLS per decade; lag-1 autocorrelation r1 of the residuals gives
     n_eff = n (1 - r1) / (1 + r1), capped at n (Santer et al. 2000); two-sided t-test with
     n_eff - 2 degrees of freedom. In the bar charts, hatching marks bands whose trend is
     significant after Benjamini-Hochberg FDR (alpha_FDR = 0.10) across the 8 bands.

SIGN CONVENTIONS
  South of the equator PV < 0 and grows in magnitude towards the pole, so dPV/dy > 0 and its
  maximum marks the sharpest PV contrast (e.g. the PV waveguide on the subtropical jet).
  G_max trend > 0: a sharper maximum.  phi_max trend < 0: the maximum moves POLEWARD.

OUTPUT
  era5_pv_gradient_max_Oct_<level>hPa.png
    left:  band map; bar charts of the G_max and phi_max trends for every band
    right: one subfigure per band (framed in its map colour): G_max(t) | phi_max(t), with trends
  era5_pv_gradient_max_Oct_series.csv  (G_max, phi_max and edge flag per year, level, band)
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
lat_s, lat_n = -35.0, -5.0                  # search range for the maximum
band_edges = np.arange(0, 45, 5)            # 0, 5, ..., 40 E -> 8 bands
smooth_deg = 1.0                            # meridional running mean before d/dy (0 = none)
margin = 2.5                                # extra latitude read beyond the search range
alpha_fdr = 0.10
earth_radius_1000km = 6.371
colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
           "#9467bd", "#8c564b", "#e377c2", "#17becf"]
pc = ccrs.PlateCarree()
bands = [f"{w}-{w + 5}°E" for w in band_edges[:-1]]


# ---------------------------------------------------------------- loading
def read_year(path):
    """October, chosen levels, 0-40 E, search range plus margin, from one yearly file (PVU)."""
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


# ---------------------------------------------------------------- gradient maximum (vectorised)
def gradient_profile(pv):
    """Band-mean, optionally smoothed dPV/dy (PVU per 1000 km) over the search range."""
    b = pv.groupby_bins("lon", band_edges, right=False).mean("lon")
    b = b.rename(lon_bins="band").assign_coords(band=bands)
    if smooth_deg > 0:
        dlat = float(np.median(np.diff(b.lat)))
        win = max(1, int(round(smooth_deg / dlat)) | 1)                   # odd window
        b = b.rolling(lat=win, center=True).mean()
    g = b.differentiate("lat") / (earth_radius_1000km * np.pi / 180)
    return g.sel(lat=slice(lat_s, lat_n))


def gradient_max(g):
    """G_max, phi_max (parabolic sub-grid refinement) and an at-the-edge flag."""
    lat = g.lat.values
    v = g.transpose(..., "lat").values
    i = np.nanargmax(np.where(np.isfinite(v), v, -np.inf), axis=-1)
    at_edge = (i == 0) | (i == lat.size - 1)
    ic = np.clip(i, 1, lat.size - 2)                                      # neighbours exist
    f0 = np.take_along_axis(v, ic[..., None], -1)[..., 0]
    fm = np.take_along_axis(v, (ic - 1)[..., None], -1)[..., 0]
    fp = np.take_along_axis(v, (ic + 1)[..., None], -1)[..., 0]
    curv = fm - 2 * f0 + fp
    delta = np.where(curv < 0, 0.5 * (fm - fp) / np.where(curv < 0, curv, 1), 0.0)
    delta = np.clip(delta, -0.5, 0.5)
    dlat = lat[1] - lat[0]
    phi = np.where(at_edge, lat[i], lat[ic] + delta * dlat)
    gmax = np.where(at_edge, np.take_along_axis(v, i[..., None], -1)[..., 0],
                    f0 - 0.25 * (fm - fp) * delta)
    dims = [d for d in g.dims if d != "lat"]
    coords = {d: g[d] for d in dims}
    out = xr.Dataset({"G_max": (dims, gmax), "phi_max": (dims, phi), "at_edge": (dims, at_edge)},
                     coords=coords)
    return out


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
    out = xr.Dataset({"trend": 10 * slope, "trend_se": 10 * se, "slope": slope,
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
def fmt_p(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def band_map(sf):
    ax = sf.add_subplot(projection=pc)
    for w, name, c in zip(band_edges[:-1], bands, colours):
        ax.add_patch(Rectangle((w, lat_s), 5, lat_n - lat_s, facecolor=c, alpha=0.4,
                               edgecolor=c, linewidth=2, transform=pc, zorder=3))
        ax.text(w + 2.5, lat_n + 0.8, name.replace("°E", ""), color=c, fontsize=7,
                fontweight="bold", ha="center", va="bottom", rotation=90, transform=pc)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.8, zorder=4)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.4, zorder=4)
    ax.set_extent([-5, 45, -38, 0], crs=pc)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4, linestyle="--")
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"fontsize": 8}
    ax.set_title(f"Search range for the maximum: 5-35°S\n5° bands, 0-40°E", fontsize=10,
                 fontweight="bold")


def trend_bars(sf, st_lev):
    axes = sf.subplots(2, 1)
    sf.subplots_adjust(left=0.2, right=0.95, top=0.93, bottom=0.12, hspace=0.55)
    x = np.arange(len(bands))
    for ax, var, unit, title in (
            (axes[0], "G_max", "PVU / 1000 km / decade", "Trend in strength G$_{max}$"),
            (axes[1], "phi_max", "° latitude / decade", "Trend in position φ$_{max}$ (< 0 poleward)")):
        s = st_lev[var]
        sig = fdr(s.p)
        ax.bar(x, s.trend, yerr=1.96 * s.trend_se, color=colours, edgecolor="black",
               linewidth=0.6, capsize=2, hatch=[("///" if k else "") for k in sig],
               error_kw=dict(linewidth=0.7))
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_xticks(x, [b.replace("°E", "") for b in bands], fontsize=7, rotation=45)
        ax.set_ylabel(unit, fontsize=7.5)
        ax.set_title(title, fontsize=9.5, fontweight="bold")
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", alpha=0.25)
    axes[1].set_xlabel("band (°E)   hatched: FDR-significant;  bars: ±1.96 SE (AR(1)-corrected)",
                       fontsize=7)


def band_panel(sf, name, c, res, st, ylims):
    sf.set_facecolor(matplotlib.colors.to_rgba(c, 0.06))
    sf.set_edgecolor(c)
    sf.set_linewidth(3)
    axes = sf.subplots(1, 2, gridspec_kw={"wspace": 0.3})
    sf.subplots_adjust(left=0.1, right=0.97, bottom=0.17, top=0.78)
    yr = res.year.values
    edge = res.at_edge.values
    for ax, var, ylabel, lim in ((axes[0], "G_max", "G$_{max}$ (PVU / 1000 km)", ylims[0]),
                                 (axes[1], "phi_max", "φ$_{max}$ (°)", ylims[1])):
        v = res[var].values
        s = st[var]
        ax.plot(yr, v, color=c, linewidth=0.8, alpha=0.6)
        ax.plot(yr[~edge], v[~edge], "o", ms=3, color=c)
        ax.plot(yr[edge], v[edge], "o", ms=3.5, mfc="white", mec=c, label="at search edge")
        if np.isfinite(float(s.slope)):
            ax.plot(yr, float(s.intercept) + float(s.slope) * yr, "--", color="black",
                    linewidth=1)
            unit = "PVU/1000 km" if var == "G_max" else "°"
            ax.text(0.02, 0.97, f"{float(s.trend):+.4f} {unit}/dec, {fmt_p(float(s.p))}\n"
                    f"n$_{{eff}}$ = {float(s.n_eff):.0f}/{int(s.n)}", transform=ax.transAxes,
                    fontsize=6.5, va="top")
        ax.set_ylim(lim)
        ax.set_ylabel(ylabel, fontsize=7.5)
        ax.tick_params(labelsize=6.5)
        ax.grid(alpha=0.25)
    if edge.any():
        axes[1].text(0.98, 0.03, f"{int(edge.sum())} edge year(s) excluded",
                     transform=axes[1].transAxes, fontsize=6, ha="right", color="0.3")
    sf.suptitle(name, fontsize=11, fontweight="bold", color=c, y=0.985)


def padded(v, pad=0.12):
    v = np.asarray(v, float)
    lo, hi = np.nanmin(v), np.nanmax(v)
    d = (hi - lo) or abs(hi) or 1e-3
    return lo - pad * d, hi + 2.2 * pad * d                 # extra headroom for the text


def plot_level(lev, res, st):
    r, s = res.sel(pressure_level=lev), st.sel(pressure_level=lev)
    fig = plt.figure(figsize=(26, 14.5))
    head, body = fig.subfigures(2, 1, height_ratios=[0.07, 1])
    left, right = body.subfigures(1, 2, width_ratios=[1.15, 3.2], wspace=0.01)
    top, bottom = left.subfigures(2, 1, height_ratios=[1, 1.15])
    band_map(top)
    trend_bars(bottom, s)
    ylims = (padded(r.G_max), padded(r.phi_max))           # shared across bands
    grid = right.subfigures(4, 2, hspace=0.04, wspace=0.03).T.ravel()
    for sf, name, c in zip(grid, bands, colours):
        band_panel(sf, name, c, r.sel(band=name), s.sel(band=name), ylims)
    yrs = f"{int(r.year.min())}-{int(r.year.max())}"
    head.suptitle(f"ERA5 October {lev} hPa: strength G$_{{max}}$ and latitude φ$_{{max}}$ of the "
                  f"maximum meridional PV gradient, 5-35°S, {yrs}\n"
                  f"dPV/dy after a {smooth_deg:g}° meridional running mean; parabolic sub-grid "
                  "peak; open markers: maximum on the search edge (excluded from trends); "
                  "dashed: OLS trend (AR(1)-corrected p)", fontsize=13.5, fontweight="bold",
                  y=0.75)
    out = out_dir / f"era5_pv_gradient_max_Oct_{lev}hPa.png"
    fig.savefig(out, dpi=250, facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    pv = load()
    res = gradient_max(gradient_profile(pv))                       # (year, level, band)
    valid = res[["G_max", "phi_max"]].where(~res.at_edge)          # edge maxima excluded
    st = _StatsByLevel({v: ols_stats(valid[v]) for v in ("G_max", "phi_max")})
    for lev in levels:
        print(f"\n{lev} hPa (October)")
        for name in bands:
            r = res.sel(pressure_level=lev, band=name)
            s = st.sel(pressure_level=lev, band=name)
            g, ph = s["G_max"], s["phi_max"]
            print(f"  {name:<9} G_max {float(r.G_max.mean()):.3f} PVU/1000km, trend "
                  f"{float(g.trend):+.4f}/dec ({fmt_p(float(g.p))})   phi_max "
                  f"{float(r.phi_max.mean()):6.2f}°, trend {float(ph.trend):+.3f}°/dec "
                  f"({fmt_p(float(ph.p))})   edge years {int(r.at_edge.sum())}")
        plot_level(lev, res, st)
    res.to_dataframe().to_csv(out_dir / "era5_pv_gradient_max_Oct_series.csv")


class _StatsByLevel(dict):
    """{'G_max': Dataset, 'phi_max': Dataset} that can be .sel()-ed like one Dataset."""
    def sel(self, **kw):
        return _StatsByLevel({k: v.sel(**kw) for k, v in self.items()})


if __name__ == "__main__":
    main()