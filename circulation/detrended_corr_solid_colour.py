"""
ERA5 October: detrended correlation between vertical velocity (omega) at 300 hPa and cloud
fraction (CF) at every pressure level, in 5-degree longitude bands from 10 to 40 E
(latitude-pressure cross sections), plus what the long-term changes are where they co-vary.

DATA (current server; one file per year)
  omega  scratch/era5/global/vertical_velocity/global_vertical_velocity_YYYY.nc   w   (Pa s-1)
  CF     scratch/era5/global/fraction_of_cloud_cover/global_fraction_of_cloud_cover_YYYY.nc
         cc (0-1), longitude 0-360 (converted)
  Surface pressure, only to blank out levels below ground and draw the surface line:
         SP_FILE (ERA5 sp) if it exists, otherwise the October mean MERRA-2 PS from
         scratch/merra_2/MERRA2_*.instM_2d_asm_Nx.YYYY10.nc4; without either, nothing is blanked.
  Only October, the 300 hPa omega level, the 10-40 E / 35 S - 10 N domain are read.

METHOD (per band: omega and CF are averaged over the band's longitudes first)
  r      Pearson correlation of the linearly DETRENDED October series, omega(300 hPa, lat) vs
         CF(level, lat), at every (level, latitude).
  sig    SIG_METHOD = "fdr" (default, as in the PV-omega and trend figures): t-test on the
         regression slope with n_eff - 3 degrees of freedom (n_eff from the lag-1
         autocorrelation of the residuals, capped at n), then Benjamini-Hochberg FDR
         q < ALPHA_FDR over each panel. SIG_METHOD = "pointwise": p < P_THRESH with n - 2
         degrees of freedom, as in the original script.
  category  long-term change, from the UNdetrended series:
         omega regime at 300 hPa (per latitude): strengthening / weakening subsidence (mean > 0,
         trend > 0 / < 0), strengthening / weakening ascent (mean < 0, trend < 0 / > 0),
         new subsidence / new ascent (persistent sign change of MIN_PERSIST years; the same
         switch logic as the omega regime script)
         x CF trend at that level and latitude: increasing or decreasing.

FIGURE  <OUT_DIR>/omega300hPa_cf_crosssections_10to40E.png
  Top row     detrended r (RdBu_r, -1 to 1); black dots: significant.
  Bottom row  12 categories (omega regime x CF direction), each a solid colour (only where r is
              significant if CAT_SIG_ONLY = True); same dots.
  Right       domain map with the bands. Below ground: white; brown line: mean surface
              pressure; dotted line: the 300 hPa omega level.
"""
from __future__ import annotations
import glob
import os
import re
from pathlib import Path
import numpy as np
import xarray as xr
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ---------------------------------------------------------------- paths
OMEGA_DIR = Path("/soge-home/users/jesu4127/scratch/era5/global/vertical_velocity")
CF_DIR = Path("/soge-home/users/jesu4127/scratch/era5/global/fraction_of_cloud_cover")
SP_FILE = Path("/soge-home/users/jesu4127/scratch/era5/global_surface_pressure.nc")
MERRA2_DIR = Path("/soge-home/users/jesu4127/scratch/merra_2")
OUT_DIR = Path("/soge-home/users/jesu4127/Paper_1/circulation")

# ---------------------------------------------------------------- settings
START_YEAR, END_YEAR = 1979, 2025
MONTH = 10
OMEGA_LEV = 300.0
LAT_MIN, LAT_MAX = -35.0, 10.0
LON_BANDS = [(10.0, 15.0), (15.0, 20.0), (20.0, 25.0), (25.0, 30.0), (30.0, 35.0), (35.0, 40.0)]
SIG_METHOD = "fdr"                # "fdr" or "pointwise"
ALPHA_FDR = 0.10
P_THRESH = 0.05
CAT_SIG_ONLY = False              # True: colour categories only where r is significant
MIN_PERSIST = 10                  # years of one sign for a "new" regime
DOT_EVERY = 4                     # stipple every 4th latitude (1 deg)
C_SP_LINE = "#7B3F00"
REGIMES = ["str_sub", "wk_sub", "new_sub", "wk_asc", "str_asc", "new_asc"]
REGIME_LABELS = {"str_sub": "Strengthening subsidence", "wk_sub": "Weakening subsidence",
                 "new_sub": "New subsidence", "wk_asc": "Weakening ascent",
                 "str_asc": "Strengthening ascent", "new_asc": "New ascent"}
# (CF increasing, CF decreasing) for each omega regime; same palette as the PV-omega figure
CATEGORY_COLOURS = {
    "str_sub": ("#DC2671", "#7A0F3C"), "wk_sub": ("#FE6100", "#8C3A00"),
    "new_sub": ("#FFB000", "#8C6D00"), "wk_asc": ("#4DB5FA", "#0B4F82"),
    "str_asc": ("#785EF0", "#36238C"), "new_asc": ("#60D691", "#14703C")}
BAND_COLS = list(plt.cm.tab10.colors[:len(LON_BANDS)])

matplotlib.rcParams.update({"font.family": "sans-serif", "font.size": 13,
                            "axes.labelsize": 13, "xtick.labelsize": 11,
                            "ytick.labelsize": 11})


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------- loading
def year_files(directory, stem):
    out = {}
    for f in directory.glob(f"{stem}_*.nc"):
        m = re.search(r"_(\d{4})\.nc$", f.name)
        if m and START_YEAR <= int(m[1]) <= END_YEAR:
            out[int(m[1])] = f
    return dict(sorted(out.items()))


def read_october(path, var, levels=None):
    """October of one yearly file, cut to the domain: (level, lat, lon) or (lat, lon)."""
    with xr.open_dataset(path) as ds:
        da = ds[var].rename({"latitude": "lat", "longitude": "lon"})
        t = "valid_time" if "valid_time" in da.dims else "time"
        idx = np.flatnonzero(da[t].dt.month.values == MONTH)
        if not idx.size:
            raise ValueError("no October in file")
        da = da.isel({t: int(idx[0])}).drop_vars(["number", "expver", t], errors="ignore")
        if float(da.lon.max()) > 180:                       # 0-360 -> -180-180
            da = da.assign_coords(lon=(da.lon + 180) % 360 - 180).sortby("lon")
        da = da.sortby("lat").sel(lat=slice(LAT_MIN, LAT_MAX),
                                  lon=slice(LON_BANDS[0][0], LON_BANDS[-1][1]))
        if levels is not None:
            da = da.sel(pressure_level=levels)
        return da.load()


def load_stack(directory, stem, var, levels=None):
    files = year_files(directory, stem)
    parts, years = [], []
    for y, f in files.items():
        try:
            parts.append(read_october(f, var, levels))
            years.append(y)
        except Exception as e:
            log(f"  SKIPPED {f.name}: {type(e).__name__}: {e}")
    if not parts:
        raise RuntimeError(f"no readable {stem} files in {directory}")
    log(f"  {var}: {len(years)} Octobers ({years[0]}-{years[-1]})")
    return xr.concat(parts, "year").assign_coords(year=years)


def load_surface_pressure(lat, lon):
    """Mean October surface pressure (hPa) on (lat, lon), or None."""
    if SP_FILE.exists():
        with xr.open_dataset(SP_FILE) as ds:
            da = ds["sp"].rename({"latitude": "lat", "longitude": "lon"})
            t = "valid_time" if "valid_time" in da.dims else "time"
            da = da.isel({t: np.flatnonzero(da[t].dt.month.values == MONTH)}).mean(t)
            if float(da.lon.max()) > 180:
                da = da.assign_coords(lon=(da.lon + 180) % 360 - 180).sortby("lon")
            sp = da.sortby("lat").interp(lat=lat, lon=lon).load()
        log("  surface pressure: ERA5")
    else:
        files = sorted(glob.glob(str(MERRA2_DIR / f"MERRA2_*.instM_2d_asm_Nx.*{MONTH:02d}.nc4")))
        files = [f for f in files if re.search(rf"\.(\d{{4}}){MONTH:02d}\.nc4$", f)]
        if not files:
            log("  [WARN] no surface pressure found: levels below ground are not blanked")
            return None
        fields = []
        for f in files:
            try:
                with xr.open_dataset(f) as ds:
                    fields.append(ds["PS"].squeeze(drop=True).load())
            except Exception as e:
                log(f"  [WARN] {os.path.basename(f)}: {e}")
        sp = xr.concat(fields, "t").mean("t").interp(lat=lat, lon=lon)
        log(f"  surface pressure: MERRA-2 PS, mean of {len(fields)} Octobers")
    sp = sp.where(np.abs(sp) < 1e10)
    return sp / 100.0 if float(sp.max()) > 2000 else sp


# ---------------------------------------------------------------- statistics
def detrend(a):
    """OLS-detrended anomalies along axis 0 (years)."""
    t = np.arange(a.shape[0], dtype=float)
    t = (t - t.mean()).reshape((-1,) + (1,) * (a.ndim - 1))
    am = a - a.mean(0)
    return am - (t * am).sum(0) / (t ** 2).sum(0) * t


def slope(a):
    t = np.arange(a.shape[0], dtype=float)
    t = (t - t.mean()).reshape((-1,) + (1,) * (a.ndim - 1))
    return (t * (a - a.mean(0))).sum(0) / (t ** 2).sum(0)


def corr_and_p(x, y):
    """x (T, N) detrended omega broadcast to y (T, L, N) detrended CF: r and p per (L, N)."""
    x = np.broadcast_to(x[:, None, :], y.shape)
    n = y.shape[0]
    sxx, syy, sxy = (x * x).sum(0), (y * y).sum(0), (x * y).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = sxy / np.sqrt(sxx * syy)
        if SIG_METHOD == "pointwise":
            tt = r * np.sqrt((n - 2) / np.maximum(1 - r ** 2, 1e-30))
            return r, 2 * stats.t.sf(np.abs(tt), n - 2)
        b = sxy / sxx                                       # regression of CF' on omega'
        e = y - b * x
        sse = (e * e).sum(0)
        r1 = (e[1:] * e[:-1]).sum(0) / sse
        n_eff = np.minimum(n * (1 - r1) / (1 + r1), n)
        se = np.sqrt(sse / (n_eff - 3) / sxx)
        p = 2 * stats.t.sf(np.abs(b / se), n_eff - 3)
    return r, np.where(n_eff > 3, p, np.nan)


def significant(p):
    sig, ok = np.zeros(p.shape, bool), np.isfinite(p)
    if not ok.any():
        return sig
    if SIG_METHOD == "pointwise":
        sig[ok] = p[ok] < P_THRESH
    else:
        sig[ok] = stats.false_discovery_control(p[ok]) < ALPHA_FDR
    return sig


def regime_codes(x, slope_):
    """Omega regime per column of x (year, N): index into REGIMES, -1 none (switch logic of
    the omega regime script: first persistent sign change, else mean sign x trend sign)."""
    n, P = x.shape[0], MIN_PERSIST
    fin = np.isfinite(x)
    cum = lambda a: np.concatenate([np.zeros_like(a[:1], float), np.cumsum(a, 0)])
    c_sum, c_fin = cum(np.where(fin, x, 0.0)), cum(fin)
    c_pos, c_neg = cum(fin & (x > 0)), cum(fin & (x < 0))
    ks = np.arange(1, n - P)
    win = lambda c: c[ks + P] - c[ks]
    with np.errstate(invalid="ignore", divide="ignore"):
        pre_n, post_n = c_fin[ks], c_fin[n] - c_fin[ks]
        pre, post = c_sum[ks] / pre_n, (c_sum[n] - c_sum[ks]) / post_n
        ok = ((win(c_fin) == P) & ((win(c_pos) == P) | (win(c_neg) == P)) & (pre_n >= 2)
              & (post_n >= 2) & (pre != 0) & (post != 0) & (np.sign(pre) != np.sign(post)))
        found = ok.any(0)
        post_first = np.take_along_axis(post, ok.argmax(0)[None], 0)[0]
        mean = c_sum[n] / np.maximum(c_fin[n], 1)
    code = np.where(mean > 0, np.where(slope_ > 0, 0, 1), np.where(slope_ < 0, 4, 3))
    code = np.where(found, np.where(post_first > 0, 2, 5), code)
    return np.where(np.isfinite(slope_), code, -1)


# ---------------------------------------------------------------- plotting
def edges(v):
    d = np.diff(v) / 2
    return np.concatenate([[v[0] - d[0]], v[:-1] + d, [v[-1] + d[-1]]])


def category_colours():
    return [c for k in REGIMES for c in CATEGORY_COLOURS[k]]


def category_labels():
    return [f"{REGIME_LABELS[k]} × CF {d}" for k in REGIMES for d in ("increasing", "decreasing")]


def format_section(ax, plevs, sp, show_y, show_x=True):
    ybot = min(1025.0, np.ceil(np.nanmax(sp) / 25.0) * 25.0) if sp is not None else 1000.0
    ax.set_ylim(ybot, 100)
    ax.set_xlim(LAT_MIN, LAT_MAX)
    ax.yaxis.set_major_locator(mticker.FixedLocator(
        [v for v in (100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000) if v <= ybot]))
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.5, zorder=3)
    if show_y:
        ax.set_ylabel("Pressure (hPa)")
    else:
        ax.tick_params(labelleft=False)
    if show_x:
        ax.set_xlabel("Latitude (°N)")
    else:
        ax.tick_params(labelbottom=False)


def overlays(ax, lat, plevs, res):
    if res["sp"] is not None:
        ax.plot(lat, res["sp"], color=C_SP_LINE, lw=1.8, zorder=6)
    ax.axhline(OMEGA_LEV, color="black", lw=1.2, ls=":", zorder=7)
    sig = res["sig"].copy()
    sig[:, np.arange(sig.shape[1]) % DOT_EVERY != 0] = False
    li, ci = np.nonzero(sig)
    ax.scatter(lat[ci], plevs[li], s=4, color="black", linewidths=0, zorder=5)


def make_figure(results, lat, plevs, years, path):
    nb = len(results)
    W, left, right, gap = 30.0, 0.9, 0.3, 0.25
    map_w = 3.4
    pw = (W - left - right - nb * gap - map_w - 0.6) / nb
    ph = 4.2
    tops = [1.5, 1.5 + ph + 1.0]
    H = tops[1] + ph + 2.6
    fig = plt.figure(figsize=(W, H))
    rnorm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    lat_e, plev_e = edges(lat), edges(plevs)
    cols = category_colours()
    for j, res in enumerate(results):
        x0 = left + j * (pw + gap)
        for row, top in enumerate(tops):
            ax = fig.add_axes([x0 / W, 1 - (top + ph) / H, pw / W, ph / H])
            if row == 0:
                ax.pcolormesh(lat_e, plev_e, res["r_masked"], cmap="RdBu_r", norm=rnorm,
                              zorder=1, rasterized=True)
                ax.set_title(res["label"], fontsize=14, fontweight="bold",
                             color=BAND_COLS[j], pad=6)
            else:
                cat = res["cat"].astype(float)
                cat[(cat < 0) | ~np.isfinite(res["r_masked"])] = np.nan
                if CAT_SIG_ONLY:
                    cat[~res["sig"]] = np.nan
                ax.pcolormesh(lat_e, plev_e, cat, cmap=mcolors.ListedColormap(cols),
                              norm=mcolors.BoundaryNorm(np.arange(len(cols) + 1) - 0.5, len(cols)),
                              zorder=1, rasterized=True)
            overlays(ax, lat, plevs, res)
            format_section(ax, plevs, res["sp"], show_y=(j == 0))
            for s in ax.spines.values():
                s.set_edgecolor(BAND_COLS[j])
                s.set_linewidth(2.5)
            if j == 0:
                ax.text(-0.3, 0.5, ["Detrended r(ω300, CF)", "Long-term change"][row],
                        transform=ax.transAxes, rotation=90, ha="center", va="center",
                        fontsize=15, fontweight="bold")
    xr0 = left + nb * (pw + gap) + 0.4
    # domain map (top right)
    axm = fig.add_axes([xr0 / W, 1 - (tops[0] + ph) / H, map_w / W, ph / H], projection=ccrs.PlateCarree())
    axm.set_extent([LON_BANDS[0][0], LON_BANDS[-1][1], LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
    axm.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#d0e8f0", zorder=1)
    axm.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f0ede8", zorder=2)
    axm.coastlines(resolution="50m", linewidth=0.6, zorder=5)
    axm.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.4, edgecolor="grey", zorder=4)
    for j, (lo, hi) in enumerate(LON_BANDS):
        axm.add_patch(mpatches.Rectangle((lo, LAT_MIN), hi - lo, LAT_MAX - LAT_MIN,
                                         facecolor=BAND_COLS[j], alpha=0.35,
                                         transform=ccrs.PlateCarree(), zorder=6))
    gl = axm.gridlines(draw_labels=True, linewidth=0.3, color="grey", alpha=0.5, linestyle=":")
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 9}
    axm.set_title("Domain", fontsize=13, fontweight="bold", pad=4)
    # r colour bar (under the map)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=rnorm, cmap="RdBu_r"),
                      cax=fig.add_axes([xr0 / W, 1 - (tops[0] + ph + 0.62) / H, map_w / W, 0.18 / H]),
                      orientation="horizontal", ticks=np.linspace(-1, 1, 5))
    cb.set_label("Correlation r", fontsize=12)
    cb.ax.xaxis.set_label_position("top")
    sig_text = (f"BH-FDR q < {ALPHA_FDR} per panel, AR(1)-corrected" if SIG_METHOD == "fdr"
                else f"p < {P_THRESH}")
    handles = [mpatches.Patch(facecolor=c, label=l) for c, l in zip(cols, category_labels())]
    handles += [Line2D([], [], marker="o", ls="none", color="black", markersize=5,
                       label=f"r significant ({sig_text})"),
                Line2D([], [], color=C_SP_LINE, lw=2, label="Mean surface pressure"),
                Line2D([], [], color="black", lw=1.5, ls=":", label=f"ω level ({OMEGA_LEV:.0f} hPa)"),
                mpatches.Patch(facecolor="white", edgecolor="0.5", label="Below ground")]
    fig.legend(handles=handles, loc="upper center", ncol=6, fontsize=12, frameon=True,
               bbox_to_anchor=(0.5, 1 - (tops[1] + ph + 0.75) / H), handlelength=1.8,
               columnspacing=1.2)
    fig.suptitle(f"ERA5 October {years[0]}-{years[-1]}: detrended r(ω {OMEGA_LEV:.0f} hPa, cloud "
                 "fraction) and the long-term change it goes with",
                 fontsize=20, fontweight="bold", y=1 - 0.35 / H, va="center")
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    log(f"Saved: {path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log("Loading omega ...")
    w = load_stack(OMEGA_DIR, "global_vertical_velocity", "w", levels=[OMEGA_LEV])
    w = w.isel(pressure_level=0, drop=True)
    log("Loading cloud fraction ...")
    cf = load_stack(CF_DIR, "global_fraction_of_cloud_cover", "cc").sortby("pressure_level")
    years = sorted(set(w.year.values.tolist()) & set(cf.year.values.tolist()))
    w, cf = w.sel(year=years), cf.sel(year=years)
    cf = cf.interp(lat=w.lat, lon=w.lon)                     # same 0.25 deg grid; aligns coords
    lat, plevs = w.lat.values, cf.pressure_level.values.astype(float)
    log(f"Common years {years[0]}-{years[-1]} (n = {len(years)}); {lat.size} latitudes; "
        f"{plevs.size} CF levels")
    sp = load_surface_pressure(w.lat, w.lon)
    results = []
    for lo, hi in LON_BANDS:
        sel = (w.lon >= lo) & (w.lon < hi)
        wb = w.where(sel, drop=True).mean("lon").values                 # (year, lat)
        cfb = cf.where(sel, drop=True).mean("lon").transpose("year", "pressure_level", "lat").values
        spb = None if sp is None else sp.where(sel, drop=True).mean("lon").values
        r, p = corr_and_p(detrend(wb), detrend(cfb))
        below = (plevs[:, None] > spb[None, :]) if spb is not None else np.zeros(r.shape, bool)
        r_masked = np.where(below, np.nan, r)
        sig = significant(np.where(below, np.nan, p))
        reg = regime_codes(wb, slope(wb))                               # (lat,)
        cf_up = slope(cfb) > 0                                          # (level, lat)
        cat = np.where((reg[None, :] >= 0) & ~below, 2 * reg[None, :] + np.where(cf_up, 0, 1), -1)
        label = f"{lo:.0f}°E–{hi:.0f}°E"
        log(f"  {label}: {100 * sig.sum() / max(np.isfinite(r_masked).sum(), 1):.0f}% of "
            "level-latitude cells significant")
        results.append({"label": label, "r_masked": r_masked, "sig": sig, "cat": cat, "sp": spb})
    make_figure(results, lat, plevs, years,
                OUT_DIR / f"omega{OMEGA_LEV:.0f}hPa_cf_crosssections_10to40E.png")
    log("Done.")


if __name__ == "__main__":
    main()