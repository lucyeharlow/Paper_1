"""
Diagnostics for the Southern Africa 2 m temperature trend analysis.

Loads the main trend script (main_script below) and reuses its loading, land mask,
anomalies, year ranges, completeness rule and OLS / AR(1) code, so every diagnostic
describes exactly the data and fits behind the trend figures.

FIGURES (written to the main script's out_dir)
  diagnostics_qq_MM_Mon.png (12 figures)
      Normal Q-Q plots of the OLS residuals for each month, one panel per dataset.
      Grey line: standardised residuals of every complete land cell for that month,
      with area-weighted quantiles. Points: residuals of the regional (area-mean)
      series. Text: share of cell series failing Shapiro-Wilk at 5 % (~5 % if normal).
  diagnostics_timeseries_MM_Mon.png (12 figures)
      Area-weighted regional anomaly series for each month, one line per dataset.
  diagnostics_autocorrelation.png
      Lag-1 autocorrelation r1 of the OLS residuals, the quantity behind n_eff.
      (a) area-weighted distribution of r1 by month   (b)-(g) r1 maps, target month
      (h) autocorrelation function of each regional target-month series, 95 % band.
  diagnostics_crutem5_completeness.png  (+ .csv)
      Every CRUTEM5 5-degree box in the domain, laid out as on a map: its monthly anomaly
      series over the analysis period, missing months marked in red, % of months present,
      how many calendar months (of 12) have a complete record, and whether the box is
      land. Shaded panels are removed by the land mask.
"""
from __future__ import annotations
import calendar
import importlib.util
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
import cartopy.feature as cfeature

main_script = Path(__file__).with_name("lag_autocorr.py")
spec = importlib.util.spec_from_file_location("trends", main_script)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
target_month = getattr(m, "target_month", 10)
months = m.months


# ---------------------------------------------------------------- data
def prepare():
    """Same steps as the main script: land mask, anomalies, year ranges."""
    raw = m.load_all()
    data = {n: m.anomalies(n, da) for n, da in m.land_only(raw).items()}
    shared = set.intersection(*(m.years_of(da) for n, da in data.items() if n != "CHIRTS"))
    periods = {n: sorted(shared) for n in data}
    periods["CHIRTS"] = sorted(m.years_of(data["CHIRTS"]) & m.chirts_years)
    data = {n: m.select_years(da, periods[n]) for n, da in data.items()}
    crutem_all = m.select_years(m.anomalies("CRUTEM", raw["CRUTEM"]), periods["CRUTEM"])
    return data, periods, crutem_all


def one_month(da, month):
    return da.sel(time=da.time.dt.month == month)


def month_series(da, month):
    """Regional series exactly as in the main analysis: complete cells, cos-lat weights."""
    sel = m.complete(one_month(da, month))
    return sel.weighted(np.cos(np.deg2rad(sel.lat))).mean(("lat", "lon"))


def residuals(da):
    """Standardised OLS residuals (complete cells only), the same fit as the trends."""
    da = m.complete(da)
    x = da.time.dt.year.astype(float)
    xa, ya = x - x.mean(), da - da.mean("time")
    resid = ya - (xa * ya).sum("time") / (xa ** 2).sum() * xa
    return resid / resid.std("time", ddof=2)


def acf(resid, lags):
    """Autocorrelation on a gap-free year axis, same estimator as r1 in the main script."""
    years = resid.time.dt.year.values
    r = resid.assign_coords(time=years).reindex(time=np.arange(years.min(), years.max() + 1))
    ss = (r ** 2).sum()
    return np.array([float((r * r.shift(time=k)).sum() / ss) for k in lags])


def map_panel(ax, field, title, norm, cmap, note):
    mesh = ax.pcolormesh(field.lon, field.lat, field, transform=m.pc, cmap=cmap, norm=norm,
                         shading="nearest")
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.8, zorder=5)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.6, zorder=5)
    ax.set_extent([m.lon_min, m.lon_max, m.lat_min, m.lat_max], crs=m.pc)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4, linestyle="--")
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"fontsize": 7}
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.text(0.02, 0.03, note, transform=ax.transAxes, fontsize=7, zorder=8,
            bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none", alpha=0.8))
    return mesh


def save(fig, name):
    out = m.out_dir / name
    fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ---------------------------------------------------------------- Q-Q
def plot_qq(data):
    print("\nQ-Q plots of OLS residuals...")
    p = np.linspace(0.005, 0.995, 199)
    zq = stats.norm.ppf(p)
    for month in months:
        fig, axes = plt.subplots(2, 3, figsize=(13, 8.5), sharex=True, sharey=True)
        for ax, (name, da) in zip(axes.flat, data.items()):
            z = residuals(one_month(da, month))
            zz = z.values.reshape(z.sizes["time"], -1)
            ok = np.isfinite(zz).all(axis=0)
            fail, tested = 0, 0
            if ok.any():
                wt = np.broadcast_to(m.area_weights(z.isel(time=0)).values.ravel()[ok], zz[:, ok].shape)
                pv = stats.shapiro(zz[:, ok], axis=0).pvalue
                fail += int((pv < 0.05).sum()); tested += int(ok.sum())
                ax.plot(zq, m.wquantile(zz[:, ok].ravel(), wt.ravel(), p), color="0.45",
                        linewidth=2.5, label="complete land cells (area-weighted)")
            r = residuals(month_series(da, month)).values
            r = np.sort(r[np.isfinite(r)])
            if r.size:
                ax.scatter(stats.norm.ppf((np.arange(r.size) + 0.5) / r.size), r, s=6,
                           color=m.colours[name], zorder=3, label="regional series")
            txt = (f"Shapiro-Wilk p < 0.05:\n{100 * fail / tested:.1f}% of {tested:,} "
                   f"cell series\n(~5% expected if normal)" if tested else "no complete cells")
            ax.text(0.03, 0.97, txt, transform=ax.transAxes, fontsize=8, va="top")
            ax.set_title(m.labels[name], fontsize=11, fontweight="bold")
            ax.set(xlim=(-3.2, 3.2), ylim=(-4.5, 4.5))
            ax.grid(alpha=0.25)
            print(f"  {calendar.month_name[month]:<10}{m.labels[name]:<26}{tested:>7,} "
                  f"cell series, {100 * fail / max(tested, 1):.1f}% fail Shapiro-Wilk at 5%")
        for ax in axes[1]:
            ax.set_xlabel("Theoretical normal quantile")
        for ax in axes[:, 0]:
            ax.set_ylabel("Standardised OLS residual (°C)")
        axes[0, 0].legend(loc="lower right", fontsize=8, frameon=False)
        fig.suptitle(f"Normal Q-Q plots of {calendar.month_name[month]} OLS trend residuals "
                     "(land only)", fontsize=14, fontweight="bold")
        fig.tight_layout()
        save(fig, f"diagnostics_qq_{month:02d}_{calendar.month_abbr[month]}.png")


def plot_monthly_timeseries(data, periods):
    """Plot the area-weighted regional anomaly series separately for each month."""
    print("\nMonthly area-weighted regional time series...")
    for month in months:
        fig, ax = plt.subplots(figsize=(12, 5.5))
        for name, da in data.items():
            series = month_series(da, month)
            if series.notnull().any():
                ax.plot(series.time.dt.year, series, color=m.colours[name], linewidth=1.8,
                        label=m.labels[name])
        all_years = [year for years in periods.values() for year in years]
        ax.set_xlim(min(all_years), max(all_years))
        ax.set_title(f"{calendar.month_name[month]} area-weighted anomalies", fontsize=13)
        ax.set_xlabel("Year")
        ax.set_ylabel(f"{calendar.month_name[month]} anomaly (°C)")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", frameon=False, ncol=2, fontsize=9)
        fig.tight_layout()
        save(fig, f"diagnostics_timeseries_{month:02d}_{calendar.month_abbr[month]}.png")


# ---------------------------------------------------------------- autocorrelation
def plot_autocorr(data, periods):
    print("\nAutocorrelation of OLS residuals...")
    results = {n: da.groupby("time.month").map(m.ols_stats) for n, da in data.items()}
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(3, 6, height_ratios=[1.0, 1.2, 0.9], hspace=0.6, wspace=0.28)

    ax = fig.add_subplot(gs[0, :])                                   # (a) r1 by month
    line = dict(color="black", linewidth=0.9)
    for (name, r), off in zip(results.items(), np.linspace(-0.15, 0.15, 6)):
        boxes = {mm: m.box_stats(r.r1.sel(month=mm)) for mm in months}
        have = [mm for mm in months if boxes[mm]]
        if have:
            ax.bxp([boxes[mm] for mm in have], positions=np.array(have) + off, widths=0.06,
                   patch_artist=True, showfliers=False, manage_ticks=False, whiskerprops=line,
                   capprops=line, medianprops=line,
                   boxprops=dict(facecolor=m.colours[name], edgecolor="black", linewidth=0.9))
    ax.axhline(0, linestyle="--", linewidth=0.8, color="black")
    ax.set(xlim=(0.6, 12.8), xticks=months, xticklabels=calendar.month_abbr[1:])
    ax.set_ylabel("Lag-1 autocorrelation r$_1$\n(area-weighted)")
    ax.grid(axis="y", alpha=0.25)
    ax.text(0.99, 0.96, "(a)", transform=ax.transAxes, fontsize=12, fontweight="bold",
            ha="right", va="top")
    ax.legend(handles=[matplotlib.patches.Patch(facecolor=m.colours[n], edgecolor="black",
                                                label=m.labels[n]) for n in results],
              loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=6, frameon=False, fontsize=9)

    norm = BoundaryNorm(np.linspace(-0.6, 0.6, 13), plt.get_cmap("PuOr_r").N, extend="both")
    map_axes = [fig.add_subplot(gs[1, i], projection=m.pc) for i in range(6)]
    for ax_map, (name, r), panel in zip(map_axes, results.items(), "bcdefg"):
        rm = r.sel(month=target_month)
        d = m.describe(rm.r1)
        note = (f"area-mean r$_1$ = {d['mean_C_per_decade']:+.2f}\n"
                f"median n$_{{eff}}$ = {float(rm.n_eff.median()):.0f} of "
                f"{float(rm.n.median()):.0f}" if d["n_gridcells"] else "no complete cells")
        mesh = map_panel(ax_map, rm.r1, f"({panel}) {m.labels[name]}", norm, "PuOr_r", note)
    cbar = fig.colorbar(mesh, ax=map_axes, orientation="horizontal", fraction=0.06, pad=0.12,
                        aspect=45, shrink=0.8)
    cbar.set_label(f"{calendar.month_name[target_month]} lag-1 autocorrelation r$_1$ of OLS residuals")

    ax = fig.add_subplot(gs[2, :])                                   # (h) regional ACF
    lags = np.arange(1, 11)
    for n_years in sorted({len(p) for p in periods.values()}):
        band = 1.96 / np.sqrt(n_years)
        ax.fill_between([0.5, 10.5], -band, band, color="0.85", alpha=0.6, linewidth=0)
        ax.text(10.45, band, f"95% band, n = {n_years}", fontsize=7, ha="right", va="bottom")
    for (name, da), off in zip(data.items(), np.linspace(-0.25, 0.25, 6)):
        res = residuals(month_series(da, target_month))
        if res.notnull().any():
            ax.bar(lags + off, acf(res, lags), width=0.09, color=m.colours[name],
                   label=m.labels[name])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(xlim=(0.5, 10.5), xticks=lags, xlabel="Lag (years)", ylabel="Autocorrelation")
    ax.set_title(f"(h) Autocorrelation of the regional {calendar.month_name[target_month]} "
                 "residual series", fontsize=12)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=6, fontsize=8, frameon=False, loc="lower left")
    fig.suptitle("Autocorrelation of OLS trend residuals (land only)", fontsize=15,
                 fontweight="bold", y=0.96)
    save(fig, "diagnostics_autocorrelation.png")


# ---------------------------------------------------------------- CRUTEM5 completeness
def plot_crutem_completeness(crutem):
    print("\nCRUTEM5 completeness per grid box...")
    land = m.land_fraction(crutem)
    present = crutem.notnull()
    pct = 100 * present.mean("time")
    full_months = present.groupby("time.month").all("time")         # (month, lat, lon)
    lats, lons = crutem.lat.values[::-1], crutem.lon.values          # north at the top
    fig, axes = plt.subplots(len(lats), len(lons), figsize=(2.9 * len(lons), 1.9 * len(lats)),
                             sharex=True, sharey=True, squeeze=False)
    rows, t = [], crutem.time.values
    ylim = np.nanmax(np.abs(crutem.values)) if np.isfinite(crutem.values).any() else 1
    for i, la in enumerate(lats):
        for j, lo in enumerate(lons):
            ax = axes[i, j]
            s = crutem.sel(lat=la, lon=lo)
            is_land = float(land.sel(lat=la, lon=lo)) >= m.land_min_fraction
            k = int(full_months.sel(lat=la, lon=lo).sum())
            ok = s.notnull().values
            ax.plot(t, s.values, color=m.colours["CRUTEM"], linewidth=0.5)
            ax.plot(t[~ok], np.zeros((~ok).sum()), "|", color="red", markersize=4,
                    transform=ax.get_xaxis_transform())
            if not is_land:
                ax.set_facecolor("0.9")
            months_ok = ",".join(calendar.month_abbr[mm][0] for mm in months
                                 if bool(full_months.sel(month=mm, lat=la, lon=lo)))
            ax.set_title(f"{abs(la):.1f}°S {lo:.1f}°E", fontsize=8, pad=2)
            ax.text(0.02, 0.97, f"{float(pct.sel(lat=la, lon=lo)):.0f}% months · "
                    f"{k}/12 complete" + ("" if is_land else " · sea (masked)"),
                    transform=ax.transAxes, fontsize=6.5, va="top")
            ax.set_ylim(-ylim, ylim)
            ax.tick_params(labelsize=6)
            rows.append({"lat": la, "lon": lo, "land_fraction": round(float(land.sel(lat=la, lon=lo)), 3),
                         "used_after_land_mask": is_land, "pct_months_present": float(pct.sel(lat=la, lon=lo)),
                         "first_year": int(s.time.dt.year[ok].min()) if ok.any() else None,
                         "n_complete_calendar_months": k, "complete_months": months_ok})
    for ax in axes[:, 0]:
        ax.set_ylabel("Anomaly (°C)", fontsize=7)
    years = crutem.time.dt.year.values
    fig.suptitle(f"CRUTEM5 grid boxes, {years.min()}-{years.max()}: monthly anomalies "
                 "(red ticks = missing months; grey panels = removed by land mask;\n"
                 "'k/12 complete' = calendar months with a value in every year, "
                 "i.e. usable for trends)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "diagnostics_crutem5_completeness.png")
    table = pd.DataFrame(rows)
    table.to_csv(m.out_dir / "diagnostics_crutem5_completeness.csv", index=False)
    print(table.to_string(index=False))


def main():
    data, periods, crutem_all = prepare()
    plot_qq(data)
    plot_monthly_timeseries(data, periods)
    plot_autocorr(data, periods)
    plot_crutem_completeness(crutem_all)
    print(f"\nDiagnostics written to {m.out_dir}")


if __name__ == "__main__":
    main()