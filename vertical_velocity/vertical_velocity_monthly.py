"""
Vertical velocity (omega) trends and regime change for EVERY month: ERA5 vs MERRA-2 vs JRA-3Q.

Uses omega_regime_change_october.py (same folder) for everything except the month selection:
the same files, common 1.25-degree grid, completeness rule (no NaNs; below ground removed),
OLS + AR(1)-corrected p, BH-FDR per map (dataset x month x level), regime logic and Pettitt
check. See that script's docstring for the method.

Year handling: only years present in all three datasets are used. If one month is missing in
one dataset for a year (a missing or incomplete file), that month-year is removed from all three
datasets; the other months of that year are kept.

FIGURES  (one per level)  omega_regime_change_monthly_<level>hPa.png
  12 columns (Jan -> Dec) x 6 rows: signed trend maps for ERA5, MERRA-2, JRA-3Q, then regime
  maps for the same three, shaded by |trend| (white at zero). Black dots: FDR-significant; white
  areas: no data. One colour scale for all months, levels and datasets (98th percentile of
  |slope|).
CSV  omega_regime_change_monthly_summary.csv
  per dataset, month and level: boxes tested, % FDR-significant, significant boxes per regime,
  mean significant slope (Pa s-1 decade-1).
"""
from __future__ import annotations
import calendar
import importlib.util
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location(
    "omega_oct", Path(__file__).with_name("october_trends.py"))
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m                        # lets parallel readers find it
spec.loader.exec_module(m)
m.months = list(range(1, 13))                      # every month instead of October only


def summary(res):
    rows = []
    for n, r in res.items():
        for mo in m.months:
            for lev in m.levels:
                s = r.sel(month=mo, lev=lev)
                tested, sig = int(s.slope.notnull().sum()), s.sig.values
                rows.append({"dataset": n, "month": calendar.month_abbr[mo], "month_number": mo,
                             "level_hPa": lev, "boxes_tested": tested,
                             "pct_significant": 100 * sig.sum() / max(tested, 1),
                             **{f"n_sig_{k}": int((sig & (s.regime.values == i)).sum())
                                for i, k in enumerate(m.regime_keys)},
                             "mean_sig_slope": float(np.nanmean(s.slope.values[sig]))
                             if sig.any() else np.nan})
    return pd.DataFrame(rows)


def plot_level(lev, res, vlim, yrs, path):
    norm = mcolors.TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)
    fig, axes = plt.subplots(6, 12, figsize=(34, 16.5), subplot_kw={"projection": m.pc})
    fig.subplots_adjust(left=0.05, right=0.99, top=0.88, bottom=0.12, wspace=0.05, hspace=0.12)
    for j, mo in enumerate(m.months):
        for i, n in enumerate(m.names):
            s = res[n].sel(lev=lev, month=mo)
            for row, draw in ((i, lambda ax: m.draw_trend(ax, s, norm)),
                              (i + 3, lambda ax: m.draw_regime(ax, s, vlim))):
                ax = axes[row, j]
                draw(ax)
                m.add_geo(ax, labels=(j == 0, row == 5))
    fig.canvas.draw()                                    # final cartopy boxes, then labels
    for j, mo in enumerate(m.months):
        b = axes[0, j].get_position()
        fig.text((b.x0 + b.x1) / 2, b.y1 + 0.006, calendar.month_name[mo], ha="center",
                 va="bottom", fontsize=14, fontweight="bold")
    for row in range(6):
        b = axes[row, 0].get_position()
        kind = "Trend" if row < 3 else "Regime"
        fig.text(0.006, (b.y0 + b.y1) / 2, f"{m.names[row % 3]}\n{kind}", rotation=90,
                 ha="left", va="center", fontsize=12, fontweight="bold")
    m.trend_colourbar(fig, fig.add_axes([0.25, 0.065, 0.5, 0.012]), norm, vlim)
    fig.legend(handles=m.regime_legend(), loc="upper center", ncol=8, fontsize=13,
               frameon=True, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle(f"Vertical velocity OLS trends and regimes at {lev:.0f} hPa, every month ({yrs}); "
                 f"black dots: BH-FDR q < {m.alpha_fdr} per map, AR(1)-corrected p; "
                 "regime shade ∝ |trend|",
                 fontsize=18, fontweight="bold", y=0.99)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}", flush=True)


def main():
    out = m.out_dir / "monthly" / "regime_change"
    out.mkdir(parents=True, exist_ok=True)
    data = m.load_all()
    print("\nTrends, FDR, regimes and Pettitt for all months and levels ...", flush=True)
    res = m.analyse(data)
    vlim = m.global_vlim(res)
    yrs = f"{int(data['ERA5'].year.min())}-{int(data['ERA5'].year.max())}"
    print(f"Global colour limit: {vlim:.4e} Pa s-1 decade-1")
    table = summary(res)
    table.to_csv(out / "omega_regime_change_monthly_summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 50):
        print(table.pivot_table(index=["level_hPa", "month_number"], columns="dataset",
                                values="pct_significant").round(1))
    for lev in m.levels:
        path = out / f"omega_regime_change_monthly_{lev:06.1f}hPa.png"
        if m.overwrite or not path.exists():
            plot_level(lev, res, vlim, yrs, path)
    print("\nAll levels complete.", flush=True)


if __name__ == "__main__":
    main()