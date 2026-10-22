"""
October PV-omega coupling at one pressure level (default 300 hPa), ERA5 and MERRA-2.

Reuses pv_omega_eof_october.py (same folder), and through it the omega and PV readers: the
same domain (35 S - 15 N, 20 W - 55 E), 1.25-degree box grid, completeness rule and caches.

TOP ROW: detrended correlation r(PV', omega') at every box (both October series linearly
  detrended). Black dots: significant, t-test on the regression slope with n_eff - 3 degrees of
  freedom (n_eff from the lag-1 autocorrelation of the residuals, capped at n), then
  Benjamini-Hochberg FDR q < alpha_fdr per map (Wilks 2016).

BOTTOM ROW ("trend classification"): every box gets one of 12 categories, the combination of
  its long-term omega regime and its PV trend, each with its own colour:
    omega regime   strengthening / weakening subsidence (mean > 0, trend > 0 / < 0),
                   strengthening / weakening ascent (mean < 0, trend < 0 / > 0),
                   new subsidence / new ascent (persistent sign change, min_persist years),
                   from the long-term mean and OLS trend (same rules as the omega figures);
    PV direction   pv_direction = "magnitude": |PV| strengthening (PV trend has the same sign
                   as the mean PV; PV < 0 south of the equator, so there a negative trend is a
                   strengthening) or weakening. pv_direction = "signed": PV increasing or
                   decreasing.
  e.g. one colour = strengthening subsidence x |PV| strengthening. The colour is shaded by the
  strength of the detrended correlation, white at r = 0 to full colour at |r| = 1 (one mini
  colour bar per category). The same black dots as the top row mark significant correlations.

FIGURE  pv_omega_correlation_categories_oct_<lev>hPa.png
CSV     pv_omega_correlation_categories_oct_<lev>hPa.csv   area-weighted % of the domain in each
        category, all boxes and significant-correlation boxes only, with the mean r in each.
"""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import xarray as xr

spec = importlib.util.spec_from_file_location(
    "pv_omega_eof", Path(__file__).with_name("eof_pv_omega_exploration.py"))
e = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = e
spec.loader.exec_module(e)
om, pvm = e.om, e.pvm

# ---------------------------------------------------------------- settings
out_dir = Path("/soge-home/users/jesu4127/Paper_1/pv_omega_correlation")
e.level = 300                                  # hPa (also read by the loaders)
pv_direction = "magnitude"                     # "magnitude" (|PV|) or "signed" (PV)
names = ["ERA5", "MERRA-2"]
# one colour per (omega regime, PV direction): first |PV| strengthening / PV increasing, then
# weakening / decreasing. Subsidence warm, ascent cool; each pair shares a hue family.
category_colours = {
    "str_sub": ("#DC2671", "#7A0F3C"), "wk_sub": ("#FE6100", "#8C3A00"),
    "new_sub": ("#FFB000", "#8C6D00"), "wk_asc": ("#4DB5FA", "#0B4F82"),
    "str_asc": ("#785EF0", "#36238C"), "new_asc": ("#60D691", "#14703C")}


# ---------------------------------------------------------------- analysis
def categories(omega, pv):
    """Category code per box: 2 * regime index + (0 if PV strengthening/increasing else 1)."""
    w = e.gap_free(omega)
    st = om.ols_stats(w.expand_dims(month=[e.month]))
    regime = om.regimes(w.expand_dims(month=[e.month]), st.slope).isel(month=0)
    p = e.gap_free(pv)
    pv_slope = om.ols_stats(p.expand_dims(month=[e.month])).slope.isel(month=0)
    if pv_direction == "magnitude":
        up = np.sign(pv_slope) == np.sign(p.mean("year"))       # |PV| strengthening
    else:
        up = pv_slope > 0
    ok = (regime >= 0) & pv_slope.notnull()
    return (2 * regime + xr.where(up, 0, 1)).where(ok), regime, st.slope.isel(month=0)


def category_labels():
    pv_up, pv_down = (("|PV| strengthening", "|PV| weakening") if pv_direction == "magnitude"
                      else ("PV increasing", "PV decreasing"))
    return [f"{om.regime_labels[k]}, {d}" for k in om.regime_keys for d in (pv_up, pv_down)]


# ---------------------------------------------------------------- plotting
def colours_in_order():
    return [c for k in om.regime_keys for c in category_colours[k]]


def draw_categories(ax, cat, strength):
    """Each category in its own colour, shaded from white (r = 0) to full colour (|r| = 1)."""
    for k, colour in enumerate(colours_in_order()):
        field = np.where(cat.values == k, np.clip(strength, 0, 1), np.nan)
        if np.isfinite(field).any():
            ax.pcolormesh(cat.lon, cat.lat, field, cmap=om.shade_cmap(colour), vmin=0, vmax=1,
                          shading="nearest", transform=om.pc, zorder=2, rasterized=True)


def strength_colourbars(fig, x0, y0, width, height):
    """One thin bar per category (white -> its colour for |r| = 0 -> 1), in legend order."""
    cols = colours_in_order()
    k = len(cols)
    w = width / (k + 0.35 * (k - 1))
    for i, colour in enumerate(cols):
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=mcolors.Normalize(0, 1),
                                                cmap=om.shade_cmap(colour)),
                          cax=fig.add_axes([x0 + i * 1.35 * w, y0, w, height]),
                          orientation="vertical",
                          ticks=np.linspace(0, 1, 6) if i == k - 1 else [])
        cb.outline.set_linewidth(0.5)
        if i == k - 1:
            cb.ax.tick_params(labelsize=12)
            cb.set_label("Correlation strength |r(PV′, ω′)|\nlight = weak, full colour = strong",
                         fontsize=13)


def make_figure(res, years, path):
    W, left, right, gap = 20.0, 1.3, 3.3, 0.9
    mw = (W - left - right - gap) / 2
    mh = mw * (e.lat_n - e.lat_s) / (e.lon_e - e.lon_w)
    tops = [1.9, 1.9 + mh + 0.8]
    H = tops[1] + mh + 2.6
    fig = plt.figure(figsize=(W, H))
    rnorm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    for col, n in enumerate(names):
        r = res[n]
        x0 = (left + col * (mw + gap)) / W
        fig.text(x0 + mw / W / 2, 1 - (tops[0] - 0.4) / H, n, ha="center", va="bottom",
                 fontsize=26, fontweight="bold")
        for row, top in enumerate(tops):
            ax = fig.add_axes([x0, 1 - (top + mh) / H, mw / W, mh / H], projection=om.pc)
            e.geo(ax, (col == 0, True))
            if row == 0:
                e.mesh(ax, r["r"], rnorm, "RdBu_r")
                e.stipple(ax, r["r"], r["sig"])
            else:
                # category colour (omega regime x PV direction), shaded by |r|; same dots
                draw_categories(ax, r["cat"], np.abs(r["r"].values))
                e.stipple(ax, r["r"], r["sig"])
            ax.text(0.01, 0.02, f"{100 * r['pct_sig']:.0f}% of area: r significant",
                    transform=ax.transAxes, fontsize=10, zorder=9,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2))
            if col == 0:
                ax.text(-0.11, 0.5, ["Detrended r(PV′, ω′)", "Trend classification"][row],
                        transform=ax.transAxes, rotation=90, ha="center", va="center",
                        fontsize=15, fontweight="bold")
    xc = (W - right + 0.35) / W
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=rnorm, cmap="RdBu_r"),
                      cax=fig.add_axes([xc, 1 - (tops[0] + mh - 0.3) / H, 0.22 / W,
                                        (mh - 0.6) / H]))
    cb.set_label("Correlation", fontsize=13)
    cb.ax.tick_params(labelsize=11)
    strength_colourbars(fig, xc, 1 - (tops[1] + mh - 0.3) / H, 1.75 / W, (mh - 0.6) / H)
    handles = [mpatches.Patch(facecolor=c, label=l.replace(", ", " × "))
               for c, l in zip(colours_in_order(), category_labels())]
    handles += [Line2D([], [], marker="o", linestyle="none", markersize=6, color="black",
                       label=f"r significant (BH-FDR q < {e.alpha_fdr}, AR(1)-corrected)"),
                mpatches.Patch(facecolor="white", edgecolor="0.5", label="No data")]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=11.5, frameon=True,
               bbox_to_anchor=(0.5, 1 - (tops[1] + mh + 0.55) / H), handlelength=2.2,
               handleheight=1.5, columnspacing=1.2)
    fig.suptitle(f"October {e.level} hPa PV and vertical velocity ({years[0]}-{years[-1]}): "
                 "year-to-year coupling and long-term change", fontsize=19, fontweight="bold",
                 y=1 - 0.45 / H, va="center")
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"  Saved {path}", flush=True)


def summary(res):
    lab, rows = category_labels(), []
    for n, r in res.items():
        w = np.cos(np.deg2rad(r["cat"].lat)).broadcast_like(r["cat"]).values
        cat, sig, rr = r["cat"].values, r["sig"], r["r"].values
        valid = np.isfinite(cat)
        for k, l in enumerate(lab):
            m = cat == k
            rows.append({"dataset": n, "category": l,
                         "pct_area_all": 100 * w[m].sum() / w[valid].sum(),
                         "pct_area_sig_r": 100 * w[m & sig].sum() / w[valid].sum(),
                         "mean_r": float(np.average(rr[m], weights=w[m])) if m.any() else np.nan,
                         "mean_r_where_sig": (float(np.average(rr[m & sig], weights=w[m & sig]))
                                              if (m & sig).any() else np.nan)})
    return pd.DataFrame(rows)


def main():
    out_dir.mkdir(parents=True, exist_ok=True)
    e.configure_readers()
    omega = e.load_omega()
    years = sorted(set.intersection(*(set(d.year.values.tolist()) for d in omega.values())))
    pv = e.load_pv(years)
    res = {}
    for n in names:
        b, r, p = e.regress_pv_on_omega(pv[n], omega[n])
        sig = e.bh(p.values)
        cat, regime, w_slope = categories(omega[n], pv[n])
        w = np.cos(np.deg2rad(r.lat)).broadcast_like(r).values
        tested = np.isfinite(p.values)
        res[n] = {"r": r, "sig": sig, "cat": cat, "regime": regime.where(cat.notnull(), -1),
                  "w_slope": w_slope, "pv_up": (cat % 2 == 0) & cat.notnull(),
                  "pct_sig": (w * sig).sum() / max(w[tested].sum(), 1e-12)}
    table = summary(res)
    path = out_dir / f"pv_omega_correlation_categories_oct_{e.level}hPa.csv"
    table.to_csv(path, index=False)
    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        print(table.round(2).to_string(index=False))
    make_figure(res, years, out_dir / f"pv_omega_correlation_categories_oct_{e.level}hPa.png")


if __name__ == "__main__":
    main()