"""
Monthly version of 3_row_detrended_corr.py: the same figure (detrended r(omega300, CF),
CF decreasing, CF increasing) for each calendar month, for ERA5 and MERRA-2.
Keep this file next to 3_row_detrended_corr.py (it loads it; name set in BASE_SCRIPT).

USAGE
  python monthly_detrended_corr_3_row.py                 all 12 months, both datasets
  python monthly_detrended_corr_3_row.py 1 2 3           only these months
  python monthly_detrended_corr_3_row.py 7 --era5        one dataset (--era5 / --merra2)
  As a SLURM array (run_omega_cf_monthly.sh, --array=1-12) each task does the month
  SLURM_ARRAY_TASK_ID.

OUTPUT  <OUT_DIR>/omega300hPa_cf_crosssections_10to40E_<DATASET>_<MM>_<Mon>.png
"""
from __future__ import annotations
import calendar
import importlib.util
import os
import sys
from pathlib import Path

# the single-figure script this one reuses (a file name starting with a digit cannot be
# imported the normal way, so it is loaded from its path)
BASE_SCRIPT = "3_row_detrended_corr.py"


def load_base():
    here = Path(__file__).resolve().parent
    for name in (BASE_SCRIPT, "omega_cf_correlation_300hPa.py"):
        path = here / name
        if path.exists():
            spec = importlib.util.spec_from_file_location("base_script", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise SystemExit(f"cannot find {BASE_SCRIPT} next to {Path(__file__).name}")


base = load_base()

OUT_DIR = Path("/soge-home/users/jesu4127/Paper_1/circulation/monthly")


def parse_args(argv):
    months = [int(a) for a in argv if a.isdigit()]
    keys = [a.lstrip("-").lower().replace("-", "") for a in argv if a.startswith("--")]
    if not months and os.environ.get("SLURM_ARRAY_TASK_ID", "").isdigit():
        months = [int(os.environ["SLURM_ARRAY_TASK_ID"])]
    months = months or list(range(1, 13))
    keys = keys or list(base.DATASETS)
    bad = [m for m in months if not 1 <= m <= 12] + [k for k in keys if k not in base.DATASETS]
    if bad:
        raise SystemExit(f"bad argument(s) {bad}: months 1-12, datasets --era5 / --merra2")
    return months, keys


def main():
    months, keys = parse_args(sys.argv[1:])
    base.OUT_DIR = OUT_DIR
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    failed = []
    for m in months:
        base.MONTH = m                          # every reader and the title use this
        for key in keys:
            try:
                base.run(key)
            except Exception as e:
                base.log(f"[ERROR] {key} {calendar.month_abbr[m]}: {type(e).__name__}: {e}")
                failed.append(f"{key} {calendar.month_abbr[m]}")
    base.log("Done." + (f" Failed: {', '.join(failed)}" if failed else ""))


if __name__ == "__main__":
    main()