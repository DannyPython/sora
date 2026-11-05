"""Draw the M0 vs M1 comparison figures.

    python plot_results.py                    # all figures -> results/figures/ (PNG and PDF)
    python plot_results.py --fast             # fewer bootstrap draws, for checking the code (about a minute)
    python plot_results.py --out my_folder --formats png

The figures are recomputed from the data with the same code and seeds as run_m0.py and run_m1.py.
"""
import argparse

from ssef import plots

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--out", default="results/figures")
ap.add_argument("--fast", action="store_true", help="fewer bootstrap draws (same figures, slightly noisier p-values)")
ap.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf", "svg"])
ap.add_argument("--dpi", type=int, default=200)
ap.add_argument("--seed", type=int, default=2027)
a = ap.parse_args()

files = plots.make_all(a.out, a.fast, a.formats, a.seed, a.dpi)
print(f"\n{len(files)} files written to {a.out}/")
