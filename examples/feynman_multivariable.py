"""
Recover multivariable laws from the AI Feynman dataset with ESR.

For each equation, mock data are drawn with every input uniform in [1, 5]
(200 points, 1% Gaussian noise). ESR then fits every function in the
catalogue for the equation's variables up to a maximum complexity. The
function with the lowest description length should be the true law, found at
its expected complexity.

Catalogues are generated on demand into ``esr/function_library/<set>_<vars>``
and reused on later runs. Outputs go to ``./feynman_multivariable/<id>``.

Usage::

    mpirun -n 4 python feynman_multivariable.py            # quick set (~2 min)
    mpirun -n 4 python feynman_multivariable.py --all      # all (~1 hour)
    mpirun -n 4 python feynman_multivariable.py I.14.4 II.8.31

The quick set needs complexity at most 6 with core_maths. The others need
complexity 7 with core_maths or complexity 6 with ext_maths, which take
several minutes each to fit.
"""

import argparse
import os
import time
import zlib

import numpy as np
from mpi4py import MPI

import esr
import esr.fitting.combine_dl
import esr.fitting.match
import esr.fitting.test_all
import esr.fitting.test_all_fisher
import esr.generation.duplicate_checker
from esr.fitting.likelihood import GaussLikelihood
from esr.fitting.sympy_symbols import function_set_dirname

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# id: (law, function set, variables, f(*inputs), expected best, expected complexity,
#      maximum complexity fitted)
EQUATIONS = {
    "I.12.1": ("F = mu N", "core_maths", ["x", "y"],
               lambda a, b: a * b, "x*y", 3, 4),
    "I.12.1-names": ("F = mu N", "core_maths", ["mu", "Nn"],
                     lambda a, b: a * b, "Nn*mu", 3, 4),
    "I.25.13": ("V = q / C", "core_maths", ["x", "y"],
                lambda a, b: a / b, "x/y", 3, 4),
    "I.39.1": ("E = 3/2 p V", "core_maths", ["x", "y"],
               lambda a, b: 1.5 * a * b, "a0*x*y", 5, 6),
    "I.34.27": ("E = (h / 2 pi) omega", "core_maths", ["x", "y"],
                lambda a, b: a * b / (2 * np.pi), "a0*x*y", 5, 6),
    "I.43.31": ("D = mu k_B T", "core_maths", ["x", "y", "z"],
                lambda a, b, c: a * b * c, "x*y*z", 5, 5),
    "I.14.4": ("U = k x^2 / 2", "core_maths", ["x", "y"],
               lambda a, b: 0.5 * a * b**2, "a0*x*y**2", 7, 7),
    "II.38.14": ("mu_S = Y / (2 (1 + sigma))", "core_maths", ["x", "y"],
                 lambda a, b: a / (2 * (1 + b)), "x/(a0 + 2*y)", 7, 7),
    "II.3.24": ("flux = P / (4 pi r^2)", "ext_maths", ["x", "y"],
                lambda a, b: a / (4 * np.pi * b**2), "a0*x/y**2", 6, 6),
    "II.8.31": ("E_den = eps E^2 / 2", "ext_maths", ["x", "y"],
                lambda a, b: a * b**2 / 2, "a0*x*y**2", 6, 6),
    "I.47.23": ("c = sqrt(gamma p / rho)", "ext_maths", ["x", "y", "z"],
                lambda a, b, c: np.sqrt(a * b / c), "sqrt(x)*sqrt(y)/sqrt(z)", 6, 6),
}

QUICK = ["I.12.1", "I.12.1-names", "I.25.13", "I.39.1", "I.34.27", "I.43.31"]

OUT_ROOT = os.path.join(os.getcwd(), "feynman_multivariable")


def build_catalogues(eq_ids):
    """Generate each catalogue needed by eq_ids, skipping complexities already built."""
    needed = {}
    for eq in eq_ids:
        _, fn_set, variables, _, _, _, max_comp = EQUATIONS[eq]
        key = (fn_set, tuple(variables))
        needed[key] = max(needed.get(key, 0), max_comp)

    lib = os.path.join(os.path.dirname(esr.__file__), "function_library")
    for (fn_set, variables), max_comp in needed.items():
        dirname = function_set_dirname(fn_set, variables)
        for comp in range(1, max_comp + 1):
            fname = os.path.join(
                lib, dirname, f"compl_{comp}", f"unique_equations_{comp}.txt"
            )
            if os.path.exists(fname):
                continue
            t0 = time.time()
            esr.generation.duplicate_checker.main(fn_set, comp, variables=list(variables))
            if rank == 0:
                print(f"Built {dirname} complexity {comp} in {time.time() - t0:.0f} s",
                      flush=True)


def make_data(eq, out_dir):
    _, _, variables, f, _, _, _ = EQUATIONS[eq]
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        rng = np.random.default_rng(zlib.crc32(eq.encode()))
        inputs = [rng.uniform(1, 5, 200) for _ in variables]
        y = f(*inputs)
        err = 0.01 * np.abs(y)
        noisy = y + err * rng.standard_normal(len(y))
        # Columns: one per variable, then y, then the error on y
        np.savetxt(os.path.join(out_dir, "data.txt"),
                   np.column_stack(inputs + [noisy, err]))
    comm.Barrier()


def fit(eq):
    """Fit every function up to the maximum complexity and return the best one."""
    _, _, variables, _, _, _, max_comp = EQUATIONS[eq]
    out_dir = os.path.join(OUT_ROOT, eq)
    make_data(eq, out_dir)
    likelihood = GaussLikelihood(
        "data.txt", "fit", data_dir=out_dir, base_out_dir=out_dir, variables=variables
    )

    t0 = time.time()
    for comp in range(1, max_comp + 1):
        esr.fitting.test_all.main(comp, likelihood)
        esr.fitting.test_all_fisher.main(comp, likelihood)
        esr.fitting.match.main(comp, likelihood)
        esr.fitting.combine_dl.main(comp, likelihood)

    best = None
    if rank == 0:
        for comp in range(1, max_comp + 1):
            fname = os.path.join(likelihood.out_dir, f"final_{comp}.dat")
            with open(fname) as fh:
                row = fh.readline().split(";")
            DL = float(row[2])
            if best is None or DL < best["DL"]:
                best = {"comp": comp, "fn": row[1], "DL": DL,
                        "params": [float(p) for p in row[7:] if float(p) != 0]}
        best["time"] = time.time() - t0
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("equations", nargs="*",
                        help="equation ids to run (default: the quick set): "
                        + ", ".join(EQUATIONS))
    parser.add_argument("--all", action="store_true", help="run every equation")
    args = parser.parse_args()
    eq_ids = list(EQUATIONS) if args.all else (args.equations or QUICK)
    unknown = [eq for eq in eq_ids if eq not in EQUATIONS]
    if unknown:
        parser.error(f"unknown equation id(s): {', '.join(unknown)}")

    build_catalogues(eq_ids)
    results = {eq: fit(eq) for eq in eq_ids}

    if rank == 0:
        header = (f"{'ID':<13} {'law':<27} {'best function':<24} "
                  f"{'comp':>4} {'params':<16} {'time':>6}  ok")
        print("\n" + header + "\n" + "-" * len(header))
        n_ok = 0
        for eq, best in results.items():
            law, _, _, _, expected, exp_comp, _ = EQUATIONS[eq]
            ok = best["fn"] == expected and best["comp"] == exp_comp
            n_ok += ok
            params = ", ".join(f"{p:.5g}" for p in best["params"]) or "-"
            print(f"{eq:<13} {law:<27} {best['fn']:<24} {best['comp']:>4} "
                  f"{params:<16} {best['time']:>5.0f}s  {'yes' if ok else 'NO'}")
        print(f"\nRecovered {n_ok} of {len(results)} laws at the expected complexity.")


if __name__ == "__main__":
    main()
