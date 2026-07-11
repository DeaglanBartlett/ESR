#!/usr/bin/env python3
"""Time ESR's serial numerical-duplicate fingerprint diagnostic.

Run this against an already-generated ``unique_equations_<complexity>.txt``
catalogue. It deliberately does not generate or simplify equations: the
reported time is only for the opt-in fingerprint diagnostic.
"""

import argparse
from pathlib import Path
import resource
import sys
import time
import types


def _install_serial_mpi_stub():
    """Avoid initializing Open MPI for this deliberately serial benchmark.

    The fingerprint functions do not communicate; ESR imports mpi4py only to
    obtain rank/size for progress messages.  On Glamdring, the queue wrapper's
    ``srun`` environment is incompatible with the Open MPI used by mpi4py, so
    provide the required rank-0, size-1 interface before importing ESR.
    """
    class SerialComm:
        @staticmethod
        def Get_rank():
            return 0

        @staticmethod
        def Get_size():
            return 1

    mpi4py = types.ModuleType("mpi4py")
    mpi4py.MPI = types.SimpleNamespace(COMM_WORLD=SerialComm())
    sys.modules["mpi4py"] = mpi4py


_install_serial_mpi_stub()

from esr.generation import simplifier  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalogue", type=Path,
                        help="final unique-equation catalogue to fingerprint")
    parser.add_argument("--max-param", type=int, default=None,
                        help="optional maximum parameter count")
    args = parser.parse_args()

    catalogue = args.catalogue.resolve()
    with catalogue.open() as f:
        equations = f.read().splitlines()
    max_param = (args.max_param if args.max_param is not None
                 else simplifier.get_max_param(equations, verbose=False))

    start = time.perf_counter()
    candidates = simplifier.numerical_duplicate_candidates(
        equations, max_param=max_param, verbose=True)
    elapsed = time.perf_counter() - start
    max_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    flagged = sum(len(indexes) - 1 for _, indexes in candidates)

    print(
        "FINGERPRINT_BENCHMARK "
        f"catalogue={catalogue} equations={len(equations)} "
        f"max_param={max_param} groups={len(candidates)} "
        f"flagged={flagged} elapsed_s={elapsed:.3f} "
        f"max_rss_kib={max_rss_kib}",
        flush=True,
    )


if __name__ == "__main__":
    main()
