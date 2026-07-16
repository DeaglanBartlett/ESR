"""Generic file/IO and environment helpers shared across the fitting pipeline.

These utilities are not specific to any likelihood or catalogue: they handle
natural-sort ordering, concatenating rank-local temporary files, writing the
negative-log-likelihood output table, and raising Python's recursion limit for
deep expression trees. They live here (rather than in ``test_all``) so that
``test_all``, ``test_all_Fisher``, ``match`` and downstream code can reuse them
without importing the fitting driver. ``test_all`` re-exports them for backwards
compatibility, so existing ``test_all.combine_temp_files``-style calls continue
to work.
"""
import glob
import os
import re
import sys
import warnings

import numpy as np

# Re-exported so fitting code can write catalogue/output files atomically
# without importing the generation layer directly.
from esr.generation.utils import atomic_write  # noqa: F401


def emit_diagnostic_warning(message, category):
    """Emit an ESR diagnostic warning through the normal warnings machinery.

    The fitting modules (``test_all``, ``test_all_Fisher``, ``match``, ``plot``)
    narrow their module-level suppression to ``RuntimeWarning`` -- the bulk
    numpy/scipy fitting noise -- so a diagnostic raised as a dedicated
    ``UserWarning`` subclass is not swallowed by it. This deliberately does NOT
    force ``always``: it honours an explicit caller ``filterwarnings('ignore')``
    and uses Python's default per-location de-duplication, so a per-function
    diagnostic (whose message is kept constant for exactly this reason) is shown
    once rather than flooding a large MPI log.

    Args:
        :message (str): the warning message (keep constant across calls from the
            same site so it de-duplicates)
        :category (type): the warning category (a ``Warning`` subclass) to raise

    Returns:
        None
    """
    warnings.warn(message, category, stacklevel=2)


def natural_sort_key(path):
    """Sort key ordering embedded integers numerically (so rank_2 precedes rank_10).

    Args:
        :path (str): the string to derive a sort key from

    Returns:
        :key (list): list of interleaved string and integer chunks
    """
    return [int(text) if text.isdigit() else text
            for text in re.split(r'(\d+)', path)]


def combine_temp_files(temp_dir, pattern, output_file, remove=True):
    """Concatenate rank-local temporary files in natural-sort order.

    Args:
        :temp_dir (str): directory holding the per-rank temporary files
        :pattern (str): glob pattern (relative to ``temp_dir``) selecting the
            files to concatenate
        :output_file (str): path of the combined file to write
        :remove (bool, default=True): whether to delete the temporary files once
            they have been concatenated

    Returns:
        None
    """
    paths = sorted(glob.glob(os.path.join(temp_dir, pattern)),
                   key=natural_sort_key)
    with open(output_file, 'w') as fout:
        for path in paths:
            with open(path, 'r') as fin:
                fout.writelines(fin)
    if remove:
        for path in paths:
            os.remove(path)


def write_negloglike_file(path, chi2, params, max_param):
    """Write the negative-log-likelihood table (chi2 followed by parameters).

    Args:
        :path (str): output file path
        :chi2 (np.ndarray): negative log-likelihood per function, shape (nfun,)
        :params (np.ndarray): best-fit parameters per function, shape
            (nfun, >=max_param)
        :max_param (int): number of parameter columns to write

    Returns:
        None
    """
    out_arr = np.transpose(
        np.vstack([chi2] + [params[:, i] for i in range(max_param)]))
    np.savetxt(path, out_arr, fmt='%.7e')


def set_recursionlimit_for_comp(comp):
    """Raise Python's recursion limit for the deep trees seen at high complexity.

    The limit is only raised for ``comp >= 8``; below that the default is
    sufficient and the function is a no-op.

    Args:
        :comp (int): complexity of the functions being processed

    Returns:
        None
    """
    if comp >= 8:
        sys.setrecursionlimit(2000 + 500 * (comp - 8))
