"""Generic file/IO and environment helpers shared across the fitting pipeline.

These utilities are shared by the fitting stages: they define the common file
paths, handle natural-sort ordering and rank-local file concatenation, write the
negative-log-likelihood output table, and raise Python's recursion limit for deep
expression trees. They live here (rather than in ``test_all``) so that
``test_all``, ``test_all_Fisher``, ``match`` and downstream code can reuse them
without importing the fitting driver. ``test_all`` re-exports the established
helpers for backwards compatibility, while ESR's own modules import them from
this module directly.
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


def raw_catalogue_paths(comp, likelihood):
    """Return paths to the raw generated catalogue for one complexity.

    Args:
        :comp (int): complexity of functions to consider
        :likelihood (fitting.likelihood object): object providing ``fn_dir`` and
            optionally ``fnprior_prefix``

    Returns:
        :paths (dict): paths for the all/unique equation catalogues, matching and
            inverse-substitution files, lower-complexity exclusions and function
            prior
    """
    base = os.path.join(likelihood.fn_dir, f'compl_{comp}')
    fnprior_prefix = getattr(likelihood, 'fnprior_prefix', 'aifeyn_')
    return {
        'all': os.path.join(base, f'all_equations_{comp}.txt'),
        'unique': os.path.join(base, f'unique_equations_{comp}.txt'),
        'matches': os.path.join(base, f'matches_{comp}.txt'),
        'previous': os.path.join(base, f'previous_eqns_{comp}.txt'),
        'inv_subs': os.path.join(base, f'inv_subs_{comp}.txt'),
        'fnprior': os.path.join(base, f'{fnprior_prefix}{comp}.txt'),
    }


def likelihood_catalogue_paths(comp, likelihood):
    """Return paths to the transformed likelihood-aware catalogue.

    Args:
        :comp (int): complexity of functions to consider
        :likelihood (fitting.likelihood object): object providing ``out_dir``

    Returns:
        :paths (dict): paths for the transformed unique equations, match indices
            and cache metadata
    """
    prefix = os.path.join(
        likelihood.out_dir, f'likelihood_catalogue_comp{comp}')
    return {
        'unique': prefix + '_unique_equations.txt',
        'matches': prefix + '_matches.txt',
        'metadata': prefix + '_metadata.json',
    }


def fitting_paths(comp, likelihood, rank=None):
    """Return shared fitting input, temporary and output paths.

    This is the single filename definition used by the fitting, Fisher, matching,
    combination and plotting stages. Rank-local entries are included when
    ``rank`` is supplied; glob patterns are relative to ``likelihood.temp_dir``
    for direct use with :func:`combine_temp_files`.

    Args:
        :comp (int): complexity of functions to consider
        :likelihood (fitting.likelihood object): object providing ``out_dir`` and
            ``temp_dir`` and optionally the combination/final filename prefixes
        :rank (int or None): MPI rank for rank-local files

    Returns:
        :paths (dict): semantic mapping of fitting filenames and glob patterns
    """
    combine_prefix = getattr(likelihood, 'combineDL_prefix', 'combine_DL_')
    final_prefix = getattr(likelihood, 'final_prefix', 'final_')
    out_dir = likelihood.out_dir
    temp_dir = getattr(likelihood, 'temp_dir', None)

    paths = {
        'negloglike': os.path.join(out_dir, f'negloglike_comp{comp}.dat'),
        'negloglike_checkpoint': os.path.join(
            out_dir, f'negloglike_comp{comp}.checkpoint.dat'),
        'fisher_settings': os.path.join(
            out_dir, f'fisher_settings_comp{comp}.json'),
        'codelen': os.path.join(out_dir, f'codelen_comp{comp}_deriv.dat'),
        'derivs': os.path.join(out_dir, f'derivs_comp{comp}.dat'),
        'codelen_matches': os.path.join(
            out_dir, f'codelen_matches_comp{comp}.dat'),
        'combined': os.path.join(
            out_dir, f'{combine_prefix}comp{comp}.dat'),
        'combined_functions': os.path.join(
            out_dir, f'{combine_prefix}fcn_comp{comp}.dat'),
        'final': os.path.join(out_dir, f'{final_prefix}{comp}.dat'),
        'results_pretty': os.path.join(
            out_dir, f'results_pretty_{comp}.txt'),
        'negloglike_rank_pattern': f'chi2_comp{comp}weights_*.dat',
        'codelen_rank_pattern': f'codelen_deriv_{comp}_*.dat',
        'derivs_rank_pattern': f'derivs_{comp}_*.dat',
        'codelen_matches_rank_pattern': f'codelen_matches_{comp}_*.dat',
        'combined_rank_pattern': f'{combine_prefix}{comp}_*.dat',
        'combined_functions_rank_pattern': (
            f'{combine_prefix}fcn_{comp}_*.dat'),
    }
    if rank is not None:
        if temp_dir is None:
            raise AttributeError(
                'likelihood.temp_dir is required for rank-local fitting paths')
        paths.update({
            'negloglike_rank': os.path.join(
                temp_dir, f'chi2_comp{comp}weights_{rank}.dat'),
            'codelen_rank': os.path.join(
                temp_dir, f'codelen_deriv_{comp}_{rank}.dat'),
            'derivs_rank': os.path.join(
                temp_dir, f'derivs_{comp}_{rank}.dat'),
            'codelen_matches_rank': os.path.join(
                temp_dir, f'codelen_matches_{comp}_{rank}.dat'),
            'combined_rank': os.path.join(
                temp_dir, f'{combine_prefix}{comp}_{rank}.dat'),
            'combined_functions_rank': os.path.join(
                temp_dir, f'{combine_prefix}fcn_{comp}_{rank}.dat'),
        })
    return paths


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
