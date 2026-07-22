import numpy as np
import sympy
import warnings
import os
import json
import hashlib
from mpi4py import MPI
from scipy.optimize import minimize
import itertools

from esr.fitting.sympy_symbols import x, a0
from esr.fitting.utils import (
    atomic_write, combine_temp_files, emit_diagnostic_warning,
    fitting_paths, likelihood_catalogue_paths, raw_catalogue_paths,
    set_recursionlimit_for_comp, write_negloglike_file)
import esr.generation.simplifier as simplifier

# Suppress the numpy/scipy RuntimeWarnings (overflow, invalid value, divide by
# zero) raised in bulk while evaluating candidate functions, but leave other
# categories -- including unrelated user warnings -- untouched. Our own
# diagnostics are emitted via utils.emit_diagnostic_warning so they survive
# even this narrowed filter.
warnings.filterwarnings("ignore", category=RuntimeWarning)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class LikelihoodCatalogueWarning(UserWarning):
    """Diagnostic about the likelihood-aware catalogue build (transformation
    failures, or a transforming likelihood with no catalogue_transform_version).

    A ``UserWarning`` subclass so it is not swallowed by the module-level
    ``filterwarnings('ignore', category=RuntimeWarning)``.
    """


# MPI message tags for the dynamic (rank-0 dispatcher) scheduler in
# ``_main_dynamic``. Rank 0 hands out one equation at a time and gathers the
# fitted results, so every message is labelled by its role:
#   WORK_TAG   -- rank 0 -> worker: the next (index, function string) to fit
#   RESULT_TAG -- worker -> rank 0: the (index, chi2, params) of a finished fit
#   STOP_TAG   -- rank 0 -> worker: no work remains; leave the receive loop
WORK_TAG = 11
RESULT_TAG = 12
STOP_TAG = 13


def chi2_fcn(x, likelihood, eq_numpy, integrated, signs):
    """Compute chi2 for a function

    Args:
        :x (list): parameters to use for function
        :likelihood (fitting.likelihood object): object containing data and likelihood function
        :eq_numpy (numpy function): function to pass to likelihood object to make prediction of y(x)
        :integrated (bool): whether eq_numpy has already been integrated
        :signs (list): each entry specifies whether than parameter should be optimised logarithmically. If None, then do nothing, if '+' then optimise 10**x[i] and if '-' then optimise -10**x[i]

    Returns:
        :negloglike (float): - log(likelihood) for this function and parameters

    """
    if signs is None:
        p = x
    else:
        p = [None] * len(signs)
        for i in range(len(signs)):
            if signs[i] is None:
                p[i] = x[i]
            elif signs[i] == '+':
                p[i] = 10 ** x[i]
            elif signs[i] == '-':
                p[i] = - 10 ** x[i]
            else:
                raise ValueError
    return likelihood.negloglike(p, eq_numpy, integrated=integrated)


def ensure_output_dirs(likelihood):
    if rank == 0:
        for dirname in [likelihood.base_out_dir, likelihood.out_dir, likelihood.temp_dir]:
            if not os.path.exists(dirname):
                print('Making dir:', dirname)
            os.makedirs(dirname, exist_ok=True)
    comm.Barrier()


def function_catalogue_path(comp, likelihood, unique=True):
    """Path of the equation catalogue to load for fitting/matching.

    Returns the active likelihood-aware unique catalogue when one exists,
    otherwise the raw generated catalogue.

    Args:
        :comp (int): complexity of functions to consider
        :likelihood (fitting.likelihood object): provides the catalogue paths
        :unique (bool, default=True): return the unique catalogue (True) or the
            full all-equations catalogue (False)

    Returns:
        :path (str): path of the selected catalogue file
    """
    if unique and likelihood_catalogue_active(comp, likelihood):
        return likelihood_catalogue_paths(comp, likelihood)['unique']
    raw = raw_catalogue_paths(comp, likelihood)
    return raw['unique'] if unique else raw['all']


def _likelihood_catalogue_settings(tmax, try_integration,
                                   all_equations_hash=None,
                                   transform_version=None,
                                   transform_fingerprint=None):
    """Settings fingerprint stored alongside a cached likelihood catalogue.

    The returned dict is written into the catalogue metadata and compared on
    subsequent runs: if it differs from the stored copy, the cached catalogue
    is discarded and rebuilt (see ``ensure_likelihood_catalogue``).

    The fingerprint covers the build inputs so a cache is not silently reused
    when they change:

      * ``cache_schema_version`` -- an internal tag for the on-disk cache format
        (not a user-facing version). It is bumped whenever the catalogue layout
        or the meaning of these settings changes, which forces any catalogue
        written under an older layout to be regenerated. There is no accessible
        "older version" to fall back to: a mismatch simply triggers a rebuild.
      * ``all_equations_hash`` -- a hash of the raw ``all_equations`` file, so
        adding, removing or editing an equation invalidates the cache (otherwise
        a stale matches file could associate equations with the wrong
        representative).
      * ``transform_fingerprint`` -- a hash of the ``run_sympify`` transform
        applied to a few fixed probe expressions (see ``_transform_fingerprint``).
        This is a *best-effort heuristic*: it reliably distinguishes broad
        transform changes (e.g. a normalising likelihood swapped for an identity
        one on the same equations and output directory), but it can miss a
        transform change that only affects expressions unlike the probes. It is
        not a complete guarantee.
      * ``transform_version`` -- an optional likelihood-supplied version (read
        from ``likelihood.catalogue_transform_version``). Set/bump this for
        guaranteed invalidation on any transform change; when a transforming
        likelihood provides none, ``ensure_likelihood_catalogue`` warns because
        the probe fingerprint alone cannot guarantee the cache is fresh.

    Args:
        :tmax (float): per-expression simplification timeout used when building
            the catalogue
        :try_integration (bool): whether analytic integration was attempted
            while transforming expressions
        :all_equations_hash (str or None): content hash of the raw
            ``all_equations`` file the catalogue was built from
        :transform_version: optional JSON-serialisable version tag for the
            likelihood transformation
        :transform_fingerprint (str or None): probe-based hash of the transform
            behaviour

    Returns:
        :settings (dict): JSON-serialisable settings fingerprint
    """
    settings = {
        'cache_schema_version': 0,
        'tmax': float(tmax),
        'try_integration': bool(try_integration),
        'all_equations_hash': all_equations_hash,
        'transform_version': transform_version,
        'transform_fingerprint': transform_fingerprint,
    }
    # Normalise through JSON so the fresh settings compare equal to the stored
    # (JSON-loaded) copy: a tuple ``transform_version`` would otherwise become a
    # list on load and force perpetual rebuilds. Non-serialisable values fall
    # back to their str() form.
    return json.loads(json.dumps(settings, default=str))


def _hash_file(path):
    """Return a hex digest of a file's contents.

    Streamed in chunks so that fingerprinting a large ``all_equations`` file
    does not require loading the whole catalogue into memory.

    Args:
        :path (str): path of the file to hash

    Returns:
        :digest (str): hex SHA-1 digest of the file bytes
    """
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


_TRANSFORM_PROBES = (
    'a0', 'a0*x', 'a0 + a1*x', 'a0*x + a1*pow(x, 2)',
    'a0 + a1*pow(x, 3)', 'a0*pow(x, 4) + a1', 'a0/(a1 + x)',
    'a0 + a1*x + a2*pow(x, 2)', 'a0*inv(x) + a1')


def _transform_fingerprint(likelihood, tmax, try_integration):
    """Fingerprint a likelihood's ``run_sympify`` transform.

    Applies the transform to a few fixed probe expressions and hashes the
    canonical outputs. Two different transforms (for example the identity parse
    versus a normalising divide ``g -> g/g(1)``) produce different fingerprints,
    so the catalogue cache is invalidated when the transform changes even if the
    equation set and ``tmax``/``try_integration`` are unchanged. Combined in the
    settings fingerprint with any explicit ``catalogue_transform_version``.

    Args:
        :likelihood (fitting.likelihood object): provides ``run_sympify``
        :tmax (float): per-expression simplification timeout
        :try_integration (bool): whether to attempt analytic integration

    Returns:
        :fingerprint (str): hex SHA-1 digest identifying the transform behaviour
    """
    h = hashlib.sha1()
    for probe in _TRANSFORM_PROBES:
        try:
            _, eq, integrated = likelihood.run_sympify(
                probe, tmax=tmax, try_integration=try_integration)
            token = '%s:%s' % (bool(integrated), sympy.srepr(eq))
        except Exception as exc:
            token = 'ERR:%s' % type(exc).__name__
        h.update(token.encode('utf-8'))
        h.update(b'\x00')
    return h.hexdigest()


def _read_likelihood_catalogue_metadata(comp, likelihood):
    """Load the cached catalogue metadata, or None if it has not been written.

    Args:
        :comp (int): complexity of functions to consider
        :likelihood (fitting.likelihood object): object providing ``out_dir``

    Returns:
        :metadata (dict or None): the parsed metadata, or None if absent
    """
    paths = likelihood_catalogue_paths(comp, likelihood)
    try:
        with open(paths['metadata'], 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def likelihood_catalogue_active(comp, likelihood):
    """Whether an active likelihood-aware catalogue exists for this complexity.

    Args:
        :comp (int): complexity of functions to consider
        :likelihood (fitting.likelihood object): object providing ``out_dir``

    Returns:
        :active (bool): True if the metadata marks the catalogue active and both
            the unique-representative and matches files are present
    """
    metadata = _read_likelihood_catalogue_metadata(comp, likelihood)
    if metadata is None or not metadata.get('active', False):
        return False
    paths = likelihood_catalogue_paths(comp, likelihood)
    return os.path.exists(paths['unique']) and os.path.exists(paths['matches'])


def canonicalize_parameter_symbols(eq):
    """Rename surviving ESR parameters to a contiguous a0, a1, ... sequence.

    A likelihood transformation can leave a non-contiguous set of parameters
    (e.g. only a1 and a3 survive); this relabels them to a0, a1, ... in index
    order so the fit and Fisher analysis see a clean parameter vector.

    Args:
        :eq (sympy object): expression whose ``a*`` free symbols are relabelled

    Returns:
        :eq (sympy object): the expression with its parameters relabelled
        :active_params (list): the original surviving parameter symbols, sorted
            by index (its length is the number of free parameters)
    """
    active_params = sorted(
        [
            symbol for symbol in eq.free_symbols
            if symbol != x and symbol.name.startswith('a')
            and symbol.name[1:].isdigit()
        ],
        key=lambda symbol: int(symbol.name[1:])
    )
    if len(active_params) == 0:
        return eq, active_params

    canonical = sympy.symbols(
        ' '.join([f'a{i}' for i in range(len(active_params))]), real=True)
    canonical = list(np.atleast_1d(canonical))
    replacements = {
        active_params[i]: canonical[i]
        for i in range(len(active_params))
        if active_params[i] != canonical[i]
    }
    if replacements:
        eq = eq.subs(replacements, simultaneous=True)
    return eq, active_params


def _canonical_transformed_key(fcn_i, likelihood, tmax, try_integration):
    """Canonical key grouping raw expressions by their transformed model.

    Applies the likelihood transformation, canonicalises the parameters, and
    returns a hashable key that is identical for expressions mapping to the same
    model after transformation.

    Args:
        :fcn_i (str): raw expression string
        :likelihood (fitting.likelihood object): provides ``run_sympify``
        :tmax (float): per-expression simplification timeout
        :try_integration (bool): whether to attempt analytic integration

    Returns:
        :key (tuple): ``(integrated, srepr)`` hashable grouping key
        :active_param_names (list): names of the surviving parameters, used to
            detect a parameter-layout change against the raw tree
    """
    fcn_i, eq, integrated = likelihood.run_sympify(
        fcn_i, tmax=tmax, try_integration=try_integration)
    eq, active_params = canonicalize_parameter_symbols(eq)
    try:
        eq_key = sympy.factor(sympy.cancel(eq))
    except Exception:
        eq_key = eq
    return (bool(integrated), sympy.srepr(eq_key)), [
        symbol.name for symbol in active_params
    ]


def _transformed_keys_for_slice(functions, start_index, likelihood, tmax,
                                try_integration, max_param):
    """Canonical transformed keys for a contiguous slice of raw equations.

    This is the per-equation work of the catalogue build, factored out so it can
    be scattered across MPI ranks. It is pure (no communication) and returns,
    for each equation, ``(index, key, changed, failed)`` where ``index`` is the
    global index (``start_index`` + offset), ``key`` is the canonical
    transformed key, ``changed`` flags a parameter-layout change under the
    likelihood transformation, and ``failed`` flags a transformation that timed
    out or errored (given a unique key so it is never merged).

    Args:
        :functions (list): the raw expression strings in this rank's slice
        :start_index (int): global index of the first equation in the slice
        :likelihood (fitting.likelihood object): provides ``run_sympify``
        :tmax (float): per-expression simplification timeout
        :try_integration (bool): whether to attempt analytic integration
        :max_param (int): maximum number of parameters (for counting)

    Returns:
        :results (list): ``(index, key, changed, failed)`` tuples, one per input
    """
    results = []
    for offset, fcn_i in enumerate(functions):
        index = start_index + offset
        expected = [f'a{i}' for i in range(
            simplifier.count_params([fcn_i], max_param)[0])]
        failed = False
        try:
            with simplifier.time_limit(tmax):
                key, layout = _canonical_transformed_key(
                    fcn_i, likelihood, tmax, try_integration)
        except Exception:
            failed = True
            key = ('failed', index, fcn_i)
            layout = expected
        results.append((index, key, layout != expected, failed))
    return results


def ensure_likelihood_catalogue(comp, likelihood, tmax=5, try_integration=False):
    """Build (and cache) a likelihood-aware catalogue of fitted functions.

    Some likelihoods transform each generated expression through
    ``likelihood.run_sympify`` before evaluating it (for example dividing by the
    value at a reference point, which can remove or relabel parameters). When
    that happens two syntactically different raw expressions can collapse to the
    *same* model once transformed, and a single raw expression can end up with a
    different parameter layout than its tree implies. Fitting every raw
    expression separately would then repeat identical fits and, worse, assume
    the wrong number of free parameters.

    This routine groups the raw ``all_equations`` by their canonical transformed
    form (see ``_canonical_transformed_key``) and, on rank 0, writes:

      * ``<prefix>_unique_equations.txt`` -- one representative raw expression
        per transformed family; these are the expressions actually fitted;
      * ``<prefix>_matches.txt`` -- for every raw expression, the index of its
        representative, so fitted results can be mapped back to the full list;
      * ``<prefix>_metadata.json`` -- bookkeeping plus the settings fingerprint
        (``_likelihood_catalogue_settings``) used to decide whether a cached
        catalogue can be reused.

    The catalogue is marked ``active`` when the transformation actually makes a
    difference: either at least one expression's parameter layout changes, or two
    or more distinct raw expressions collapse to the same transformed model
    (fewer representatives than raw uniques). If neither happens ESR falls back to
    the ordinary raw unique-equation catalogue. The result is cached, but a
    rebuild happens whenever the metadata is missing, the settings fingerprint
    differs, the cached build had transform failures, or the settings carry no
    ``catalogue_transform_version`` (a versionless build is never trusted for
    reuse, since a best-effort probe fingerprint cannot prove the transform is
    unchanged).

    Building the catalogue is opt-in via ``likelihood.use_likelihood_catalogue``,
    which defaults to ``False`` (see ``Likelihood.use_likelihood_catalogue``). A
    likelihood object that does not define the attribute at all is likewise
    treated as opted out. The built-in likelihoods leave it ``False`` because
    they do not change the parameter layout, so they skip the build entirely --
    worthwhile because the build is an all-equations transformation pass,
    expensive at high complexity. A custom transforming likelihood must set it
    ``True`` (and should set ``catalogue_transform_version`` so its catalogue can
    be cached across runs).

    The per-equation canonicalisation is the expensive part of the build (a
    run_sympify and a factor/cancel per raw tree). Because ``all_equations`` can
    reach hundreds of thousands of trees at high complexity, that work is
    scattered across the MPI ranks (each canonicalises a contiguous slice) and
    only the final grouping runs on rank 0, which merges the gathered keys in
    global index order so the output is identical to a serial build.

    Args:
        :comp (int): complexity of functions to consider
        :likelihood (fitting.likelihood object): object providing
            ``run_sympify``, ``fn_dir``/``out_dir`` paths and (optionally) the
            ``use_likelihood_catalogue`` flag
        :tmax (float, default=5): maximum time in seconds allowed for the
            simplification/transformation of any one expression
        :try_integration (bool, default=False): when the likelihood requires an
            integral, whether to attempt analytic integration while transforming

    Returns:
        :active (bool): True if a likelihood-aware catalogue was built and is in
            use for this complexity, False if the raw catalogue should be used
    """
    ensure_output_dirs(likelihood)
    paths = likelihood_catalogue_paths(comp, likelihood)
    # Opt-in: a likelihood must set use_likelihood_catalogue = True to build the
    # (potentially expensive) catalogue. A missing attribute defaults to False,
    # matching the explicit Likelihood base-class default, so a duck-typed
    # likelihood is never surprised by an all-equations rebuild.
    #
    # Deliberate choice: the default here (False) is opt-out, so a *duck-typed*
    # transforming likelihood that never sets the attribute will silently skip
    # the catalogue and be fitted with the raw parameter count. We accept this
    # because (a) the catalogue is introduced in this same change, so no released
    # opt-in-by-default behaviour is being broken; (b) every built-in and every
    # Likelihood subclass carries the explicit, documented attribute; and (c) the
    # alternative default (True) would make every duck-typed likelihood pay a
    # silent all-equations transform pass -- expensive at high complexity -- for a
    # feature only custom transforming likelihoods need. The README and tutorial
    # state that a transforming likelihood must set use_likelihood_catalogue=True.
    if not getattr(likelihood, 'use_likelihood_catalogue', False):
        if rank == 0:
            for path in [paths['unique'], paths['matches']]:
                if os.path.exists(path):
                    os.remove(path)
            metadata = {
                'active': False,
                'settings': _likelihood_catalogue_settings(tmax, try_integration),
                'disabled_by_likelihood': True,
            }
            with atomic_write(paths['metadata']) as f:
                json.dump(metadata, f, indent=2, sort_keys=True)
        comm.Barrier()
        return False

    # Fingerprint the raw all_equations file (streamed, so a cache hit does not
    # load the whole catalogue) so the cache is invalidated whenever the
    # equation set -- or the likelihood transform -- changes, not only when
    # tmax/try_integration change.
    raw_paths = raw_catalogue_paths(comp, likelihood)
    if rank == 0:
        all_equations_hash = _hash_file(raw_paths['all'])
        transform_fingerprint = _transform_fingerprint(
            likelihood, tmax, try_integration)
    else:
        all_equations_hash = None
        transform_fingerprint = None
    all_equations_hash = comm.bcast(all_equations_hash, root=0)
    transform_fingerprint = comm.bcast(transform_fingerprint, root=0)
    transform_version = getattr(likelihood, 'catalogue_transform_version', None)
    settings = _likelihood_catalogue_settings(
        tmax, try_integration, all_equations_hash,
        transform_version, transform_fingerprint)

    metadata = _read_likelihood_catalogue_metadata(comp, likelihood)
    # A cache is reused only when its inputs match, it was built cleanly (no
    # failed transformations), AND the likelihood supplied a
    # catalogue_transform_version. Without an explicit version the probe
    # fingerprint is only a best-effort heuristic that can miss a transform
    # change affecting expressions unlike the probes, so the catalogue is rebuilt
    # every call rather than risk a stale mapping (correct but slower -- set
    # catalogue_transform_version to cache it, or use_likelihood_catalogue=False
    # to skip it).
    if (metadata is not None and transform_version is not None
            and metadata.get('settings') == settings
            and metadata.get('failed_count', 0) == 0
            and (not metadata.get('active', False)
                 or likelihood_catalogue_active(comp, likelihood))):
        comm.Barrier()
        return metadata.get('active', False)

    max_param = int(max(4, np.floor((comp - 1) / 2)))
    if rank == 0:
        with open(raw_paths['all'], 'r') as f:
            all_functions = [line.strip() for line in f]
        with open(raw_paths['unique'], 'r') as f:
            raw_unique_count = sum(1 for _ in f)
        n_all = len(all_functions)
        per = int(np.ceil(n_all / size)) if n_all else 0
        # Contiguous slices, so gathering in rank order reproduces global index
        # order and the representative selection is deterministic.
        chunks = [
            (min(r * per, n_all), all_functions[min(r * per, n_all):min((r + 1) * per, n_all)])
            for r in range(size)]
    else:
        chunks = None

    # The per-equation canonicalisation is the expensive part of the build
    # (run_sympify + factor/cancel on every raw tree). Measured serially on one
    # core over local complexities 4-6: ~1.7-2.0 ms/equation for a
    # non-transforming likelihood and ~2.3-3.2 ms for a parameter-removing one,
    # with the per-equation cost growing ~1.3x per unit complexity. all_equations
    # has 772,515 trees at complexity 9 (99,406 at 8, 19,860 at 7), so even at
    # the flat complexity-6 rate that is ~26-40 min serial at c9, rising to
    # ~1.5 h once the per-equation growth (to ~7 ms/eq at c9) is extrapolated.
    # So scatter the work across the ranks rather than looping on rank 0.
    # Each rank buckets its own slice by canonical transformed form: a
    # difference between the tree's ``expected`` layout and the layout surviving
    # the transformation means run_sympify altered the parameter space (the
    # catalogue is genuinely needed), and equations sharing a transformed key
    # are duplicate fits collapsed onto one representative. Rank 0 then merges
    # the gathered keys in global index order; the build is cached, so this is a
    # one-time cost per settings combination.
    start_index, my_functions = comm.scatter(chunks, root=0)
    local_results = _transformed_keys_for_slice(
        my_functions, start_index, likelihood, tmax, try_integration, max_param)
    gathered = comm.gather(local_results, root=0)

    if rank == 0:
        results = sorted(
            (item for sub in gathered for item in sub),
            key=lambda entry: entry[0])

        key_to_unique = {}
        unique_representatives = []
        matches = []
        changed_layout_count = 0
        failed_count = 0
        for index, key, changed, failed in results:
            if failed:
                failed_count += 1
            if changed:
                changed_layout_count += 1
            if key not in key_to_unique:
                key_to_unique[key] = len(unique_representatives)
                unique_representatives.append(all_functions[index])
            matches.append(key_to_unique[key])

        # Surface swallowed transformation failures. A handful of genuinely
        # pathological expressions can legitimately fail, but a large fraction
        # failing usually signals a systematic problem (e.g. an invalid tmax
        # that makes every run_sympify raise), which would otherwise be cached
        # silently.
        if failed_count > 0:
            emit_diagnostic_warning(
                f'{failed_count} of {len(all_functions)} equations failed to '
                f'transform while building the likelihood-aware catalogue for '
                f'complexity {comp}; each such equation is treated as its own '
                f'family.', LikelihoodCatalogueWarning)

        # Activate when the transform removes/relabels a parameter (correctness)
        # OR collapses raw-distinct expressions onto the same transformed model
        # (a dedup benefit: fewer fits than the raw unique catalogue).
        active = (changed_layout_count > 0
                  or len(unique_representatives) < raw_unique_count)
        if transform_version is None:
            emit_diagnostic_warning(
                f'Likelihood-aware catalogue for complexity {comp} is built '
                f'from a run_sympify transform with no '
                f'catalogue_transform_version, so it is rebuilt on every call '
                f'rather than cached: the probe fingerprint alone cannot '
                f'guarantee cache freshness. Set '
                f'likelihood.catalogue_transform_version to cache it, or '
                f'use_likelihood_catalogue=False to skip it.',
                LikelihoodCatalogueWarning)
        if active:
            with atomic_write(paths['unique']) as f:
                for fcn_i in unique_representatives:
                    f.write(fcn_i + '\n')
            with atomic_write(paths['matches']) as f:
                for match in matches:
                    f.write(str(match) + '\n')
        else:
            for path in [paths['unique'], paths['matches']]:
                if os.path.exists(path):
                    os.remove(path)

        metadata = {
            'active': bool(active),
            'settings': settings,
            'n_all': len(all_functions),
            'n_unique': len(unique_representatives),
            'raw_unique_count': raw_unique_count,
            'changed_layout_count': changed_layout_count,
            'failed_count': failed_count,
        }
        with atomic_write(paths['metadata']) as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
        if active:
            print('Using likelihood-aware catalogue: '
                  f"{len(unique_representatives)} transformed families from "
                  f"{len(all_functions)} equations; "
                  f"{changed_layout_count} parameter-layout changes.",
                  flush=True)
    comm.Barrier()
    metadata = _read_likelihood_catalogue_metadata(comp, likelihood)
    return bool(metadata is not None and metadata.get('active', False))


def get_functions(comp, likelihood, unique=True):
    """Load all functions for a given complexity to use and distribute among ranks

    Args:
        :comp (int): complexity of functions to consider
        :likelihood (fitting.likelihood object): object containing data, functions to convert SR expressions to variable of data and file path
        :unique (bool, default=True): whether to load just the unique functions (True) or all functions (False)

    Returns:
        :fcn_list (list): list of strings representing functions to be used by given rank
        :data_start (int): first index of function used by rank
        :data_end (int): last index of function used by rank

    """

    unifn_file = function_catalogue_path(comp, likelihood, unique=unique)
    set_recursionlimit_for_comp(comp)

    ensure_output_dirs(likelihood)

    if rank == 0:
        print("Number of cores:", size, flush=True)

    # First, count total number of lines without loading into memory
    if rank == 0:
        with open(unifn_file, "r") as f:
            total_lines = sum(1 for _ in f)
    else:
        total_lines = None
    # Broadcast total number of lines to all processes
    total_lines = comm.bcast(total_lines, root=0)
    # Number of lines per file for this rank.  Keep at least one line per
    # populated rank; otherwise small catalogues with more MPI ranks than
    # functions put all work on the final rank and can stall at startup.
    nLs = int(np.ceil(total_lines / float(size))) if total_lines else 0
    if total_lines and size > total_lines and rank == 0:
        print("Correcting for many cores.", flush=True)

    if rank == 0:
        print("Total number of functions: ", total_lines, flush=True)
        print("Number of test points per proc: ", nLs, flush=True)

    data_start = min(rank * nLs, total_lines)
    data_end = min((rank + 1) * nLs, total_lines)

    # Load the functions for this rank
    with open(unifn_file, "r") as f:
        # Skip lines up to data_start quickly
        for _ in range(data_start):
            next(f)

        # Now read only lines from data_start to data_end-1
        fcn_list = []
        for i in range(data_end - data_start):
            line = next(f)
            fcn_list.append(line.strip())

    return fcn_list, data_start, data_end


def get_all_functions(comp, likelihood, unique=True):
    """Load every function for a complexity into a list (for dynamic scheduling).

    Args:
        :comp (int): complexity of functions to consider
        :likelihood (fitting.likelihood object): provides the catalogue paths
        :unique (bool, default=True): load the unique catalogue (True) or the
            full all-equations catalogue (False)

    Returns:
        :fcn_list (list): the function strings
    """
    unifn_file = function_catalogue_path(comp, likelihood, unique=unique)
    set_recursionlimit_for_comp(comp)
    with open(unifn_file, "r") as f:
        return [line.strip() for line in f]


def get_function_count(comp, likelihood, unique=True):
    """Return the number of functions in the active catalogue (bcast to all ranks).

    Args:
        :comp (int): complexity of functions to consider
        :likelihood (fitting.likelihood object): provides the catalogue paths
        :unique (bool, default=True): count the unique catalogue (True) or the
            full all-equations catalogue (False)

    Returns:
        :count (int): number of functions in the selected catalogue
    """
    set_recursionlimit_for_comp(comp)
    if rank == 0:
        unifn_file = function_catalogue_path(comp, likelihood, unique=unique)
        with open(unifn_file, "r") as f:
            total_lines = sum(1 for _ in f)
    else:
        total_lines = None
    return comm.bcast(total_lines, root=0)


def _fit_function_with_timeout(fcn_i, likelihood, tmax, pmin, pmax, comp,
                               try_integration, log_opt, max_param,
                               Niter_params, Nconv_params,
                               ignore_previous_eqns):
    """Fit one function under a wall-clock timeout, returning NaN on failure.

    Wraps ``optimise_fun`` in ``simplifier.time_limit`` and a broad exception
    guard so a single pathological function cannot stall or crash the run; on a
    timeout or error it returns ``(nan, zeros)``. If integration was requested
    but the integrated form is not implemented in numpy (``NameError``), it
    retries without integration.

    Args:
        :fcn_i (str): function string to fit
        :likelihood (fitting.likelihood object): data and likelihood
        :tmax (float): maximum seconds for any one simplification step
        :pmin (float): minimum initial-guess value for each parameter
        :pmax (float): maximum initial-guess value for each parameter
        :comp (int): complexity of the function
        :try_integration (bool): whether to attempt analytic integration
        :log_opt (bool): whether to optimise 1/2-parameter cases in log space
        :max_param (int): maximum number of parameters (sets array widths)
        :Niter_params (list): polynomial coefficients for the restart count
        :Nconv_params (list): polynomial coefficients for the converged count
        :ignore_previous_eqns (bool): skip equations seen at lower complexity

    Returns:
        :chi2_i (float): the minimum -log(likelihood), or NaN on failure
        :params (np.ndarray): best-fit parameters (length ``max_param``)
    """
    params = np.zeros(max_param)
    chi2_i = np.nan
    try:
        with simplifier.time_limit(tmax):
            try:
                chi2_i, params = optimise_fun(
                    fcn_i,
                    likelihood,
                    tmax,
                    pmin,
                    pmax,
                    comp=comp,
                    try_integration=try_integration,
                    log_opt=log_opt,
                    max_param=max_param,
                    Niter_params=Niter_params,
                    Nconv_params=Nconv_params,
                    ignore_previous_eqns=ignore_previous_eqns)
            except NameError:
                if try_integration:
                    chi2_i, params = optimise_fun(
                        fcn_i,
                        likelihood,
                        tmax,
                        pmin,
                        pmax,
                        comp=comp,
                        try_integration=False,
                        log_opt=log_opt,
                        max_param=max_param,
                        Niter_params=Niter_params,
                        Nconv_params=Nconv_params,
                        ignore_previous_eqns=ignore_previous_eqns)
                else:
                    raise NameError
    except Exception as e:
        print(e, flush=True)
        chi2_i = np.nan
        params[:] = 0.
    return chi2_i, params


def _main_dynamic(comp, likelihood, fcn_list, tmax, pmin, pmax,
                  print_frequency, try_integration, log_opt, max_param,
                  Niter_params, Nconv_params, ignore_previous_eqns):
    """Fit all functions using a rank-0 coordinator/worker MPI schedule.

    The static partitioning in ``main`` gives each rank a fixed contiguous
    slice of the catalogue. Because individual fits vary enormously in cost,
    that leaves fast ranks idle while a few slow ones finish (the "straggler"
    problem). Here rank 0 acts as a dispatcher instead: it holds the full
    function list, sends one equation at a time to whichever worker is free
    (tagged ``WORK_TAG``), receives each result (``RESULT_TAG``), and hands out
    the next equation until the list is exhausted, then tells each worker to
    stop (``STOP_TAG``). Rank 0 performs no fitting itself, so this needs at
    least two worker ranks (``size >= 3``); ``main`` falls back to the static
    path otherwise. Results are written to ``negloglike_comp<comp>.dat`` on
    rank 0, with periodic checkpoints to a ``.checkpoint.dat`` file (removed on
    success) so a long run can be inspected while in progress.

    Args:
        :comp (int): complexity of functions to consider
        :likelihood (fitting.likelihood object): object containing data,
            likelihood functions and file paths
        :fcn_list (list or None): the full list of function strings to fit; only
            read on rank 0 (pass None on worker ranks)
        :tmax (float): maximum time in seconds for any one simplification step
        :pmin (float): minimum value for each parameter's initial guess
        :pmax (float): maximum value for each parameter's initial guess
        :print_frequency (int): print progress every this many completed fits
        :try_integration (bool): when the likelihood requires an integral,
            whether to attempt analytic integration (True) or integrate
            numerically (False)
        :log_opt (bool): whether to optimise the 1- and 2-parameter cases in
            log space
        :max_param (int): maximum number of parameters; sets the array widths
        :Niter_params (list): coefficients setting the maximum number of
            optimisation restarts as a polynomial in the parameter count
        :Nconv_params (list): coefficients setting the number of converged
            restarts required before stopping, as a polynomial in the count
        :ignore_previous_eqns (bool): whether to skip equations already seen at
            a lower complexity

    Returns:
        None
    """
    if rank == 0:
        n_functions = len(fcn_list)
        n_workers = size - 1
        print(
            f"Dynamic scheduling: {n_functions} functions across "
            f"{n_workers} workers",
            flush=True)

        chi2 = np.full(n_functions, np.nan)
        params = np.zeros([n_functions, max_param])
        paths = fitting_paths(comp, likelihood)
        checkpoint_file = paths['negloglike_checkpoint']
        output_file = paths['negloglike']
        next_index = 0
        active = 0

        for worker in range(1, size):
            if next_index < n_functions:
                comm.send(
                    (next_index, fcn_list[next_index]),
                    dest=worker,
                    tag=WORK_TAG)
                next_index += 1
                active += 1
            else:
                comm.send(None, dest=worker, tag=STOP_TAG)

        completed = 0
        checkpoint_frequency = max(print_frequency, 50)
        while active:
            status = MPI.Status()
            index, chi2_i, params_i = comm.recv(
                source=MPI.ANY_SOURCE, tag=RESULT_TAG, status=status)
            worker = status.Get_source()
            chi2[index] = chi2_i
            params[index, :] = params_i
            completed += 1
            if completed == 1 or completed % print_frequency == 0:
                print(f"{completed} of {n_functions}", flush=True)
            if completed % checkpoint_frequency == 0:
                write_negloglike_file(
                    checkpoint_file, chi2, params, max_param)

            if next_index < n_functions:
                comm.send(
                    (next_index, fcn_list[next_index]),
                    dest=worker,
                    tag=WORK_TAG)
                next_index += 1
            else:
                comm.send(None, dest=worker, tag=STOP_TAG)
                active -= 1

        write_negloglike_file(output_file, chi2, params, max_param)
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
    else:
        while True:
            status = MPI.Status()
            payload = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            if status.Get_tag() == STOP_TAG:
                break
            index, fcn_i = payload
            chi2_i, params_i = _fit_function_with_timeout(
                fcn_i,
                likelihood,
                tmax,
                pmin,
                pmax,
                comp,
                try_integration,
                log_opt,
                max_param,
                Niter_params,
                Nconv_params,
                ignore_previous_eqns)
            comm.send((index, chi2_i, params_i), dest=0, tag=RESULT_TAG)

    comm.Barrier()
    return


def optimise_fun(fcn_i, likelihood, tmax, pmin, pmax, comp=0, try_integration=False, log_opt=False, max_param=4, Niter_params=[40, 60], Nconv_params=[5, 20], test_success=False, ignore_previous_eqns=True):
    """Optimise the parameters of a function to fit data

    The list of parameters, P, passed as Niter_params and Nconv_params compute these values, N, to be
    N = P[0] + P[1] * nparam + P[2] * nparam ** 2 + ...
    where nparam is the number of parameters of the function. The order of the polynomial is determined by
    the length of P, so P can be arbirary in length.

    Args:
        :fcn_i (str): string representing function we wish to fit to data
        :likelihood (fitting.likelihood object): object containing data and likelihood function
        :tmax (float): maximum time in seconds to run any one part of simplification procedure for a given function
        :pmin (float): minimum value for each parameter to consider when generating initial guess
        :pmax (float): maximum value for each parameter to consider when generating initial guess
        :comp (float, default=0): Complexity. Deafault of 0 because it is not provided when fitting a single function
        :try_integration (bool, default=False): when likelihood requires integral, whether to try to analytically integrate (True) or just numerically integrate (False)
        :log_opt (bool, default=False): whether to optimise 1 and 2 parameter cases in log space
        :max_param (int, default=4): The maximum number of parameters considered. This sets the shapes of arrays used.
        :Niter_params (list, default=[40, 60]): Parameters determining maximum number of parameter optimisation iterations to attempt.
        :Nconv_params (list, default=[-5, 20]): If we find Nconv solutions for the parameters which are within a logL of 0.5 of the best, we say we have converged and stop optimising parameters. These parameters determine Nconv.
        :test_sucess (bool, default=False): Whether to test whether the optimisation was successful using scipy's criteria
        :ignore_previous_eqns (bool, default=True): If we have seen an equation at lower complexity, whether to ignore the equation in this routine.

    Returns:
        :chi2_i (float): the minimum value of -log(likelihood) (corresponding to the maximum likelihood)
        :params (list): the maximum likelihood values of the parameters

    """

    xvar = getattr(likelihood, 'xvar', None)

    params = np.zeros(max_param)

    if comp > 1 and ignore_previous_eqns:
        previous_fns_file = raw_catalogue_paths(comp, likelihood)['previous']
        with open(previous_fns_file, "r") as f:
            previous_fns = f.readlines()
        # discard repeat of lower complexity (e.g. [inv, inv, ...])
        if fcn_i in previous_fns:
            return np.inf, params

    try:
        fcn_i, eq, integrated = likelihood.run_sympify(
            fcn_i, tmax=tmax, try_integration=try_integration)

        # A likelihood transformation can remove a parameter without retaining
        # a prefix, e.g. a0*(a1+x) / g(1) leaves a1. Fit a canonical parameter
        # vector for the transformed expression rather than the original tree.
        eq, active_params = canonicalize_parameter_symbols(eq)
        nparam = len(active_params)
        Niter = int(np.sum(nparam ** np.arange(len(Niter_params))
                    * np.array(Niter_params)))
        Nconv = int(np.sum(nparam ** np.arange(len(Nconv_params))
                    * np.array(Nconv_params)))
        if (Nconv <= 0) or (Niter <= 0) or (Nconv > Niter):
            raise ValueError("Nconv and/or Niter have unacceptable values")

        if nparam == 0:
            eq_numpy = sympy.lambdify(x, eq, modules=["numpy"])
            chi2_i = likelihood.negloglike([], eq_numpy, integrated=integrated)
            return chi2_i, params

        flag_three = False

        mult_arr = np.ones(max_param)
        count_lowest = 0
        inf_count = 0

        if nparam > 1:
            all_a = ' '.join([f'a{i}' for i in range(nparam)])
            all_a = list(sympy.symbols(all_a, real=True))
            eq_numpy = sympy.lambdify([x] + all_a, eq, modules=["numpy"])
        else:
            eq_numpy = sympy.lambdify([x, a0], eq, modules=["numpy"])

        bad_fun = True
        if xvar is not None:
            for p in itertools.product([1, -1], repeat=nparam):
                if not (np.sum(np.isnan(eq_numpy(xvar, *p))) > 0):
                    bad_fun = False
                    break

        if bad_fun:
            # Don't bother trying to optimise bc this fcn is clearly really bad
            chi2_i = np.inf
            return chi2_i, params

        # Reset chi2
        chi2_min = np.inf

        if nparam > 2:
            flag_three = True

        for j in range(Niter):

            if nparam > 2:
                inpt = [np.random.uniform(pmin, pmax) for _ in range(nparam)]
                res = minimize(chi2_fcn, inpt, args=(likelihood, eq_numpy, integrated,
                               # Default=3000
                                                     None), method="BFGS", options={'maxiter': 7000})
            elif nparam == 2:
                if log_opt:
                    # These are now in log-space, so this is 1e-1 -- 1e1; was -10:10
                    inpt = [np.random.uniform(
                        pmin, pmax), np.random.uniform(pmin, pmax)]
                    res_pp = minimize(chi2_fcn, inpt, args=(
                        likelihood, eq_numpy, integrated, ['+', '+']), method="BFGS")
                    res_mp = minimize(chi2_fcn, inpt, args=(
                        likelihood, eq_numpy, integrated, ['-', '+']), method="BFGS")
                    res_pm = minimize(chi2_fcn, inpt, args=(
                        likelihood, eq_numpy, integrated, ['+', '-']), method="BFGS")
                    res_mm = minimize(chi2_fcn, inpt, args=(
                        likelihood, eq_numpy, integrated, ['-', '-']), method="BFGS")

                    choose = np.argmin(
                        [res_pp['fun'], res_mp['fun'], res_pm['fun'], res_mm['fun']])
                    mult_arr = np.ones(max_param)
                    if choose == 0:
                        res = res_pp
                    elif choose == 1:
                        res = res_mp
                        mult_arr[0] = -1
                    elif choose == 2:
                        res = res_pm
                        mult_arr[1] = -1
                    elif choose == 3:
                        res = res_mm
                        mult_arr[0] = -1
                        mult_arr[1] = -1
                    else:
                        print("Some ambiguity in choose", eq, flush=True)
                        res = res_pp
                else:
                    flag_three = True
                    inpt = [np.random.uniform(
                        pmin, pmax), np.random.uniform(pmin, pmax)]
                    res = minimize(chi2_fcn, inpt, args=(likelihood, eq_numpy, integrated,
                                   # Default=3000
                                                         None), method="BFGS", options={'maxiter': 5000})

            else:
                if log_opt:
                    inpt = np.random.uniform(pmin, pmax)
                    res_p = minimize(chi2_fcn, inpt, args=(
                        likelihood, eq_numpy, integrated, ['+']), method="BFGS")
                    res_m = minimize(chi2_fcn, inpt, args=(
                        likelihood, eq_numpy, integrated, ['-']), method="BFGS")

                    mult_arr = np.ones(max_param)
                    if res_p['fun'] < res_m['fun']:
                        res = res_p
                    elif res_p['fun'] > res_m['fun']:
                        res = res_m
                        mult_arr[0] = -1
                    else:
                        # Both give same. Choose positive (arbitrarily)
                        res = res_p
                else:
                    flag_three = True
                    inpt = np.random.uniform(pmin, pmax)
                    res = minimize(chi2_fcn, inpt, args=(
                        likelihood, eq_numpy, integrated, None), method="BFGS", options={'maxiter': 5000})

            if test_success and (not res.success):
                continue

            if np.isinf(res['fun']):
                inf_count += 1

            # Failure if first 50 all give inf
            if inf_count == 50 and np.isinf(chi2_min):
                break

            # Reset count if log-like improves by 2
            if res['fun']-chi2_min < -2.:
                count_lowest = 0

            #  If within 0.5 of lowest, say converged to that value
            if abs(res['fun']-chi2_min) < 0.5:
                count_lowest += 1

            if res['fun'] < chi2_min:
                best = res
                mult_arr_best = mult_arr
                chi2_min = res['fun']

            # Converged the required number of times, so a success
            if count_lowest == Nconv:
                break

        if chi2_min < 1.e100:
            # Optimisation happened. Print something
            if flag_three:
                params = np.pad(np.array(best.x), (0, max_param-len(best.x)))
            else:
                # Params put in linear space and sign added back in
                params = np.pad(10.**np.array(best.x),
                                (0, max_param-len(best.x))) * mult_arr_best
        elif not np.isfinite(chi2_min):
            print('\tFailed to find parameters for function:', fcn_i)

        # This is after all the iterations, so it's the best we have; reduced chi2
        chi2_i = chi2_min

    except NameError:
        print(NameError)
        # Occurs if function produced not implemented in numpy
        raise NameError

    except simplifier.TimeoutException:
        print('TIMED OUT:', fcn_i, flush=True)
        try:
            if chi2_min < 1.e100:
                if flag_three:
                    params = np.pad(np.array(best.x),
                                    (0, max_param-len(best.x)))
                else:
                    params = np.pad(10.**np.array(best.x),
                                    (0, max_param-len(best.x))) * mult_arr_best
                chi2_i = chi2_min
            else:
                chi2_i = np.nan
                params[:] = 0.
        except Exception:
            chi2_i = np.nan
            params[:] = 0.

    except Exception as e:
        print(e)
        return np.nan, params

    return chi2_i, params


def main(comp, likelihood, tmax=5, pmin=0, pmax=3, print_frequency=50, try_integration=False, log_opt=False, Niter_params=[40, 60], Nconv_params=[5, 20], ignore_previous_eqns=False, dynamic=True):
    """Optimise all functions for a given complexity and save results to file.

    This can optimise in log-space, with separate +ve and -ve branch (except when there are >=3 params in which case it does it in linear)

    The list of parameters, P, passed as Niter_params and Nconv_params compute these values, N, to be
    N = P[0] + P[1] * nparam + P[2] * nparam ** 2 + ...
    where nparam is the number of parameters of the function. The order of the polynomial is determined by
    the length of P, so P can be arbirary in length.

    Args:
        :comp (int): complexity of functions to consider
        :likelihood (fitting.likelihood object): object containing data, likelihood functions and file paths
        :tmax (float, default=5.): maximum time in seconds to run any one part of simplification procedure for a given function
        :pmin (float, default=0.): minimum value for each parameter to considered when generating initial guess
        :pmax (float, default=3.): maximum value for each parameter to considered when generating initial guess
        :print_frequency (int, default=50): the status of the fits will be printed every ``print_frequency`` number of iterations
        :try_integration (bool, default=False): when likelihood requires integral, whether to try to analytically integrate (True) or just numerically integrate (False)
        :log_opt (bool, default=False): whether to optimise 1 and 2 parameter cases in log space
        :Niter_params (list, default=[40, 60]): Parameters determining maximum number of parameter optimisation iterations to attempt.
        :Nconv_params (list, default=[-5, 20]): If we find Nconv solutions for the parameters which are within a logL of 0.5 of the best, we say we have converged and stop optimising parameters. These parameters determine Nconv.
        :ignore_previous_eqns (bool, default=False): If we have seen an equation at lower complexity, whether to ignore the equation in this routine.
        :dynamic (bool, default=True): Use rank-0 work dispatch for MPI runs with at least two worker ranks. This avoids long idle tails when individual functions have very different runtimes. Set to False to use the original static rank partitioning.

    Returns:
        None

    """

    if rank == 0:
        print('\nRunning fits', flush=True)

    ensure_likelihood_catalogue(comp, likelihood, tmax, try_integration)

    if rank == 0 and ignore_previous_eqns:
        previous_unifn_list = []
        if comp > 1:
            for compl in range(1, comp):
                unifn_file_i = raw_catalogue_paths(
                    compl, likelihood)['unique']
                with open(unifn_file_i, "r") as f:
                    fcn_list_i = f.readlines()
                previous_unifn_list += fcn_list_i
        previous_unifn_list = np.array(previous_unifn_list)
        np.savetxt(
            raw_catalogue_paths(comp, likelihood)['previous'],
            previous_unifn_list,
            fmt='%s')

    comm.Barrier()

    # Set max param >=4 for backwards compatibility
    max_param = int(max(4, np.floor((comp - 1) / 2)))

    n_functions = get_function_count(comp, likelihood)
    if dynamic and size >= 3 and n_functions > size:
        fcn_list_all = get_all_functions(comp, likelihood) if rank == 0 else None
        _main_dynamic(
            comp,
            likelihood,
            fcn_list_all,
            tmax,
            pmin,
            pmax,
            print_frequency,
            try_integration,
            log_opt,
            max_param,
            Niter_params,
            Nconv_params,
            ignore_previous_eqns)
        return

    fcn_list_proc, _, _ = get_functions(comp, likelihood)
    chi2 = np.zeros(len(fcn_list_proc))     # This is now only for this proc
    params = np.zeros([len(fcn_list_proc), max_param])
    for i in range(len(fcn_list_proc)):           # Consider all possible complexities
        if rank == 0 and ((i == 0) or ((i+1) % print_frequency == 0)):
            print(f'{i+1} of {len(fcn_list_proc)}', flush=True)
        chi2[i], params[i, :] = _fit_function_with_timeout(
            fcn_list_proc[i],
            likelihood,
            tmax,
            pmin,
            pmax,
            comp,
            try_integration,
            log_opt,
            max_param,
            Niter_params,
            Nconv_params,
            ignore_previous_eqns)

    # Save the data for this proc in Partial
    paths = fitting_paths(comp, likelihood, rank=rank)
    write_negloglike_file(paths['negloglike_rank'], chi2, params, max_param)

    comm.Barrier()

    if rank == 0:
        combine_temp_files(
            likelihood.temp_dir,
            paths['negloglike_rank_pattern'],
            paths['negloglike'])

    comm.Barrier()

    return
