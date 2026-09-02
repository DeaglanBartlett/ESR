import numpy as np
import math
import sympy
from mpi4py import MPI
import warnings
import itertools
import json
import numdifftools as nd
from scipy.optimize import minimize
from scipy.stats import mode

import esr.fitting.test_all as test_all
from esr.fitting.utils import (
    combine_temp_files, emit_diagnostic_warning, fitting_paths,
    set_recursionlimit_for_comp)
from esr.fitting.sympy_symbols import x, a0


class HighConditionNumberWarning(UserWarning):
    """Diagnostic warning that a fitted Hessian is badly conditioned."""


class ProjectedEigenbasisWarning(UserWarning):
    """Diagnostic warning for snap_choice=2: the Hessian has (near-)degenerate
    eigenvalues (so the projected codelength is basis-sensitive) or the snapped
    likelihood re-evaluation raised an unexpected exception."""


class DiagonalSnapDeterminantWarning(UserWarning):
    """Diagnostic warning that determinant scoring has been paired with diagonal
    snapping, which cannot remove an unconstrained direction from det(H)."""


# A consecutive pair of positive eigenvalues is treated as (near-)degenerate for
# snap_choice=2 -- where the eigenbasis is then ambiguous -- when their gap
# relative to the larger of the pair falls below this. 1e-3 is a heuristic
# threshold, not a rigorous stability bound: it flags pairs agreeing to about
# three significant figures, at which point the eigenvector directions (and hence
# the codelength) become increasingly basis-sensitive. It errs towards warning so
# that clustered spectra are surfaced rather than silently scored.
CLUSTER_REL_TOL = 1e-3


# Suppress the numpy/scipy RuntimeWarnings raised in bulk while fitting, but
# leave other categories (including unrelated user warnings) untouched. Our own
# diagnostics are emitted via emit_diagnostic_warning so they survive this.
warnings.filterwarnings("ignore", category=RuntimeWarning)

use_relative_dx = True              # CHANGE

# An eigenvalue of the *correlation-normalised* Hessian below this is treated as
# a degenerate (unconstrained) direction in parameter space. This prevents
# det(H)→0 from corrupting the codelen when parameters are structurally
# redundant. See :func:`_correlation_eigenvalues` for why the test is applied to
# the normalised matrix rather than to the raw eigenvalues.
#
# On the raw eigenvalues no threshold works, because lambda_min/lambda_max
# depends on the units the parameters happen to be measured in as much as on
# whether the model is degenerate. Measured across the fitting likelihoods, a
# structurally redundant parameterisation and a well determined but badly scaled
# one both land around 1e-8 of the largest eigenvalue, and cannot be told apart.
# After normalisation they separate cleanly: redundant parameterisations and
# genuinely unresolvable parameters sit at 1e-5 and below, while well determined
# fits -- including strongly correlated ones such as the Pantheon exponentials,
# at 2e-3 -- stay far above. 1e-4 sits in that gap, and corresponds to a
# parameter combination determined some 1e4 times less well than the individual
# parameter scales.
#
# The same threshold serves both the degeneracy test and the negative-curvature
# (saddle) test. A mathematically flat direction comes out of finite
# differencing as a small positive or a small negative number at random, so if
# the two tests disagreed about it the sign of numerical noise would decide
# whether a degenerate parameterisation is rejected outright or kept and
# rewarded with a short determinant codelength.
EIGENVALUE_REL_THRESHOLD = 1e-4


def _validate_scoring_options(snap_choice):
    """Validate the parameter-snapping mode; raise on unsupported values.

    ``snap_choice`` selects the snapping strategy: 0 = diagonal, 1 = eigenbasis
    identification with original-parameter zeroing, 2 = projected eigenbasis.
    See :func:`convert_params` for the full definitions.
    The ``use_det_I``/``snap_choice`` *combination* is validated separately by
    ``_validate_snap_and_det`` where both are available, so this range check can
    be used on its own (for example in ``_compute_snap_mask``).

    Args:
        :snap_choice (int): snapping mode: 0 = diagonal, 1 = eigenbasis
            identification with original-parameter zeroing, 2 = projected
            eigenbasis. See :func:`convert_params` for details
    """
    if snap_choice not in (0, 1, 2):
        raise ValueError("snap_choice must be 0, 1 or 2.")


def _validate_snap_and_det(use_det_I, snap_choice):
    """Validate the snapping mode together with the determinant-scoring flag.

    ``snap_choice=2`` (projected eigenbasis) evaluates the codelength in the
    Hessian eigenbasis, where it is consistent only with the determinant score;
    it is therefore permitted only with ``use_det_I=True``, not the
    basis-dependent diagonal hybrid.

    ``use_det_I=True`` with ``snap_choice=0`` is allowed but warned about. It is
    a useful comparison setting -- holding the published snapping rule fixed is
    the only way to attribute a change to the determinant alone -- but it is not
    safe for ranking a catalogue, because diagonal snapping cannot remove an
    unconstrained direction and the determinant then rewards one. The warning is
    emitted through :func:`emit_diagnostic_warning`, so it appears once per run
    and can be silenced by the caller.

    Args:
        :use_det_I (bool): whether the determinant codelength is in use
        :snap_choice (int): snapping mode: 0 = diagonal, 1 = eigenbasis
            identification with original-parameter zeroing, 2 = projected
            eigenbasis. See :func:`convert_params` for details
    """
    _validate_scoring_options(snap_choice)
    if snap_choice == 2 and not use_det_I:
        raise ValueError(
            "snap_choice=2 (projected eigenbasis) requires use_det_I=True.")
    if use_det_I and snap_choice == 0:
        emit_diagnostic_warning(
            'use_det_I=True with snap_choice=0: diagonal snapping tests one '
            'parameter axis at a time, so it cannot remove an unconstrained '
            'direction lying between the axes. That direction stays in det(H), '
            'where the smaller its eigenvalue the shorter the codelength, so a '
            'redundant parameterisation can score better than the model it is a '
            'redundant copy of. Use this pairing to isolate the effect of the '
            'determinant in comparison runs, not to rank a catalogue; '
            'snap_choice=1 is the setting for that.',
            DiagonalSnapDeterminantWarning)


def _symmetrized_hessian(Hmat):
    """Return the symmetric part 0.5 (H + H^T) of a (possibly noisy) Hessian.

    Args:
        :Hmat (np.ndarray): square matrix to symmetrise

    Returns:
        :Hsym (np.ndarray): the symmetrised matrix as float
    """
    Hmat = np.asarray(Hmat, dtype=float)
    return 0.5 * (Hmat + Hmat.T)


def _correlation_eigenvalues(Hmat):
    """Eigenvalues of the Hessian normalised by its own diagonal.

    ``D^-1/2 H D^-1/2`` for ``D = diag(H)`` is the correlation matrix of the
    Fisher information. Rescaling a parameter (a change of units, or writing
    ``2*a0`` for ``a0``) rescales a row and column of ``H`` and leaves this
    matrix unchanged, so its smallest eigenvalue measures how degenerate the fit
    is and nothing else. The raw eigenvalues do not: a well determined fit whose
    parameters differ in magnitude by a few orders of magnitude has a raw
    ``lambda_min/lambda_max`` as small as a genuinely redundant parameterisation,
    so the two cannot be separated by any threshold on the raw spectrum.

    A parameter with zero curvature of its own has no scale to normalise by. It
    is an exactly flat direction, so it is reported as a zero eigenvalue --
    degenerate, but not a saddle. A parameter with *negative* curvature along its
    own axis is a saddle, and is reported through the second return value rather
    than being normalised.

    Args:
        :Hmat (np.ndarray): Hessian of the negative log-likelihood
            (nparam x nparam); symmetrised internally

    Returns:
        :eigenvalues (np.ndarray or None): eigenvalues of the normalised matrix
            in ascending order, with one zero for each exactly flat parameter
            direction, or None if the spectrum is unusable
        :unusable (bool): True if the Hessian is non-finite, has negative
            curvature along a parameter axis, or cannot be decomposed. The fit
            is then a saddle (or broken) rather than merely degenerate
    """
    Hsym = _symmetrized_hessian(Hmat)
    if not np.all(np.isfinite(Hsym)):
        return None, True
    diagonal = np.diag(Hsym)
    if np.any(diagonal < 0):
        return None, True
    constrained = diagonal > 0
    n_flat = int(np.sum(~constrained))
    if not np.any(constrained):
        return np.zeros(len(diagonal)), False
    block = Hsym[np.ix_(constrained, constrained)]
    block_diagonal = np.diag(block)
    try:
        eigenvalues = np.linalg.eigvalsh(
            block / np.sqrt(np.outer(block_diagonal, block_diagonal)))
    except np.linalg.LinAlgError:
        return None, True
    return np.concatenate([np.zeros(n_flat), eigenvalues]), False


def _is_saddle(eigenvalues, unusable):
    """Whether a correlation-normalised spectrum indicates a saddle.

    Args:
        :eigenvalues (np.ndarray or None): as returned by
            :func:`_correlation_eigenvalues`
        :unusable (bool): as returned by :func:`_correlation_eigenvalues`

    Returns:
        :is_saddle (bool): True for resolved negative curvature or an unusable
            Hessian, False for a minimum (however degenerate)
    """
    if unusable or eigenvalues is None:
        return True
    return bool(np.min(eigenvalues) < -EIGENVALUE_REL_THRESHOLD)


def _has_negative_curvature(Hmat):
    """Return True if the Hessian has a genuinely negative eigendirection.

    Used to reject saddle points: at a true fitted minimum every eigenvalue of
    the Hessian of the negative log-likelihood should be positive. We do not
    test ``np.any(eigenvalues <= 0)`` directly because the Hessian is obtained
    by finite differencing, so eigenvalues that are mathematically zero (flat,
    unconstrained directions) or tiny and positive routinely come out as small
    negative numbers from numerical noise. Flagging those as negative curvature
    would spuriously reject well-behaved fits that merely have a redundant
    parameter. A direction counts as negative only if it falls below
    ``-EIGENVALUE_REL_THRESHOLD`` in the correlation-normalised spectrum, which
    is the same scale on which degeneracy is judged: below it the sign carries
    no information, and the direction is left for the snapping modes to remove
    as an unconstrained one rather than being read as a saddle.

    Args:
        :Hmat (np.ndarray): Hessian of the negative log-likelihood
            (nparam x nparam); symmetrised internally

    Returns:
        :has_negative (bool): True if a resolved negative eigendirection exists
            (or the normalised spectrum is undefined), False otherwise
    """
    return _is_saddle(*_correlation_eigenvalues(Hmat))


def save_scoring_settings(comp, likelihood, use_det_I, snap_choice):
    """Record Fisher scoring settings so matching cannot silently change them.

    Args:
        :comp (int): complexity of functions to consider
        :likelihood (fitting.likelihood object): object providing ``out_dir``
        :use_det_I (bool): whether the determinant codelength is in use
        :snap_choice (int): snapping mode: 0 = diagonal, 1 = eigenbasis
            identification with original-parameter zeroing, 2 = projected
            eigenbasis. See :func:`convert_params` for details
    """
    _validate_snap_and_det(use_det_I, snap_choice)
    with open(fitting_paths(comp, likelihood)['fisher_settings'], 'w') as f:
        json.dump({'use_det_I': bool(use_det_I), 'snap_choice': int(snap_choice)}, f)


def load_scoring_settings(comp, likelihood):
    """Read the Fisher scoring settings saved by ``save_scoring_settings``.

    Args:
        :comp (int): complexity of functions to consider
        :likelihood (fitting.likelihood object): object providing ``out_dir``

    Returns:
        :settings (dict or None): ``{'use_det_I': bool, 'snap_choice': int}``, or
            None if no settings file has been written
    """
    try:
        with open(fitting_paths(comp, likelihood)['fisher_settings'], 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def _compute_codelen(Hmat, Fisher_diag, theta, kept_mask, use_det_I):
    """Compute parametric codelen for a given set of kept parameters.

    The determinant score applies a floor of 0 to each
    ln(|theta_i|/Delta_i) contribution, where Delta_i = sqrt(12/H_ii).
    The diagonal score retains the published diagonal codelength formula
    for explicit comparison runs.

    Args:
        :Hmat (np.ndarray): full Hessian matrix (nparam x nparam)
        :Fisher_diag (np.ndarray): diagonal of the Hessian (nparam,)
        :theta (np.ndarray): parameter values (nparam,)
        :kept_mask (np.ndarray): boolean mask of which parameters to include
        :use_det_I (bool): if True, use det(H); if False, use prod(diag(H))

    Returns:
        :codelen (float): the parametric contribution to description length
    """
    k = int(np.sum(kept_mask))
    if k == 0:
        return 0.0
    theta_active = theta[kept_mask]
    diag_active = Fisher_diag[kept_mask]

    if use_det_I:
        H_active = _symmetrized_hessian(Hmat[np.ix_(kept_mask, kept_mask)])
        try:
            np.linalg.cholesky(H_active)
        except np.linalg.LinAlgError:
            return np.inf
        log_theta_floored = np.empty(k)
        for j in range(k):
            log_delta = 0.5 * np.log(12. / diag_active[j])
            log_theta_floored[j] = max(
                np.log(np.abs(theta_active[j])), log_delta)
        _, logdet = np.linalg.slogdet(H_active)
        return -k/2. * math.log(3.) + 0.5 * logdet + \
            np.sum(log_theta_floored)
    else:
        return -k/2. * math.log(3.) + np.sum(0.5*np.log(diag_active) +
                                              np.log(np.abs(theta_active)))


def _unresolved_directions(Hmat, eigenvalues, eigenvectors):
    """Which eigendirections of the Hessian carry unresolved curvature.

    An eigenvalue is only meaningful next to the scale of the parameters it is
    measured in, so each direction is compared against its own Rayleigh quotient
    over ``diag(H)``: ``lambda_j / (v_j . diag(H) v_j)``. That ratio is unchanged
    when a parameter is rescaled, and its minimum over directions is the smallest
    eigenvalue of the correlation-normalised Hessian, so this is the per-direction
    form of the test :func:`_correlation_eigenvalues` applies to the matrix as a
    whole.

    The one-precision-step test cannot stand in for this. A direction can be
    unresolved and still clear one step, if ``theta`` projects far enough along
    it; scoring it then leaves the unresolved eigenvalue inside ``det H``, where
    the smaller it is the shorter the code it produces.

    Args:
        :Hmat (np.ndarray): Hessian of the negative log-likelihood (nparam x nparam)
        :eigenvalues (np.ndarray): its eigenvalues (nparam,)
        :eigenvectors (np.ndarray): its eigenvectors as columns (nparam x nparam)

    Returns:
        :unresolved (np.ndarray): boolean mask of the unresolved directions
    """
    diagonal = np.diag(_symmetrized_hessian(Hmat))
    scale = np.einsum('ij,i,ij->j', eigenvectors, diagonal, eigenvectors)
    with np.errstate(invalid='ignore', divide='ignore'):
        ratio = np.where(scale > 0, eigenvalues / scale, -np.inf)
    return ~(ratio >= EIGENVALUE_REL_THRESHOLD)


def _compute_snap_mask(Hmat, Fisher_diag, theta, Nsteps, snap_choice):
    """Compute which parameters to snap to zero based on snap_choice.

    With ``snap_choice=0`` the supplied diagonal mask is returned unchanged.
    With ``snap_choice=1`` this eigendecomposes ``Hmat`` to identify weak
    directions, then maps them back to original parameters. ``snap_choice=2`` is
    handled by :func:`_score_projected_eigenbasis` before this helper is reached
    in the fitting pipeline. See :func:`convert_params` for all three modes.

    Args:
        :Hmat (np.ndarray): full Hessian matrix (nparam x nparam)
        :Fisher_diag (np.ndarray): diagonal of the Hessian (nparam,)
        :theta (np.ndarray): parameter values (nparam,)
        :Nsteps (np.ndarray): diagonal-based Nsteps (used for snap_choice=0 and as fallback)
        :snap_choice (int): 0 = diagonal, 1 = eigenbasis identification with
            original-parameter zeroing, 2 = projected eigenbasis (handled by
            :func:`_score_projected_eigenbasis` in the fitting pipeline)

    Returns:
        :Nsteps (np.ndarray): updated Nsteps array (values < 1 indicate parameters to snap)
        :has_degenerate_eig (bool): True if any eigenvalue is unresolved
            (below EIGENVALUE_REL_THRESHOLD relative to the largest), so that the
            unsnapped determinant cannot be trusted and the snap is mandatory.
    """
    nparam = len(theta)
    has_degenerate_eig = False

    _validate_scoring_options(snap_choice)
    if snap_choice == 0:
        return Nsteps, has_degenerate_eig

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(
            _symmetrized_hessian(Hmat[:nparam, :nparam]))
        # Degeneracy and negative curvature are judged on the correlation-
        # normalised spectrum, which does not depend on how the parameters are
        # scaled; the snap itself still uses the eigenvectors of the Hessian, so
        # that the parameter it zeros is the one that actually carries the
        # unconstrained direction.
        scaled, unusable = _correlation_eigenvalues(Hmat[:nparam, :nparam])
        if _is_saddle(scaled, unusable):
            return Nsteps, has_degenerate_eig
        theta_rot = eigenvectors.T @ theta
        # An unconstrained direction (e.g. g and c*g having the same
        # f_DE = g/g(1)) leaves the unsnapped determinant untrustworthy: the
        # smaller its eigenvalue comes out, the shorter the code it produces. So
        # the snap is mandatory and the caller must not fall back on the
        # unsnapped score. A *resolved* but weakly occupied direction (small
        # theta_rot, healthy eigenvalue) is an ordinary snap candidate and stays
        # subject to the description-length comparison in convert_params.
        has_degenerate_eig = bool(np.min(scaled) < EIGENVALUE_REL_THRESHOLD)
        #  Only guards the 12/lambda division below; a tiny positive eigenvalue
        #  gives a huge precision step and so is flagged by the one-step test.
        good_eig = eigenvalues > 0
        Nsteps_rot = np.zeros(nparam)
        Nsteps_rot[good_eig] = np.abs(theta_rot[good_eig]) / np.sqrt(12. / eigenvalues[good_eig])
        # Map unconstrained eigendirections back to original parameters: for each
        # bad eigendirection, snap the original param with largest projection. A
        # direction is bad if it carries fewer than one precision step, or if its
        # curvature is unresolved -- the latter regardless of the former, since a
        # direction the data cannot measure must leave det(H) however far theta
        # happens to project along it.
        unresolved = _unresolved_directions(
            Hmat[:nparam, :nparam], eigenvalues, eigenvectors)
        bad_eig = np.where((Nsteps_rot < 1) | unresolved)[0]
        snap_set = set()
        for ei in bad_eig:
            snap_set.add(np.argmax(np.abs(eigenvectors[:, ei])))
        Nsteps = np.ones(nparam)
        for j in snap_set:
            Nsteps[j] = 0.
    except np.linalg.LinAlgError:
        has_degenerate_eig = True  # can't decompose — treat as degenerate

    return Nsteps, has_degenerate_eig


def _refit_after_snap(fop, theta, kept_mask):
    """Re-optimise the retained parameters with the snapped ones held at zero.

    Snapping removes a parameter, so the point to encode is the maximum
    likelihood of the *reduced* model, not the full model's fit with one entry
    overwritten by zero. Without this re-optimisation a snap can wreck the
    likelihood -- zeroing an intercept is not the same as removing it, since the
    slope was fitted alongside it -- and the description-length comparison then
    rejects a snap that was in fact the right move.

    The search starts from the snapped vector and keeps the result only if it is
    finite and no worse, so this can only improve the reduced fit. Nelder-Mead is
    used because the fitting likelihoods contain kinks (``Abs``, ``pow``) that
    make gradient-based refinement unreliable this close to a boundary.

    Args:
        :fop (callable): maps a full-length parameter vector to its -log(L)
        :theta (np.ndarray): parameter vector with the snapped entries already
            set to zero (nparam,)
        :kept_mask (np.ndarray): boolean mask of the parameters left free

    Returns:
        :theta (np.ndarray): the re-optimised vector, snapped entries still zero
        :negloglike (float): -log(L) at that vector
    """
    theta = np.asarray(theta, dtype=float).copy()
    negloglike = fop(theta)
    if not np.any(kept_mask):
        return theta, negloglike

    def objective(free):
        full = np.zeros_like(theta)
        full[kept_mask] = free
        return fop(full)

    try:
        result = minimize(objective, theta[kept_mask], method='Nelder-Mead',
                          options={'maxiter': 2000})
    except Exception:
        return theta, negloglike
    if np.isfinite(result.fun) and not (result.fun > negloglike):
        theta[kept_mask] = result.x
        negloglike = float(result.fun)
    return theta, negloglike


def _score_projected_eigenbasis(Hmat, theta, negloglike, use_det_I,
                                eval_negloglike):
    """Score a fit in the Hessian eigenbasis (``snap_choice=2``).

    Rotate ``theta`` into the eigenbasis of the Hessian (``b = V^T theta``),
    drop the weakly constrained *projected* coordinates (those with fewer than
    one precision step, or an unconstrained/non-positive eigenvalue), transform
    the retained vector back to the original basis and re-evaluate the
    likelihood there. Unlike ``snap_choice=1`` -- which flags weak eigen-
    directions but then zeros an *original* parameter -- this zeros the
    projected coordinate itself, so the snap and the codelength live in the same
    basis. The parametric codelength is computed in the eigenbasis, where the
    Hessian is diagonal (its eigenvalues), so the determinant volume
    (``sum ln lambda_j``) and the per-direction precision floor
    (``0.5 ln(12/lambda_j)`` against ``|b_j|``) are consistent. This is the
    treatment discussed for ``snap_choice=2`` in the documentation.

    Limitation: the eigenvectors of a repeated (or nearly repeated) eigenvalue
    are not unique -- any rotation within the (near-)degenerate subspace is an
    equally valid eigenbasis -- so both which directions are snapped and the
    per-direction floor term (hence the codelength) become basis-sensitive when
    the Hessian has clustered eigenvalues. For a given Hessian the result is
    deterministic (``np.linalg.eigh`` returns a fixed basis), but a
    re-parameterisation that yields the same clustered spectrum can change both
    ``k`` and the codelength; a diagnostic ``ProjectedEigenbasisWarning`` is
    emitted in that case (see ``CLUSTER_REL_TOL``). Note this is not unique to
    mode 2: the per-coordinate precision floor makes *every* snap mode
    coordinate-dependent, so ``snap_choice=1`` is not strictly
    re-parameterisation-invariant either -- it is simply always evaluated in the
    fixed original basis, so it is the more predictable choice for clustered
    spectra.

    Args:
        :Hmat (np.ndarray): Hessian of -log(L) at the fit (nparam x nparam)
        :theta (np.ndarray): fitted parameters (nparam,)
        :negloglike (float): -log(L) at ``theta`` with no snapping
        :use_det_I (bool): must be True (enforced by the callers)
        :eval_negloglike (callable): maps a back-transformed parameter vector to
            its -log(L). Only called if a projected coordinate is actually
            snapped, so callers may build the numpy function lazily.

    Returns:
        :theta_final (np.ndarray): parameters after any projected snap (nparam,)
        :negloglike_final (float): -log(L) at ``theta_final``
        :k (int): number of retained eigendirections
        :codelen (float): eigenbasis parametric codelength (``inf`` if the fit is
            a saddle, or degenerate and the projected snap breaks the likelihood)
    """
    theta = np.asarray(theta, dtype=float)
    nparam = len(theta)
    if not np.all(np.isfinite(Hmat)):
        # A non-finite Hessian has no usable eigenbasis.
        return theta, negloglike, nparam, np.inf
    try:
        eigenvalues, V = np.linalg.eigh(_symmetrized_hessian(Hmat))
    except np.linalg.LinAlgError:
        return theta, negloglike, nparam, np.inf

    #  As in _compute_snap_mask, both verdicts come from the correlation-
    #  normalised spectrum so that they do not depend on the parameter scaling,
    #  while the projection itself uses the Hessian's own eigenbasis.
    scaled, unusable = _correlation_eigenvalues(Hmat)
    if _is_saddle(scaled, unusable):
        # Resolved negative curvature: a saddle, not a fitted minimum. Curvature
        # smaller than this is not resolved by finite differencing at all, so it
        # falls through to be projected out below rather than having the sign of
        # numerical noise decide whether the function is scored or discarded.
        return theta, negloglike, nparam, np.inf

    b = V.T @ theta
    #  A direction whose curvature is unresolved relative to the parameter scales
    #  is not retained however far b projects along it: keeping it would leave the
    #  unresolved eigenvalue in the volume term, where the smaller it is the
    #  shorter the code.
    good = (eigenvalues > 0) & ~_unresolved_directions(Hmat, eigenvalues, V)
    has_degenerate = bool(np.min(scaled) < EIGENVALUE_REL_THRESHOLD)

    # (Near-)degenerate positive eigenvalues make the eigenbasis ill-conditioned:
    # any rotation within a near-equal subspace is an equally valid eigenbasis,
    # so both the retained-direction count and the codelength become
    # basis-sensitive. Warn (this is the "clustered eigenvalue" policy); the
    # result stays deterministic for the given Hessian. The clustering test is
    # per *pair* -- each consecutive gap relative to the larger eigenvalue of
    # that pair -- so a small eigenvalue near another small one is caught while
    # two well-separated small eigenvalues under a huge one (e.g. [1, 2, 1e6])
    # are not.
    good_vals = np.sort(eigenvalues[good])
    if good_vals.size >= 2:
        rel_gaps = np.diff(good_vals) / good_vals[1:]
        if np.min(rel_gaps) < CLUSTER_REL_TOL:
            emit_diagnostic_warning(
                'snap_choice=2 encountered (near-)degenerate Hessian '
                'eigenvalues for one or more functions; the projected '
                'codelength is basis-sensitive there -- prefer snap_choice=1.',
                ProjectedEigenbasisWarning)

    Nsteps_rot = np.zeros(nparam)
    Nsteps_rot[good] = np.abs(b[good]) / np.sqrt(12. / eigenvalues[good])
    kept = good & (Nsteps_rot >= 1)

    Hdiag = np.diag(eigenvalues)

    def eigen_codelen(mask):
        return _compute_codelen(Hdiag, eigenvalues, b, mask, use_det_I)

    # Unsnapped reference retains every constrained direction. A degenerate
    # Hessian has an undefined unsnapped determinant, so snapping is mandatory.
    codelen_nosnap = np.inf if has_degenerate else eigen_codelen(good)

    if np.array_equal(kept, good) and not has_degenerate:
        # Every direction is well constrained: nothing to snap.
        return theta, negloglike, int(np.sum(good)), codelen_nosnap

    # Snap: zero the weak projected coordinates and transform back.
    theta_snapped = V @ np.where(kept, b, 0.0)
    try:
        negloglike_snapped = eval_negloglike(theta_snapped)
    except Exception:
        # A broken evaluation at the snapped point is treated the same as a
        # non-finite value below (fall back to no-snap for a well-conditioned
        # Hessian, or an infinite codelength if it is degenerate), so one
        # pathological function cannot abort a whole-catalogue run. But an
        # *unexpected* exception is surfaced as a diagnostic rather than silently
        # becoming a score.
        emit_diagnostic_warning(
            'snap_choice=2 likelihood re-evaluation raised an exception for one '
            'or more functions; falling back to the unsnapped score, or to an '
            'infinite codelength if the Hessian is degenerate.',
            ProjectedEigenbasisWarning)
        negloglike_snapped = np.nan
    k = int(np.sum(kept))

    if not np.isfinite(negloglike_snapped):
        # The projected snap broke the likelihood. If the Hessian is degenerate
        # the unsnapped determinant is undefined, so the codelength is too.
        if has_degenerate:
            return theta, negloglike, int(np.sum(good)), np.inf
        return theta, negloglike, int(np.sum(good)), codelen_nosnap

    codelen_snap = eigen_codelen(kept)
    if has_degenerate:
        # Mandatory snap — the unsnapped determinant is not trustworthy.
        return theta_snapped, negloglike_snapped, k, codelen_snap
    if k > 0 and negloglike_snapped + codelen_snap < negloglike + codelen_nosnap:
        return theta_snapped, negloglike_snapped, k, codelen_snap
    # Well-conditioned Hessian but snapping did not help — keep all directions.
    return theta, negloglike, int(np.sum(good)), codelen_nosnap


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def load_loglike(comp, likelihood, data_start, data_end, split=True):
    """Load results of optimisation completed by test_all.py

    Args:
        :comp (int): complexity of functions to consider
        :likelihood (fitting.likelihood object): object containing data, likelihood functions and file paths
        :data_start (int): minimum index of results we want to load (only if split=True)
        :data_end (int): maximum index of results we want to load (only if split=True)
        :split (bool, deault=True): whether to return subset of results given by data_start and data_end (True) or all data (False)

    Returns:
        :negloglike (list): list of minimum log-likelihoods
        :params (np.ndarray): list of parameters at maximum likelihood points. Shape = (nfun, nparam).

    """
    fname = fitting_paths(comp, likelihood)['negloglike']
    if rank == 0:
        print(fname, flush=True)
    if split:
        with open(fname, 'r') as f:
            selected_lines = [line for i, line in enumerate(
                f) if data_start <= i < data_end]
        expected_rows = data_end - data_start
        if len(selected_lines) != expected_rows:
            raise ValueError(
                f'{fname} contains {len(selected_lines)} rows for requested '
                f'range [{data_start}, {data_end}); expected {expected_rows}. '
                'Rerun test_all.main with the current catalogue/settings.')
        if expected_rows == 0:
            return np.empty(0), np.zeros((0, 0))
        data = np.genfromtxt(selected_lines)
    else:
        data = np.genfromtxt(fname)
    data = np.atleast_2d(data)
    if data.size == 0:
        # Partial files are empty (e.g., parameterless functions).
        # Read NLLs from the main negloglike file instead.
        all_data = np.atleast_2d(np.genfromtxt(fname))
        if all_data.size == 0:
            nfun = data_end - data_start
            return np.full(nfun, np.inf), np.zeros((nfun, 0))
        negloglike = np.atleast_1d(all_data[data_start:data_end, 0])
        return negloglike, np.zeros((len(negloglike), 0))
    negloglike = np.atleast_1d(data[:, 0])
    params = np.atleast_2d(data[:, 1:])
    return negloglike, params


def convert_params(fcn_i, eq, integrated, theta_ML, likelihood, negloglike, max_param=4, use_det_I=True, snap_choice=1):
    """Compute Fisher, correct MLP and find parametric contirbution to description length for single function

    Args:
        :fcn_i (str): string representing function we wish to fit to data
        :eq (sympy object): sympy object for the function we wish to fit to data
        :integrated (bool): whether eq_numpy has already been integrated
        :theta_ML (list): the maximum likelihood values of the parameters
        :likelihood (fitting.likelihood object): object containing data, likelihood functions and file paths
        :negloglike (float): the minimum log-likelihood for this function
        :max_param (int, default=4): The maximum number of parameters considered. This sets the shapes of arrays used.
        :use_det_I (bool, default=True): If True, use the positive-definite full Hessian determinant for codelen. If False, use the published diagonal parameter-codelength formula for comparison.
        :snap_choice (int, default=1): Controls candidate parameter snapping.
            With 0, each parameter is assessed independently using its Hessian
            diagonal element. With 1, ESR diagonalises the full Hessian to
            identify directions with fewer than one precision step, then snaps
            the original parameter with the largest projection onto each such
            direction. With 2 (projected eigenbasis), ESR rotates into the
            Hessian eigenbasis, zeros the weak *projected* coordinates and
            transforms the retained vector back, re-evaluating the likelihood at
            that point *in the original parameterisation*; only the snap decision
            and the codelength are expressed in the eigenbasis. This requires
            ``use_det_I=True``.

    Returns:
        :params (list): the corrected maximum likelihood values of the parameters
        :negloglike (float): the corrected minimum log-likelihood for this function
        :deriv (list): flattened version of the Hessian of -log(likelihood) at the maximum likelihood point
        :codelen (float): the parameteric contribution to the description length of this function

    """

    _validate_snap_and_det(use_det_I, snap_choice)
    eq, active_params = test_all.canonicalize_parameter_symbols(eq)
    nparam = len(active_params)

    if nparam > 0:
        def fop(x):
            return likelihood.negloglike(x, eq_numpy, integrated=integrated)
    else:
        def fop(x):
            return likelihood.negloglike([x], eq_numpy, integrated=integrated)

    params = np.zeros(max_param)
    deriv = np.full(int(max_param * (max_param + 1) / 2), np.nan)

    #  Step-sizes to try in case the function misbehvaes
    d_list = [1.e-5, 10.**(-5.5), 10.**(-4.5), 1.e-6, 1.e-4, 10.**(-6.5), 10.**(-3.5),
              1.e-7, 1.e-3, 10.**(-7.5), 10.**(-2.5), 1.e-8, 1.e-2, 1.e-9, 1.e-10, 1.e-11]

    method_list = ["central", "forward", "backward"]

    if nparam == 0:
        codelen = 0
        return params, negloglike, deriv, codelen

    try:
        if nparam > 1:
            all_a = ' '.join([f'a{i}' for i in range(nparam)])
            all_a = list(sympy.symbols(all_a, real=True))
            eq_numpy = sympy.lambdify([x] + all_a, eq, modules=["numpy"])
        else:
            eq_numpy = sympy.lambdify([x, a0], eq, modules=["numpy"])
    except Exception:
        print("BAD:", fcn_i, negloglike, np.isfinite(negloglike))
        deriv[:] = np.nan
        return params, negloglike, deriv, np.inf

    # Get Hessian
    theta_ML = theta_ML[:nparam]
    Hfun = nd.Hessian(fop)
    Hmat = Hfun(theta_ML)
    Hmat_best = Hmat.copy()

    # 2nd derivatives of -log(L) wrt params
    Fisher_diag = np.array([Hmat[i, i] for i in range(nparam)])

    for i in range(nparam):
        start = int(i * max_param - (i - 1) * i / 2)
        deriv[start:start+nparam-i] = Hmat[i, i:]

    if snap_choice == 2:
        # Projected eigenbasis: score straight from the Hessian eigen-
        # decomposition, which removes zero/degenerate directions and rejects
        # saddles / non-finite curvature itself. This must run before the
        # diagonal retry/rejection below, whose ``np.all(np.diagonal > 0)``
        # filter would otherwise discard an exact flat direction that mode 2 can
        # legitimately project out (returning nan). eq_numpy is already built, so
        # fop can be passed directly.
        theta_snapped, negloglike, _, codelen = _score_projected_eigenbasis(
            Hmat_best, theta_ML[:nparam], negloglike, use_det_I, fop)
        params[:] = np.pad(theta_snapped, (0, max_param - len(theta_snapped)))
        return params, negloglike, deriv, codelen

    #  Precision to know constants (diagonal snap modes only; the mode-2
    #  branch above returns first, so a flat direction never divides by zero)
    Delta = np.sqrt(12./Fisher_diag)
    Nsteps = abs(np.array(theta_ML))/Delta

    n_iter = len(d_list)*len(method_list)
    # or (np.sum(Nsteps<1) > 0):
    if (np.sum(Fisher_diag <= 0.) > 0.) or (np.sum(np.isnan(Fisher_diag)) > 0) or (np.sum(np.isinf(Fisher_diag)) > 0):
        Fisher_array = np.empty((n_iter, nparam))
        Hmat_array = np.empty((n_iter, nparam, nparam))
        e = 0
        for d2, meth in itertools.product(d_list, method_list):
            if use_relative_dx:
                Hfun = nd.Hessian(fop, step=np.abs(
                    d2*theta_ML)+1.e-15, method=meth)
            else:
                Hfun = nd.Hessian(fop, step=d2, method=meth)
            Hmat = Hfun(theta_ML)
            Hmat_array[e] = Hmat
            e += 1

        Hmat_array_f = []  # filter array
        for matrix in Hmat_array:
            if not np.any(np.isnan(matrix)) and not np.any(np.isinf(matrix)) and np.all(np.diagonal(matrix) > 0):
                Hmat_array_f.append(matrix)
        Hmat_array_f = np.array(Hmat_array_f)
        Fisher_array = np.array(
            [np.array([mat[i, i] for i in range(nparam)]) for mat in Hmat_array_f])
        Delta_array = np.sqrt(12./Fisher_array)
        Delta_array_round = [[format(num, ".3e")
                              for num in row] for row in Delta_array]
        Delta_array_round = np.array(Delta_array_round, dtype=float)
        if len(Delta_array_round.shape) < 2:
            repeated_elements_exist = False
        else:
            repeated_elements_exist = len(Delta_array_round[:, 0]) != len(
                set(Delta_array_round[:, 0]))

        if repeated_elements_exist:
            Delta_mode = mode(Delta_array_round)[0][0]
            mode_ind = np.where(Delta_array_round == Delta_mode)[0][0]
            Fisher_diag = np.atleast_1d(Fisher_array[mode_ind])
            # Delta, Nsteps = np.atleast_1d(Delta_array[mode_ind]), np.atleast_1d(Nsteps_array[mode_ind])
            Delta = np.sqrt(12./Fisher_diag)
            Nsteps = abs(np.array(theta_ML))/Delta
            Hmat_best = Hmat_array_f[mode_ind].copy()
            for i in range(nparam):
                start = int(i * max_param - (i - 1) * i / 2)
                deriv[start:start+nparam-i] = Hmat_array_f[mode_ind][i, i:]

        else:  # try again with less precision
            Delta_array_round = [[format(num, ".1e")
                                  for num in row] for row in Delta_array]
            Delta_array_round = np.array(Delta_array_round, dtype=float)
            if len(Delta_array_round.shape) < 2:
                repeated_elements_exist = False
            else:
                repeated_elements_exist = len(Delta_array_round[:, 0]) != len(
                    set(Delta_array_round[:, 0]))
            if not repeated_elements_exist:
                codelen = np.nan
                return params, negloglike, deriv, codelen
            else:
                Delta_mode = mode(Delta_array_round)[0][0]
                mode_ind = np.where(Delta_array_round == Delta_mode)[0][0]
                Fisher_diag = np.atleast_1d(Fisher_array[mode_ind])
                Delta = np.sqrt(12./Fisher_diag)
                Nsteps = abs(np.array(theta_ML))/Delta
                Hmat_best = Hmat_array_f[mode_ind].copy()
                for i in range(nparam):
                    start = int(i * max_param - (i - 1) * i / 2)
                    deriv[start:start+nparam-i] = Hmat_array_f[mode_ind][i, i:]

    if (np.sum(Fisher_diag <= 0.) > 0.) or (np.sum(np.isnan(Fisher_diag)) > 0):
        return params, negloglike, deriv, np.inf

    # Require all Hessian eigenvalues to be positive (a genuine minimum).
    if use_det_I and _has_negative_curvature(Hmat_best):
        return params, negloglike, deriv, np.inf

    k = nparam
    theta_ML_orig = np.copy(theta_ML)
    negloglike_orig = np.copy(negloglike)

    Nsteps, has_degenerate_eig = _compute_snap_mask(Hmat_best, Fisher_diag, theta_ML, Nsteps, snap_choice)

    # Compute unsnapped DL (for comparison if snapping is attempted)
    all_mask = np.ones(nparam, dtype=bool)
    codelen_nosnap = _compute_codelen(Hmat_best, Fisher_diag, theta_ML, all_mask, use_det_I)
    DL_nosnap = negloglike + codelen_nosnap

    # See whether we can snap any parameters to zero
    if np.sum(Nsteps < 1) > 0:

        # First try setting any parameter to 0 that doesn't have at least
        # one precision step, and recompute -log(L).
        theta_ML[Nsteps < 1] = 0.
        if snap_choice == 0:
            #  Published diagonal behaviour: score at the zeroed vector itself
            negloglike = fop(theta_ML)
        else:
            #  The eigenbasis modes snap a parameter chosen from a rotated
            #  direction, so the remaining parameters are generally no longer at
            #  their own optimum once it is removed. Re-fit them, or an
            #  unresolved direction gets rejected purely because the reduced
            #  model was evaluated at the wrong point.
            theta_ML, negloglike = _refit_after_snap(fop, theta_ML, Nsteps >= 1)

        # For the codelen, we effectively don't have the parameter that had Nsteps<1
        if np.isfinite(negloglike):
            k -= np.sum(Nsteps < 1)
            kept_mask = Nsteps >= 1
        else:
            #  Let's see if setting any of the parameters to zero is ok
            try_idx = np.arange(nparam)[Nsteps < 1]
            for r in reversed(range(1, len(try_idx))):
                for idx in itertools.combinations(try_idx, r):
                    theta_ML = np.copy(theta_ML_orig)
                    for idx_ in idx:
                        theta_ML[idx_] = 0.
                    negloglike = fop(theta_ML)
                    if np.isfinite(negloglike):
                        break
            kept_mask = np.ones(len(theta_ML), dtype=bool)
            if np.isfinite(negloglike):
                k -= len(idx)
                kept_mask[idx] = 0
            else:
                theta_ML = theta_ML_orig
                negloglike = negloglike_orig
                k = nparam
                if has_degenerate_eig:
                    # No way of removing the unconstrained direction leaves a
                    # usable likelihood, and the unsnapped determinant still
                    # contains that direction, so there is no codelength here
                    # that can be trusted. Reverting silently would hand the
                    # function the very score the mandatory snap exists to
                    # prevent.
                    params[:] = np.pad(
                        theta_ML_orig, (0, max_param - len(theta_ML_orig)))
                    return params, negloglike_orig, deriv, np.inf

        if k < 0:
            print("This shouldn't have happened", flush=True)
            quit()

        # Compute snapped codelen and compare DL. theta_ML is theta_ML_orig with
        # the snapped entries zeroed, plus any re-fit of the retained ones, and
        # _compute_codelen reads only the retained entries.
        codelen_snap = _compute_codelen(Hmat_best, Fisher_diag, theta_ML, kept_mask, use_det_I)
        DL_snap = negloglike + codelen_snap

        if has_degenerate_eig:
            # Mandatory snap — an unresolved direction is still in the unsnapped
            # determinant, where the smaller its eigenvalue the shorter the code
            # it produces, so the DL comparison cannot be trusted here.
            pass
        elif k == 0 or DL_snap >= DL_nosnap:
            # Well-conditioned Hessian but snapping didn't help — revert
            theta_ML = theta_ML_orig
            negloglike = negloglike_orig
            k = nparam
            kept_mask = np.ones(nparam, dtype=bool)
    else:
        kept_mask = np.ones(len(theta_ML), dtype=bool)

    # Log condition number for diagnostics
    H_active = Hmat_best[np.ix_(kept_mask, kept_mask)]
    if H_active.size > 0:
        try:
            cond = np.linalg.cond(H_active)
            if cond > 1e10:
                emit_diagnostic_warning(
                    'One or more fitted Hessians are badly conditioned '
                    '(condition number > 1e10); their parameter codelengths may '
                    'be unreliable.', HighConditionNumberWarning)
        except np.linalg.LinAlgError:
            pass

    # Compute final codelen. theta_ML holds the reverted, snapped or re-fitted
    # vector depending on which branch above was taken, so the encoded
    # parameters and the reported ones are the same numbers.
    codelen = _compute_codelen(Hmat_best, Fisher_diag, theta_ML, kept_mask, use_det_I)

    # New params after the setting to 0, padded to length max_param as always
    theta_ML = np.asarray(theta_ML, dtype=float).copy()
    theta_ML[~kept_mask] = 0.
    params[:] = np.pad(theta_ML, (0, max_param-len(theta_ML)))

    return params, negloglike, deriv, codelen


def main(comp, likelihood, tmax=5, print_frequency=50, try_integration=False, use_det_I=True, snap_choice=1):
    """Compute Fisher, correct MLP and find parametric contirbution to description length for all functions and save to file

    Args:
        :comp (int): complexity of functions to consider
        :likelihood (fitting.likelihood object): object containing data, likelihood functions and file paths
        :tmax (float, default=5.): maximum time in seconds to run any one part of simplification procedure for a given function
        :print_frequency (int, default=50): the status of the fits will be printed every ``print_frequency`` number of iterations
        :try_integration (bool, default=False): when likelihood requires integral, whether to try to analytically integrate (True) or just numerically integrate (False)
        :use_det_I (bool, default=True): If True, use a positive-definite full Hessian determinant for codelen. If False, use the published diagonal parameter-codelength formula for comparison.
        :snap_choice (int, default=1): Controls parameter snapping. 0: diagonal,
            1: eigenbasis (zeros the original parameter with the largest
            projection onto each weak direction), 2: projected eigenbasis (zeros
            the weak projected coordinate and scores in the eigenbasis; requires
            ``use_det_I=True``). See :func:`convert_params` for the detailed
            definitions.

    Returns:
        None

    """

    if likelihood.is_mse:
        raise ValueError('Cannot use MSE with description length')
    _validate_snap_and_det(use_det_I, snap_choice)

    if rank == 0:
        print('\nComputing Fisher', flush=True)

    set_recursionlimit_for_comp(comp)

    test_all.ensure_likelihood_catalogue(comp, likelihood, tmax, try_integration)
    fcn_list_proc, data_start, data_end = test_all.get_functions(
        comp, likelihood)
    if rank == 0:
        save_scoring_settings(comp, likelihood, use_det_I, snap_choice)
    comm.Barrier()
    negloglike, params_proc = load_loglike(
        comp, likelihood, data_start, data_end)
    max_param = params_proc.shape[1]

    # This is now only for this proc
    codelen = np.zeros(len(fcn_list_proc))
    params = np.zeros([len(fcn_list_proc), max_param])
    deriv = np.zeros([len(fcn_list_proc), int(max_param * (max_param+1) / 2)])

    for i in range(len(fcn_list_proc)):           # Consider all possible complexities
        if rank == 0 and ((i == 0) or ((i+1) % print_frequency == 0)):
            print(f'{i+1} of {len(fcn_list_proc)}', flush=True)

        if np.isnan(negloglike[i]) or np.isinf(negloglike[i]):
            codelen[i] = np.nan
            continue

        theta_ML = params_proc[i, :]

        try:
            fcn_i = fcn_list_proc[i].replace('\n', '')
            fcn_i = fcn_list_proc[i].replace('\'', '')
            fcn_i, eq, integrated = likelihood.run_sympify(
                fcn_i, tmax=tmax, try_integration=try_integration)
            params[i, :], negloglike[i], deriv[i, :], codelen[i] = convert_params(
                fcn_i, eq, integrated, theta_ML, likelihood, negloglike[i], max_param=max_param, use_det_I=use_det_I, snap_choice=snap_choice)
        except NameError:
            # Occurs if function produced not implemented in numpy
            if try_integration:
                fcn_i = fcn_list_proc[i].replace('\n', '')
                fcn_i = fcn_list_proc[i].replace('\'', '')
                fcn_i, eq, integrated = likelihood.run_sympify(
                    fcn_i, tmax=tmax, try_integration=False)
                params[i, :], negloglike[i], deriv[i, :], codelen[i] = convert_params(
                    fcn_i, eq, integrated, theta_ML, likelihood, negloglike[i], max_param=max_param, use_det_I=use_det_I, snap_choice=snap_choice)
            else:
                params[i, :] = 0.
                deriv[i, :] = np.nan
                codelen[i] = np.inf

        except Exception:
            params[i, :] = 0.
            deriv[i, :] = np.nan
            codelen[i] = np.inf

    n_nonposdef = np.sum(np.isinf(codelen))
    total_nonposdef = comm.reduce(int(n_nonposdef), op=MPI.SUM, root=0)
    if rank == 0 and total_nonposdef > 0:
        print(f'Warning: {total_nonposdef} functions had non-positive-definite Hessian (codelen=inf)', flush=True)

    out_arr = np.transpose(
        np.vstack([codelen, negloglike] + [params[:, i] for i in range(max_param)]))

    if deriv.shape[1] > 0:
        out_arr_deriv = np.transpose(
            np.vstack([deriv[:, i] for i in range(deriv.shape[1])]))
    else:
        out_arr_deriv = np.empty((len(codelen), 0))

    paths = fitting_paths(comp, likelihood, rank=rank)
    np.savetxt(paths['codelen_rank'], out_arr, fmt='%.7e')
    np.savetxt(paths['derivs_rank'], out_arr_deriv, fmt='%.7e')

    comm.Barrier()

    if rank == 0:
        combine_temp_files(
            likelihood.temp_dir,
            paths['codelen_rank_pattern'],
            paths['codelen'])
        combine_temp_files(
            likelihood.temp_dir,
            paths['derivs_rank_pattern'],
            paths['derivs'])

    comm.Barrier()

    return
