import numpy as np
import math
import sympy
from mpi4py import MPI
import warnings
import os
import sys
import itertools
import numdifftools as nd
from scipy.stats import mode

import esr.fitting.test_all as test_all
from esr.fitting.sympy_symbols import x, a0
import esr.generation.simplifier as simplifier

warnings.filterwarnings("ignore")

use_relative_dx = True              # CHANGE

# Eigenvalues below this fraction of the largest are treated as degenerate
# (unconstrained direction in parameter space). This prevents det(H)→0
# from corrupting the codelen when parameters are structurally redundant.
EIGENVALUE_REL_THRESHOLD = 1e-10


def _compute_codelen(Hmat, Fisher_diag, theta, kept_mask, use_det_I):
    """Compute parametric codelen for a given set of kept parameters.

    Applies a floor of 0 to each ln(|theta_i|/Delta_i) contribution,
    where Delta_i = sqrt(12/H_ii). This prevents parameters with
    |theta| < Delta (i.e. value smaller than measurement precision)
    from artificially reducing the description length.

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

    # Floor: ln|theta_i| >= ln(Delta_i) where Delta_i = sqrt(12/H_ii)
    # This ensures each parameter contributes >= 0 to ln(|theta|/Delta).
    log_theta_floored = np.empty(k)
    for j in range(k):
        if diag_active[j] > 0:
            log_delta = 0.5 * np.log(12. / diag_active[j])
            log_theta_floored[j] = max(np.log(np.abs(theta_active[j])), log_delta)
        else:
            log_theta_floored[j] = np.log(np.abs(theta_active[j]))

    if use_det_I:
        H_active = Hmat[np.ix_(kept_mask, kept_mask)]
        sign, logdet = np.linalg.slogdet(H_active)
        if sign > 0:
            return -k/2. * math.log(3.) + 0.5 * logdet + \
                np.sum(log_theta_floored)
        else:
            return np.inf
    else:
        return -k/2. * math.log(3.) + np.sum(0.5*np.log(diag_active) +
                                              log_theta_floored)


def _compute_snap_mask(Hmat, Fisher_diag, theta, Nsteps, snap_choice):
    """Compute which parameters to snap to zero based on snap_choice.

    For snap_choice 1 or 2, eigendecomposes Hmat to identify unconstrained
    directions, then maps them back to original parameters. Falls back to the
    diagonal-based Nsteps if eigendecomposition fails.

    Args:
        :Hmat (np.ndarray): full Hessian matrix (nparam x nparam)
        :Fisher_diag (np.ndarray): diagonal of the Hessian (nparam,)
        :theta (np.ndarray): parameter values (nparam,)
        :Nsteps (np.ndarray): diagonal-based Nsteps (used for snap_choice=0 and as fallback)
        :snap_choice (int): 0=diagonal, 1=eigen-informed (uses diagonal Nsteps in eigenbasis),
            2=full eigenbasis (uses rotated theta for Nsteps)

    Returns:
        :Nsteps (np.ndarray): updated Nsteps array (values < 1 indicate parameters to snap)
        :has_degenerate_eig (bool): True if any eigenvalue is below EIGENVALUE_REL_THRESHOLD
            relative to the largest. Used to decide whether snap is mandatory.
    """
    nparam = len(theta)
    has_degenerate_eig = False

    if snap_choice not in (1, 2):
        return Nsteps, has_degenerate_eig

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(Hmat[:nparam, :nparam])
        theta_rot = eigenvectors.T @ theta
        # Eigenvalues that are non-positive OR negligibly small relative to the
        # largest indicate degenerate/unconstrained directions. Use a relative
        # threshold to catch near-zero eigenvalues from parameter redundancies
        # (e.g. g and c*g having the same f_DE = g/g(1)).
        eig_threshold = max(eigenvalues.max(), 1.0) * EIGENVALUE_REL_THRESHOLD
        good_eig = eigenvalues > eig_threshold
        has_degenerate_eig = not np.all(good_eig)
        Nsteps_rot = np.zeros(nparam)
        Nsteps_rot[good_eig] = np.abs(theta_rot[good_eig]) / np.sqrt(12. / eigenvalues[good_eig])
        # Map unconstrained eigendirections back to original parameters:
        # for each bad eigendirection, snap the original param with largest projection
        bad_eig = np.where(Nsteps_rot < 1)[0]
        snap_set = set()
        for ei in bad_eig:
            snap_set.add(np.argmax(np.abs(eigenvectors[:, ei])))
        Nsteps = np.ones(nparam)
        for j in snap_set:
            Nsteps[j] = 0.
    except np.linalg.LinAlgError:
        has_degenerate_eig = True  # can't decompose — treat as degenerate

    return Nsteps, has_degenerate_eig

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
    fname = likelihood.out_dir + "/negloglike_comp" + str(comp) + ".dat"
    if rank == 0:
        print(fname, flush=True)
    if split:
        with open(fname, 'r') as f:
            selected_lines = [line for i, line in enumerate(
                f) if data_start <= i < data_end]
        data = np.genfromtxt(selected_lines)
    else:
        data = np.genfromtxt(fname)
    data = np.atleast_2d(data)
    negloglike = np.atleast_1d(data[:, 0])
    params = np.atleast_2d(data[:, 1:])
    return negloglike, params


def convert_params(fcn_i, eq, integrated, theta_ML, likelihood, negloglike, max_param=4, use_det_I=True, snap_choice=2):
    """Compute Fisher, correct MLP and find parametric contirbution to description length for single function

    Args:
        :fcn_i (str): string representing function we wish to fit to data
        :eq (sympy object): sympy object for the function we wish to fit to data
        :integrated (bool): whether eq_numpy has already been integrated
        :theta_ML (list): the maximum likelihood values of the parameters
        :likelihood (fitting.likelihood object): object containing data, likelihood functions and file paths
        :negloglike (float): the minimum log-likelihood for this function
        :max_param (int, default=4): The maximum number of parameters considered. This sets the shapes of arrays used.
        :use_det_I (bool, default=True): If True, use full Hessian determinant for codelen (captures parameter degeneracies). If False, use diagonal elements only (original ESR behaviour).
        :snap_choice (int, default=2): Controls how parameters are snapped to zero. 0: original diagonal approach (Nsteps_i = |theta_i|/sqrt(12/H_ii)). 1: eigendecompose H, identify unconstrained eigendirections, snap the original parameter with largest projection onto each. 2: same as 1 but Nsteps computed in rotated eigenbasis (theta_rot = V^T @ theta, Nsteps_i = |theta_rot_i|/sqrt(12/d_i)). Codelen formula is the same for all modes.

    Returns:
        :params (list): the corrected maximum likelihood values of the parameters
        :negloglike (float): the corrected minimum log-likelihood for this function
        :deriv (list): flattened version of the Hessian of -log(likelihood) at the maximum likelihood point
        :codelen (float): the parameteric contribution to the description length of this function

    """

    nparam = simplifier.count_params([fcn_i], max_param)[0]

    # If run_sympify reduced the expression (integrated=True), the actual
    # number of free parameters may be less than what the original string
    # suggests (e.g. g(x)=a0 -> f_DE=1, eliminating a0).
    if integrated:
        try:
            eq_free = eq.free_symbols - {x}
            nparam_actual = len(eq_free)
            if nparam_actual < nparam:
                nparam = nparam_actual
        except Exception:
            pass

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
        Fisher_diag = np.nan
        deriv[:] = np.nan
        return params, negloglike, deriv, codelen

    # Get Hessian
    theta_ML = theta_ML[:nparam]
    Hfun = nd.Hessian(fop)
    Hmat = Hfun(theta_ML)
    Hmat_best = Hmat.copy()

    # 2nd derivatives of -log(L) wrt params
    Fisher_diag = np.array([Hmat[i, i] for i in range(nparam)])

    #  Precision to know constants
    Delta = np.sqrt(12./Fisher_diag)
    Nsteps = abs(np.array(theta_ML))/Delta

    for i in range(nparam):
        start = int(i * max_param - (i - 1) * i / 2)
        deriv[start:start+nparam-i] = Hmat[i, i:]

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

    # Fisher_diag <= 0 means we're not at a minimum.  Before giving up,
    # re-optimise: try Nelder-Mead from the current point, then a grid of
    # starting points in log-space to catch cases where test_all landed far
    # from the global minimum.
    if (np.sum(Fisher_diag <= 0.) > 0.) or (np.sum(np.isnan(Fisher_diag)) > 0):
        from scipy.optimize import minimize as _minimize

        best_nll = negloglike
        best_theta = theta_ML.copy()

        # Phase 1: Nelder-Mead from current point
        try:
            res = _minimize(fop, theta_ML, method='Nelder-Mead',
                            options={'xatol': 1e-8, 'fatol': 1e-10,
                                     'maxiter': 5000 * nparam})
            if np.isfinite(res.fun) and res.fun < best_nll:
                best_nll = res.fun
                best_theta = res.x.copy()
        except Exception:
            pass

        # Phase 2: multi-start from log-spaced grid covering |a| in [0.1, 100]
        # with both signs, to catch minima far from the test_all result
        _log_starts = np.linspace(-1, 2, 7)  # 10^[-1..2] = [0.1, 100]
        _sign_combos = list(itertools.product([1, -1], repeat=nparam))
        for signs in _sign_combos:
            for log_vals in itertools.product(_log_starts, repeat=nparam):
                x0 = np.array([s * 10**lv for s, lv in zip(signs, log_vals)])
                try:
                    res = _minimize(fop, x0, method='Nelder-Mead',
                                    options={'xatol': 1e-8, 'fatol': 1e-10,
                                             'maxiter': 3000 * nparam})
                    if np.isfinite(res.fun) and res.fun < best_nll:
                        best_nll = res.fun
                        best_theta = res.x.copy()
                except Exception:
                    pass

        if best_nll <= negloglike + 0.01:
            theta_ML = best_theta
            negloglike = best_nll
            # Recompute Hessian at the new point
            Hfun_retry = nd.Hessian(fop)
            Hmat_best = Hfun_retry(theta_ML)
            Fisher_diag = np.array([Hmat_best[i, i] for i in range(nparam)])
            for i in range(nparam):
                start = int(i * max_param - (i - 1) * i / 2)
                deriv[start:start+nparam-i] = Hmat_best[i, i:]
            if use_relative_dx:
                # Also try relative step sizes if default Hessian fails
                if (np.sum(Fisher_diag <= 0.) > 0.) or (np.sum(np.isnan(Fisher_diag)) > 0):
                    for d2 in [1.e-5, 1.e-4, 1.e-3, 1.e-6]:
                        Hfun_retry = nd.Hessian(fop, step=np.abs(d2 * theta_ML) + 1e-15)
                        Hmat_try = Hfun_retry(theta_ML)
                        diag_try = np.array([Hmat_try[i, i] for i in range(nparam)])
                        if np.all(diag_try > 0) and not np.any(np.isnan(diag_try)):
                            Hmat_best = Hmat_try
                            Fisher_diag = diag_try
                            for i in range(nparam):
                                start = int(i * max_param - (i - 1) * i / 2)
                                deriv[start:start+nparam-i] = Hmat_best[i, i:]
                            break
            Delta = np.sqrt(12. / Fisher_diag)
            Nsteps = abs(np.array(theta_ML)) / Delta
            print(f'Re-optimised {fcn_i}: NLL {negloglike:.4f}, '
                  f'theta={theta_ML}, Fisher_diag={Fisher_diag}', flush=True)

        # If still bad after re-optimisation, give up
        if (np.sum(Fisher_diag <= 0.) > 0.) or (np.sum(np.isnan(Fisher_diag)) > 0):
            codelen = np.nan
            return params, negloglike, deriv, codelen

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
        negloglike = fop(theta_ML)

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

        if k < 0:
            print("This shouldn't have happened", flush=True)
            quit()

        # Compute snapped codelen and compare DL.
        # If Hessian has degenerate eigenvalues (detected by _compute_snap_mask),
        # snap is mandatory — reverting would allow det(H)→0 to give
        # artificially low codelen.
        codelen_snap = _compute_codelen(Hmat_best, Fisher_diag, theta_ML_orig, kept_mask, use_det_I)
        DL_snap = negloglike + codelen_snap

        if has_degenerate_eig:
            # Mandatory snap — Hessian is degenerate, don't trust DL comparison
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
                print(f'Warning: high condition number {cond:.2e} for {fcn_i}', flush=True)
        except np.linalg.LinAlgError:
            pass

    # Compute final codelen
    codelen = _compute_codelen(Hmat_best, Fisher_diag, theta_ML_orig, kept_mask, use_det_I)

    # New params after the setting to 0, padded to length max_param as always
    theta_ML = theta_ML_orig
    theta_ML[~kept_mask] = 0.
    params[:] = np.pad(theta_ML, (0, max_param-len(theta_ML)))

    return params, negloglike, deriv, codelen


def main(comp, likelihood, tmax=5, print_frequency=50, try_integration=False, use_det_I=True, snap_choice=2):
    """Compute Fisher, correct MLP and find parametric contirbution to description length for all functions and save to file

    Args:
        :comp (int): complexity of functions to consider
        :likelihood (fitting.likelihood object): object containing data, likelihood functions and file paths
        :tmax (float, default=5.): maximum time in seconds to run any one part of simplification procedure for a given function
        :print_frequency (int, default=50): the status of the fits will be printed every ``print_frequency`` number of iterations
        :try_integration (bool, default=False): when likelihood requires integral, whether to try to analytically integrate (True) or just numerically integrate (False)
        :use_det_I (bool, default=True): If True, use full Hessian determinant for codelen. If False, use diagonal elements only.
        :snap_choice (int, default=2): Controls parameter snapping. 0: diagonal, 1: eigen-informed original-space, 2: full eigenbasis.

    Returns:
        None

    """

    if likelihood.is_mse:
        raise ValueError('Cannot use MSE with description length')

    if rank == 0:
        print('\nComputing Fisher', flush=True)

    if comp >= 8:
        sys.setrecursionlimit(2000 + 500 * (comp - 8))

    fcn_list_proc, data_start, data_end = test_all.get_functions(
        comp, likelihood)
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
                deriv[i, :] = 0.
                codelen[i] = 0

        except Exception:
            params[i, :] = 0.
            deriv[i, :] = 0.
            codelen[i] = 0

    n_nonposdef = np.sum(np.isinf(codelen))
    total_nonposdef = comm.reduce(int(n_nonposdef), op=MPI.SUM, root=0)
    if rank == 0 and total_nonposdef > 0:
        print(f'Warning: {total_nonposdef} functions had non-positive-definite Hessian (codelen=inf)', flush=True)

    out_arr = np.transpose(
        np.vstack([codelen, negloglike] + [params[:, i] for i in range(max_param)]))

    out_arr_deriv = np.transpose(np.vstack([deriv[:, 0], deriv[:, 1], deriv[:, 2], deriv[:, 3],
                                 deriv[:, 4], deriv[:, 5], deriv[:, 6], deriv[:, 7], deriv[:, 8], deriv[:, 9]]))
    out_arr_deriv = np.transpose(
        np.vstack([deriv[:, i] for i in range(deriv.shape[1])]))

    np.savetxt(likelihood.temp_dir + '/codelen_deriv_' +
               str(comp)+'_'+str(rank)+'.dat', out_arr, fmt='%.7e')
    np.savetxt(likelihood.temp_dir + '/derivs_'+str(comp) +
               '_'+str(rank)+'.dat', out_arr_deriv, fmt='%.7e')

    comm.Barrier()

    if rank == 0:
        string = 'cat `find ' + likelihood.temp_dir + '/ -name "codelen_deriv_' + \
            str(comp)+'_*.dat" | sort -V` > ' + likelihood.out_dir + \
            '/codelen_comp'+str(comp)+'_deriv.dat'
        os.system(string)
        string = 'rm ' + likelihood.temp_dir + \
            '/codelen_deriv_'+str(comp)+'_*.dat'
        os.system(string)

        string = 'cat `find ' + likelihood.temp_dir + '/ -name "derivs_' + \
            str(comp)+'_*.dat" | sort -V` > ' + \
            likelihood.out_dir + '/derivs_comp'+str(comp)+'.dat'
        os.system(string)
        string = 'rm ' + likelihood.temp_dir + '/derivs_'+str(comp)+'_*.dat'
        os.system(string)

    comm.Barrier()

    return
