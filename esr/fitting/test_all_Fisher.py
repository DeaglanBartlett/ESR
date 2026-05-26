import numpy as np
import math
import sympy
from mpi4py import MPI
import warnings
import os
import sys
import itertools
import json
import numdifftools as nd
from scipy.stats import mode

import esr.fitting.test_all as test_all
from esr.fitting.sympy_symbols import x, a0

warnings.filterwarnings("ignore")

use_relative_dx = True              # CHANGE

# Eigenvalues below this fraction of the largest are treated as degenerate
# (unconstrained direction in parameter space). This prevents det(H)→0
# from corrupting the codelen when parameters are structurally redundant.
EIGENVALUE_REL_THRESHOLD = 1e-10


def _validate_scoring_options(use_det_I, snap_choice):
    if snap_choice not in (0, 1):
        raise ValueError("snap_choice must be 0 or 1.")


def _symmetrized_hessian(Hmat):
    Hmat = np.asarray(Hmat, dtype=float)
    return 0.5 * (Hmat + Hmat.T)


def _has_negative_curvature(Hmat):
    """Return True for a Hessian with a resolved negative eigendirection."""
    try:
        eigenvalues = np.linalg.eigvalsh(_symmetrized_hessian(Hmat))
    except np.linalg.LinAlgError:
        return True
    scale = max(np.max(np.abs(eigenvalues)), 1.0)
    return np.min(eigenvalues) < -scale * EIGENVALUE_REL_THRESHOLD


def _settings_file(comp, likelihood):
    return os.path.join(likelihood.out_dir, 'fisher_settings_comp' + str(comp) + '.json')


def save_scoring_settings(comp, likelihood, use_det_I, snap_choice):
    """Record Fisher scoring settings so matching cannot silently change them."""
    _validate_scoring_options(use_det_I, snap_choice)
    with open(_settings_file(comp, likelihood), 'w') as f:
        json.dump({'use_det_I': bool(use_det_I), 'snap_choice': int(snap_choice)}, f)


def load_scoring_settings(comp, likelihood):
    try:
        with open(_settings_file(comp, likelihood), 'r') as f:
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


def _compute_snap_mask(Hmat, Fisher_diag, theta, Nsteps, snap_choice):
    """Compute which parameters to snap to zero based on snap_choice.

    For snap_choice 1, eigendecomposes Hmat to identify unconstrained
    directions, then maps them back to original parameters.

    Args:
        :Hmat (np.ndarray): full Hessian matrix (nparam x nparam)
        :Fisher_diag (np.ndarray): diagonal of the Hessian (nparam,)
        :theta (np.ndarray): parameter values (nparam,)
        :Nsteps (np.ndarray): diagonal-based Nsteps (used for snap_choice=0 and as fallback)
        :snap_choice (int): 0=diagonal, 1=full eigenbasis (uses rotated
            theta for Nsteps).

    Returns:
        :Nsteps (np.ndarray): updated Nsteps array (values < 1 indicate parameters to snap)
        :has_degenerate_eig (bool): True if any eigenvalue is below EIGENVALUE_REL_THRESHOLD
            relative to the largest. Used to decide whether snap is mandatory.
    """
    nparam = len(theta)
    has_degenerate_eig = False

    _validate_scoring_options(True, snap_choice)
    if snap_choice == 0:
        return Nsteps, has_degenerate_eig

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(
            _symmetrized_hessian(Hmat[:nparam, :nparam]))
        scale = max(np.max(np.abs(eigenvalues)), 1.0)
        if np.min(eigenvalues) < -scale * EIGENVALUE_REL_THRESHOLD:
            return Nsteps, has_degenerate_eig
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
        :snap_choice (int, default=1): Controls how parameters are snapped
            to zero. 0: diagonal approach. 1: use the rotated eigenbasis.

    Returns:
        :params (list): the corrected maximum likelihood values of the parameters
        :negloglike (float): the corrected minimum log-likelihood for this function
        :deriv (list): flattened version of the Hessian of -log(likelihood) at the maximum likelihood point
        :codelen (float): the parameteric contribution to the description length of this function

    """

    _validate_scoring_options(use_det_I, snap_choice)
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

    if (np.sum(Fisher_diag <= 0.) > 0.) or (np.sum(np.isnan(Fisher_diag)) > 0):
        return params, negloglike, deriv, np.inf

    # A positive determinant is insufficient: an even number of negative
    # eigenvalues is a saddle, not a fitted minimum.
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


def main(comp, likelihood, tmax=5, print_frequency=50, try_integration=False, use_det_I=True, snap_choice=1):
    """Compute Fisher, correct MLP and find parametric contirbution to description length for all functions and save to file

    Args:
        :comp (int): complexity of functions to consider
        :likelihood (fitting.likelihood object): object containing data, likelihood functions and file paths
        :tmax (float, default=5.): maximum time in seconds to run any one part of simplification procedure for a given function
        :print_frequency (int, default=50): the status of the fits will be printed every ``print_frequency`` number of iterations
        :try_integration (bool, default=False): when likelihood requires integral, whether to try to analytically integrate (True) or just numerically integrate (False)
        :use_det_I (bool, default=True): If True, use a positive-definite full Hessian determinant for codelen. If False, use the published diagonal parameter-codelength formula for comparison.
        :snap_choice (int, default=1): Controls parameter snapping. 0: diagonal, 1: eigenbasis.

    Returns:
        None

    """

    if likelihood.is_mse:
        raise ValueError('Cannot use MSE with description length')
    _validate_scoring_options(use_det_I, snap_choice)

    if rank == 0:
        print('\nComputing Fisher', flush=True)

    if comp >= 8:
        sys.setrecursionlimit(2000 + 500 * (comp - 8))

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

    np.savetxt(likelihood.temp_dir + '/codelen_deriv_' +
               str(comp)+'_'+str(rank)+'.dat', out_arr, fmt='%.7e')
    np.savetxt(likelihood.temp_dir + '/derivs_'+str(comp) +
               '_'+str(rank)+'.dat', out_arr_deriv, fmt='%.7e')

    comm.Barrier()

    if rank == 0:
        test_all.combine_temp_files(
            likelihood.temp_dir,
            'codelen_deriv_' + str(comp) + '_*.dat',
            likelihood.out_dir + '/codelen_comp' + str(comp) + '_deriv.dat')
        test_all.combine_temp_files(
            likelihood.temp_dir,
            'derivs_' + str(comp) + '_*.dat',
            likelihood.out_dir + '/derivs_comp' + str(comp) + '.dat')

    comm.Barrier()

    return
