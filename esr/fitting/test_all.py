import numpy as np
import sympy
import warnings
import os
import sys
import json
import glob
import re
from mpi4py import MPI
from scipy.optimize import minimize
import itertools

from esr.fitting.sympy_symbols import x, a0
import esr.generation.simplifier as simplifier

warnings.filterwarnings("ignore")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

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


def _natural_sort_key(path):
    return [int(text) if text.isdigit() else text
            for text in re.split(r'(\d+)', path)]


def combine_temp_files(temp_dir, pattern, output_file, remove=True):
    """Concatenate rank-local temporary files in natural-sort order."""
    paths = sorted(glob.glob(os.path.join(temp_dir, pattern)),
                   key=_natural_sort_key)
    with open(output_file, 'w') as fout:
        for path in paths:
            with open(path, 'r') as fin:
                fout.writelines(fin)
    if remove:
        for path in paths:
            os.remove(path)


def write_negloglike_file(path, chi2, params, max_param):
    out_arr = np.transpose(
        np.vstack([chi2] + [params[:, i] for i in range(max_param)]))
    np.savetxt(path, out_arr, fmt='%.7e')


def function_catalogue_path(comp, likelihood, unique=True):
    if unique and likelihood_catalogue_active(comp, likelihood):
        return likelihood_catalogue_paths(comp, likelihood)['unique']
    if unique:
        return likelihood.fn_dir + \
            "/compl_%i/unique_equations_%i.txt" % (comp, comp)
    return likelihood.fn_dir + \
        "/compl_%i/all_equations_%i.txt" % (comp, comp)


def set_recursionlimit_for_comp(comp):
    if comp >= 8:
        sys.setrecursionlimit(2000 + 500 * (comp - 8))


def likelihood_catalogue_paths(comp, likelihood):
    prefix = os.path.join(likelihood.out_dir, 'likelihood_catalogue_comp' + str(comp))
    return {
        'unique': prefix + '_unique_equations.txt',
        'matches': prefix + '_matches.txt',
        'metadata': prefix + '_metadata.json',
    }


def _likelihood_catalogue_settings(tmax, try_integration):
    return {
        'version': 2,
        'tmax': float(tmax),
        'try_integration': bool(try_integration),
    }


def _read_likelihood_catalogue_metadata(comp, likelihood):
    paths = likelihood_catalogue_paths(comp, likelihood)
    try:
        with open(paths['metadata'], 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def likelihood_catalogue_active(comp, likelihood):
    metadata = _read_likelihood_catalogue_metadata(comp, likelihood)
    if metadata is None or not metadata.get('active', False):
        return False
    paths = likelihood_catalogue_paths(comp, likelihood)
    return os.path.exists(paths['unique']) and os.path.exists(paths['matches'])


def _canonical_transformed_key(fcn_i, likelihood, tmax, try_integration):
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


def ensure_likelihood_catalogue(comp, likelihood, tmax=5, try_integration=False):
    """Build a likelihood-aware catalogue when run_sympify changes parameter space."""
    ensure_output_dirs(likelihood)
    paths = likelihood_catalogue_paths(comp, likelihood)
    settings = _likelihood_catalogue_settings(tmax, try_integration)
    if not getattr(likelihood, 'use_likelihood_catalogue', True):
        if rank == 0:
            for path in [paths['unique'], paths['matches']]:
                if os.path.exists(path):
                    os.remove(path)
            metadata = {
                'active': False,
                'settings': settings,
                'disabled_by_likelihood': True,
            }
            with open(paths['metadata'], 'w') as f:
                json.dump(metadata, f, indent=2, sort_keys=True)
        comm.Barrier()
        return False

    metadata = _read_likelihood_catalogue_metadata(comp, likelihood)
    if metadata is not None and metadata.get('settings') == settings and (
            not metadata.get('active', False) or likelihood_catalogue_active(comp, likelihood)):
        comm.Barrier()
        return metadata.get('active', False)

    if rank == 0:
        allfn_file = likelihood.fn_dir + \
            "/compl_%i/all_equations_%i.txt" % (comp, comp)
        unifn_file = likelihood.fn_dir + \
            "/compl_%i/unique_equations_%i.txt" % (comp, comp)
        with open(allfn_file, 'r') as f:
            all_functions = [line.strip() for line in f]
        with open(unifn_file, 'r') as f:
            raw_unique_count = sum(1 for _ in f)

        key_to_unique = {}
        unique_representatives = []
        matches = []
        changed_layout_count = 0
        failed_count = 0

        max_param = int(max(4, np.floor((comp - 1) / 2)))
        for index, fcn_i in enumerate(all_functions):
            expected = [f'a{i}' for i in range(
                simplifier.count_params([fcn_i], max_param)[0])]
            try:
                with simplifier.time_limit(tmax):
                    key, active = _canonical_transformed_key(
                        fcn_i, likelihood, tmax, try_integration)
            except Exception:
                failed_count += 1
                key = ('failed', index, fcn_i)
                active = expected

            if active != expected:
                changed_layout_count += 1

            if key not in key_to_unique:
                key_to_unique[key] = len(unique_representatives)
                unique_representatives.append(fcn_i)
            matches.append(key_to_unique[key])

        active = changed_layout_count > 0
        if active:
            with open(paths['unique'], 'w') as f:
                for fcn_i in unique_representatives:
                    f.write(fcn_i + '\n')
            with open(paths['matches'], 'w') as f:
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
        with open(paths['metadata'], 'w') as f:
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
    # Number of lines per file for given thread
    nLs = int(np.ceil(total_lines / float(size)))

    while nLs*(size-1) > total_lines:
        if rank == 0:
            print("Correcting for many cores.", flush=True)
        nLs -= 1

    if rank == 0:
        print("Total number of functions: ", total_lines, flush=True)
        print("Number of test points per proc: ", nLs, flush=True)

    data_start = rank*nLs
    data_end = (rank+1)*nLs

    if rank == size-1:
        data_end = total_lines

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
    """Load all functions for dynamic MPI scheduling."""
    unifn_file = function_catalogue_path(comp, likelihood, unique=unique)
    set_recursionlimit_for_comp(comp)
    with open(unifn_file, "r") as f:
        return [line.strip() for line in f]


def get_function_count(comp, likelihood, unique=True):
    """Return the number of functions in the active catalogue."""
    set_recursionlimit_for_comp(comp)
    if rank == 0:
        unifn_file = function_catalogue_path(comp, likelihood, unique=unique)
        with open(unifn_file, "r") as f:
            total_lines = sum(1 for _ in f)
    else:
        total_lines = None
    return comm.bcast(total_lines, root=0)


def canonicalize_parameter_symbols(eq):
    """Rename surviving ESR parameters to a contiguous a0, a1, ... sequence."""
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


def _fit_function_with_timeout(fcn_i, likelihood, tmax, pmin, pmax, comp,
                               try_integration, log_opt, max_param,
                               Niter_params, Nconv_params,
                               ignore_previous_eqns):
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
    """Run test_all with rank-0 work dispatch to avoid straggler ranks."""
    if rank == 0:
        n_functions = len(fcn_list)
        n_workers = size - 1
        print(
            f"Dynamic scheduling: {n_functions} functions across "
            f"{n_workers} workers",
            flush=True)

        chi2 = np.full(n_functions, np.nan)
        params = np.zeros([n_functions, max_param])
        checkpoint_file = (
            likelihood.out_dir + '/negloglike_comp' + str(comp)
            + '.checkpoint.dat')
        output_file = (
            likelihood.out_dir + '/negloglike_comp' + str(comp) + '.dat')
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
        previous_fns_file = likelihood.fn_dir + "/compl_" + \
            str(comp)+"/previous_eqns_"+str(comp)+".txt"
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
                unifn_file_i = likelihood.fn_dir + \
                    "/compl_%i/unique_equations_%i.txt" % (compl, compl)
                with open(unifn_file_i, "r") as f:
                    fcn_list_i = f.readlines()
                previous_unifn_list += fcn_list_i
        previous_unifn_list = np.array(previous_unifn_list)
        np.savetxt(likelihood.fn_dir + "/compl_"+str(comp) +
                   "/previous_eqns_"+str(comp)+".txt", previous_unifn_list, fmt='%s')

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
    write_negloglike_file(
        likelihood.temp_dir + '/chi2_comp'+str(comp) +
        'weights_'+str(rank)+'.dat',
        chi2,
        params,
        max_param)

    comm.Barrier()

    if rank == 0:
        combine_temp_files(
            likelihood.temp_dir,
            'chi2_comp' + str(comp) + 'weights_*.dat',
            likelihood.out_dir + '/negloglike_comp' + str(comp) + '.dat')

    comm.Barrier()

    return
