import numpy as np
import sys
from mpi4py import MPI
import csv
import os
import gc
import pprint

import esr.generation.generator as generator
import esr.generation.simplifier as simplifier
import esr.generation.utils as utils

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def _validate_inverse_substitution_pairs(round_index, idx, inv):
    """Reject incomplete paired inverse-substitution artifacts.

    ``inv_idx`` and ``inv_subs`` are written from the same rank-0 list, so a
    length mismatch means that one of the files is stale or truncated. Pairing
    only a shared prefix could assign substitutions to the wrong equations.
    """
    if len(idx) != len(inv):
        raise ValueError(
            'inv_idx/inv_subs length mismatch in round %i: len(idx)=%i '
            'len(inv)=%i. Delete the incomplete round artifacts and rerun '
            'duplicate_checker.' % (round_index, len(idx), len(inv)))


def main(runname, compl, track_memory=False, search_tmax=60, expand_tmax=1,
         seed=1234, diagnose_numerical_duplicates=False, fn_dir=None):
    """Run the generation of functions for a given complexity and set of basis functions

    Args:
        :runname (str): name of run, which defines the basis functions used
        :compl (int): complexity of functions to consider
        :track_memory (bool, default=True): whether to compute and print memory statistics (True) or not (False)
        :search_tmax (float, default=60.): maximum time in seconds to run any one part of simplification procedure for a given function
        :expand_tmax (float, default=1.): maximum time in seconds to run any one part of expand/simplify procedure for a given function
        :seed (int, default=1234): seed to set random number generator for shuffling functions (used to prevent one rank having similar, hard to simplify functions)
        :diagnose_numerical_duplicates (bool, default=False): whether to
            write a report of expressions with matching numerical
            fingerprints. This is diagnostic only and never removes or
            remaps equations. Candidate groups require explicit verification
            of variable/parameter domains, boundaries, singularities, and
            description-length semantics before any manual catalogue change.
        :fn_dir (str, default=None): directory in which to store the generated
            catalogue for this run (the ``compl_<compl>`` subdirectory is created
            inside it). If None, the default ``function_library/<runname>`` is
            used. Pass ``likelihood.fn_dir`` here to generate straight into the
            location a likelihood object reads from (e.g. an isolated directory
            for a test), keeping generation and fitting consistent.

    Returns:
        None

    """

    if runname == 'keep_duplicates':
        basis_functions = [["x", "a"],  #  type0
                           ["square", "exp", "inv", "sqrt_abs", "log_abs"],  # type1
                           ["+", "*", "-", "/", "pow"]]  # type2
    elif runname == 'core_maths':
        basis_functions = [["x", "a"],  #  type0
                           ["inv"],  # type1
                           ["+", "*", "-", "/", "pow"]]  # type2
    elif runname == 'ext_maths':
        basis_functions = [["x", "a"],  #  type0
                           ["inv", "sqrt_abs", "square", "exp"],  # type1
                           ["+", "*", "-", "/", "pow"]]  # type2
    elif runname == 'osc_maths':
        basis_functions = [["x", "a"],  #  type0
                           ["inv", "sin"],  # type1
                           ["+", "*", "-", "/", "pow"]]  # type2
    elif runname == 'base10_maths':
        basis_functions = [["x", "a"],  #  type0
                           ["tenexp", "inv", "log10_abs"],  # type1
                           ["+", "*", "-", "/", "pow"]]  # type2
    elif runname == 'base_e_maths':
        basis_functions = [["x", "a"],  # type0
                           ["inv", "exp", "log_abs"],  # type1
                           ["+", "*", "-", "/", "pow"]]  # type2

    if fn_dir is None:
        dirname = os.path.abspath(os.path.join(os.path.dirname(
            generator.__file__), '..', 'function_library'))
        if (not os.path.isdir(dirname)) and (rank == 0):
            os.makedirs(dirname, exist_ok=True)
        dirname += '/' + runname + '/'
    else:
        dirname = os.path.abspath(fn_dir) + '/'
    if (rank == 0) and (not os.path.isdir(dirname)):
        print('Making output directory:', dirname)
        os.makedirs(dirname, exist_ok=True)
    sys.stdout.flush()
    comm.Barrier()

    dirname += 'compl_%i/' % compl
    if (rank == 0) and (not os.path.isdir(dirname)):
        print('Making output directory:', dirname)
        os.makedirs(dirname, exist_ok=True)
    sys.stdout.flush()
    comm.Barrier()

    all_fun, extra_orig = generator.generate_equations(
        compl, basis_functions, dirname)

    if rank == 0 and track_memory:
        utils.using_mem("generated")
        utils.locals_size(locals())

    max_param = simplifier.get_max_param(all_fun)
    nparam = simplifier.count_params(all_fun, max_param)
    nparam = [np.sum(nparam == i) for i in range(max_param+1)]
    param_list = ['a%i' % i for i in range(max_param)]

    if rank == 0 and track_memory:
        utils.using_mem("pre sympify")
        utils.locals_size(locals())

    nextra = len(extra_orig)

    # Get the mapping between original and new
    if rank == 0:
        print('\nGetting extra_orig indices')
        sys.stdout.flush()

    # Convert list of equations to list of indices
    extra_orig = utils.get_match_indexes(all_fun, extra_orig)

    if nextra > 0:
        # Get str and sympy of the original equations
        all_fun[:-nextra], all_sym = simplifier.initial_sympify(all_fun[:-nextra],
                                                                max_param,
                                                                track_memory=track_memory)

        # Get str but not sympy of the extra equations
        all_fun[-nextra:], _ = simplifier.initial_sympify(all_fun[-nextra:],
                                                          max_param,
                                                          track_memory=track_memory,
                                                          save_sympy=False,
                                                          verbose=False)
    else:
        all_fun, all_sym = simplifier.initial_sympify(all_fun,
                                                      max_param,
                                                      track_memory=track_memory)

    if rank == 0 and track_memory:
        utils.using_mem("post sympify")
        utils.locals_size(locals())

    if rank == 0:
        print('\nSaving all equations')
        sys.stdout.flush()
        with utils.atomic_write(dirname + '/all_equations_%i.txt' % compl) as f:
            w = 80
            pp = pprint.PrettyPrinter(width=w, stream=f)
            for s in all_fun:
                if len(s + '\n') > w / 2:
                    w = 2 * len(s)
                    pp = pprint.PrettyPrinter(width=w, stream=f)
                pp.pprint(s)

    # We know the extra equations are duplicates, so will use this
    if nextra > 0:
        all_fun[-nextra:] = [all_fun[f] for f in extra_orig]

    del extra_orig
    gc.collect()

    if rank == 0 and track_memory:
        utils.using_mem("pre do sympy")
        utils.locals_size(locals())

    all_fun, _, nround = simplifier.do_sympy(all_fun,
                                             all_sym,
                                             compl,
                                             search_tmax,
                                             expand_tmax,
                                             dirname,
                                             track_memory=track_memory)

    uniq, match = utils.get_unique_indexes(all_fun)
    uniq_fun = list(uniq.keys())

    if rank == 0:

        #  Shuffle the unique equations
        print('\nShuffling')
        sys.stdout.flush()
        np.random.seed(seed)
        i = np.arange(len(uniq_fun))
        np.random.shuffle(i)
        inv = {i[j]: j for j in range(len(i))}
        uniq_fun = [uniq_fun[ii] for ii in i]
        match_idx = [inv[match[f]] for f in all_fun]

        ntot = len(all_fun)
        del all_fun
        gc.collect()

        uniq_nparam = simplifier.count_params(uniq_fun, max_param)

        stars = '\n' + ''.join(['*']*35) + '\n'
        print(stars)
        print('For complexity %i:' % compl)
        print('Total unique: %i (%i)' % (len(uniq_fun), ntot))

        for i in range(max_param+1):
            if i == 1:
                print('Functions with 1 parameter: %i (%i)' %
                      (np.sum(uniq_nparam == 1), nparam[1]))
            else:
                print('Functions with %i parameters: %i (%i)' %
                      (i, np.sum(uniq_nparam == i), nparam[i]))
        del uniq_nparam, nparam
        gc.collect()
        print(stars)

        print('\nSaving results:')
        sys.stdout.flush()

        print('\tUnique equations')
        with utils.atomic_write(dirname + '/unique_equations_%i.txt' % compl) as f:
            w = 80
            pp = pprint.PrettyPrinter(width=w, stream=f)
            for s in uniq_fun:
                if len(s + '\n') > w / 2:
                    w = 2 * len(s)
                    pp = pprint.PrettyPrinter(width=w, stream=f)
                pp.pprint(s)
        del uniq_fun
        gc.collect()

        print('\tMatches')
        with utils.atomic_write(dirname + '/matches_%i.txt' % compl) as f:
            for i in range(len(match_idx)):
                print(match_idx[i], file=f)
        del match_idx
        gc.collect()

    # Now combine the inverse subs calculations
    del all_sym, _, uniq, match
    gc.collect()

    if rank == 0:
        all_inv_subs = [[]] * ntot
        print('\nCombining Inverse Subs')
        sys.stdout.flush()

    for r in range(nround):
        if rank == 0:
            print('Round %i of %i' % (r+1, nround))
            sys.stdout.flush()

        inv = simplifier.load_subs(dirname + '/inv_subs_%i_round_%i.txt' % (compl, r),
                                   max_param,
                                   use_sympy=False)

        if rank == 0:
            if len(inv) != 0:
                idx = np.atleast_1d(np.loadtxt(
                    dirname + '/inv_idx_%i_round_%i.txt' % (compl, r), dtype=int))
            else:
                idx = []

        if rank == 0:
            _validate_inverse_substitution_pairs(r, idx, inv)
            for i in range(len(idx)):
                all_inv_subs[idx[i]] = all_inv_subs[idx[i]] + inv[i]
            del idx, inv
            gc.collect()

    if rank == 0:

        # Remove subs which are followed by their inverse
        print('\nRemoving unnecessary inv_subs')
        sys.stdout.flush()
        all_dup = simplifier.get_all_dup(max_param)
        sys.stdout.flush()

        for i in range(len(all_inv_subs)):
            all_inv_subs[i] = simplifier.simplify_inv_subs(
                all_inv_subs[i], all_dup)

        print('\nSaving Inverse Subs')
        sys.stdout.flush()
        for i in range(len(all_inv_subs)):
            if all_inv_subs[i] is None:
                all_inv_subs[i] = []
        with utils.atomic_write(dirname + '/inv_subs_%i.txt' % compl) as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerows(all_inv_subs)

        all_fname = ['unique_equations_', 'all_equations_',
                     'trees_', 'orig_trees_', 'extra_trees_']
        for fname in all_fname:
            s = "sed 's/.$//; s/^.//' %s/%s%i.txt > %s/temp_%i.txt" % (
                dirname, fname, compl, dirname, compl)
            os.system(s)
            s = "mv %s/temp_%i.txt %s/%s%i.txt" % (
                dirname, compl, dirname, fname, compl)
            os.system(s)

        del all_inv_subs
        gc.collect()

    if rank == 0:
        print('\nChecking Results', flush=True)
    if compl > 2:
        simplifier.check_results(dirname, compl)

    if rank == 0 and diagnose_numerical_duplicates:
        with open(dirname + '/unique_equations_%i.txt' % compl, 'r') as f:
            final_uniq_fun = f.read().splitlines()
        candidate_groups = simplifier.numerical_duplicate_candidates(
            final_uniq_fun, max_param=max_param)
        report_file = os.path.join(
            dirname, 'numerical_duplicate_candidates_%i.txt' % compl)
        with utils.atomic_write(report_file) as f:
            f.write('# Candidate numerical fingerprint collisions only.\n')
            f.write('# No equations were removed or remapped by this diagnostic.\n')
            f.write('# Verify exact equality over the required variable and '
                    'parameter domains, including boundaries and singularities, '
                    'before merging any expressions.\n')
            for group_id, (fingerprint, indexes) in enumerate(candidate_groups):
                f.write('\nGROUP %i HASH %s\n' % (group_id, fingerprint))
                for index in indexes:
                    f.write('%i\t%s\n' % (index, final_uniq_fun[index]))
        print('Wrote numerical duplicate candidate report:', report_file, flush=True)

    sys.stdout.flush()
    comm.Barrier()

    return


if __name__ == "__main__":
    compl = int(sys.argv[1])
    runname = 'core_maths'
    main(runname, compl)
