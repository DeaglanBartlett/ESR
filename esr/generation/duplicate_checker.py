import numpy as np
import sys
from mpi4py import MPI
import csv
import os
import gc
import pprint
import glob
import heapq
from collections import defaultdict
import textwrap

import esr.generation.generator as generator
import esr.generation.simplifier as simplifier
import esr.generation.utils as utils

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def line_iterator(fname):
    with open(fname) as f:
        for line in f:
            k, v = line.rstrip('\n').split(';', 1)
            yield int(k.strip("'")), v.strip("'")
        

def main(runname, compl, track_memory=False, search_tmax=60, expand_tmax=1, seed=1234):
    """Run the generation of functions for a given complexity and set of basis functions
    
    Args:
        :runname (str): name of run, which defines the basis functions used
        :compl (int): complexity of functions to consider
        :track_memory (bool, default=True): whether to compute and print memory statistics (True) or not (False)
        :search_tmax (float, default=60.): maximum time in seconds to run any one part of simplification procedure for a given function
        :expand_tmax (float, default=1.): maximum time in seconds to run any one part of expand/simplify procedure for a given function
        :seed (int, default=1234): seed to set random number generator for shuffling functions (used to prevent one rank having similar, hard to simplify functions)
    
    Returns:
        None
    
    """

    if runname == 'keep_duplicates':
        basis_functions = [["x", "a"],  # type0
                ["square", "exp", "inv", "sqrt_abs", "log_abs"],  # type1
                ["+", "*", "-", "/", "pow"]]  # type2
    elif runname == 'core_maths':
        basis_functions = [["x", "a"],  # type0
                ["inv"],  # type1
                ["+", "*", "-", "/", "pow"]]  # type2
    elif runname == 'ext_maths':
        basis_functions = [["x", "a"],  # type0
                ["inv", "sqrt_abs", "square", "exp"],  # type1
                ["+", "*", "-", "/", "pow"]]  # type2
    elif runname == 'osc_maths':
        basis_functions = [["x", "a"],  # type0
                ["inv", "sin"],  # type1
                ["+", "*", "-", "/", "pow"]]  # type2
    elif runname == 'base10_maths':
        basis_functions = [["x", "a"],  # type0
                ["tenexp", "inv", "log10_abs"],  # type1
                ["+", "*", "-", "/", "pow"]]  # type2
    elif runname == 'base_e_maths':
        basis_functions = [["x", "a"], # type0
                ["inv","exp","log_abs"], # type1
                ["+", "*", "-", "/", "pow"]]  # type2

    dirname = os.path.abspath(os.path.join(os.path.dirname(generator.__file__), '..', 'function_library'))
    if (not os.path.isdir(dirname)) and (rank == 0):
        os.mkdir(dirname)
    dirname += '/' + runname + '/'
    if (not os.path.isdir(dirname)) and (rank == 0):
        os.mkdir(dirname)
    
    if (rank == 0) and (not os.path.isdir(dirname)):
        print('Making output directory:', dirname)
        os.mkdir(dirname)
    sys.stdout.flush()
    comm.Barrier()

    dirname += 'compl_%i/'%compl
    if (rank == 0) and (not os.path.isdir(dirname)):
        print('Making output directory:', dirname)
        os.mkdir(dirname)
    sys.stdout.flush()
    comm.Barrier()

    all_fun, extra_orig = generator.generate_equations(compl, basis_functions, dirname)
    
    if rank == 0 and track_memory:
        utils.using_mem("generated")
        utils.locals_size(locals())

    # NOTE: all_fun and extra_orig are rank-specific dicts

    # Get maximum number of parameters in any function on any rank
    max_param = simplifier.get_max_param(all_fun.values())
    max_param = comm.allreduce(max_param, op=MPI.MAX)
    # nparam = simplifier.count_params(all_fun, max_param)
    # nparam = [np.sum(nparam == i) for i in range(max_param+1)]
    
    if rank == 0 and track_memory:
        utils.using_mem("pre sympify")
        utils.locals_size(locals())

    # TO DO: Redo this bit -----

    # nextra = len(extra_orig)
    
    # # Get the mapping between original and new
    # if rank == 0:
    #     print('\nGetting extra_orig indices')
    #     sys.stdout.flush()

    # # Convert list of equations to list of indices
    # extra_orig = utils.get_match_indexes(all_fun, extra_orig)

    # --------------

    # if nextra > 0:
    #     # Get str and sympy of the original equations
    #     all_fun[:-nextra], all_sym = simplifier.initial_sympify(all_fun[:-nextra],
    #                                         max_param,
    #                                         track_memory=track_memory)

    #     # Get str but not sympy of the extra equations
    #     all_fun[-nextra:], _ = simplifier.initial_sympify(all_fun[-nextra:],
    #                                         max_param,
    #                                         track_memory=track_memory,
    #                                         save_sympy=False,
    #                                         verbose=False)
    # else:
    all_fun, all_sym = simplifier.initial_sympify(all_fun,
                                        max_param,
                                        track_memory=track_memory)
    
    # all_fun contains strings of the original equations for this rank
    # all_sym contains sympy objects of the the unique equations for this rank,
    # which are keyed by string representations of the equation

    if rank == 0 and track_memory:
        utils.using_mem("post sympify")
        utils.locals_size(locals())

    if rank == 0:
        print('All fun:', type(all_fun), len(all_fun), type(list(all_fun.keys())[0]), type(list(all_fun.values())[0]))
        print('All sym:', type(all_sym), len(all_sym), type(list(all_sym.keys())[0]), type(list(all_sym.values())[0]))
        sys.stdout.flush()

    # Save all equations in a rank dependent way. First delete the files
    # if they exist
    if rank == 0:
        print('\nSaving all equations')
        for fname in glob.glob(dirname + 'all_equations_%i_rank_*.txt'%compl):
            os.remove(fname)
    comm.Barrier()
    rank_fname = dirname + 'all_equations_%i_rank_%i.txt'%(compl, rank)
    with open(rank_fname, "w") as f:
        w = 80
        pp = pprint.PrettyPrinter(width=w, stream=f)
        for k in sorted(all_fun.keys()):
            s = f'{k}; {all_fun[k]}'
            if len(s + '\n') > w / 2:
                w = 2 * len(s)
                pp = pprint.PrettyPrinter(width=w, stream=f)
            pp.pprint(s)
    comm.Barrier()
    if rank == 0:
        all_iters = [line_iterator(fname) for fname in glob.glob(dirname + 'all_equations_%i_rank_*.txt'%compl)]
        merged = heapq.merge(*all_iters)
        next_key = 0
        with open(dirname + 'all_equations_%i.txt'%compl, "w") as out:
            for k, v in merged:
                while next_key < k:
                    out.write("\n")  # empty line for missing key
                    next_key += 1
                out.write(f"{v}\n")
                next_key += 1
        # Now delete the rank-specific files
        for fname in glob.glob(dirname + 'all_equations_%i_rank_*.txt'% compl):
            os.remove(fname)
    comm.Barrier()

    # # We know the extra equations are duplicates, so will use this
    # if nextra > 0:
    #     all_fun[-nextra:] = [all_fun[f] for f in extra_orig]

    del extra_orig
    gc.collect()

    if rank == 0 and track_memory:
        utils.using_mem("pre do sympy")
        utils.locals_size(locals())
        
    old_all_fun = all_fun.copy()

    all_fun, uniq, nround = simplifier.do_sympy(all_fun, 
                                        all_sym, 
                                        compl, 
                                        search_tmax, 
                                        expand_tmax,
                                        dirname,
                                        track_memory=track_memory)
    ntot = comm.allreduce(len(all_fun), op=MPI.SUM)

    # Shuffle the unique equations across ranks
    if rank == 0:
        print('\nShuffling', flush=True)
    uniq = utils.shuffle_dict_mpi(uniq, seed)

    # Print summary of results
    new_len = comm.allreduce(len(uniq), op=MPI.SUM)
    if rank == 0:
        stars = '\n' + ''.join(['*']*35) + '\n'
        print(stars)
        print('For complexity %i:'%compl)
        print('Total unique: %i (%i)'%(new_len, ntot))
    uniq_nparam = list(simplifier.count_params_dict(uniq.keys()).values())
    nparam = list(simplifier.count_params_dict(all_fun.items()).values())
    for i in range(max_param+1):
        n_uniq = comm.allreduce(np.sum([u == i for u in uniq_nparam]), op=MPI.SUM)
        n = comm.allreduce(np.sum([p == i for p in nparam]), op=MPI.SUM)
        if rank == 0:
            if i == 1:
                print('Functions with 1 parameter: %i (%i)'%(n_uniq, n))
            else:
                print('Functions with %i parameters: %i (%i)'%(i, n_uniq, n))

    # Get the match index for the unique equations
    uniq = utils.get_dict_index_mpi(uniq)
    print(f"Rank {rank} has {len(uniq)} unique equations:")
    match = utils.get_match_index_mpi(uniq, all_fun)
    # quit()

    # if rank == 0:
    #     for i in old_all_fun.keys():
    #         if old_all_fun[i] != all_fun[i]:
    #             print(f"{i}: {old_all_fun[i]} -> {all_fun[i]}")
    # quit()

    # Save the unique equations
    if rank == 0:
        print(stars, flush=True)
        print('\nSaving results:', flush=True)
    
    # Save the equations
    fname_str = dirname + 'unique_equations_%i.txt'%compl
    if rank == 0:
        print('\tUnique equations')
        if os.path.exists(fname_str):
            os.remove(fname_str)
    comm.Barrier()
    for r in range(size):
        comm.Barrier()
        if rank != r:
            continue
        # Keep to sort the equations by their value
        uniq_sorted = sorted(uniq.keys(), key=uniq.get)
        if rank == 0:
            print('uniq sorted', uniq_sorted)
        # buf = io.StringIO()
        with open(fname_str, "a") as f_str:
            w = 80
            for eq in uniq_sorted:
                if len(eq + '\n') > w / 2:
                    w = 2 * len(eq)
                wrapped = textwrap.fill(eq, width=w)
                f_str.write(wrapped + "\n")

    # Save the match index
    fname_match = dirname + 'matches_%i.txt'%compl
    if rank == 0:
        print('\tMatches')
        if os.path.exists(fname_match):
            os.remove(fname_match)
    comm.Barrier()
    root = fname_match[:fname_match.index('.txt')]
    if rank == 0:
        print('\nSaving all equations')
        for fname in glob.glob(f'{root}_rank_*.txt'):
            os.remove(fname)
    comm.Barrier()
    rank_fname = f'{root}_rank_{rank}.txt'

    local_items = np.array([(k, match[k]) for k in sorted(match.keys())])
    np.savetxt(rank_fname, local_items, fmt='%s', delimiter=';')
    comm.Barrier()
    if rank == 0:
        all_iters = [line_iterator(fname) for fname in glob.glob(f'{root}_rank_*.txt')]
        merged = heapq.merge(*all_iters)
        next_key = 0
        with open(fname_match, "w") as out:
            for k, v in merged:
                while next_key < k:
                    out.write("\n")  # empty line for missing key
                    next_key += 1
                out.write(f"{v}\n")
                next_key += 1
        # Now delete the rank-specific files
        for fname in glob.glob(f'{root}_rank_*.txt'):
            os.remove(fname)

        print('Matches saved to:', fname_match)
    comm.Barrier()

    if rank == 0:
        print('\nCombining Inverse Subs')
        sys.stdout.flush()

    # TO DO: Print the inverse substitutions
    # TO DO: Combine the duplicate inverse substitutions
    # TO DO: Reintroduce expand in the search

    # Load the inverse substitutions
    all_inv_subs = defaultdict(list)
    for r in range(nround):
        if rank == 0:
            print('Round %i of %i'%(r+1, nround))
            sys.stdout.flush()
        inv = simplifier.load_subs_dict(
                all_fun.keys(),
                dirname,
                compl,
                r,
                max_param,
                use_sympy=False)
        for k, v in inv.items():
            all_inv_subs[k] += v

    # Remove the inverse substitutions which are not required
    # if rank == 0:
    #     print('\nRemoving unnecessary inv_subs', flush=True)
    # all_dup = simplifier.get_all_dup(max_param)
    # for i in all_inv_subs.keys():
        # all_inv_subs[i] = simplifier.simplify_inv_subs(all_inv_subs[i], all_dup)

    # Save the inverse substitutions to file
    if rank == 0:
        print('\nSaving Inverse Subs', flush=True)
    root = dirname + 'inv_subs_%i'%compl
    if rank == 0:
        print('\tInverse substitutions')
        if os.path.exists(root + '.txt'):
            os.remove(root + '.txt')
        for fname in glob.glob(f'{root}_rank_*.txt'):
            os.remove(fname)
    comm.Barrier()
    rank_fname = f'{root}_rank_{rank}.txt'
    with open(rank_fname, "w") as f:
        writer = csv.writer(f, delimiter=';')
        for k in sorted(all_fun.keys()):
            if k in all_inv_subs:
                subs = all_inv_subs[k]
                if 'nan' in str(subs):
                    subs = [None]
            else:
                subs = [None]
            writer.writerow([k] + subs)
    comm.Barrier()
    if rank == 0:
        all_iters = [line_iterator(fname) for fname in glob.glob(f'{root}_rank_*.txt')]
        merged = heapq.merge(*all_iters)
        next_key = 0
        with open(f'{root}.txt', "w") as out:
            for k, v in merged:
                while next_key < k:
                    out.write("\n")  # empty line for missing key
                    next_key += 1
                out.write(f"{v}\n")
                next_key += 1
        # Now delete the rank-specific files
        # for fname in glob.glob(f'{root}_rank_*.txt'):
            # os.remove(fname)
        print('Inverse substitutions saved to:', f'{root}.txt')

    # quit()

    # if rank == 0:

    #     # Remove subs which are followed by their inverse
    #     print('\nRemoving unnecessary inv_subs')
    #     sys.stdout.flush()
    #     all_dup = simplifier.get_all_dup(max_param)
    #     sys.stdout.flush()

    #     for i in range(len(all_inv_subs)):
    #         all_inv_subs[i] = simplifier.simplify_inv_subs(all_inv_subs[i], all_dup)

    #     print('\nSaving Inverse Subs')
    #     sys.stdout.flush()
    #     for i in range(len(all_inv_subs)):
    #         if all_inv_subs[i] is None:
    #             all_inv_subs[i] = []
    #     with open(dirname + '/inv_subs_%i.txt'%compl, "w") as f:
    #         writer = csv.writer(f, delimiter=';')
    #         writer.writerows(all_inv_subs)

    #     all_fname = ['unique_equations_', 'all_equations_', 'trees_', 'orig_trees_', 'extra_trees_']
    #     for fname in all_fname:
    #         s = "sed 's/.$//; s/^.//' %s/%s%i.txt > %s/temp_%i.txt"%(dirname,fname,compl,dirname,compl)
    #         os.system(s)
    #         s = "mv %s/temp_%i.txt %s/%s%i.txt"%(dirname,compl,dirname,fname,compl)
    #         os.system(s)

    #     del all_inv_subs
    #     gc.collect()

    if rank == 0:
        print('\nChecking Results', flush=True)
    if compl > 2:
        simplifier.check_results(dirname, compl)
        
    sys.stdout.flush()
    comm.Barrier()

    return


if __name__ == "__main__":
    compl = int(sys.argv[1])
    runname = 'core_maths'
    main(runname, compl)
