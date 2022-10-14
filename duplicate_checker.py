import numpy as np
import sys
from mpi4py import MPI
import sympy
import csv
import os
import gc
import time
import pprint

import generator
import simplifier
import utils
from custom_printer import ESRPrinter

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def main(compl, track_memory=False):

    seed = 1234
    search_tmax = 60
    expand_tmax = 1

    """
    dirname = 'keep_duplicates/'
    basis_functions = [["x", "a"],  # type0
                    ["square", "exp", "inv", "sqrt_abs", "log_abs"],  # type1
                    ["+", "*", "-", "/", "pow"]]  # type2
    """
    #"""
    dirname = 'core_maths/'
    basis_functions = [["x", "a"],  # type0
                    ["inv"],  # type1
                    ["+", "*", "-", "/", "pow"]]  # type2
    #"""
    """
    dirname = 'ext_maths/'
    basis_functions = [["x", "a"],  # type0
                    ["inv", "sqrt_abs", "square", "exp"],  # type1
                    ["+", "*", "-", "/", "pow"]]  # type2
    """
    """
    dirname = 'osc_maths/'
    basis_functions = [["x", "a"],  # type0
                    ["inv", "sin"],  # type1
                    ["+", "*", "-", "/", "pow"]]  # type2
    """
    #dirname = 'test_dir/'
    
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

    max_param = simplifier.get_max_param(all_fun)
    nparam = simplifier.count_params(all_fun, max_param)
    nparam = [np.sum(nparam == i) for i in range(max_param+1)]
    param_list = ['a%i'%i for i in range(max_param)]
    
    if rank == 0 and track_memory:
        utils.using_mem("pre sympify")
        utils.locals_size(locals())

    nextra = len(extra_orig)
    
    # Get the mapping between original and new
    if rank == 0:
        print('\nGetting extra_orig indices')
        sys.stdout.flush()

    start = time.time()
    # Convert list of equations to list of indices
    extra_orig = utils.get_match_indexes(all_fun, extra_orig)
    end = time.time()

    if rank == 0:
        print(end - start)
        sys.stdout.flush()

    if nextra > 0:
        # Get str and sympy of the original equations
        all_fun[:-nextra], all_sym = simplifier.initial_sympify(all_fun[:-nextra],
                                                max_param,
                                                track_memory=track_memory)

        # Get str but not sympy of the extra equations
        all_fun[-nextra:], _ = simplifier.initial_sympify(all_fun[-nextra:],
                                                max_param,
                                                track_memory=track_memory,
                                                save_sympy=False)
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
        with open(dirname + '/all_equations_%i.txt'%compl, "w") as f:
            w = 80
            pp = pprint.PrettyPrinter(width=w, stream=f)
            for s in all_fun:
                if len(s + '\n') > w / 2:
                    w = 2 * len(s)
                    pp = pprint.PrettyPrinter(width=w, stream=f)
                pp.pprint(s)
            #for s in all_fun:
            #    print(s, file=f)

    #sys.exit(0)

    # We know the extra equations are duplicates, so will use this
    if nextra > 0:
        all_fun[-nextra:] = [all_fun[f] for f in extra_orig]

    del extra_orig; gc.collect()

    if rank == 0 and track_memory:
        utils.using_mem("pre do sympy")
        utils.locals_size(locals())

    # For testing
    """
    nparam = simplifier.count_params(all_fun, max_param)
    idx = [i for i in range(len(all_fun)) if nparam[i] == 3]
    all_fun = [all_fun[i] for i in idx]
    all_fun, all_sym = simplifier.initial_sympify(all_fun, max_param)
    """
    """
    all_fun = ['(a0 + sqrt(x))**2',
                'a0**2 - 2*a0*sqrt(x) + x',
                'a0**2 + a0*sqrt(x) + x',
                '(a0 - sqrt(x))**2']
    all_fun, all_sym = simplifier.initial_sympify(all_fun, max_param)
    if rank == 0:
        print('\nSaving all equations')
        sys.stdout.flush()
        with open(dirname + '/all_equations_%i.txt'%compl, "w") as f:
            w = 80
            pp = pprint.PrettyPrinter(width=w, stream=f)
            for s in all_fun:
                if len(s + '\n') > w / 2:
                    w = 2 * len(s)
                    pp = pprint.PrettyPrinter(width=w, stream=f)
                pp.pprint(s)
    """

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
    
        # Shuffle the unique equations
        print('\nShuffling')
        sys.stdout.flush()
        np.random.seed(seed)
        i = np.arange(len(uniq))
        np.random.shuffle(i)
        inv = {i[j]:j for j in range(len(i))}
        uniq_fun = [uniq_fun[ii] for ii in i]
        match_idx = [inv[match[f]] for f in all_fun]

        ntot = len(all_fun)
        del all_fun; gc.collect()
        
        uniq_nparam = simplifier.count_params(uniq_fun, max_param)
    
        stars = '\n' + ''.join(['*']*35) + '\n'
        print(stars)
        print('For complexity %i:'%compl)
        print('Total unique: %i (%i)'%(len(uniq_fun), ntot))
        
        for i in range(max_param+1):
            if i == 1:
                print('Functions with 1 parameter: %i (%i)'%(np.sum(uniq_nparam == 1), nparam[1]))
            else:
                print('Functions with %i parameters: %i (%i)'%(i, np.sum(uniq_nparam == i), nparam[i]))
        del uniq_nparam, nparam; gc.collect()
        print(stars)

        print('\nSaving results:')
        sys.stdout.flush()

        print('\tUnique equations')
        with open(dirname + '/unique_equations_%i.txt'%compl, "w") as f:
            w = 80
            pp = pprint.PrettyPrinter(width=w, stream=f)
            for s in uniq_fun:
                if len(s + '\n') > w / 2:
                    w = 2 * len(s)
                    pp = pprint.PrettyPrinter(width=w, stream=f)
                pp.pprint(s)
            #for u in uniq_fun:
            #    print(u, file=f)
        del uniq_fun; gc.collect()
               
        print('\tMatches')
        with open(dirname + '/matches_%i.txt'%compl, "w") as f:
            for i in range(len(match_idx)):
                print(match_idx[i], file=f)
        del match_idx; gc.collect() 


    # Now combine the inverse subs calculations
    del all_sym, _, uniq, match
    gc.collect()

    if rank == 0:
        all_inv_subs = [[]] * ntot
        print('\nCombining Inverse Subs')
        sys.stdout.flush()

    for r in range(nround):
        if rank == 0:
            print('Round %i of %i'%(r+1, nround))
            sys.stdout.flush()

        inv = simplifier.load_subs(dirname + '/inv_subs_%i_round_%i.txt'%(compl,r),
                                    max_param,
                                    use_sympy=False)

        if rank == 0:
            if len(inv) != 0:
                idx = np.atleast_1d(np.loadtxt(dirname + '/inv_idx_%i_round_%i.txt'%(compl,r), dtype=int))
            else:
                idx = []
        
        if rank == 0:
            for i, j in enumerate(idx):
                all_inv_subs[j] = all_inv_subs[j] + inv[i]
            del idx, inv; gc.collect()

    if rank == 0:

        # Remove subs which are followed by their inverse
        print('\nRemoving unnecessary inv_subs')
        sys.stdout.flush()
        all_dup = simplifier.get_all_dup(max_param)
        sys.stdout.flush()

        for i in range(len(all_inv_subs)):
            all_inv_subs[i] = simplifier.simplify_inv_subs(all_inv_subs[i], all_dup)

        print('\nSaving Inverse Subs')
        sys.stdout.flush()
        for i in range(len(all_inv_subs)):
            if all_inv_subs[i] is None:
                all_inv_subs[i] = []
        with open(dirname + '/inv_subs_%i.txt'%compl, "w") as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerows(all_inv_subs)

        all_fname = ['unique_equations_', 'all_equations_', 'trees_', 'orig_trees_', 'extra_trees_']
        for fname in all_fname:
            s = "sed 's/.$//; s/^.//' %s/%s%i.txt > %s/temp_%i.txt"%(dirname,fname,compl,dirname,compl)
            print(s)
            os.system(s)
            s = "mv %s/temp_%i.txt %s/%s%i.txt"%(dirname,compl,dirname,fname,compl)
            print(s)
            os.system(s)

        del all_inv_subs; gc.collect()


    if rank == 0:
        print('\nChecking Results', flush=True)
    simplifier.check_results(dirname, compl)

    """
    # For testing
    if rank == 0:
        matches = np.loadtxt(dirname + '/matches_%i.txt'%compl).astype(int)
        with open(dirname + '/all_equations_%i.txt'%compl, 'r') as f:
            all_equations = f.read().splitlines()
        with open(dirname + '/unique_equations_%i.txt'%compl, 'r') as f:
            uniq_equations = f.read().splitlines()
        with open(dirname + '/inv_subs_%i.txt'%compl, 'r') as f:
            inv_subs = f.read().splitlines()

        print('\nMATCHES', flush=True)
        for i in range(len(matches)):
            print(all_equations[i], '\t~~~~~\t', uniq_equations[matches[i]], '\t~~~~~\t', inv_subs[i], flush=True)
    """



    return
    
    
if __name__ == "__main__":
#    try:
#        main(4)
#    except:
#        comm.Abort()
    #main(5)
    #main(7)
    #for c in [4, 5, 6, 7]:
    #for c in [8, 9]:
    #    main(c)
    #for c in range(4, 11):
    #for c in range(4, 9):
    #    main(c)
    #main(5)
    #main(6)
    #main(7)
    #simplifier.check_results('test_dir/compl_7/', 7)
    #simplifier.check_results('osc_maths/compl_9/', 9)
    #main(10)
    #main(9, track_memory=True)
    #main(9)
    for c in [2, 3]:
        main(c)

"""
TO DO
- Save after each iteration so can restart?

- CHECK THE INV_SUBS at each interation. If it doesn't work, then don't keep. Better than redoing at the end

- Change param converter to check if inv_subs is made of sympy objects
"""
