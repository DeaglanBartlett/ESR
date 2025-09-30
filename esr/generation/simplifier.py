import numpy as np
import sympy
import signal
import sys
import itertools
from mpi4py import MPI
from contextlib import contextmanager
import csv
import ast
import gc
from collections import OrderedDict
import pprint
import os

import esr.generation.utils as utils
from esr.generation.custom_printer import ESRPrinter
from esr.fitting.sympy_symbols import (
    sympy_locs, square, cube, pow_abs, sqrt_abs, log_abs
)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    """ Check function call does not exceed allotted time

    Args:
        :seconds (float): maximum time function can run in seconds

    Raises:
        TimeoutException if time exceeds seconds
    """

    def signal_handler(signum, frame):
        raise TimeoutException("Timed out")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def get_max_param(all_fun, verbose=True):
    """ Find maximum number of free parameters in list of functions

    Args:
        :all_fun (list): list of strings containing functions
        :verbose (bool, default=True): Whether to print result (True) or not (False)

    Returns:
        :max_param (int): maximum number of free parameters in any equation in all_fun
    """

    max_param = -1

    with_ai = all_fun.copy()
    while len(with_ai) > 0:
        max_param += 1
        with_ai = [f for f in with_ai if 'a%i' % max_param in f]
    if max_param < 0:
        max_param = 0
    if verbose and rank == 0:
        print('\nMax number of parameters:', max_param)
    sys.stdout.flush()

    return max_param


def count_params(all_fun, max_param):
    """ Count the number of free parameters in each member of a list of functions

    Args:
        :all_fun (list): list of strings containing functions
        :max_param (int): maximum number of free parameters in any equation in all_fun

    Returns:
        :nparam (np.array): array of ints containing number of free parameters in corresponding member of all_fun
    """

    nparam = np.zeros(len(all_fun), dtype=int)
    param_list = ['a%i' % i for i in range(max_param)]

    for i in range(len(nparam)):
        for j in range(max_param-1, -1, -1):
            if param_list[j] in all_fun[i]:
                nparam[i] = j+1
                break

    return nparam


def make_changes(all_fun, all_sym, all_inv_subs, str_fun, sym_fun, inv_subs_fun):
    """ Update global variables of functions and symbolic expressions by combining rank
    calculations

    Args:
        :all_fun (list): list of strings containing all functions
        :all_sym (list): list of sympy objects containing all functions
        :all_inv_subs (list): list of dictionaries giving subsitutions to be applied to all functions
        :str_fun (list): list of strings containing functions considered by rank
        :sym_fun (list): list of sympy objects containing functions considered by rank
        :inv_subs_fun (list): list of dictionaries giving subsitutions to be applied to functions considered by rank

    Returns:
        :all_fun: list of strings containing all (updated) functions
        :all_sym (list): list of sympy objects containing all (updated) functions
        :all_inv_subs: list of dictionaries giving subsitutions to be applied to all (updated) functions
    """

    i = utils.split_idx(len(all_fun), rank, size)
    if len(i) > 0:
        imin = i[0]
        imax = i[-1] + 1
        start_idx = imax - imin
    else:
        start_idx = 0
        imin = len(all_fun)

    start_idx = comm.gather(start_idx, root=0)
    if rank == 0:
        start_idx = np.array([0] + start_idx)
        start_idx = np.cumsum(start_idx)
    start_idx = comm.bcast(start_idx, root=0)

    chidx = [i for i in range(len(str_fun)) if str_fun[i] != all_fun[imin+i]]
    str_changes = [str_fun[c] for c in chidx]
    sym_changes = [sym_fun[c] for c in chidx]
    inv_changes = [inv_subs_fun[c] for c in chidx]

    chidx = comm.gather(chidx, root=0)
    str_changes = comm.gather(str_changes, root=0)
    sym_changes = comm.gather(sym_changes, root=0)
    inv_changes = comm.gather(inv_changes, root=0)

    chidx = comm.bcast(chidx, root=0)
    str_changes = comm.bcast(str_changes, root=0)
    sym_changes = comm.bcast(sym_changes, root=0)
    inv_changes = comm.bcast(inv_changes, root=0)

    for i in range(size):
        j = chidx[i] + start_idx[i]
        for k in range(len(inv_changes[i])):
            all_fun[j[k]] = str_changes[i][k]
            all_sym[j[k]] = sym_changes[i][k]
            if inv_changes[i][k] is None:
                all_inv_subs[j[k]] = None
            else:
                all_inv_subs[j[k]] = inv_changes[i][k].copy()

    return all_fun, all_sym, all_inv_subs


def initial_sympify(all_fun, max_param, verbose=True, parallel=True, track_memory=False, save_sympy=True):
    """Convert list of strings of functions into list of sympy objects

    Args:
        :all_fun (list): list of strings containing functions
        :max_param (int): maximum number of free parameters in any equation in all_fun
        :verbose (bool, default=True): whether to print progress (True) or not (False)
        :parallel (bool, default=True): whether to split equations amongst ranks (True) or each equation considered by all ranks (False)
        :track_memory (bool, default=True): whether to compute and print memory statistics (True) or not (False)
        :save_sympy (bool, default=True): whether to return sympy objects (True) or not (False)

    Returns:
        :str_fun (list): list of strings containing functions
        :sym_fun (OrderedDict): dictionary of sympy objects which can be accessed by their string representations. If save_sympy is False, then sym_fun is None.
    """

    if rank == 0 and verbose:
        if track_memory:
            utils.using_mem("start initial sympify")
            utils.locals_size(locals())
        print('\nSympy simplify')
    sys.stdout.flush()

    x, x0, y = sympy.symbols('x x0 y', positive=True)
    if max_param > 0:
        param_list = ['a%i' % i for i in range(max_param)]
        all_a = sympy.symbols(" ".join(param_list), real=True)
        if max_param == 1:
            all_a = [all_a]
    else:
        param_list = []

    sympy.init_printing(use_unicode=True)
    locs = sympy_locs

    if max_param > 0:
        for i in range(len(all_a)):
            locs["a%i" % i] = all_a[i]

    if parallel:
        i = np.atleast_1d(utils.split_idx(len(all_fun), rank, size))
        if len(i) == 0:
            str_fun = []
        else:
            str_fun = all_fun[i[0]:i[-1]+1]
    else:
        str_fun = all_fun

    if save_sympy:
        sym_fun = OrderedDict()
    else:
        sym_fun = None

    p = ESRPrinter()
    for i in range(len(str_fun)):
        try:
            s = sympy.sympify(str_fun[i], locals=locs)
        except Exception:
            print('Making %s a zoo' % str_fun[i])
            s = sympy.zoo
        str_fun[i] = p.doprint(s)
        if save_sympy:
            if str_fun[i] not in sym_fun:
                sym_fun[str_fun[i]] = s

    # We have to gather these, although won't do this again
    if parallel:

        #  First find which ranks contain which indices
        start_idx = len(str_fun)
        start_idx = comm.gather(start_idx, root=0)
        if rank == 0:
            start_idx = np.array([0] + start_idx, dtype=int)
            start_idx = np.squeeze(np.cumsum(start_idx))
        start_idx = comm.bcast(start_idx, root=0)

        # Now send each rank to everyone else
        all_fun = [None] * start_idx[-1]
        for r in range(size):
            all_fun[start_idx[r]:start_idx[r+1]] = comm.bcast(str_fun, root=r)
        str_fun = all_fun

        if save_sympy:
            all_sym = OrderedDict()
            for r in range(size):
                sym_keys = comm.bcast(list(sym_fun.keys()), root=r)
                sym_vals = comm.bcast(list(sym_fun.values()), root=r)
                for i in range(len(sym_keys)):
                    key = sym_keys[i]
                    if key not in all_sym:
                        all_sym[key] = sym_vals[i]

            sym_fun = all_sym

    return str_fun, sym_fun


def sympy_simplify(all_fun, all_sym, all_inv_subs, max_param, expand_fun=True, tmax=1, check_perm=False):
    """Simplify equations and find duplicates.

    Args:
        :all_fun (list): list of strings containing all functions
        :all_sym (list): list of sympy objects containing all functions
        :all_inv_subs (list): list of dictionaries giving subsitutions to be applied to all functions
        :max_param (int): maximum number of free parameters in any equation in all_fun
        :expand_fun (bool, default=True): whether to run the sympy expand options (True) or not (False)
        :tmax (float, default=1.): maximum time in seconds to run any one part of simplification procedure for a given function
        :check_perm (bool, default=False): whether to check all possible permutations and inverses of constants (True) or not (False)

    Returns:
        :all_fun: list of strings containing all (updated) functions
        :all_sym (list): list of sympy objects containing all (updated) functions
        :all_inv_subs: list of dictionaries giving subsitutions to be applied to all (updated) functions

    """

    if max_param == 0:
        return all_fun, all_sym, all_inv_subs

    if len(all_fun) == 0:
        return all_fun, all_sym, all_inv_subs

    esrp = ESRPrinter()

    if max_param > 0:
        param_list = ['a%i' % i for i in range(max_param)]
        all_a = sympy.symbols(" ".join(param_list), real=True)
        if max_param == 1:
            all_a = [all_a]
    else:
        param_list = []

    i = np.atleast_1d(utils.split_idx(len(all_inv_subs), rank, size))
    if len(i) == 0:
        str_fun = []
        sym_fun = []
        inv_subs_fun = []
    else:
        str_fun = all_fun[i[0]:i[-1]+1]
        sym_fun = all_sym[i[0]:i[-1]+1]
        inv_subs_fun = all_inv_subs[i[0]:i[-1]+1]

    identity_subs = {a: a for a in all_a}

    #  Do some substitutions to simplify
    comm.Barrier()
    comb = list(itertools.combinations(np.flip(np.arange(max_param)), 2))
    if max_param > 1:
        for c in comb:
            #  Second number = 0 if normal subs, = 1 if abs subs
            all_expr = [[all_a[c[0]] + all_a[c[1]], 0],
                        [all_a[c[0]] - all_a[c[1]], 0],
                        [all_a[c[1]] - all_a[c[0]], 0],
                        [all_a[c[0]] * all_a[c[1]], 0],
                        [all_a[c[0]] / all_a[c[1]], 0],
                        [all_a[c[1]] / all_a[c[0]], 0],
                        [all_a[c[0]] + sympy.Abs(all_a[c[1]]), 0],
                        [all_a[c[0]] - sympy.Abs(all_a[c[1]]), 0],
                        [sympy.Abs(all_a[c[1]]) - all_a[c[0]], 0],
                        [all_a[c[0]] * sympy.Abs(all_a[c[1]]), 0],
                        [all_a[c[0]] / sympy.Abs(all_a[c[1]]), 0],
                        [all_a[c[1]] / sympy.Abs(all_a[c[0]]), 0],
                        [all_a[c[1]] + sympy.Abs(all_a[c[0]]), 0],
                        [all_a[c[1]] - sympy.Abs(all_a[c[0]]), 0],
                        [sympy.Abs(all_a[c[0]]) - all_a[c[1]], 0],
                        [all_a[c[1]] * sympy.Abs(all_a[c[0]]), 0],
                        [all_a[c[1]] / sympy.Abs(all_a[c[0]]), 0],
                        [all_a[c[0]] / sympy.Abs(all_a[c[1]]), 0],
                        [sympy.Abs(all_a[c[0]]) * sympy.Abs(all_a[c[1]]), 1],
                        [sympy.Abs(all_a[c[0]]) + sympy.Abs(all_a[c[1]]), 1],
                        [sympy.Abs(all_a[c[0]]) - sympy.Abs(all_a[c[1]]), 0],
                        [sympy.Abs(all_a[c[1]]) - sympy.Abs(all_a[c[0]]), 0],
                        [sympy.Abs(all_a[c[0]]) / sympy.Abs(all_a[c[1]]), 1],
                        [sympy.Abs(all_a[c[1]]) / sympy.Abs(all_a[c[0]]), 1],
                        [pow_abs(all_a[c[1]], all_a[c[0]]), 1],
                        [pow_abs(all_a[c[0]], all_a[c[1]]), 1],
                        [pow_abs(all_a[c[1]], sympy.Abs(all_a[c[0]])), 1],
                        [pow_abs(all_a[c[0]], sympy.Abs(all_a[c[1]])), 1],
                        ]

            for i in range(len(str_fun)):
                orig_fun = str_fun[i]
                orig_sym = sym_fun[i]
                try:
                    with time_limit(tmax):
                        if (all_a[c[0]] in sym_fun[i].free_symbols) and (all_a[c[1]] in sym_fun[i].free_symbols):
                            #  Make sure symbols only appear once in sym version
                            if sym_fun[i].count(all_a[c[1]]) == 1:
                                if sym_fun[i].count(all_a[c[0]]) == 1:
                                    v = 1
                                    keep = False
                                else:
                                    v = None    # Don't have to ignore this combination
                                    keep = True
                            # or str_fun[i].count(param_list[c[0]]) == 1:
                            elif sym_fun[i].count(all_a[c[0]]) == 1:
                                if sym_fun[i].count(all_a[c[1]]) == 1:
                                    v = 0
                                    keep = False
                                else:
                                    v = None    # Don't have to ignore this combination
                                    keep = True
                            else:
                                v = None

                            if v is not None:
                                for expr in all_expr:
                                    if sym_fun[i].has(expr[0]):
                                        s = sym_fun[i]
                                        f1 = str(sym_fun[i])
                                        if expr[1] == 0:
                                            sym_fun[i] = sym_fun[i].subs(
                                                expr[0], all_a[c[v]])
                                            f2 = str(sym_fun[i])
                                            if inv_subs_fun[i] is None:
                                                if keep:
                                                    inv_subs_fun[i] = [
                                                        str({expr[0]: all_a[c[v]]})]
                                                else:
                                                    inv_subs_fun[i] = [
                                                        str(np.nan)]
                                            else:
                                                if keep:
                                                    inv_subs_fun[i].append(
                                                        str({expr[0]: all_a[c[v]]}))
                                                else:
                                                    inv_subs_fun[i].append(
                                                        str(np.nan))
                                        elif expr[1] == 1:
                                            sym_fun[i] = sym_fun[i].subs(
                                                expr[0], sympy.Abs(all_a[c[v]], evaluate=False))
                                            f2 = str(sym_fun[i])
                                            if inv_subs_fun[i] is None:
                                                if keep:
                                                    inv_subs_fun[i] = [
                                                        str({expr[0]: sympy.Abs(all_a[c[v]])})]
                                                else:
                                                    inv_subs_fun[i] = [
                                                        str(np.nan)]
                                            else:
                                                if keep:
                                                    inv_subs_fun[i].append(
                                                        str({expr[0]: sympy.Abs(all_a[c[v]])}))
                                                else:
                                                    inv_subs_fun[i].append(
                                                        str(np.nan))
                                        if expand_fun:
                                            str_fun[i] = esrp.doprint(
                                                sym_fun[i].expand())
                                        else:
                                            str_fun[i] = esrp.doprint(
                                                sym_fun[i])
                except TimeoutException:
                    print('TIMED OUT:', orig_fun)
                    str_fun[i] = orig_fun
                    sym_fun[i] = orig_sym

    # See if multiples of constants appear
    comm.Barrier()
    if max_param > 0:
        for i in range(len(str_fun)):
            orig_fun = str_fun[i]
            orig_sym = sym_fun[i]
            try:
                with time_limit(tmax):
                    # Can't use force=True since sometimes pulls out factor -1 and computes log(-1)
                    if expand_fun:
                        sym_fun[i] = sympy.expand_log(sym_fun[i])

                    numbers = [atom for atom in sym_fun[i].atoms(
                    ) if atom.is_number and atom.is_finite]
                    even = [n for n in numbers if n.is_Integer and n.is_even]
                    odd = [n for n in numbers if n.is_Integer and n.is_odd]

                    for j in range(len(param_list)):
                        if str_fun[i].count(param_list[j]) > 0:
                            all_expr = [
                                [n*all_a[j], all_a[j], 0, str({all_a[j]: all_a[j]/n})] for n in numbers]
                            all_expr += [[all_a[j]**n, sympy.Abs(all_a[j], evaluate=False), 0, str(
                                {all_a[j]: pow_abs(all_a[j], 1/n)})] for n in even]
                            all_expr += [[all_a[j]**n, all_a[j], 0,
                                          str({all_a[j]: all_a[j] ** (1/n)})] for n in odd]
                            all_expr += [[all_a[j]**n * sympy.Abs(all_a[j]), sympy.Abs(
                                all_a[j], evaluate=False), 0, str({all_a[j]: pow_abs(all_a[j], 1/(n+1))})] for n in even]
                            all_expr += [[all_a[j]**n * sympy.Abs(all_a[j]), all_a[j], 0, str(
                                {all_a[j]: pow_abs(all_a[j], 1/(n+1)) * sympy.sign(all_a[j])})] for n in odd]
                            all_expr += [[square(all_a[j]), sympy.Abs(all_a[j], evaluate=False), 0, str({all_a[j]: sqrt_abs(all_a[j])})],
                                         [cube(all_a[j]), all_a[j], 0, str(
                                             {all_a[j]: all_a[j]**(1/3)})],
                                         [square(sympy.Abs(all_a[j])), sympy.Abs(
                                             all_a[j], evaluate=False), 0, str({all_a[j]: sqrt_abs(all_a[j])})],
                                         [cube(sympy.Abs(all_a[j])), sympy.Abs(all_a[j], evaluate=False), 0, str(
                                             {all_a[j]: pow_abs(all_a[j], 1/3)})],
                                         [sqrt_abs(all_a[j]), sympy.Abs(all_a[j], evaluate=False), 0, str(
                                             {all_a[j]: square(all_a[j])})],
                                         [log_abs(all_a[j]), all_a[j], 1, str(
                                             {all_a[j]: sympy.exp(all_a[j])})],
                                         [sympy.exp(all_a[j]), sympy.Abs(all_a[j], evaluate=False), 0, str(
                                             {all_a[j]: log_abs(all_a[j])})]
                                         ]

                            for expr in all_expr:
                                if sym_fun[i].has(expr[0]):
                                    ss = str(sym_fun[i]).replace(" ", "")
                                    ee = str(expr[0]).replace(" ", "")
                                    # Make sure variable only appears in this form in the sym version
                                    if ss.count(param_list[j]) in [1, ss.count(ee), ee.count(param_list[j])]:
                                        f0 = sym_fun[i].copy()
                                        sym_fun[i] = sym_fun[i].subs(
                                            {expr[0]: expr[1]})

                                        try:

                                            if expr[2] == 0:
                                                if 'zoo' in str(expr[3]):
                                                    #  Don't make this substitution
                                                    sym_fun[i] = f0.copy()
                                                else:
                                                    if inv_subs_fun[i] is None:
                                                        inv_subs_fun[i] = [
                                                            expr[3]]
                                                    else:
                                                        inv_subs_fun[i].append(
                                                            expr[3])

                                            elif expr[2] == 1:
                                                # These cases can be tricky with Abs
                                                s = {expr[1]: expr[0]}
                                                f1 = sym_fun[i].subs(
                                                    {sympy.Abs(all_a[j]): all_a[j]}).subs(s)
                                                if f0.equals(f1):
                                                    #  It worked, so append original subs
                                                    if inv_subs_fun[i] is None:
                                                        inv_subs_fun[i] = [
                                                            expr[3]]
                                                    else:
                                                        inv_subs_fun[i].append(
                                                            expr[3])
                                                else:
                                                    s = {expr[1]: sympy.Abs(
                                                        expr[0], evaluate=False)}
                                                    f2 = sym_fun[i].subs(
                                                        {sympy.Abs(all_a[j]): all_a[j]}).subs(s)
                                                    if f0.equals(f2):
                                                        if inv_subs_fun[i] is None:
                                                            inv_subs_fun[i] = [
                                                                expr[3]]
                                                        else:
                                                            inv_subs_fun[i].append(
                                                                expr[3])
                                                    else:
                                                        # Can't undo the simplification, so we won't do it
                                                        sym_fun[i] = f0.copy()
                                        except Exception:
                                            print('Bad comparison:', f0, f1)
                                            sys.stdout.flush()
                                            sym_fun[i] = f0.copy()

                                        if expand_fun:
                                            str_fun[i] = esrp.doprint(
                                                sym_fun[i].expand())
                                        else:
                                            str_fun[i] = esrp.doprint(
                                                sym_fun[i])

                                        break
            except TimeoutException:
                print('TIMED OUT:', orig_fun)
                str_fun[i] = orig_fun
                sym_fun[i] = orig_sym

    comm.Barrier()
    all_fun, all_sym, all_inv_subs = make_changes(all_fun, all_sym, all_inv_subs,
                                                  str_fun, sym_fun, inv_subs_fun)

    i = np.atleast_1d(utils.split_idx(len(all_inv_subs), rank, size))
    if len(i) == 0:
        str_fun = []
        sym_fun = []
        inv_subs_fun = []
    else:
        str_fun = all_fun[i[0]:i[-1]+1]
        sym_fun = all_sym[i[0]:i[-1]+1]
        inv_subs_fun = all_inv_subs[i[0]:i[-1]+1]

    change_indices = []
    ref_indices = []
    new_inv_subs = []

    # Check permutations and inverses of constants
    comm.Barrier()
    if max_param > 1 and check_perm:
        use_a = list(all_a) + [1/a for a in all_a]
        perm = list(itertools.permutations(
            np.flip(np.arange(len(use_a))), len(all_a)))

        for i in range(len(str_fun)):
            orig_fun = str_fun[i]
            orig_sym = sym_fun[i]
            s = list(sym_fun[i].free_symbols)
            s = list(set(s).intersection(all_a))
            perm = list(itertools.permutations(
                np.flip(np.arange(len(s))), len(s)))
            perm.remove(tuple(range(len(s))))
            try_subs = [{s[i]: s[p[i]]
                         for i in range(len(p)) if i != p[i]} for p in perm]
            try:
                with time_limit(tmax):
                    for p in range(len(try_subs)):
                        if all([a in sym_fun[i].free_symbols for a in list(try_subs[p].keys())]):
                            expr = sym_fun[i].subs(
                                try_subs[p], simultaneous=True)
                            if expand_fun:
                                str_expand = esrp.doprint(expr.expand())
                            else:
                                str_expand = esrp.doprint(expr)
                            if str_expand in all_fun:
                                m = all_fun.index(str_expand)
                                n = all_fun.index(str_fun[i])
                                if n != m:
                                    change_indices.append(n)
                                    ref_indices.append(m)
                                    new_inv_subs.append(str(try_subs[p]))
                                    break
            except TimeoutException:
                print('TIMED OUT:', orig_fun)
                str_fun[i] = orig_fun
                sym_fun[i] = orig_sym

        comm.Barrier()

        change_indices = comm.gather(change_indices, root=0)
        ref_indices = comm.gather(ref_indices, root=0)
        new_inv_subs = comm.gather(new_inv_subs, root=0)
        if rank == 0:
            change_indices = list(itertools.chain(*change_indices))
            ref_indices = list(itertools.chain(*ref_indices))
            new_inv_subs = list(itertools.chain(*new_inv_subs))
        change_indices = comm.bcast(change_indices, root=0)
        ref_indices = comm.bcast(ref_indices, root=0)
        new_inv_subs = comm.bcast(new_inv_subs, root=0)

        for i in range(len(change_indices)):
            # Check we haven't already made the change
            if (ref_indices[i] not in change_indices[:i]) and (change_indices[i] not in change_indices[:i]):
                all_fun[change_indices[i]] = all_fun[ref_indices[i]]
                all_sym[change_indices[i]] = all_sym[ref_indices[i]]
                if all_inv_subs[change_indices[i]] is None:
                    all_inv_subs[change_indices[i]] = []
                all_inv_subs[change_indices[i]].append(new_inv_subs[i])

        i = np.atleast_1d(utils.split_idx(len(all_inv_subs), rank, size))
        if len(i) == 0:
            str_fun = []
            sym_fun = []
            inv_subs_fun = []
        else:
            str_fun = all_fun[i[0]:i[-1]+1]
            sym_fun = all_sym[i[0]:i[-1]+1]
            inv_subs_fun = all_inv_subs[i[0]:i[-1]+1]

    comm.Barrier()
    if max_param > 0:

        change_indices = []
        ref_indices = []
        new_inv_subs = []

        # See if function with a0 -> -a0 already in list
        for i in range(len(str_fun)):
            orig_fun = str_fun[i]
            orig_sym = sym_fun[i]
            try:
                with time_limit(tmax):
                    for j in range(len(all_a)):
                        if param_list[j] in str_fun[i]:
                            expr = sym_fun[i].subs(all_a[j], -all_a[j])
                            if str(expr) != str_fun[i]:
                                if expand_fun:
                                    str_expand = str(expr.expand())
                                else:
                                    str_expand = str(expr)

                                if str_expand in all_fun:
                                    m = all_fun.index(str_expand)
                                    n = all_fun.index(str_fun[i])
                                    if n != m:
                                        change_indices.append(n)
                                        ref_indices.append(m)
                                        new_inv_subs.append(
                                            str({all_a[j]: -all_a[j]}))
                                        break
            except TimeoutException:
                print('TIMED OUT:', orig_fun)
                str_fun[i] = orig_fun
                sym_fun[i] = orig_sym

        comm.Barrier()
        change_indices = comm.gather(change_indices, root=0)
        ref_indices = comm.gather(ref_indices, root=0)
        new_inv_subs = comm.gather(new_inv_subs, root=0)
        if rank == 0:
            change_indices = list(itertools.chain(*change_indices))
            ref_indices = list(itertools.chain(*ref_indices))
            new_inv_subs = list(itertools.chain(*new_inv_subs))
        change_indices = comm.bcast(change_indices, root=0)
        ref_indices = comm.bcast(ref_indices, root=0)
        new_inv_subs = comm.bcast(new_inv_subs, root=0)

        for i in range(len(change_indices)):
            # Check we haven't already made the change
            if (ref_indices[i] not in change_indices[:i]) and (change_indices[i] not in change_indices[:i]):
                all_fun[change_indices[i]] = all_fun[ref_indices[i]]
                all_sym[change_indices[i]] = all_sym[ref_indices[i]]
                if all_inv_subs[change_indices[i]] is None:
                    all_inv_subs[change_indices[i]] = []
                all_inv_subs[change_indices[i]].append(new_inv_subs[i])

        i = np.atleast_1d(utils.split_idx(len(all_inv_subs), rank, size))
        if len(i) == 0:
            str_fun = []
            sym_fun = []
            inv_subs_fun = []
        else:
            str_fun = all_fun[i[0]:i[-1]+1]
            sym_fun = all_sym[i[0]:i[-1]+1]
            inv_subs_fun = all_inv_subs[i[0]:i[-1]+1]

        # Check parameters are in correct order
        comm.Barrier()
        for i in range(len(str_fun)):
            orig_fun = str_fun[i]
            orig_sym = sym_fun[i]
            try:
                with time_limit(tmax):
                    vars = list(sym_fun[i].free_symbols)
                    vars = [str(v) for v in vars]
                    param_list = ['a%i' % i for i in range(max_param)]
                    common = list(set(param_list).intersection(vars))
                    if len(common) > 0:
                        common.sort()
                        if common[-1] != param_list[len(common)-1]:
                            common = [int(v[1:]) for v in common]
                            s = {all_a[common[i]]: all_a[i]
                                 for i in range(len(common))}
                            sym_fun[i] = sym_fun[i].subs(s, simultaneous=True)
                            if expand_fun:
                                str_fun[i] = esrp.doprint(sym_fun[i].expand())
                            else:
                                str_expand = esrp.doprint(expr)
                            if s != identity_subs:
                                if inv_subs_fun[i] is None:
                                    inv_subs_fun[i] = [str(s)]
                                else:
                                    inv_subs_fun[i].append(str(s))
            except TimeoutException:
                print('TIMED OUT:', orig_fun)
                str_fun[i] = orig_fun
                sym_fun[i] = orig_sym

    # If we find a zoo, let's make this a nan
    for i in range(len(sym_fun)):
        if sympy.zoo in sym_fun[i].atoms():
            sym_fun[i] = sympy.core.numbers.NaN
            str_fun[i] = str(sympy.core.numbers.NaN)

    comm.Barrier()

    all_fun, all_sym, all_inv_subs = make_changes(all_fun, all_sym, all_inv_subs,
                                                  str_fun, sym_fun, inv_subs_fun)

    return all_fun, all_sym, all_inv_subs


def expand_or_factor(all_sym, tmax=1, method='expand'):
    """Run the sympy expand or factor functions

    Args:
        :all_sym (OrderedDict): dictionary of sympy objects which can be accessed by their string representations.
        :tmax (float, default=1.): maximum time in seconds to run any one part of expand/simplify procedure for a given function
        :method (str, default='expand'): whether to run expand ('expand') or simplify ('simplify'). All other options are ignored

    Returns:
        :all_sym (OrderedDict): dictionary of (updated) sympy objects which can be accessed by their string representations.
    """

    vals = list(all_sym.values())
    keys = list(all_sym.keys())

    i = np.atleast_1d(utils.split_idx(len(vals), rank, size))

    change_vals = []
    change_idx = []

    p = ESRPrinter()
    if len(i) > 0:
        for j in range(i[0], i[-1]+1):
            if vals[j] is sympy.core.numbers.NaN:
                continue
            try:
                with time_limit(tmax):
                    if method == 'expand':
                        v = vals[j].expand()
                    elif method == 'factor':
                        v = vals[j].powsimp()
                        v = v.factor()
                    if p.doprint(v) != keys[j]:
                        change_idx.append(j)
                        change_vals.append(v)
            except TimeoutException:
                print('Terminated expanding:', j, rank, vals[j])

    change_vals = comm.gather(change_vals, root=0)
    change_idx = comm.gather(change_idx, root=0)
    if rank == 0:
        change_vals = list(itertools.chain(*change_vals))
        change_idx = list(itertools.chain(*change_idx))
    change_vals = comm.bcast(change_vals, root=0)
    change_idx = comm.bcast(change_idx, root=0)

    for i in range(len(change_idx)):
        all_sym[keys[change_idx[i]]] = change_vals[i]

    return all_sym


def do_sympy(all_fun, all_sym, compl, search_tmax, expand_tmax, dirname, track_memory=False):
    """Run the duplicate checking procedure

    Args:
        :all_fun (list): list of strings containing all functions
        :all_sym (OrderedDict): dictionary of sympy objects which can be accessed by their string representations.
        :compl (int):
        :search_tmax (float, default=1.): maximum time in seconds to run any one part of simplification procedure for a given function
        :expand_tmax (float, default=1.): maximum time in seconds to run any one part of expand/simplify procedure for a given function
        :dirname (str): directory path to save results in
        :track_memory (bool, default=True): whether to compute and print memory statistics (True) or not (False)
    Returns:
        :all_fun (list): list of strings containing all (updated) functions
        :all_sym (list): dictionary of (updated) sympy objects which can be accessed by their string representations.
        :count (int): number of rounds of optimisation which were performed
    """

    if rank == 0 and track_memory:
        utils.using_mem("start do_sympy")
        utils.locals_size(locals())

    max_param = get_max_param(all_fun)

    # Split by number of parameters
    old_nuniq = 0
    new_nuniq = len(all_fun)
    count = 0

    # Initial optimisation
    while old_nuniq != new_nuniq:

        all_inv_subs = [None] * len(all_fun)

        old_nuniq = new_nuniq

        if rank == 0:
            print('Optimisation', count, old_nuniq, len(all_fun))
        sys.stdout.flush()

        if rank == 0 and track_memory:
            utils.using_mem("start")
            utils.locals_size(locals())

        # (1) Get unique functions and matches
        if rank == 0:
            print('\tGetting unique functions')
        sys.stdout.flush()

        uniq, match = utils.get_unique_indexes(all_fun)
        uniq_fun = list(uniq.keys())

        if rank == 0:
            if track_memory:
                utils.using_mem("end")
                utils.locals_size(locals())
            print('\tGetting unique sympy')
        sys.stdout.flush()

        all_sym = [all_sym[u] for u in uniq_fun]

        if rank == 0:
            print('\tGetting unique inverse subs')
        sys.stdout.flush()

        uniq_inv_subs = [all_inv_subs[i] for i in uniq.values()]

        del uniq
        gc.collect()

        # (2) Simplify the unique functions
        add_inv_subs = [None] * len(uniq_inv_subs)
        nparam = count_params(uniq_fun, max_param)
        for i in range(max_param+1):
            if rank == 0:
                print('\t\tnparam = %i' % i)
            sys.stdout.flush()

            check_perm = (count != 0)

            m = nparam == i
            j = np.atleast_1d(np.squeeze(np.argwhere(m)))
            f = [uniq_fun[jj] for jj in j]
            e = [all_sym[jj] for jj in j]
            t = [None if uniq_inv_subs[jj] is None else uniq_inv_subs[jj].copy()
                 for jj in j]
            f, e, t = sympy_simplify(
                f, e, t, i, expand_fun=False, tmax=search_tmax, check_perm=check_perm)
            for k in range(len(t)):
                old_fun = uniq_fun[j[k]]
                uniq_fun[j[k]] = f[k]
                all_sym[j[k]] = e[k]
                if uniq_inv_subs[j[k]] is None:
                    add_inv_subs[j[k]] = t[k]
                else:
                    add_inv_subs[j[k]] = t[k][len(uniq_inv_subs[j[k]]):]

        del f, e, t, m, j, k, uniq_inv_subs, nparam
        gc.collect()

        # (3) Make replacements to full functions list
        for i in range(len(all_fun)):
            old_fun = all_fun[i]
            m = match[all_fun[i]]
            all_fun[i] = uniq_fun[m]
            if add_inv_subs[m] is not None and len(add_inv_subs[m]) > 0:
                if all_inv_subs[i] is None:
                    all_inv_subs[i] = add_inv_subs[m].copy()
                else:
                    all_inv_subs[i] = all_inv_subs[i].copy() + \
                        add_inv_subs[m].copy()

        del add_inv_subs, match
        gc.collect()

        if rank == 0:
            print('\tMaking dict')
        sys.stdout.flush()

        all_sym = dict(zip(uniq_fun, all_sym))

        if rank == 0:
            new_nuniq = len(set(all_fun))
        else:
            new_nuniq = None
        new_nuniq = comm.bcast(new_nuniq, root=0)

        if rank == 0:

            print('\tPrinting inv_subs to file')
            data = [i for i in range(len(all_inv_subs))
                    if all_inv_subs[i] is not None]
            with open(dirname + '/inv_idx_%i_round_%i.txt' % (compl, count), "w") as f:
                for i in data:
                    print(i, file=f)

            print('\tPrinting inv to file')
            data = [all_inv_subs[i] for i in data]
            with open(dirname + '/inv_subs_%i_round_%i.txt' % (compl, count), "w") as f:
                writer = csv.writer(f, delimiter=';')
                writer.writerows(data)

            del data
            gc.collect()

            if track_memory:
                utils.using_mem("end of round")
                utils.locals_size(locals())

        count += 1

    round1_count = count

    if rank == 0 and track_memory:
        utils.using_mem("END")
        utils.locals_size(locals())

    #  Expand functions
    if rank == 0:
        print('\nExpanding')
    sys.stdout.flush()
    all_sym = expand_or_factor(all_sym, tmax=expand_tmax, method='expand')
    count = 0
    old_nuniq = 0

    # Now replace functions by their expanded form
    if rank == 0:
        print('\nRewriting')
    sys.stdout.flush()

    while old_nuniq != new_nuniq:

        all_inv_subs = [None] * len(all_fun)
        old_nuniq = new_nuniq

        if rank == 0:
            print('Optimisation', count, new_nuniq, len(all_fun))
        sys.stdout.flush()

        if rank == 0 and track_memory:
            utils.using_mem("start")
            utils.locals_size(locals())

        # (1) Get unique functions and matches
        if rank == 0:
            print('\tGetting unique functions')
        sys.stdout.flush()

        uniq, match = utils.get_unique_indexes(all_fun)
        uniq_fun = list(uniq.keys())

        if rank == 0:
            if track_memory:
                utils.using_mem("end")
                utils.locals_size(locals())
            print('\tGetting unique sympy')
        sys.stdout.flush()

        all_sym = [all_sym[u] for u in uniq_fun]

        if rank == 0:
            print('\tGetting unique inverse subs')
        sys.stdout.flush()

        uniq_inv_subs = [all_inv_subs[i] for i in uniq.values()]

        del uniq
        gc.collect()

        # (2) Simplify the unique functions
        add_inv_subs = [None] * len(uniq_inv_subs)
        nparam = count_params(uniq_fun, max_param)
        for i in range(max_param+1):
            if rank == 0:
                print('\t\tnparam = %i' % i)
            sys.stdout.flush()

            check_perm = True

            m = nparam == i
            j = np.atleast_1d(np.squeeze(np.argwhere(m)))
            f = [uniq_fun[jj] for jj in j]
            e = [all_sym[jj] for jj in j]
            t = [None if uniq_inv_subs[jj] is None else uniq_inv_subs[jj].copy()
                 for jj in j]
            f, e, t = sympy_simplify(
                f, e, t, i, expand_fun=True, tmax=search_tmax)
            for k in range(len(t)):
                old_fun = uniq_fun[j[k]]
                uniq_fun[j[k]] = f[k]
                all_sym[j[k]] = e[k]
                if uniq_inv_subs[j[k]] is None:
                    add_inv_subs[j[k]] = t[k]
                else:
                    add_inv_subs[j[k]] = t[k][len(uniq_inv_subs[j[k]]):]

        del f, e, t, m, j, k, uniq_inv_subs, nparam
        gc.collect()

        # (3) Make replacements to full functions list
        for i in range(len(all_fun)):
            old_fun = all_fun[i]
            m = match[all_fun[i]]
            all_fun[i] = uniq_fun[m]
            if add_inv_subs[m] is not None and len(add_inv_subs[m]) > 0:
                if all_inv_subs[i] is None:
                    all_inv_subs[i] = add_inv_subs[m].copy()
                else:
                    all_inv_subs[i] = all_inv_subs[i].copy() + \
                        add_inv_subs[m].copy()

        del add_inv_subs, match
        gc.collect()

        if rank == 0:
            print('\tMaking dict')
        sys.stdout.flush()

        all_sym = dict(zip(uniq_fun, all_sym))

        if rank == 0:
            new_nuniq = len(set(all_fun))
        else:
            new_nuniq = None
        new_nuniq = comm.bcast(new_nuniq, root=0)

        if rank == 0:

            print('\tPrinting inv_subs to file')
            data = [i for i in range(len(all_inv_subs))
                    if all_inv_subs[i] is not None]
            with open(dirname + '/inv_idx_%i_round_%i.txt' % (compl, round1_count + count), "w") as f:
                for i in data:
                    print(i, file=f)

            print('\tPrinting inv to file')
            data = [all_inv_subs[i] for i in data]
            with open(dirname + '/inv_subs_%i_round_%i.txt' % (compl, round1_count + count), "w") as f:
                writer = csv.writer(f, delimiter=';')
                writer.writerows(data)

            del data
            gc.collect()

            if track_memory:
                utils.using_mem("end of round")
                utils.locals_size(locals())

        count += 1

    if rank == 0:
        print('\nFinal factorisation')
    sys.stdout.flush()
    all_sym = expand_or_factor(all_sym, tmax=expand_tmax, method='factor')

    if rank == 0 and track_memory:
        utils.using_mem("END")
        utils.locals_size(locals())

    return all_fun, all_sym, round1_count + count


def get_all_dup(max_param):
    """Finds self-inverse transformations of parameters, to be used
    in simplify_inv_subs(inv_subs, all_dup)

    Args:
        :max_param (int): maximum number of parameters to consider

    Returns:
        :all_dup (list): list of dictionaries giving subsitutions which are self-inverse
    """

    if max_param == 0:
        return []
    param_list = ['a%i' % i for i in range(max_param)]
    all_a = sympy.symbols(" ".join(param_list), real=True)
    if max_param == 1:
        all_a = [all_a]

    all_dup = [str({a: -a}) for a in all_a]
    all_dup += [str({a: 1/a}) for a in all_a]

    comb = list(itertools.combinations(np.flip(np.arange(max_param)), 2))
    all_dup += [str({all_a[c[0]]: all_a[c[1]], all_a[c[1]]: all_a[c[0]]})
                for c in comb]
    all_dup += [str({all_a[c[1]]: all_a[c[0]], all_a[c[0]]: all_a[c[1]]})
                for c in comb]

    return all_dup


def simplify_inv_subs(inv_subs, all_dup):
    """Find if two consecutive {a0: -a0} or {a0: a1, a1: a0} or {a0: 1/a0}
    and then remove both of these

    Args:
        :inv_subs (list): list of dictionaries giving subsitutions to check
        :all_dup (list): list of dictionaries giving subsitutions which are self-inverse

    Returns:
        :all_subs (list): list of dictionaries giving subsitutions without consecutive self-inverses
    """

    if inv_subs is None or len(inv_subs) == 0:
        return inv_subs

    del_idx = []

    i = 0

    while i < len(inv_subs) - 1:
        if inv_subs[i] in all_dup:
            if inv_subs[i+1] == inv_subs[i]:
                del_idx.append(i)
                del_idx.append(i+1)
                i += 2
            else:
                i += 1
        else:
            i += 1

    new_inv = [inv_subs[i] for i in range(len(inv_subs)) if i not in del_idx]
    if len(new_inv) == 0:
        new_inv = None

    return new_inv


def count_lines(fname):
    """
    Count the number of lines in a file.

    Args:
        :fname (str): file name to count lines in

    Returns:
        :int: number of lines in the file
    """
    with open(fname, 'r') as f:
        return sum(1 for _ in f)


def get_line_range(n_lines):
    """
    Return (imin, imax) inclusive range of lines for a given rank.

    Args:
        :n_lines (int): total number of lines in the file

    Returns:
        :tuple: (imin, imax) where imin is the first line index for this rank and 
            imax is the exclusive last line index for this rank
    """
    counts = [n_lines // size + (1 if i < n_lines % size else 0)
              for i in range(size)]
    offsets = np.cumsum([0] + counts[:-1])
    imin = offsets[rank]
    imax = imin + counts[rank]  # exclusive
    return imin, imax


def load_subs(fname, max_param, use_sympy=True, bcast_res=True):
    """Load the subsitutions required to convert between all and unique functions

    Args:
        :fname (str): file name containing the subsitutions
        :max_param (int): maximum number of parameters to consider
        :use_sympy (bool, default=True): whether to convert substituions to sympy objects (True) or leave as strings (False)
        :bcast_res (bool, default=True): whether to allow all ranks to have the substitutions (True) or just the 0th rank (False)

    Returns:
        :all_subs (dict): dict of substitutions required to convert between all and unique functions. 
            Each item is either a dictionary with sympy objects as keys and values (use_sympy=True) or 
            a string version of this dictionary (use_sympy=False). If bcast_res=True, then all ranks have this dict, 
            otherwise all ranks receive a chunk of the dict corresponding to their rank.

    """

    if rank == 0:
        n_lines = count_lines(fname)
    else:
        n_lines = None
    n_lines = comm.bcast(n_lines, root=0)

    imin, imax = get_line_range(n_lines)
    all_subs = {}  # Use a dict instead of a list
    with open(fname, 'r') as f:
        for i, line in enumerate(f):
            if i >= imax:
                break
            if i >= imin:
                sub = line.strip().split(';')
                if sub != ['']:
                    all_subs[i] = sub

    param_list = ['a%i' % i for i in range(max_param)]
    all_a = sympy.symbols(" ".join(param_list), real=True)
    if max_param == 1:
        all_a = [all_a]

    locs = sympy_locs

    if max_param > 0:
        for i in range(len(all_a)):
            locs["a%i" % i] = all_a[i]

    for i in all_subs.keys():
        for j in range(len(all_subs[i])):
            all_subs[i][j] = all_subs[i][j].replace("{", "{'")
            all_subs[i][j] = all_subs[i][j].replace("}", "'}")
            all_subs[i][j] = all_subs[i][j].replace(", ", "', '")
            all_subs[i][j] = all_subs[i][j].replace(": ", "': '")
            if all_subs[i][j] == 'nan':
                all_subs[i][j] = np.nan
            else:
                d = ast.literal_eval(all_subs[i][j])
                k = list(d.keys())
                v = list(d.values())
                k = [sympy.sympify(kk, locals=locs) for kk in k]
                v = [sympy.sympify(vv, locals=locs) for vv in v]
                all_subs[i][j] = dict(zip(k, v))
                if not use_sympy:
                    all_subs[i][j] = str(all_subs[i][j])
    comm.Barrier()

    if bcast_res:

        gathered = comm.gather(all_subs, root=0)
        if rank == 0:
            all_subs = {}
            for d in gathered:
                all_subs.update(d)
            # [all_subs.get(i, []) for i in range(total_lines)]

        all_subs = comm.bcast(all_subs, root=0)
        # Fix MPI4PY bug for empty lists
        # if isinstance(all_subs, int):
        #     all_subs = [[] for _ in range(all_subs)]

    return all_subs


def convert_params(p_meas, fish_meas, inv_subs, n=4):
    """Convert parameters from those in unique function to those in actual function

    Args:
        :p_meas (list): list of measured parameters in unique function
        :fish_meas (list): flattened version of the Hessian of -log(likelihood) at the maximum likelihood point
        :inv_subs (list): list of substitutions required to convert between all and unique functions
        :n (int, default=4): the number of dimensions of the array from which fish_meas was computed using

    Returns:
        :p_new (list): list of parameters for the actual function
        :diag_fish (np.array): the diagonal entries of the Fisher matrix of the actual function at the maximum likelihood point

    """

    max_param = len(p_meas)

    if np.nan in inv_subs:
        return np.array([np.nan]*max_param), np.array([np.nan]*max_param)

    fish = np.zeros((n, n))
    fish[np.triu_indices(n)] = fish_meas
    fish = np.where(fish, fish, fish.T)
    fish = fish[:max_param, :max_param]

    param_list = ['a%i' % i for i in range(max_param)]
    all_a = sympy.symbols(" ".join(param_list), real=True)
    if max_param == 1:
        all_a = [all_a]

    p = sympy.Array(sympy.symbols(" ".join(param_list), real=True))
    for i in range(len(inv_subs)):
        p = p.subs(inv_subs[i], simultaneous=True)

    jac = sympy.Matrix(p).jacobian(all_a)

    if max_param == 1:
        p_lam = sympy.lambdify(all_a[0], str(p))
    else:
        p_lam = sympy.lambdify(all_a[:len(p_meas)], p)
    p_new = p_lam(*p_meas)

    j_lam = sympy.lambdify(all_a[:len(p_meas)], jac)
    j = j_lam(*p_meas)
    jinv = np.linalg.inv(j)

    fish_new = np.dot(jinv.T, np.dot(fish, jinv))

    diag_fish = np.array([fish_new[i, i] for i in range(fish_new.shape[0])])

    return p_new, diag_fish


def check_results(dirname, compl, tmax=10):
    """Check that all functions can be recovered by applying the subsitutions to the unique functions.
    If not, define a new unique function and save results to file.

    Args:
        :dirname (str): name of directory containing all the functions to consider
        :compl (int): complexity of functions to consider
        :tmax (float, default=10.): maximum time in seconds to run the substitutions

    Returns:
        None

    """

    if rank == 0:
        print('\tLoading all equations', flush=True)
        with open(dirname + '/all_equations_%i.txt' % compl, 'r') as f:
            all_fun = f.read().splitlines()
        max_param = get_max_param(all_fun)
    else:
        all_fun = None
        max_param = None
    max_param = comm.bcast(max_param, root=0)

    if rank == 0:
        print('\tLoading inverse subs')
        with open(dirname + '/inv_subs_%i.txt' % compl, 'r') as f:
            reader = csv.reader(f, delimiter=';')
            inv_subs = [row for row in reader]

        # Only need functions with non-trivial inverse subs
        shufidx = np.array(
            [i for i in range(len(inv_subs)) if len(inv_subs[i]) != 0])
        np.random.seed(1234)
        #  Shuffle to make each rank more similar
        np.random.shuffle(shufidx)

        all_fun = [all_fun[ii] for ii in shufidx]
        nfun = len(all_fun)
        inv_subs = [inv_subs[ii] for ii in shufidx]
    else:
        nfun = None
    comm.Barrier()
    nfun = comm.bcast(nfun, root=0)

    imin, imax = utils.split_idx(nfun, rank, size)
    imax += 1
    imin = comm.gather(imin, root=0)
    imax = comm.gather(imax, root=0)

    if rank == 0:
        all_fun = [all_fun[imin[i]:imax[i]] for i in range(size)]
        inv_subs = [inv_subs[imin[i]:imax[i]] for i in range(size)]
    else:
        all_fun = None
        inv_subs = None
    all_fun = comm.scatter(all_fun, root=0)
    inv_subs = comm.scatter(inv_subs, root=0)

    all_nparam = count_params(all_fun, max_param)

    if rank == 0:
        print('\tLoading unique equations', flush=True)
        with open(dirname + '/unique_equations_%i.txt' % compl, 'r') as f:
            uniq_fun = f.read().splitlines()
    else:
        uniq_fun = None
    uniq_fun = comm.bcast(uniq_fun, root=0)
    uniq_nparam = count_params(uniq_fun, max_param)

    if rank == 0:
        print('\tLoading matches')
        matches = np.loadtxt(dirname + '/matches_%i.txt' % compl).astype(int)
        matches = matches[shufidx]
        matches = np.array_split(matches, size)
    else:
        matches = None
    matches = comm.scatter(matches, root=0)

    param_list = ['a%i' % i for i in range(max_param)]
    all_a = sympy.symbols(" ".join(param_list), real=True)
    if max_param == 1:
        all_a = [all_a]
    locs = sympy_locs
    if max_param > 0:
        for i in range(len(all_a)):
            locs["a%i" % i] = all_a[i]

    to_change = []
    imin, imax = utils.split_idx(nfun, rank, size)

    for i in range(len(all_fun)):
        if rank == 0 and (i % 100) == 0:
            print(i, len(all_fun))
        if all_nparam[i] != uniq_nparam[matches[i]]:
            continue
        s1 = sympy.sympify(all_fun[i], locals=locs)
        try:
            s2 = sympy.sympify(uniq_fun[matches[i]], locals=locs)
        except Exception:
            print(
                f'Could not check {uniq_fun[matches[i]]} so will keep equation')
            s2 = None

        p = sympy.Array(sympy.symbols(" ".join(param_list), real=True))

        try:
            if s2 is None:
                raise ValueError
            with time_limit(tmax):
                for j in range(len(inv_subs[i])):
                    inv_subs[i][j] = inv_subs[i][j].replace("{", "{'")
                    inv_subs[i][j] = inv_subs[i][j].replace("}", "'}")
                    inv_subs[i][j] = inv_subs[i][j].replace(", ", "', '")
                    inv_subs[i][j] = inv_subs[i][j].replace(": ", "': '")
                    d = ast.literal_eval(inv_subs[i][j])
                    k = list(d.keys())
                    v = list(d.values())
                    k = [sympy.sympify(kk, locals=locs) for kk in k]
                    v = [sympy.sympify(vv, locals=locs) for vv in v]
                    p = p.subs(dict(zip(k, v)), simultaneous=True)
                sub = {all_a[j]: p[j] for j in range(len(all_a))}
                s1 = s1.subs(sub, simultaneous=True)
                if (not str(s1) == str(s2)) and (not s1.equals(s2)):
                    raise ValueError
        except Exception:
            to_change.append([i+imin, all_fun[i]])

    del inv_subs, all_fun
    gc.collect()

    to_change = comm.gather(to_change, root=0)

    if rank == 0:

        to_change = list(itertools.chain(*to_change))

        # Change indices to how they were before
        for r in to_change:
            r[0] = shufidx[r[0]]
        del shufidx

        print('\nNeed to change %i functions' % len(to_change))
        for r in to_change:
            print(r)

        print('\nLoading all equations', flush=True)
        with open(dirname + '/all_equations_%i.txt' % compl, 'r') as f:
            all_fun = f.read().splitlines()
        for r in to_change:
            r[1] = all_fun[r[0]]
        del all_fun
        gc.collect()

        print('\nAppending new unique equations')
        with open(dirname + '/unique_equations_%i.txt' % compl, 'r') as f:
            uniq_fun = f.read().splitlines()
        nuniq = len(uniq_fun)

        new_fun = [r[1] for r in to_change]
        new_uniq, new_match = utils.get_unique_indexes(new_fun)
        new_uniq_fun = list(new_uniq.keys())

        with open(dirname + '/unique_equations_%i.txt' % compl, 'w') as f:
            w = 80
            pp = pprint.PrettyPrinter(width=w, stream=f)

            for s in uniq_fun:
                if len(s + '\n') > w / 2:
                    w = 2 * len(s)
                    pp = pprint.PrettyPrinter(width=w, stream=f)
                pp.pprint(s)

            for s in new_uniq_fun:
                if len(s + '\n') > w / 2:
                    w = 2 * len(s)
                    pp = pprint.PrettyPrinter(width=w, stream=f)
                pp.pprint(s)
        del uniq_fun
        gc.collect()
        s = "sed 's/.$//; s/^.//' %s/%s%i.txt > %s/temp_%i.txt" % (
            dirname, 'unique_equations_', compl, dirname, compl)
        os.system(s)
        s = "mv %s/temp_%i.txt %s/%s%i.txt" % (dirname,
                                               compl, dirname, 'unique_equations_', compl)
        os.system(s)

        print('\nChanging inverse subs')
        with open(dirname + '/inv_subs_%i.txt' % compl, 'r') as f:
            reader = csv.reader(f, delimiter=';')
            inv_subs = [row for row in reader]
        for r in to_change:
            inv_subs[r[0]] = ""
        with open(dirname + '/inv_subs_%i.txt' % compl, 'w') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerows(inv_subs)
        del inv_subs
        gc.collect()

        print('\nChanging matches')
        matches = np.loadtxt(dirname + '/matches_%i.txt' % compl).astype(int)
        for i in range(len(to_change)):
            matches[to_change[i][0]] = nuniq + new_match[to_change[i][1]]
        np.savetxt(dirname + '/matches_%i.txt' % compl, matches)

    return
