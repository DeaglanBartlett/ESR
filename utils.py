import numpy as np
import sys
from pympler import asizeof
import psutil
from psutil._common import bytes2human
from collections import OrderedDict
from custom_printer import ESRPrinter

def split_idx(Ntotal, r, indices_or_sections):
    """
    Returns the rth set indices for numpy.array_split(a,indices_or_sections)
    where len(a) = Ntotal
    """
    try:
        # handle array case.
        Nsections = len(indices_or_sections) + 1
        div_points = [0] + list(indices_or_sections) + [Ntotal]
    except TypeError:
        # indices_or_sections is a scalar, not an array.
        Nsections = int(indices_or_sections)
        if Nsections <= 0:
            raise ValueError('number sections must be larger than 0.') from None
        Neach_section, extras = divmod(Ntotal, Nsections)
        section_sizes = ([0] +
                         extras * [Neach_section+1] +
                         (Nsections-extras) * [Neach_section])
        div_points = np.array(section_sizes, dtype=np.intp).cumsum()
        
    imin = div_points[r]
    imax = div_points[r + 1]
    if imin >= imax:
        i = []
    else:
        i = [imin, imax-1]
        
    #return np.arange(imin, imax)
    return i

def pprint_ntuple(nt):
    for name in nt._fields:
        value = getattr(nt, name)
        if name != 'percent':
            value = bytes2human(value)
        print('\t%-10s : %7s' % (name.capitalize(), value))
    sys.stdout.flush()


def using_mem(point=""):
    print('\n%s:'%point)
    pprint_ntuple(psutil.virtual_memory())
    return


def locals_size(loc):

    keys = list(loc.keys())
    mem = np.empty(len(keys))

    for i, x in enumerate(keys):
        try:
            mem[i] = asizeof.asizeof(loc[x])
        except:
            mem[i] = 0

    j = np.argsort(-mem)

    value = bytes2human(mem.sum())
    print('\n\t%-15s : %7s' % ('LOCALS', value))
    for i in j:
        if mem[i] > 0:
            value = bytes2human(mem[i])
            print('\t%-15s : %7s' % (keys[i], value))
        sys.stdout.flush()

    return


def get_unique_indexes(l):
    result = OrderedDict()
    for i in range(len(l)):
        val = l[i]
        if not val in result:
            result[val] = i
    match = {v:i for i, v in enumerate(result.keys())}
    return result, match


def get_match_indexes(a, b):
    """
    Returns indices in a of items in b
    """
    bb = set(b)
    result = OrderedDict()
    for i in range(len(a)):
        val = a[i]
        if (val in bb) and (not val in result):
            result[val] = i
    result = [result[f] for f in b]
    return result


def merge_keys(all_fun, all_sym):
    """
    Convert all_fun so that different values which give same
    item in all_sym now have the same value
    """
    p = ESRPrinter()
    for i in range(len(all_fun)):
        s = p.doprint(all_sym[all_fun[i]])
        if s != all_fun[i]:
            if s not in all_sym:
                all_sym[s] = all_sym[all_fun[i]]
            all_sym.pop(all_fun[i])
            all_fun[i] = s
    return


