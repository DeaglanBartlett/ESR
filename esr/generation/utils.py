import numpy as np
import sys
from pympler import asizeof
import psutil
from psutil._common import bytes2human
from collections import OrderedDict

from esr.generation.custom_printer import ESRPrinter

def split_idx(Ntotal, r, indices_or_sections):
    """ Returns the rth set indices for numpy.array_split(a,indices_or_sections)
    where len(a) = Ntotal
    
    Args:
        :Ntotal (int): length of array to split
        :r (int): rank whose indices are required
        :indices_or_sections (int): how many parts to split array into
        
    Returns:
        :i (list): [min, max] index used by rank
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
        
    return i

def pprint_ntuple(nt):
    """Printing function for memory diagnostics
    
    Args:
        :nt (tuple): tuple of memory statistics returned by psutil.virtual_memory()
        
    Returns:
        None
    """
    for name in nt._fields:
        value = getattr(nt, name)
        if name != 'percent':
            value = bytes2human(value)
        print('\t%-10s : %7s' % (name.capitalize(), value))
    sys.stdout.flush()


def using_mem(point=""):
    """Find and print current virtual memory usage
    
    Args:
        :point (str): string to print to identify where memory diagnostics calculated
        
    Returns:
        None
    
    """
    print('\n%s:'%point)
    pprint_ntuple(psutil.virtual_memory())
    return


def locals_size(loc):
    """Find and print the total memory used by locals()
    
    Args:
        :loc (dict): dictionary of locals (obtained calling locals() in another script)
        
    Returns:
        None
    
    """

    keys = list(loc.keys())
    mem = np.empty(len(keys))

    for i, x in enumerate(keys):
        try:
            mem[i] = asizeof.asizeof(loc[x])
        except Exception:
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


def get_unique_indexes(L):
    """Find the indices of the unique items in a list
    
    Args:
        :L (list): list from which we want to find unique indices
        
    Returns:
        :result (OrderedDict): dictionary which returns index of unique item in l, accessed by unique item
        :match (dict): dictionary which returns index of unique item in result, accessed by unique item
    
    """
    result = OrderedDict()
    for i in range(len(L)):
        val = L[i]
        if val not in result:
            result[val] = i
    match = {v:i for i, v in enumerate(result.keys())}
    return result, match


def get_match_indexes(a, b):
    """Returns indices in a of items in b
    
    Args:
        :a (list): list of values whose index in b we wish to determine
        :b (list): list of values whose indices we wish to find
        
    Returns:
        :result (list): indices where corresponding value of a appears in b
    """
    bb = set(b)
    result = OrderedDict()
    for i in range(len(a)):
        val = a[i]
        if (val in bb) and (val not in result):
            result[val] = i
    result = [result[f] for f in b]
    return result

