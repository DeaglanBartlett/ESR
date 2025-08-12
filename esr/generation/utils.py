import numpy as np
import sys
from pympler import asizeof
import psutil
from psutil._common import bytes2human
from collections import OrderedDict
from mpi4py import MPI
import random
import pickle

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

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
        :result (OrderedDict): dictionary which returns index of unique item in L, accessed by unique item
        :match (dict): dictionary which returns index of unique item in result, accessed by unique item
    
    """
    result = OrderedDict()
    for i in range(len(L)):
        val = L[i]
        if val not in result:
            result[val] = i
    match = {v:i for i, v in enumerate(result.keys())}
    return result, match


def get_unique_dict(d):
    """
    Find the unique items in a dictionary and return a new dictionary with unique values as keys,
    and the original keys as values.
    
    Args:
        :d (dict): dictionary to process
        
    Returns:
        :result (dict): dictionary which returns key of unique item in d, accessed by unique item

    """
    result = {}
    for k, v in d.items():
        if v not in result:
            result[v] = k
    return result


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


def shuffle_dict_mpi(d, global_seed):
    """
    Shuffle a dictionary across MPI ranks.
    """

    # Step 1: Turn into list of (key, value) pairs
    items = list(d.items())

    # Step 2: Decide destination rank for each item (same RNG seed for all ranks)
    random.seed(global_seed + rank)  # can use global seed + rank to avoid collisions
    dest_ranks = [random.randrange(size) for _ in items]

    # Step 3: Create a list of items to send to each destination rank
    send_data = [[] for _ in range(size)]
    for dest, kv in zip(dest_ranks, items):
        send_data[dest].append(kv)

    # Step 3: Communicate items to their destination ranks
    dnew = {}
    for r in range(size):
        if rank == r:
            # This rank is receiving from everyone
            for src in range(size):
                if src == rank:
                    data = send_data[rank]
                else:
                    data = comm.recv(source=src)

                # Warn if there are duplicate keys
                for key, value in data:
                    if key in dnew:
                        print(f"Warning: Duplicate key '{key}' received by rank {rank} from rank {src}")
                    dnew[key] = value
        else:
            # This rank sends its r-th piece to rank r
            comm.send(send_data[r], dest=r)
        comm.Barrier()

    return dnew


def get_dict_index_mpi(d):
    """
    Given a dictionary split across MPI ranks, we assign an index to each key
    and return a dictionary mapping keys to their indices.
    Rank 0 will have indices from 0 to len(d)-1, and other ranks will have
    indices starting from the offset of previous ranks.

    Args:
        :d (dict): dictionary to index

    Returns:
        :x (dict): dictionary of indices of functions, with keys as function names
    """
    
    # Step 1: Get the offset for each rank
    local_offset = comm.gather(len(d), root=0)
    if rank == 0:
        # Step 2: Calculate the cumulative offset for each rank
        local_offset = np.array(local_offset, dtype=int)
        local_offset = np.concatenate(([0], np.cumsum(local_offset[:-1])))
    else:
        local_offset = None
    local_offset = int(comm.scatter(local_offset, root=0))

    # Step 2: Create the index dictionary
    x = {k: v + local_offset for v, k in enumerate(d.keys())}

    return x


def get_match_index_mpi(uniq, all_fun):
    """
    Given a dictionary of unique functions and a list of all functions,
    return a dictionary mapping each function in all_fun to its index in uniq.

    Args:
        :uniq (dict): dictionary of unique functions
        :all_fun (dict): dictionary of all functions

    Returns:
        :match (dict): dictionary mapping each function in all_fun to its index in uniq
    """

    x = {}
    
    for r in range(size):
        new_uniq = comm.bcast(uniq, root=r)
        for i, f in all_fun.items():
            if f in new_uniq:
                x[i] = new_uniq[f]
    
    if not len(x) == len(all_fun):
        missing_keys = set(all_fun.keys()) - set(x.keys())
        missing = [all_fun[k] for k in missing_keys]
        print('Missing:', set(missing))
        print(f"Warning: Length of match dictionary {len(x)} does not match length of all_fun {len(all_fun)} on rank {rank}.")
        comm.Abort(1)
    comm.Barrier()

    return x
