import subprocess
import numpy as np
import esr.generation.simplifier as simplifier

all_compl = np.arange(4, 11)
dirname = 'core_maths/'
nfortab = 4

def file_len(fname):

    p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE,
                                              stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])


def count_fun(fname):

    ntotal = file_len(fname)
    with open(fname, "r") as f:
        all_fun = f.read().splitlines()
    assert ntotal == len(all_fun), 'Maybe missing some equations'
    max_param = simplifier.get_max_param(all_fun, verbose=False)
    nparam = simplifier.count_params(all_fun, max_param)
    count = np.empty(max_param + 1, dtype=int)
    for i in range(len(count)):
        count[i] = (nparam == i).sum()
    assert ntotal == count.sum(), 'Count does not match total'

    return count


s = ['l'] + ['c'] * (nfortab+1)
s = '\t\\begin{tabular}{' + '|'.join(s) + '}'
print(s)
print('\t\t\\hline')
s = ['%i parameters'%i if i != 1 else '%i parameter'%i for i in range(nfortab+1)]
s = '\t\t& ' + ' & '.join(s) + '\\\\'
print(s)
print('\t\t\\hline')

for compl in all_compl:

    try:
    
        fname = dirname + '/compl_%i/all_equations_%i.txt'%(compl,compl)
        total_count = count_fun(fname)

        fname = dirname + '/compl_%i/unique_equations_%i.txt'%(compl,compl)
        uniq_count = count_fun(fname)

        if (len(uniq_count) < nfortab+1) or (len(total_count) < nfortab+1):
            t = np.zeros(nfortab+1, dtype=int)
            t[:len(uniq_count)] = uniq_count
            uniq_count = t

            t = np.zeros(nfortab+1, dtype=int)
            t[:len(total_count)] = total_count
            total_count = t

        total_count = [str(x) for x in total_count[:nfortab+1]]
        uniq_count = [str(x) for x in uniq_count[:nfortab+1]]
    except:
        total_count = ['XXX' for _ in range(nfortab+1)]
        uniq_count = ['YYY' for _ in range(nfortab+1)]

    print('\t\t\\rule{0pt}{3ex}')
    s = ['%s (%s)'%(total_count[i],uniq_count[i]) for i in range(len(total_count))]
    s = ' & '.join(s)
    s = '\t\t%i & '%compl + s + '\\\\'
    print(s)
print('\t\t\\hline')
print('\t\\end{tabular}')
    

