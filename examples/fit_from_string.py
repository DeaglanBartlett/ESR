# Copyright 2023 Deaglan J. Bartlett
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.

"""
Script to generate mock data according to some function and re-fit using the ESR
single_function() function, defining the function as a string.
"""

import numpy as np
import sympy
import esr.generation.generator as generator
from esr.fitting.fit_single import single_function
from esr.fitting.likelihood import GaussLikelihood
import os


def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


# ESR hyperparams
basis_functions = [["x", "a"],  # type0
                   ["inv"],  # type1
                   ["+", "*", "-", "/", "pow"]]  # type2
maxvar = 20

# Make some mock data and define likelihood
np.random.seed(123)
x = np.random.uniform(0.1, 5, 500)
y = 1.1 * x ** 4 + 2 * x ** 3 + 4 * x ** 2 + 3 * x + 5
yerr = np.full(x.shape, 1.0)
y = y + yerr * np.random.normal(size=len(x))
np.savetxt('data.txt', np.array([x, y, yerr]).T)
likelihood = GaussLikelihood('data.txt', 'gauss_example', data_dir=os.getcwd())

# Define all models
all_models = ['1.1 * x ** 4 + 2 * x ** 3 + 4 * x ** 2 + 3 * x + 5']

#  Make string to sympy mapping
x = sympy.symbols('x', real=True)
a = sympy.symbols([f'a{i}' for i in range(maxvar)], real=True)
d1 = {'x': x}
d2 = {f'a{i}': a[i] for i in range(maxvar)}
locs = {**d1, **d2}

# Empty arrays to store results
all_logl = np.empty(len(all_models))
all_dl = np.empty(len(all_models))
all_comp = np.empty(len(all_models))

for i, fun in enumerate(all_models):

    # Change string to list
    expr, nodes, all_comp[i] = generator.string_to_node(
        fun, basis_functions, locs=locs, evalf=True)
    labels = nodes.to_list(basis_functions)

    # Prepare to get parents
    new_labels = [None] * len(labels)
    for j, lab in enumerate(labels):
        if lab == 'Mul':
            new_labels[j] = '*'
            labels[j] = '*'
        elif lab == 'Add':
            new_labels[j] = '+'
            labels[j] = '+'
        else:
            new_labels[j] = lab.lower()
            labels[j] = lab.lower()
    param_idx = [j for j, lab in enumerate(new_labels) if is_float(lab)]
    assert len(param_idx) <= maxvar
    for k, j in enumerate(param_idx):
        new_labels[j] = f'a{k}'

    # Get parent operators
    s = generator.labels_to_shape(new_labels, basis_functions)
    success, _, tree = generator.check_tree(s)
    parents = [None] + [labels[p.parent] for p in tree[1:]]

    # Replace floats with symbols (except exponents)
    param_idx = [j for j, lab in enumerate(labels) if is_float(
        lab) and not (parents[j].lower() == 'pow')]
    for k, j in enumerate(param_idx):
        labels[j] = f'a{k}'
    fstr = generator.node_to_string(0, tree, labels)

    all_logl[i], all_dl[i] = single_function(labels,
                                             basis_functions,
                                             likelihood,
                                             verbose=False)

    print(expr)
    print('\t-logL', all_logl[-1])
    print('\tL(D)', all_dl[-1])
