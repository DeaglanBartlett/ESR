import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import unittest

import esr.generation.duplicate_checker
import esr.generation.generator as generator
import esr.fitting.test_all
import esr.fitting.test_all_Fisher
import esr.fitting.match
import esr.fitting.combine_DL
import esr.fitting.plot
from esr.fitting.likelihood import (
    CCLikelihood, PanthLikelihood, GaussLikelihood,
    PoissonLikelihood, MockLikelihood, MSE)
from esr.fitting.fit_single import single_function, fit_from_string, tree_to_aifeyn, string_to_aifeyn
import esr.plotting.plot


def test_cc(monkeypatch):

    if monkeypatch is not None:
        monkeypatch.setattr(plt, 'show', lambda: None)

    comp = 3
    likelihood = CCLikelihood()
    esr.generation.duplicate_checker.main('core_maths', comp)

    for log_opt in [True, False]:
        esr.fitting.test_all.main(comp, likelihood, log_opt=log_opt)
        esr.fitting.test_all_Fisher.main(comp, likelihood)
        esr.fitting.match.main(comp, likelihood)
        esr.fitting.combine_DL.main(comp, likelihood)
        esr.fitting.plot.main(comp, likelihood)

        # Test results match Table 1 of arXiv:2211.11461
        assert os.path.exists(likelihood.out_dir)
        fname = os.path.join(likelihood.out_dir, f'final_{comp}.dat')
        with open(fname, 'r') as f:
            best = f.readline().split(';')
        assert int(best[0]) == 0  #  Rank
        assert best[1] == 'a0*x'  # best function
        assert np.isclose(
            float(best[2]), 29.959725666004328, atol=2e-2)  #  logL
        assert np.isclose(float(best[4]), 24.138886, atol=2e-2)  #  Residuals
        assert np.isclose(float(best[5]), 2.5250028, atol=2e-2)  #  Parameter
        assert np.isclose(
            float(best[6]), 3.295836866004329, atol=2e-2)   # Function
        assert np.isclose(float(best[7]), 5638.4157, atol=10)  #  Best-fit a0
        assert np.all(np.array(best[8:], dtype=float) == 0)  # Other parameters

    #  Test single_function using the mock likelihood
    likelihood = MockLikelihood(320, 0.2)
    labels = ["+", "a0", "*", "a1", "pow", "x", "3"]
    basis_functions = [["x", "a"],  # type0
                       ["inv"],  # type1
                       ["+", "*", "-", "/", "pow"]]  # type2

    # test using labels
    nll_0, DL_0 = single_function(labels,
                                  basis_functions,
                                  likelihood,
                                  verbose=True)
    aifeyn_0, comp_0 = tree_to_aifeyn(labels, basis_functions)

    # test using string
    fun = "a0 + a1 * x**3"
    nll_1, DL_1, labels_1 = fit_from_string(fun,
                                            basis_functions,
                                            likelihood,)
    aifeyn_1, comp_1 = string_to_aifeyn(fun, basis_functions)

    assert np.isclose(nll_0, nll_1, atol=2e-2)
    assert np.all(np.isclose(DL_0, DL_1, atol=2e-2))
    assert labels_1 == labels
    assert comp_0 == comp_1
    assert aifeyn_0 == aifeyn_1

    return


def test_pantheon(monkeypatch):

    if monkeypatch is not None:
        monkeypatch.setattr(plt, 'show', lambda: None)

    comp = 5
    likelihood = PanthLikelihood()
    esr.generation.duplicate_checker.main('core_maths', comp)
    esr.fitting.test_all.main(
        comp, likelihood, Niter_params=[4], Nconv_params=[2])
    esr.fitting.test_all_Fisher.main(comp, likelihood)
    esr.fitting.match.main(comp, likelihood)
    esr.fitting.combine_DL.main(comp, likelihood)
    esr.fitting.plot.main(comp, likelihood)

    assert esr.plotting.plot.pareto_plot(
        likelihood.out_dir, "no_name", do_DL=False, do_logL=False) is None
    esr.plotting.plot.pareto_plot(
        likelihood.out_dir, "no_name", do_DL=True, do_logL=False)
    esr.plotting.plot.pareto_plot(
        likelihood.out_dir, "no_name", do_DL=False, do_logL=True)

    # Test results match Table 2 of arXiv:2211.11461
    assert os.path.exists(likelihood.out_dir)
    fname = os.path.join(likelihood.out_dir, f'final_{comp}.dat')
    with open(fname, 'r') as f:
        best = f.readline().split(';')
    assert int(best[0]) == 0  #  Rank
    assert best[1] == 'a0*pow(x,x)'  # best function
    assert np.isclose(float(best[2]), 718.22, atol=2e-2)  #  logL
    assert np.isclose(float(best[4]), 706.18, atol=2e-2)  #  Residuals
    assert np.isclose(float(best[5]), 5.11, atol=2e-2)  #  Parameter
    assert np.isclose(float(best[6]), 6.93, atol=2e-2)   # Function
    assert np.isclose(float(best[7]), 5345.02, atol=10)  #  Best-fit a0
    assert np.all(np.array(best[8:], dtype=float) == 0)  # Other parameters

    return


def test_gaussian(monkeypatch):

    if monkeypatch is not None:
        monkeypatch.setattr(plt, 'show', lambda: None)

    # Run the Gaussian example
    np.random.seed(123)
    x = np.random.uniform(0.1, 5, 100)
    y = 0.5 * x ** 2
    yerr = np.full(x.shape, 1.0)
    y = y + yerr * np.random.normal(size=len(x))
    np.savetxt('data.txt', np.array([x, y, yerr]).T)
    sys.stdout.flush()
    likelihood = GaussLikelihood(
        'data.txt', 'gauss_example', data_dir=os.getcwd())
    comp = 3
    esr.generation.duplicate_checker.main('core_maths', comp)
    esr.fitting.test_all.main(comp, likelihood)
    esr.fitting.test_all_Fisher.main(comp, likelihood)
    esr.fitting.match.main(comp, likelihood)
    esr.fitting.combine_DL.main(comp, likelihood)
    esr.fitting.plot.main(comp, likelihood)

    return


def test_poisson(monkeypatch):

    if monkeypatch is not None:
        monkeypatch.setattr(plt, 'show', lambda: None)

    # Run the Poisson examples
    np.random.seed(123)
    x = np.random.uniform(0.1, 5, 100)
    y = 0.5 * x ** 2
    y = np.random.poisson(y)
    np.savetxt('data.txt', np.array([x, y]).T)
    sys.stdout.flush()

    likelihood = PoissonLikelihood(
        'data.txt', 'poisson_example', data_dir=os.getcwd())
    comp = 3
    esr.generation.duplicate_checker.main('core_maths', comp)
    esr.fitting.test_all.main(comp, likelihood)
    esr.fitting.test_all_Fisher.main(comp, likelihood)
    esr.fitting.match.main(comp, likelihood)
    esr.fitting.combine_DL.main(comp, likelihood)
    esr.fitting.plot.main(comp, likelihood)

    # Plot the pareto front for the Poisson example
    esr.plotting.plot.pareto_plot(
        likelihood.out_dir, 'pareto.png', do_DL=True, do_logL=True)

    return


def test_mse():

    # Run the MSE example
    np.random.seed(123)
    x = np.random.uniform(0.1, 5, 100)
    y = 0.5 * x ** 2
    yerr = np.zeros(x.shape)
    np.savetxt('data.txt', np.array([x, y, yerr]).T)
    sys.stdout.flush()

    likelihood = MSE('data.txt', 'mse_example', data_dir=os.getcwd())
    comp = 3
    esr.generation.duplicate_checker.main('core_maths', comp)
    esr.fitting.test_all.main(comp, likelihood)
    unittest.TestCase().assertRaises(
        ValueError,
        esr.fitting.test_all_Fisher.main,
        comp=comp,
        likelihood=likelihood
    )

    return


def test_function_making():

    for basis_set in ['core_maths', 'ext_maths', 'keep_duplicates', 'osc_maths']:
        for comp in range(1, 5):
            esr.generation.duplicate_checker.main(basis_set, comp)

    for comp in [5, 6]:
        esr.generation.duplicate_checker.main(
            'core_maths', comp, track_memory=True)

    return


def test_node():

    labels = ["-", "+", "/", "+", "+", "a0", "*", "a1", "pow", "x",
              "3", "*", "a2", "pow", "x", "2", "*", "a3", "x", "pow", "x", "0.5",
              "*", "a4", "pow", "x", "-1"]
    basis_functions = [["x", "a"],  # type0
                       ["inv"],  # type1
                       ["+", "*", "-", "/", "pow"]]  # type2
    sympy_numerics = ['Number', 'Float', 'Rational', 'Integer', 'AlgebraicNumber',
                      'NumberSymbol', 'RealNumber', 'igcd', 'ilcm', 'seterr', 'Zero',
                      'One', 'NegativeOne', 'Half', 'NaN', 'Infinity', 'NegativeInfinity',
                      'ComplexInfinity', 'Exp1', 'ImaginaryUnit', 'Pi', 'EulerGamma',
                      'Catalan', 'GoldenRatio', 'TribonacciConstant', 'mod_inverse']

    labels_changed = labels.copy()
    for i, lab in enumerate(labels):
        if lab.lower() in sympy_numerics or generator.is_float(lab):
            labels_changed[i] = 'a'

    # Get parent operators
    s = generator.labels_to_shape(labels_changed, basis_functions)
    success, _, tree = generator.check_tree(s)
    assert success

    for i, lab in enumerate(labels_changed):
        tree[i].assign_op(lab)

    nodes = generator.DecoratedNode(None, basis_functions)
    nodes.from_node_list(0, tree, basis_functions)
    assert nodes.to_list(basis_functions) == labels_changed

    # Test DecoratedNode __init__
    basis_functions = [["x", "a"],  # type0
                       ["inv", "square", "sqrt", "cube"],  # type1
                       ["+", "*", "-", "/", "pow"]]  # type2
    s = generator.labels_to_shape(labels, basis_functions)
    fcn_i = generator.node_to_string(0, tree, labels)
    likelihood = CCLikelihood()
    fcn_i, eq, _ = likelihood.run_sympify(fcn_i)
    nodes = generator.DecoratedNode(eq, basis_functions)

    # Test unity
    assert not nodes.is_unity()
    _, unit_nodes, _ = generator.string_to_node(
        "1", basis_functions, evalf=True)
    assert unit_nodes.is_unity()

    # Test counting
    assert nodes.count_nodes(basis_functions) == len(labels)
    mylist = nodes.to_list(basis_functions)
    assert len(mylist) == len(labels)

    # Check other functions work
    nodes.get_lineage()
    nodes.get_sibling_lineage()
    nodes.get_siblings()

    assert generator.check_operators(nodes, basis_functions)

    # Check Node functions
    success, _, tree = generator.check_tree(s)
    assert success
    assert all([t.is_used() for t in tree])
    for i, lab in enumerate(labels):
        tree[i].assign_op(lab)
        tree[i] = tree[i].copy()
    s += [2, 1, 0]
    success, _, tree = generator.check_tree(s)
    assert not success
    check_used = [t.is_used() for t in tree]
    assert all(check_used[:-3]) and not any(check_used[-3:])

    return
