import numpy as np
import os
import shutil
import subprocess
import sys
import textwrap
import matplotlib.pyplot as plt
import unittest
import pytest

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
        esr.fitting.match.check_match_results(comp, likelihood)
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
    esr.fitting.match.check_match_results(comp, likelihood)
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
    # a0*pow(Abs(a1),x) and a0/pow(Abs(a1),x) are the same function (sign flip on a1)
    # which variant ranks first depends on numerical precision across environments
    assert best[1] in ('a0*pow(Abs(a1),x)', 'a0/pow(Abs(a1),x)')  # best function (det(I) codelen)
    assert np.isclose(float(best[2]), 716.34, atol=2e-2)  #  logL
    assert np.isclose(float(best[4]), 701.79, atol=2e-2)  #  Residuals
    assert np.isclose(float(best[5]), 7.62, atol=2e-2)  #  Parameter
    assert np.isclose(float(best[6]), 6.93, atol=2e-2)   # Function


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
    esr.fitting.match.check_match_results(comp, likelihood)
    esr.fitting.combine_DL.main(comp, likelihood)
    esr.fitting.plot.main(comp, likelihood)

    return


def test_gaussian_dynamic_mpi(tmp_path):

    if shutil.which('mpiexec') is None:
        pytest.skip('mpiexec not available')
    if esr.fitting.test_all.size > 1:
        pytest.skip('do not launch nested MPI jobs')

    script = tmp_path / 'dynamic_smoke.py'
    script.write_text(textwrap.dedent("""
        import os
        import sys

        import numpy as np
        from mpi4py import MPI

        sys.path.insert(0, os.environ['ESR_REPO'])

        from esr.fitting.likelihood import GaussLikelihood
        import esr.fitting.test_all as test_all

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        data_dir = os.environ['ESR_MPI_TEST_DIR']

        if rank == 0:
            x = np.linspace(0.2, 1.0, 6)
            y = 2.0 * x
            yerr = np.ones_like(x)
            np.savetxt(
                os.path.join(data_dir, 'data.txt'),
                np.column_stack([x, y, yerr]))

        comm.Barrier()
        likelihood = GaussLikelihood(
            'data.txt', 'dynamic_smoke', data_dir=data_dir)
        test_all.main(
            3,
            likelihood,
            tmax=1,
            Niter_params=[1],
            Nconv_params=[1],
            dynamic=True,
            print_frequency=2)

        if rank == 0:
            output = os.path.join(
                likelihood.out_dir, 'negloglike_comp3.dat')
            checkpoint = output.replace('.dat', '.checkpoint.dat')
            arr = np.loadtxt(output)
            with open(os.path.join(
                    likelihood.fn_dir,
                    'compl_3',
                    'unique_equations_3.txt'), 'r') as f:
                n_functions = sum(1 for _ in f)
            assert arr.shape[0] == n_functions
            assert not os.path.exists(checkpoint)
            print('MPI_DYNAMIC_SMOKE_OK', arr.shape, flush=True)
    """))

    env = os.environ.copy()
    env['ESR_REPO'] = os.getcwd()
    env['ESR_MPI_TEST_DIR'] = str(tmp_path)
    env.setdefault('OMPI_ALLOW_RUN_AS_ROOT', '1')
    env.setdefault('OMPI_ALLOW_RUN_AS_ROOT_CONFIRM', '1')
    result = subprocess.run(
        ['mpiexec', '--oversubscribe', '-n', '3', sys.executable, str(script)],
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
        timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'Dynamic scheduling:' in result.stdout
    assert 'MPI_DYNAMIC_SMOKE_OK' in result.stdout

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
    esr.fitting.match.check_match_results(comp, likelihood)
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
              "3", "*", "a2", "pow", "x", "2.1", "*", "a3", "x", "pow", "x", "0.5",
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


def test_snap_choices():
    """Test the supported snap modes and reject the unimplemented mode."""

    likelihood = MockLikelihood(320, 0.2)
    labels = ["+", "a0", "*", "a1", "pow", "x", "3"]
    basis_functions = [["x", "a"],
                       ["inv"],
                       ["+", "*", "-", "/", "pow"]]

    results = {}
    for sc in [0, 1]:
        for det_I in [True, False]:
            nll, DL = single_function(labels, basis_functions, likelihood,
                                      verbose=False, use_det_I=det_I, snap_choice=sc)
            assert np.isfinite(nll), f"snap_choice={sc}, use_det_I={det_I}: nll not finite"
            assert np.isfinite(DL), f"snap_choice={sc}, use_det_I={det_I}: DL not finite"
            results[(sc, det_I)] = (nll, DL)

    nlls = [results[(sc, True)][0] for sc in [0, 1]]
    assert np.allclose(nlls, nlls[0], atol=1e-4), f"negloglike differs across snap_choices: {nlls}"

    with pytest.raises(ValueError, match="snap_choice must be 0 or 1"):
        single_function(labels, basis_functions, likelihood,
                        verbose=False, use_det_I=True, snap_choice=-1)

    return


def test_compute_codelen():
    """Unit tests for _compute_codelen with known analytic cases."""
    from esr.fitting.test_all_Fisher import _compute_codelen
    import math

    # 1-parameter case: both det and diagonal should agree (det of 1x1 = the element)
    H = np.array([[100.0]])
    diag = np.array([100.0])
    theta = np.array([5.0])
    mask = np.array([True])
    expected = -0.5 * math.log(3.) + 0.5 * math.log(100.) + math.log(5.)
    assert np.isclose(_compute_codelen(H, diag, theta, mask, True), expected)
    assert np.isclose(_compute_codelen(H, diag, theta, mask, False), expected)

    # 2-parameter diagonal case: det(H) = prod(diag), so both should agree
    H = np.diag([100.0, 200.0])
    diag = np.array([100.0, 200.0])
    theta = np.array([3.0, 7.0])
    mask = np.array([True, True])
    cl_det = _compute_codelen(H, diag, theta, mask, True)
    cl_diag = _compute_codelen(H, diag, theta, mask, False)
    assert np.isclose(cl_det, cl_diag), f"Diagonal Hessian: det={cl_det}, diag={cl_diag}"

    # 2-parameter correlated case: det(H) < prod(diag), so det codelen < diagonal codelen
    H = np.array([[100.0, 50.0], [50.0, 100.0]])
    diag = np.array([100.0, 100.0])
    theta = np.array([3.0, 7.0])
    cl_det = _compute_codelen(H, diag, theta, mask, True)
    cl_diag = _compute_codelen(H, diag, theta, mask, False)
    assert cl_det < cl_diag, f"Correlated: det={cl_det} should be < diag={cl_diag}"
    # Check exact value: det = 100*100 - 50*50 = 7500
    expected_det = -1.0 * math.log(3.) + 0.5 * math.log(7500.) + math.log(3.) + math.log(7.)
    assert np.isclose(cl_det, expected_det)

    # Empty mask: codelen should be 0
    assert _compute_codelen(H, diag, theta, np.array([False, False]), True) == 0.0

    # Non-positive-definite: codelen should be inf
    H_bad = np.array([[100.0, 200.0], [200.0, 100.0]])  # det = -30000
    assert _compute_codelen(H_bad, diag, theta, mask, True) == np.inf
    # A saddle can have positive determinant if it has two negative directions.
    H_even_saddle = np.array([[1.0, 2.0, 2.0],
                              [2.0, 1.0, 2.0],
                              [2.0, 2.0, 1.0]])
    assert _compute_codelen(
        H_even_saddle, np.diag(H_even_saddle), np.ones(3),
        np.ones(3, dtype=bool), True) == np.inf

    # The determinant scorer floors unresolved parameter contributions; the
    # explicit diagonal comparison retains the published raw formula.
    theta_small = np.array([1.e-8])
    assert _compute_codelen(
        np.array([[100.0]]), np.array([100.0]), theta_small,
        np.array([True]), True) > _compute_codelen(
            np.array([[100.0]]), np.array([100.0]), theta_small,
            np.array([True]), False)

    # Partial mask: only keep first parameter
    mask1 = np.array([True, False])
    cl = _compute_codelen(H, diag, theta, mask1, True)
    expected_1 = -0.5 * math.log(3.) + 0.5 * math.log(100.) + math.log(3.)
    assert np.isclose(cl, expected_1)

    return


def test_compute_snap_mask():
    """Unit tests for _compute_snap_mask with known analytic cases."""
    from esr.fitting.test_all_Fisher import _compute_snap_mask

    # Well-constrained 2-param case: no snapping for either supported mode
    H = np.array([[1000.0, 0.0], [0.0, 1000.0]])
    diag = np.array([1000.0, 1000.0])
    theta = np.array([5.0, 3.0])
    Nsteps_diag = np.abs(theta) / np.sqrt(12. / diag)  # both >> 1
    for sc in [0, 1]:
        result, degen = _compute_snap_mask(H, diag, theta, Nsteps_diag.copy(), sc)
        assert np.all(result >= 1), f"snap_choice={sc}: should not snap well-constrained params"
        if sc == 1:
            assert not degen, f"snap_choice={sc}: well-conditioned Hessian should not be degenerate"

    # snap_choice=0: returns input Nsteps unchanged, not degenerate
    Nsteps_in = np.array([0.5, 2.0])
    result, degen = _compute_snap_mask(H, diag, theta, Nsteps_in.copy(), 0)
    assert np.allclose(result, Nsteps_in)
    assert not degen

    # Poorly constrained eigendirection: one eigenvalue near zero
    H_degen = np.array([[100.0, 99.0], [99.0, 100.0]])  # eigenvalues: 1, 199
    diag_degen = np.array([100.0, 100.0])
    theta_small = np.array([0.001, 0.001])
    Nsteps_diag_small = np.abs(theta_small) / np.sqrt(12. / diag_degen)
    result, degen = _compute_snap_mask(
        H_degen, diag_degen, theta_small, Nsteps_diag_small.copy(), 1)
    assert np.sum(result < 1) >= 1

    # Negative curvature is not treated as a removable degeneracy.
    H_nonposdef = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues: -1, 3
    diag_npd = np.array([1.0, 1.0])
    theta_npd = np.array([5.0, 5.0])
    Nsteps_npd = np.abs(theta_npd) / np.sqrt(12. / diag_npd)
    result, degen = _compute_snap_mask(
        H_nonposdef, diag_npd, theta_npd, Nsteps_npd.copy(), 1)
    assert np.allclose(result, Nsteps_npd)
    assert not degen

    with pytest.raises(ValueError, match="snap_choice must be 0 or 1"):
        _compute_snap_mask(H, diag, theta, Nsteps_diag.copy(), -1)

    return


def test_reduced_parameters_are_canonicalized():
    import sympy
    from esr.fitting.sympy_symbols import x

    a0, a1 = sympy.symbols('a0 a1', real=True)
    eq, active = esr.fitting.test_all.canonicalize_parameter_symbols(
        (a1 + x) / (a1 + 1))
    assert [symbol.name for symbol in active] == ['a1']
    assert {symbol.name for symbol in eq.free_symbols} == {'x', 'a0'}


def test_fisher_settings_are_persisted(tmp_path):
    class Likelihood:
        out_dir = str(tmp_path)

    esr.fitting.test_all_Fisher.save_scoring_settings(7, Likelihood, True, 1)
    assert esr.fitting.test_all_Fisher.load_scoring_settings(7, Likelihood) == {
        'use_det_I': True, 'snap_choice': 1}


def test_likelihood_aware_catalogue_groups_transformed_models(tmp_path):
    import sympy
    from esr.fitting.sympy_symbols import x
    from esr.fitting import test_all

    class NormalisingLikelihood:
        is_mse = False
        base_out_dir = str(tmp_path / 'out_base')
        out_dir = str(tmp_path / 'out')
        temp_dir = str(tmp_path / 'tmp')
        fn_dir = str(tmp_path / 'functions')

        def run_sympify(self, fcn_i, **kwargs):
            a0, a1, a2 = sympy.symbols('a0 a1 a2', real=True)
            eq = sympy.sympify(fcn_i, locals={'x': x, 'a0': a0, 'a1': a1, 'a2': a2})
            return fcn_i, sympy.cancel(eq / eq.subs(x, 1)), True

    comp = 1
    compl_dir = tmp_path / 'functions' / f'compl_{comp}'
    compl_dir.mkdir(parents=True)
    all_functions = ['a0*(a1 + x)', 'a2 + x', 'a0*x']
    (compl_dir / f'all_equations_{comp}.txt').write_text('\n'.join(all_functions) + '\n')
    (compl_dir / f'unique_equations_{comp}.txt').write_text('\n'.join(all_functions) + '\n')

    likelihood = NormalisingLikelihood()
    assert test_all.ensure_likelihood_catalogue(comp, likelihood, tmax=5,
                                                try_integration=False)
    paths = test_all.likelihood_catalogue_paths(comp, likelihood)
    with open(paths['matches']) as f:
        matches = [int(line) for line in f]
    assert matches[0] == matches[1]
    assert matches[2] != matches[0]


def test_likelihood_can_disable_likelihood_aware_catalogue(tmp_path):
    import json
    import sympy
    from esr.fitting.sympy_symbols import x
    from esr.fitting import test_all

    class DirectLikelihood:
        is_mse = False
        use_likelihood_catalogue = False
        base_out_dir = str(tmp_path / 'out_base')
        out_dir = str(tmp_path / 'out')
        temp_dir = str(tmp_path / 'tmp')
        fn_dir = str(tmp_path / 'functions')

        def run_sympify(self, fcn_i, **kwargs):
            # This transformation would normally trigger the catalogue because
            # a0 is removed. Direct likelihoods can opt out when their fitted
            # NLL files already correspond to the raw unique catalogue.
            a0, a1 = sympy.symbols('a0 a1', real=True)
            eq = sympy.sympify(fcn_i, locals={'x': x, 'a0': a0, 'a1': a1})
            return fcn_i, sympy.cancel(eq / eq.subs(x, 1)), True

    comp = 1
    compl_dir = tmp_path / 'functions' / f'compl_{comp}'
    compl_dir.mkdir(parents=True)
    all_functions = ['a0*(a1 + x)', 'a0*x']
    (compl_dir / f'all_equations_{comp}.txt').write_text(
        '\n'.join(all_functions) + '\n')
    (compl_dir / f'unique_equations_{comp}.txt').write_text(
        '\n'.join(all_functions) + '\n')

    likelihood = DirectLikelihood()
    os.makedirs(likelihood.out_dir)
    paths = test_all.likelihood_catalogue_paths(comp, likelihood)
    for key in ('unique', 'matches'):
        with open(paths[key], 'w') as f:
            f.write('stale\n')

    assert not test_all.ensure_likelihood_catalogue(
        comp, likelihood, tmax=5, try_integration=False)
    assert not os.path.exists(paths['unique'])
    assert not os.path.exists(paths['matches'])
    with open(paths['metadata']) as f:
        metadata = json.load(f)
    assert metadata['active'] is False
    assert metadata['disabled_by_likelihood'] is True

    fcn_list, data_start, data_end = test_all.get_functions(comp, likelihood)
    assert fcn_list == all_functions
    assert (data_start, data_end) == (0, len(all_functions))


def test_likelihood_aware_match_uses_transformed_representatives(tmp_path):
    import sympy
    from esr.fitting.sympy_symbols import x
    from esr.fitting import match, test_all, test_all_Fisher

    class NormalisingLikelihood:
        is_mse = False
        base_out_dir = str(tmp_path / 'out_base')
        out_dir = str(tmp_path / 'out')
        temp_dir = str(tmp_path / 'tmp')
        fn_dir = str(tmp_path / 'functions')

        def run_sympify(self, fcn_i, **kwargs):
            a0, a1, a2 = sympy.symbols('a0 a1 a2', real=True)
            eq = sympy.sympify(fcn_i, locals={'x': x, 'a0': a0, 'a1': a1, 'a2': a2})
            return fcn_i, sympy.cancel(eq / eq.subs(x, 1)), True

    comp = 1
    compl_dir = tmp_path / 'functions' / f'compl_{comp}'
    compl_dir.mkdir(parents=True)
    all_functions = ['a0*(a1 + x)', 'a2 + x', 'a0*x']
    (compl_dir / f'all_equations_{comp}.txt').write_text('\n'.join(all_functions) + '\n')
    (compl_dir / f'unique_equations_{comp}.txt').write_text('\n'.join(all_functions) + '\n')

    likelihood = NormalisingLikelihood()
    assert test_all.ensure_likelihood_catalogue(comp, likelihood, tmax=5,
                                                try_integration=False)
    test_all_Fisher.save_scoring_settings(comp, likelihood, True, 1)
    # match.main needs a negloglike file only to infer the parameter-column width
    # in the likelihood-aware branch; it takes NLLs/parameters from codelen_comp.
    np.savetxt(tmp_path / 'out' / f'negloglike_comp{comp}.dat',
               np.array([[10.0, 2.0, 0.0, 0.0, 0.0],
                         [20.0, 3.0, 0.0, 0.0, 0.0]]))
    np.savetxt(tmp_path / 'out' / f'codelen_comp{comp}_deriv.dat',
               np.array([[1.5, 10.0, 2.0, 0.0, 0.0, 0.0],
                         [2.5, 20.0, 3.0, 0.0, 0.0, 0.0]]))

    match.main(comp, likelihood)

    matched = np.loadtxt(tmp_path / 'out' / f'codelen_matches_comp{comp}.dat')
    assert matched.shape == (3, 7)
    assert matched[0, 2] == matched[1, 2] == 0
    assert matched[2, 2] == 1
    np.testing.assert_allclose(matched[0, :], matched[1, :])
    assert matched[2, 0] == 20.0
    assert matched[2, 1] == 2.5


def test_numerical_fingerprint():
    """Unit tests for the numerical fingerprint diagnostic."""
    import sympy
    from esr.generation.simplifier import (
        fingerprint_to_hash, numerical_duplicate_candidates,
        numerical_fingerprint)

    x = sympy.Symbol('x', positive=True)
    a0, a1 = sympy.symbols('a0 a1', real=True)

    # Commutative equivalents should hash identically
    fp1 = numerical_fingerprint(x * a0)
    fp2 = numerical_fingerprint(a0 * x)
    assert fp1 is not None
    assert fp1 == fp2
    assert fingerprint_to_hash(fp1) == fingerprint_to_hash(fp2)

    # Algebraically equivalent expressions should hash identically
    fp3 = numerical_fingerprint(a0 + a0)
    fp4 = numerical_fingerprint(2 * a0)
    assert fp3 is not None
    assert fingerprint_to_hash(fp3) == fingerprint_to_hash(fp4)

    # Different functions should hash differently
    fp_lin = numerical_fingerprint(a0 * x)
    fp_quad = numerical_fingerprint(a0 * x**2)
    assert fp_lin is not None and fp_quad is not None
    assert fingerprint_to_hash(fp_lin) != fingerprint_to_hash(fp_quad)

    # Constant expression should work
    fp_const = numerical_fingerprint(sympy.Integer(5))
    assert fp_const is not None
    assert all(v == 5.0 for v in fp_const if v is not None)

    # Expression with no free symbols
    fp_num = numerical_fingerprint(sympy.Rational(3, 7))
    assert fp_num is not None

    # None input returns None
    assert numerical_fingerprint(None) is None

    # fingerprint_to_hash of None returns None
    assert fingerprint_to_hash(None) is None

    # Hash is deterministic
    fp = numerical_fingerprint(a0 * x + a1)
    h1 = fingerprint_to_hash(fp)
    h2 = fingerprint_to_hash(fp)
    assert h1 == h2
    assert isinstance(h1, str) and len(h1) == 32  # MD5 hex digest

    # A matching fingerprint is not proof of expression/model equivalence.
    # These pairs agree on the positive diagnostic grid but differ at an
    # unsampled boundary or for negative real parameter values.
    diagnostic_only = [
        '1 - x**2',
        'x*(-x + 1/x)',
        'a1/(x + Abs(a0))',
        'Abs(a1)/(a0 + x)',
    ]
    groups = numerical_duplicate_candidates(diagnostic_only, max_param=2,
                                            verbose=False)
    grouped = [{diagnostic_only[i] for i in indexes} for _, indexes in groups]
    assert {'1 - x**2', 'x*(-x + 1/x)'} in grouped
    assert {'a1/(x + Abs(a0))', 'Abs(a1)/(a0 + x)'} in grouped
    assert diagnostic_only == [
        '1 - x**2',
        'x*(-x + 1/x)',
        'a1/(x + Abs(a0))',
        'Abs(a1)/(a0 + x)',
    ]

    return


def test_numerical_duplicate_diagnostic_does_not_change_catalogue():
    """The optional collision report must not filter or remap equations."""
    comp = 3
    fn_dir = os.path.join(os.path.dirname(generator.__file__), '..',
                          'function_library', 'core_maths',
                          f'compl_{comp}')
    unique_path = os.path.join(fn_dir, f'unique_equations_{comp}.txt')
    match_path = os.path.join(fn_dir, f'matches_{comp}.txt')

    esr.generation.duplicate_checker.main('core_maths', comp)
    with open(unique_path) as f:
        unique_default = f.read()
    with open(match_path) as f:
        matches_default = f.read()

    esr.generation.duplicate_checker.main(
        'core_maths', comp, diagnose_numerical_duplicates=True)
    with open(unique_path) as f:
        assert f.read() == unique_default
    with open(match_path) as f:
        assert f.read() == matches_default
    report_path = os.path.join(
        fn_dir, f'numerical_duplicate_candidates_{comp}.txt')
    with open(report_path) as f:
        report = f.read()
    assert 'Candidate numerical fingerprint collisions only' in report
    assert 'No equations were removed or remapped' in report

    return
