import numpy as np
import os
import shutil
import subprocess
import sys
import textwrap
from types import SimpleNamespace
import matplotlib.pyplot as plt
import unittest
import warnings
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
from esr.fitting.utils import (
    fitting_paths, likelihood_catalogue_paths, raw_catalogue_paths)
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
    if (os.cpu_count() or 1) < 3 and not os.environ.get('ESR_RUN_MPI_TESTS'):
        pytest.skip('launches 3 ranks; needs >=3 cores or ESR_RUN_MPI_TESTS=1')

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


def test_likelihood_catalogue_parallel_matches_serial(tmp_path):
    """The MPI-parallel catalogue build must reproduce the serial result.

    Uses a parameter-removing likelihood on an input whose transformed-model
    collisions span the rank slices, then checks that an ``mpiexec -n 3`` build
    writes byte-identical ``unique``/``matches`` files to an in-process serial
    build.
    """
    if shutil.which('mpiexec') is None:
        pytest.skip('mpiexec not available')
    if esr.fitting.test_all.size > 1:
        pytest.skip('do not launch nested MPI jobs')
    if (os.cpu_count() or 1) < 3 and not os.environ.get('ESR_RUN_MPI_TESTS'):
        pytest.skip('launches 3 ranks; needs >=3 cores or ESR_RUN_MPI_TESTS=1')

    import sympy
    from esr.fitting.sympy_symbols import x
    from esr.fitting import test_all

    comp = 5
    compl_dir = tmp_path / 'functions' / f'compl_{comp}'
    compl_dir.mkdir(parents=True)
    # Even indices collide to one transformed model (the scale a0 is removed);
    # odd indices are all distinct. Interleaving makes collisions cross the
    # contiguous rank slices.
    all_functions = []
    for k in range(1, 13):
        all_functions.append(f'{k}*a0*(a1 + a2*x)')
        all_functions.append(f'a0 + {k}*a1*x')
    (compl_dir / f'all_equations_{comp}.txt').write_text(
        '\n'.join(all_functions) + '\n')
    (compl_dir / f'unique_equations_{comp}.txt').write_text(
        '\n'.join(all_functions) + '\n')
    (compl_dir / f'matches_{comp}.txt').write_text(
        '\n'.join(str(i) for i in range(len(all_functions))) + '\n')

    class NormalisingLikelihood:
        use_likelihood_catalogue = True

        def __init__(self, out_name):
            self.fn_dir = str(tmp_path / 'functions')
            self.base_out_dir = str(tmp_path / out_name)
            self.out_dir = str(tmp_path / out_name / 'out')
            self.temp_dir = str(tmp_path / out_name / 'tmp')

        def run_sympify(self, fcn_i, **kwargs):
            a0, a1, a2 = sympy.symbols('a0 a1 a2', real=True)
            eq = sympy.sympify(
                fcn_i, locals={'x': x, 'a0': a0, 'a1': a1, 'a2': a2})
            return fcn_i, sympy.cancel(eq / eq.subs(x, 1)), False

    # Serial reference (this process, size == 1).
    serial = NormalisingLikelihood('serial')
    assert test_all.ensure_likelihood_catalogue(comp, serial, tmax=5)
    serial_paths = likelihood_catalogue_paths(comp, serial)
    serial_unique = open(serial_paths['unique']).read()
    serial_matches = open(serial_paths['matches']).read()

    # Parallel build under mpiexec -n 3, writing to a separate output dir.
    script = tmp_path / 'parallel_build.py'
    script.write_text(textwrap.dedent(f"""
        import os, sys
        sys.path.insert(0, os.environ['ESR_REPO'])
        import sympy
        from esr.fitting.sympy_symbols import x
        from esr.fitting import test_all

        class NormalisingLikelihood:
            use_likelihood_catalogue = True
            fn_dir = {str(tmp_path / 'functions')!r}
            base_out_dir = {str(tmp_path / 'parallel')!r}
            out_dir = {str(tmp_path / 'parallel' / 'out')!r}
            temp_dir = {str(tmp_path / 'parallel' / 'tmp')!r}
            def run_sympify(self, fcn_i, **kwargs):
                a0, a1, a2 = sympy.symbols('a0 a1 a2', real=True)
                eq = sympy.sympify(
                    fcn_i, locals={{'x': x, 'a0': a0, 'a1': a1, 'a2': a2}})
                return fcn_i, sympy.cancel(eq / eq.subs(x, 1)), False

        active = test_all.ensure_likelihood_catalogue(
            {comp}, NormalisingLikelihood(), tmax=5)
        if test_all.rank == 0:
            assert active
            print('PARALLEL_BUILD_OK', flush=True)
    """))

    env = os.environ.copy()
    env['ESR_REPO'] = os.getcwd()
    env.setdefault('OMPI_ALLOW_RUN_AS_ROOT', '1')
    env.setdefault('OMPI_ALLOW_RUN_AS_ROOT_CONFIRM', '1')
    result = subprocess.run(
        ['mpiexec', '--oversubscribe', '-n', '3', sys.executable, str(script)],
        cwd=os.getcwd(), env=env, text=True, capture_output=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'PARALLEL_BUILD_OK' in result.stdout

    parallel = NormalisingLikelihood('parallel')
    parallel_paths = likelihood_catalogue_paths(comp, parallel)
    assert open(parallel_paths['unique']).read() == serial_unique
    assert open(parallel_paths['matches']).read() == serial_matches

    # Dedup happened but did not collapse everything, and the first equation's
    # transformed group genuinely spans all three rank slices (per = ceil(24/3)
    # = 8), so the cross-rank gather/merge is really exercised.
    assert 1 < len(serial_unique.splitlines()) < len(all_functions)
    match_lines = [int(m) for m in serial_matches.split()]
    group0 = [i for i, m in enumerate(match_lines) if m == match_lines[0]]
    assert any(i < 8 for i in group0)
    assert any(8 <= i < 16 for i in group0)
    assert any(i >= 16 for i in group0)

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
    """Test the supported snap modes and reject unsupported combinations."""

    likelihood = MockLikelihood(320, 0.2)
    labels = ["+", "a0", "*", "a1", "pow", "x", "3"]
    basis_functions = [["x", "a"],
                       ["inv"],
                       ["+", "*", "-", "/", "pow"]]

    results = {}
    for sc in [0, 1, 2]:
        for det_I in [True, False]:
            if sc == 2 and not det_I:
                # The projected eigenbasis is only defined with det(I) scoring.
                with pytest.raises(ValueError, match="requires use_det_I=True"):
                    single_function(labels, basis_functions, likelihood,
                                    verbose=False, use_det_I=det_I, snap_choice=sc)
                continue
            nll, DL = single_function(labels, basis_functions, likelihood,
                                      verbose=False, use_det_I=det_I, snap_choice=sc)
            assert np.isfinite(nll), f"snap_choice={sc}, use_det_I={det_I}: nll not finite"
            assert np.isfinite(DL), f"snap_choice={sc}, use_det_I={det_I}: DL not finite"
            results[(sc, det_I)] = (nll, DL)

    # A well-constrained fit is not snapped in any mode, so the likelihood agrees.
    nlls = [results[(sc, True)][0] for sc in [0, 1, 2]]
    assert np.allclose(nlls, nlls[0], atol=1e-4), f"negloglike differs across snap_choices: {nlls}"

    with pytest.raises(ValueError, match="snap_choice must be 0, 1 or 2"):
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

    with pytest.raises(ValueError, match="snap_choice must be 0, 1 or 2"):
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


def test_fitting_path_helpers_centralize_shared_filenames(tmp_path):
    """Every fitting stage obtains shared filenames from the utility helpers."""
    likelihood = SimpleNamespace(
        fn_dir=str(tmp_path / 'functions'),
        out_dir=str(tmp_path / 'output'),
        temp_dir=str(tmp_path / 'partial'),
        fnprior_prefix='prior_',
        combineDL_prefix='combined_',
        final_prefix='ranked_',
    )
    comp = 6
    rank = 3

    raw = raw_catalogue_paths(comp, likelihood)
    raw_base = tmp_path / 'functions' / 'compl_6'
    assert raw == {
        'all': str(raw_base / 'all_equations_6.txt'),
        'unique': str(raw_base / 'unique_equations_6.txt'),
        'matches': str(raw_base / 'matches_6.txt'),
        'previous': str(raw_base / 'previous_eqns_6.txt'),
        'inv_subs': str(raw_base / 'inv_subs_6.txt'),
        'fnprior': str(raw_base / 'prior_6.txt'),
    }

    catalogue = likelihood_catalogue_paths(comp, likelihood)
    prefix = tmp_path / 'output' / 'likelihood_catalogue_comp6'
    assert catalogue == {
        'unique': str(prefix) + '_unique_equations.txt',
        'matches': str(prefix) + '_matches.txt',
        'metadata': str(prefix) + '_metadata.json',
    }

    paths = fitting_paths(comp, likelihood, rank=rank)
    assert paths['negloglike'] == str(
        tmp_path / 'output' / 'negloglike_comp6.dat')
    assert paths['negloglike_checkpoint'] == str(
        tmp_path / 'output' / 'negloglike_comp6.checkpoint.dat')
    assert paths['fisher_settings'] == str(
        tmp_path / 'output' / 'fisher_settings_comp6.json')
    assert paths['codelen'] == str(
        tmp_path / 'output' / 'codelen_comp6_deriv.dat')
    assert paths['derivs'] == str(
        tmp_path / 'output' / 'derivs_comp6.dat')
    assert paths['codelen_matches'] == str(
        tmp_path / 'output' / 'codelen_matches_comp6.dat')
    assert paths['combined'] == str(
        tmp_path / 'output' / 'combined_comp6.dat')
    assert paths['combined_functions'] == str(
        tmp_path / 'output' / 'combined_fcn_comp6.dat')
    assert paths['final'] == str(
        tmp_path / 'output' / 'ranked_6.dat')
    assert paths['results_pretty'] == str(
        tmp_path / 'output' / 'results_pretty_6.txt')
    assert paths['negloglike_rank'] == str(
        tmp_path / 'partial' / 'chi2_comp6weights_3.dat')
    assert paths['codelen_rank'] == str(
        tmp_path / 'partial' / 'codelen_deriv_6_3.dat')
    assert paths['derivs_rank'] == str(
        tmp_path / 'partial' / 'derivs_6_3.dat')
    assert paths['codelen_matches_rank'] == str(
        tmp_path / 'partial' / 'codelen_matches_6_3.dat')
    assert paths['combined_rank'] == str(
        tmp_path / 'partial' / 'combined_6_3.dat')
    assert paths['combined_functions_rank'] == str(
        tmp_path / 'partial' / 'combined_fcn_6_3.dat')


def test_likelihood_aware_catalogue_groups_transformed_models(tmp_path):
    import sympy
    from esr.fitting.sympy_symbols import x
    from esr.fitting import test_all

    class NormalisingLikelihood:
        use_likelihood_catalogue = True
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
    (compl_dir / f'matches_{comp}.txt').write_text(
        '\n'.join(str(i) for i in range(len(all_functions))) + '\n')

    likelihood = NormalisingLikelihood()
    assert test_all.ensure_likelihood_catalogue(comp, likelihood, tmax=5,
                                                try_integration=False)
    paths = likelihood_catalogue_paths(comp, likelihood)
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
    (compl_dir / f'matches_{comp}.txt').write_text(
        '\n'.join(str(i) for i in range(len(all_functions))) + '\n')

    likelihood = DirectLikelihood()
    os.makedirs(likelihood.out_dir)
    paths = likelihood_catalogue_paths(comp, likelihood)
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
        use_likelihood_catalogue = True
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
    (compl_dir / f'matches_{comp}.txt').write_text(
        '\n'.join(str(i) for i in range(len(all_functions))) + '\n')

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


def test_unresolved_curvature_is_scored_independently_of_its_sign():
    """A noise-level eigenvalue must not decide whether a fit is usable.

    Finite-differenced Hessians resolve eigenvalues only to about
    EIGENVALUE_REL_THRESHOLD of the largest, so a mathematically flat direction
    comes out as a small positive or a small negative number at random. Whether
    a function is rejected as a saddle, snapped, or scored must not depend on
    which of those it happened to be.
    """
    from esr.fitting.test_all_Fisher import (
        EIGENVALUE_REL_THRESHOLD,
        _compute_snap_mask,
        _has_negative_curvature,
        _score_projected_eigenbasis,
    )

    scale = 1.0e5
    theta = np.array([2.0, 3.0])

    def almost_singular(excess):
        """Hessian whose normalised spectrum has smallest eigenvalue -excess."""
        return scale * np.array([[1.0, 1.0 + excess], [1.0 + excess, 1.0]])

    unresolved = 0.1 * EIGENVALUE_REL_THRESHOLD

    #  Curvature below the resolution of the Hessian is not a saddle...
    assert not _has_negative_curvature(almost_singular(+unresolved))
    assert not _has_negative_curvature(almost_singular(-unresolved))
    #  ...but resolved negative curvature still is.
    assert _has_negative_curvature(
        almost_singular(100. * EIGENVALUE_REL_THRESHOLD))

    #  Both signs are an unresolved direction, so both make the snap mandatory.
    for excess in (+unresolved, -unresolved):
        H = almost_singular(excess)
        diag = np.diag(H)
        Nsteps = np.abs(theta) / np.sqrt(12. / diag)
        _, degenerate = _compute_snap_mask(H, diag, theta, Nsteps.copy(), 1)
        assert degenerate, f'excess {excess}: unresolved direction must snap'

    def flat_negloglike(t):
        return 100.0

    scores = [
        _score_projected_eigenbasis(almost_singular(excess), theta, 100.0,
                                    True, flat_negloglike)[3]
        for excess in (+unresolved, -unresolved)
    ]
    assert np.isfinite(scores[0]), "unresolved curvature should still be scored"
    assert np.isclose(scores[0], scores[1]), (
        f"codelen depends on the sign of unresolved curvature: {scores}")


def test_likelihood_catalogue_keeps_the_simplifiers_representatives(tmp_path):
    """The catalogue groups the unique equations, not every generated tree.

    A likelihood transformation cannot split one of the simplifier's families --
    its members differ only by a parameter redefinition, which the
    transformation carries through with them -- so the catalogue can only merge
    families further. Building from all_equations instead would re-admit the
    redundant parameterisations the simplifier removed, and fit them as separate
    models: a near-degenerate Hessian then earns such a form a shorter
    codelength than the family it is a redundant copy of.
    """
    import sympy

    from esr.fitting import test_all
    from esr.fitting.sympy_symbols import x

    class NormalisingLikelihood:
        is_mse = False
        use_likelihood_catalogue = True
        catalogue_transform_version = 'test'

        def __init__(self, fn_dir, out_dir):
            self.fn_dir = fn_dir
            self.base_out_dir = out_dir
            self.out_dir = self.temp_dir = self.fig_dir = out_dir

        def run_sympify(self, fcn_i, **kwargs):
            a0, a1 = sympy.symbols('a0 a1', real=True)
            eq = sympy.sympify(fcn_i, locals={'x': x, 'a0': a0, 'a1': a1})
            return fcn_i, sympy.cancel(eq / eq.subs(x, 1)), False

    comp = 1
    compl_dir = tmp_path / 'functions' / f'compl_{comp}'
    compl_dir.mkdir(parents=True)
    out_dir = tmp_path / 'out'
    out_dir.mkdir()

    #  pow(x,(a0*a1)) is the same one-parameter family as pow(x,a0), written
    #  with a spare parameter; the simplifier has already folded it away, so it
    #  appears in all_equations but not among the unique equations.
    all_functions = ['pow(x,a0)', 'pow(x,(a0*a1))', 'a0*x']
    unique_functions = ['pow(x,a0)', 'a0*x']
    (compl_dir / f'all_equations_{comp}.txt').write_text(
        '\n'.join(all_functions) + '\n')
    (compl_dir / f'unique_equations_{comp}.txt').write_text(
        '\n'.join(unique_functions) + '\n')
    (compl_dir / f'matches_{comp}.txt').write_text('0\n0\n1\n')

    likelihood = NormalisingLikelihood(str(compl_dir.parent), str(out_dir))
    assert test_all.ensure_likelihood_catalogue(comp, likelihood, tmax=5)

    paths = likelihood_catalogue_paths(comp, likelihood)
    with open(paths['unique']) as f:
        representatives = f.read().split()
    assert 'pow(x,(a0*a1))' not in representatives, (
        'the catalogue re-admitted a parameterisation the simplifier removed')
    assert set(representatives) <= set(unique_functions)

    #  Still one match per generated tree, and the redundant spelling inherits
    #  the family its simplifier representative belongs to.
    with open(paths['matches']) as f:
        matches = [int(v) for v in f.read().split()]
    assert len(matches) == len(all_functions)
    assert matches[0] == matches[1]

    metadata = test_all._read_likelihood_catalogue_metadata(comp, likelihood)
    assert metadata['n_all'] == len(all_functions)
    assert metadata['raw_unique_count'] == len(unique_functions)
    assert metadata['n_unique'] <= len(unique_functions)


def test_determinant_with_diagonal_snapping_warns_but_is_allowed():
    """use_det_I=True with snap_choice=0 is a comparison setting, not an error.

    Holding the published snapping rule fixed is the only way to attribute a
    change to the determinant alone, so the pairing stays available; but
    diagonal snapping cannot remove an unconstrained direction from det(H), so
    it must say so.
    """
    from esr.fitting.test_all_Fisher import (
        DiagonalSnapDeterminantWarning,
        _validate_snap_and_det,
    )

    with pytest.warns(DiagonalSnapDeterminantWarning, match='snap_choice=1'):
        _validate_snap_and_det(True, 0)

    #  Every other supported pairing is silent.
    with warnings.catch_warnings():
        warnings.simplefilter('error', DiagonalSnapDeterminantWarning)
        for use_det_I, snap_choice in [(False, 0), (True, 1), (True, 2)]:
            _validate_snap_and_det(use_det_I, snap_choice)


def test_degeneracy_verdict_does_not_depend_on_parameter_scaling():
    """Rescaling a parameter must not turn a good fit into a degenerate one.

    Writing a parameter in different units multiplies a row and column of the
    Hessian, which can drive lambda_min/lambda_max arbitrarily small without
    making the fit any less determined. The degeneracy test therefore has to be
    invariant under that rescaling, or well determined models with parameters of
    very different magnitudes (the Pantheon exponentials, say) get stripped of a
    parameter.
    """
    from esr.fitting.test_all_Fisher import (
        _compute_snap_mask,
        _has_negative_curvature,
    )

    H = np.array([[240.0, 420.0], [420.0, 860.0]])   # ordinary linear fit
    theta = np.array([1.0, 2.0])

    for stretch in [1.0, 1.0e3, 1.0e-3, 1.0e6]:
        S = np.diag([stretch, 1.0])
        H_scaled = S @ H @ S
        theta_scaled = np.array([theta[0] / stretch, theta[1]])
        assert not _has_negative_curvature(H_scaled)
        eigenvalues = np.linalg.eigvalsh(H_scaled)
        diag = np.diag(H_scaled)
        Nsteps = np.abs(theta_scaled) / np.sqrt(12. / diag)
        _, degenerate = _compute_snap_mask(H_scaled, diag, theta_scaled,
                                           Nsteps.copy(), 1)
        assert not degenerate, (
            f'stretch {stretch:g}: a rescaled well-determined fit was called '
            f'degenerate (raw eigenvalue ratio '
            f'{eigenvalues.min() / eigenvalues.max():.2e})')


def test_weakly_occupied_but_resolved_direction_is_not_forced_to_snap():
    """A small projection onto a healthy eigendirection is an optional snap.

    Only an eigenvalue the Hessian cannot resolve makes the determinant
    untrustworthy. A well-conditioned direction that theta simply happens to be
    nearly orthogonal to stays subject to the description-length comparison, so
    that ordinary fits are not stripped of a parameter.
    """
    from esr.fitting.test_all_Fisher import (
        EIGENVALUE_REL_THRESHOLD,
        _compute_snap_mask,
    )

    #  Eigenvalues 1 and 199: both comfortably resolved, but theta projects onto
    #  the weaker one by far less than one precision step.
    H = np.array([[100.0, 99.0], [99.0, 100.0]])
    diag = np.array([100.0, 100.0])
    theta = np.array([0.001, 0.001])
    Nsteps = np.abs(theta) / np.sqrt(12. / diag)
    eigenvalues = np.linalg.eigvalsh(H)
    assert eigenvalues.min() > eigenvalues.max() * EIGENVALUE_REL_THRESHOLD

    result, degenerate = _compute_snap_mask(H, diag, theta, Nsteps.copy(), 1)
    assert np.sum(result < 1) >= 1, 'the weak direction should be a candidate'
    assert not degenerate, 'a resolved direction must not force the snap'


def test_snapping_refits_the_retained_parameters():
    """Zeroing a parameter is not the same as removing it.

    The remaining parameters were fitted alongside the one being snapped, so
    they have to be re-optimised or the reduced model is evaluated at the wrong
    point and its likelihood collapses.
    """
    from esr.fitting.test_all_Fisher import _refit_after_snap

    #  Minimum of (a0 - 1)^2 + (a1 - 2 a0)^2 over a1, at a0 = 0, is a1 = 0.
    def negloglike(t):
        return (t[0] - 1.0) ** 2 + (t[1] - 2.0 * t[0]) ** 2

    kept = np.array([False, True])
    theta, refitted = _refit_after_snap(negloglike, np.array([0.0, 4.0]), kept)
    assert theta[0] == 0.0, "a snapped parameter must stay at zero"
    assert np.isclose(theta[1], 0.0, atol=1e-4)
    assert np.isclose(refitted, 1.0, atol=1e-6)
    assert refitted < negloglike(np.array([0.0, 4.0]))

    #  Nothing left free: return the snapped point unchanged.
    theta, refitted = _refit_after_snap(
        negloglike, np.zeros(2), np.array([False, False]))
    assert np.allclose(theta, 0.0)
    assert np.isclose(refitted, 1.0)


def test_unresolved_intercept_is_snapped_without_destroying_the_fit(tmp_path):
    """An intercept that the data cannot resolve is removed, not broken.

    Data taken far from x = 0 constrain the slope but not the intercept. The
    eigenbasis modes must drop the intercept and keep fitting the slope, rather
    than zeroing the intercept while holding the slope at a value that was only
    optimal alongside it.
    """
    import sympy

    from esr.fitting.sympy_symbols import x as xsym
    from esr.fitting.test_all_Fisher import convert_params

    rng = np.random.default_rng(0)
    xvar = np.linspace(1.0e4, 1.0e4 + 40.0, 100)
    yerr = np.full_like(xvar, 0.5)
    yvar = 3.0 + 1.7 * xvar + rng.normal(scale=yerr)
    data_file = tmp_path / 'unresolved.txt'
    np.savetxt(str(data_file), np.array([xvar, yvar, yerr]).T)
    likelihood = GaussLikelihood('unresolved.txt', 'unresolved',
                                 data_dir=str(tmp_path),
                                 base_out_dir=str(tmp_path))

    fcn = 'a0 + a1*x'
    _, eq, integrated = likelihood.run_sympify(fcn, tmax=5,
                                               try_integration=False)
    a0s, a1s = sympy.symbols('a0 a1', real=True)
    eq_numpy = sympy.lambdify([xsym, a0s, a1s], eq, 'numpy')
    theta = np.array([3.0, 1.7])
    unsnapped = likelihood.negloglike(theta, eq_numpy, integrated=integrated)

    for snap_choice in [1, 2]:
        params, negloglike, _, codelen = convert_params(
            fcn, eq, integrated, theta.copy(), likelihood, unsnapped,
            max_param=2, use_det_I=True, snap_choice=snap_choice)
        assert np.isfinite(codelen)
        assert abs(params[0]) < 1.0e-3, (
            f'snap_choice={snap_choice}: intercept should be dropped, '
            f'got {params[0]}')
        assert np.isclose(params[1], 1.7, atol=1e-2)
        assert negloglike < unsnapped + 1.0, (
            f'snap_choice={snap_choice}: snapping degraded -log(L) from '
            f'{unsnapped} to {negloglike}')


def test_convert_params_preserves_diagonal_default_and_supports_full_fisher():
    """The correlated Fisher path is opt-in; the public default is diagonal."""
    import sympy
    from esr.generation.simplifier import convert_params

    a0, a1 = sympy.symbols('a0 a1', real=True)
    measured = np.array([2.0, 3.0])
    # Flattened upper triangle of [[2, 1], [1, 3]].
    fish_measured = np.array([2.0, 1.0, 3.0])
    substitutions = [{a0: a0 + a1}]

    p_default, fish_diagonal = convert_params(
        measured, fish_measured, substitutions, n=2)
    p_full, fish_full = convert_params(
        measured, fish_measured, substitutions, n=2, full_fisher=True)

    # J = [[1, 1], [0, 1]], so J^-T F J^-1 has a non-zero off diagonal.
    expected_full = np.array([[2.0, -1.0], [-1.0, 3.0]])
    np.testing.assert_allclose(p_default, [5.0, 3.0])
    np.testing.assert_allclose(p_full, p_default)
    assert fish_diagonal.shape == (2,)
    np.testing.assert_allclose(fish_diagonal, np.diag(expected_full))
    assert fish_full.shape == (2, 2)
    np.testing.assert_allclose(fish_full, expected_full)


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


def test_inverse_substitution_pair_mismatch_raises():
    """Incomplete paired substitution artifacts must not be silently paired."""
    with pytest.raises(ValueError, match='inv_idx/inv_subs length mismatch'):
        esr.generation.duplicate_checker._validate_inverse_substitution_pairs(
            3, np.array([2, 7]), [['a0: a1']])


def _combine_dl_test_likelihood(tmp_path, codelen_lines, aifeyn_lines,
                                function_lines):
    """Create the smallest single-rank input set for ``combine_DL.main``."""
    comp = 1
    fn_dir = tmp_path / 'functions'
    comp_dir = fn_dir / f'compl_{comp}'
    comp_dir.mkdir(parents=True)
    out_dir = tmp_path / 'out'
    out_dir.mkdir()
    temp_dir = tmp_path / 'temp'
    temp_dir.mkdir()
    (comp_dir / f'all_equations_{comp}.txt').write_text(function_lines)
    (comp_dir / f'aifeyn_{comp}.txt').write_text(aifeyn_lines)
    (out_dir / f'codelen_matches_comp{comp}.dat').write_text(codelen_lines)
    return comp, SimpleNamespace(
        is_mse=False,
        fn_dir=str(fn_dir),
        out_dir=str(out_dir),
        temp_dir=str(temp_dir),
        fnprior_prefix='aifeyn_',
        combineDL_prefix='combineDL_',
        final_prefix='final_',
    )


def test_combine_dl_warns_on_malformed_rows_and_preserves_parameter_width(
        tmp_path, monkeypatch):
    comp, likelihood = _combine_dl_test_likelihood(
        tmp_path,
        '1 2 0 10\nbad 2 1 3\n3 4 2 20 30\n',
        '1\n1\n1\n',
        'f0\nf1\nf2\n',
    )
    monkeypatch.setattr(
        esr.fitting.combine_DL.test_all, 'get_functions',
        lambda *args, **kwargs: ([], 0, 3))

    with pytest.warns(RuntimeWarning, match='non-numeric'):
        esr.fitting.combine_DL.main(comp, likelihood)

    rows = (tmp_path / 'out' / 'final_1.dat').read_text().splitlines()
    assert len(rows) == 2
    parsed = [row.split(';') for row in rows]
    assert [row[1] for row in parsed] == ['f0', 'f2']
    assert [float(value) for value in parsed[0][-2:]] == [10.0, 0.0]
    assert [float(value) for value in parsed[1][-2:]] == [20.0, 30.0]


def test_combine_dl_rejects_unequal_companion_lengths(tmp_path, monkeypatch):
    comp, likelihood = _combine_dl_test_likelihood(
        tmp_path, '1 2 0\n', '1\n2\n', 'f0\n')
    monkeypatch.setattr(
        esr.fitting.combine_DL.test_all, 'get_functions',
        lambda *args, **kwargs: ([], 0, 1))

    with pytest.raises(ValueError, match='unequal line counts'):
        esr.fitting.combine_DL.main(comp, likelihood)


def test_combine_dl_clears_stale_final_when_no_rows_are_valid(tmp_path, monkeypatch):
    comp, likelihood = _combine_dl_test_likelihood(
        tmp_path, 'nan 2 0\n', '1\n', 'f0\n')
    final_path = tmp_path / 'out' / 'final_1.dat'
    final_path.write_text('stale result\n')
    monkeypatch.setattr(
        esr.fitting.combine_DL.test_all, 'get_functions',
        lambda *args, **kwargs: ([], 0, 1))

    esr.fitting.combine_DL.main(comp, likelihood)

    assert final_path.read_text() == ''


def _hessian_from_deriv(deriv, nparam, max_param):
    """Rebuild a symmetric Hessian from ``test_all_Fisher``'s flattened upper
    triangle (the ``deriv`` row it writes per function)."""
    H = np.zeros((nparam, nparam))
    for i in range(nparam):
        start = int(i * max_param - (i - 1) * i / 2)
        row = deriv[start:start + nparam - i]
        H[i, i:] = row
        H[i:, i] = row
    return H


def _gaussian_design_hessian(basis_cols, yerr):
    """Analytic Hessian of a Gaussian -log(L) for a model linear in its params.

    For ``f(x) = sum_j theta_j g_j(x)`` the Hessian of the Gaussian negative
    log-likelihood is the design-matrix Gram matrix
    ``H_jk = sum_i g_j(x_i) g_k(x_i) / sigma_i^2``, independent of the data y.
    """
    w = 1.0 / np.asarray(yerr) ** 2
    G = np.column_stack(basis_cols)
    return (G * w[:, None]).T @ G


def test_convert_params_reconstructs_known_hessian_from_data(tmp_path):
    """convert_params must recover a known Hessian computed from data.

    Deaglan asked for a check that the Hmat itself is correctly computed from
    data (most snapping tests instead assume a hand-set Hmat). We use a Gaussian
    likelihood with a model linear in its parameters, whose Hessian is the
    analytic design-matrix Gram matrix.
    """
    import sympy
    from esr.fitting import test_all_Fisher
    from esr.fitting.sympy_symbols import x as xsym

    rng = np.random.default_rng(0)
    xvar = np.linspace(0.5, 3.0, 60)
    yerr = np.full_like(xvar, 0.5)
    yvar = 1.0 + 2.0 * xvar + rng.normal(0.0, 0.5, xvar.size)
    np.savetxt(tmp_path / 'data.txt', np.column_stack([xvar, yvar, yerr]))

    likelihood = GaussLikelihood('data.txt', 'known_hessian',
                                 data_dir=str(tmp_path))

    a0s, a1s = sympy.symbols('a0 a1', real=True)
    eq = a0s + a1s * xsym

    # Analytic Gram/Hessian and exact least-squares MLE for the linear model.
    basis = [np.ones_like(xvar), xvar]
    Gram = _gaussian_design_hessian(basis, yerr)
    w = 1.0 / yerr ** 2
    rhs = np.array([np.sum(w * yvar), np.sum(w * xvar * yvar)])
    theta_mle = np.linalg.solve(Gram, rhs)

    eq_numpy = sympy.lambdify([xsym, a0s, a1s], eq, 'numpy')
    nll = likelihood.negloglike(theta_mle, eq_numpy)

    params, nll_out, deriv, codelen = test_all_Fisher.convert_params(
        'a0 + a1*x', eq, False, np.pad(theta_mle, (0, 2)), likelihood, nll,
        max_param=4, use_det_I=True, snap_choice=1)

    H = _hessian_from_deriv(deriv, 2, 4)
    np.testing.assert_allclose(H, Gram, rtol=1e-3)
    assert np.isfinite(codelen)
    expected = test_all_Fisher._compute_codelen(
        Gram, np.diag(Gram), theta_mle, np.ones(2, dtype=bool), True)
    assert np.isclose(codelen, expected, rtol=1e-3)
    # Well-constrained fit: no parameter should have been snapped away.
    np.testing.assert_allclose(params[:2], theta_mle, rtol=1e-3)


def test_determinant_survives_parameter_removal(tmp_path):
    """The determinant score keeps working when canonicalisation drops or
    relabels a removed parameter of a 3- or 4-parameter model.

    Each case supplies an expression with a gap in its parameter indices (as if
    a parameter had been removed upstream); canonicalisation must relabel it to
    a contiguous set, and the full-Hessian determinant codelen must remain
    finite and match an independent computation over the reduced parameters.
    """
    import sympy
    from esr.fitting import test_all_Fisher
    from esr.fitting.sympy_symbols import x as xsym

    rng = np.random.default_rng(1)
    xvar = np.linspace(0.4, 3.0, 80)
    yerr = np.full_like(xvar, 0.4)

    cases = [
        (sympy.symbols('a0 a1 a3', real=True),
         [np.ones_like(xvar), xvar, xvar ** 2], [1.0, 2.0, 0.5]),
        (sympy.symbols('a0 a1 a3 a5', real=True),
         [np.ones_like(xvar), xvar, xvar ** 2, xvar ** 3], [1.0, 2.0, 0.5, -0.3]),
    ]

    for syms, basis, coeffs in cases:
        nparam = len(syms)
        # eq with non-contiguous parameter labels (a3, a5 stand in for the
        # removed parameters); the canonical model is a plain polynomial.
        eq = sum(sym * xsym ** k for k, sym in enumerate(syms))
        G = np.column_stack(basis)
        yvar = G @ np.array(coeffs) + rng.normal(0.0, 0.4, xvar.size)
        np.savetxt(tmp_path / 'data.txt', np.column_stack([xvar, yvar, yerr]))
        likelihood = GaussLikelihood('data.txt', f'removal_{nparam}',
                                     data_dir=str(tmp_path))

        Gram = _gaussian_design_hessian(basis, yerr)
        w = 1.0 / yerr ** 2
        theta_mle = np.linalg.solve(Gram, (G * w[:, None]).T @ yvar)

        eq_numpy = sympy.lambdify([xsym, *syms], eq, 'numpy')
        nll = likelihood.negloglike(theta_mle, eq_numpy)

        params, nll_out, deriv, codelen = test_all_Fisher.convert_params(
            'reduced', eq, False, np.pad(theta_mle, (0, 4 - nparam)),
            likelihood, nll, max_param=4, use_det_I=True, snap_choice=1)

        H = _hessian_from_deriv(deriv, nparam, 4)
        np.testing.assert_allclose(H, Gram, rtol=1e-2)
        assert np.isfinite(codelen), f'{nparam}-param determinant codelen not finite'
        expected = test_all_Fisher._compute_codelen(
            Gram, np.diag(Gram), theta_mle, np.ones(nparam, dtype=bool), True)
        assert np.isclose(codelen, expected, rtol=1e-2)
        # Canonicalisation collapses the gapped labels to exactly nparam params.
        np.testing.assert_allclose(params[:nparam], theta_mle, rtol=1e-2)


def test_determinant_scoring_and_matching_with_parameter_removal(
        tmp_path, monkeypatch):
    """End-to-end determinant scoring and matching when the likelihood transform
    removes a parameter from 3-parameter models.

    This also exercises the likelihood-aware early return in ``match.main``:
    two raw expressions that collapse to the same transformed model must inherit
    the representative's determinant codelen/negloglike/params verbatim, with no
    second (inverse-substitution) transformation applied.
    """
    import sympy
    from esr.fitting import test_all, test_all_Fisher, match
    from esr.fitting.sympy_symbols import x as xsym

    if monkeypatch is not None:
        monkeypatch.setattr(plt, 'show', lambda: None)

    rng = np.random.default_rng(3)
    xvar = np.linspace(0.4, 3.0, 60)
    yerr = np.full_like(xvar, 0.4)
    yvar = 1.5 + 0.8 * xvar + 0.3 * xvar ** 2 + rng.normal(0.0, 0.4, xvar.size)
    np.savetxt(tmp_path / 'data.txt', np.column_stack([xvar, yvar, yerr]))

    class OffsetRemovingGauss(GaussLikelihood):
        # Subtract the value at x=1, removing the constant (a0) term of
        # a0 + g(x). This changes the parameter layout, so we opt back in to the
        # likelihood-aware catalogue (the built-in GaussLikelihood opts out).
        use_likelihood_catalogue = True

        def run_sympify(self, fcn_i, **kwargs):
            fcn_i, eq, _ = super().run_sympify(fcn_i, **kwargs)
            return fcn_i, sympy.expand(eq - eq.subs(xsym, 1)), False

    likelihood = OffsetRemovingGauss('data.txt', 'det_removal',
                                     data_dir=str(tmp_path))
    likelihood.fn_dir = str(tmp_path / 'functions')

    comp = 5
    compl_dir = tmp_path / 'functions' / f'compl_{comp}'
    compl_dir.mkdir(parents=True)
    all_functions = [
        'a0 + a1*x + a2*pow(x, 2)',
        '2*a0 + a1*x + a2*pow(x, 2)',   # same transformed model as the first
        'a0 + a1*x',
    ]
    (compl_dir / f'all_equations_{comp}.txt').write_text(
        '\n'.join(all_functions) + '\n')
    (compl_dir / f'unique_equations_{comp}.txt').write_text(
        '\n'.join(all_functions) + '\n')
    (compl_dir / f'matches_{comp}.txt').write_text(
        '\n'.join(str(i) for i in range(len(all_functions))) + '\n')

    assert test_all.ensure_likelihood_catalogue(comp, likelihood, tmax=5)
    test_all.main(comp, likelihood, Niter_params=[4], Nconv_params=[2])
    test_all_Fisher.main(comp, likelihood, use_det_I=True, snap_choice=1)
    match.main(comp, likelihood)

    matched = np.atleast_2d(np.loadtxt(
        os.path.join(likelihood.out_dir, f'codelen_matches_comp{comp}.dat')))
    # Columns are [negloglike_all, codelen, index, params...].
    assert matched.shape[0] == len(all_functions)
    # The first two raw expressions collapse to the same representative.
    assert matched[0, 2] == matched[1, 2]
    assert matched[2, 2] != matched[0, 2]
    # Inheritance is a straight copy: no second, inverse transformation.
    np.testing.assert_allclose(matched[0, :], matched[1, :])
    # Determinant scoring produced a finite codelen for the reduced models.
    assert np.isfinite(matched[0, 1])
    assert np.isfinite(matched[2, 1])

    # The match output must reproduce the likelihoods it claims.
    assert match.check_match_results(comp, likelihood) == 0


def test_legacy_diagonal_settings_reproduce_published_values(
        monkeypatch, tmp_path):
    """The published (old-method) cosmic-chronometer values are reproduced when
    the pipeline is run with the pre-det(I) settings.

    Running the hard-coded CC dataset with ``use_det_I=False, snap_choice=0``
    must recover the Table-1 values of arXiv:2211.11461 for the best function,
    every function's stored likelihood must still re-evaluate correctly, and the
    old-method description lengths must be finite and correctly ranked across the
    whole range of complexity-3 functions, not just the best row. (Complexity 3
    contains only 0- and 1-parameter functions; the genuinely multi-parameter
    old-method check lives in
    ``test_old_method_codelen_matches_diagonal_formula_from_data``.)
    """
    if monkeypatch is not None:
        monkeypatch.setattr(plt, 'show', lambda: None)

    comp = 3
    # Isolated catalogue + output directories, so this test is safe to run
    # concurrently with others (e.g. under pytest-xdist).
    likelihood = CCLikelihood(fn_dir=str(tmp_path / 'functions'),
                              base_out_dir=str(tmp_path / 'output'))
    esr.generation.duplicate_checker.main(
        'core_maths', comp, fn_dir=likelihood.fn_dir)

    esr.fitting.test_all.main(comp, likelihood)
    esr.fitting.test_all_Fisher.main(
        comp, likelihood, use_det_I=False, snap_choice=0)
    esr.fitting.match.main(comp, likelihood)
    assert esr.fitting.match.check_match_results(comp, likelihood) == 0
    esr.fitting.combine_DL.main(comp, likelihood)

    settings = esr.fitting.test_all_Fisher.load_scoring_settings(comp, likelihood)
    assert settings == {'use_det_I': False, 'snap_choice': 0}

    # final_<comp>.dat columns are: rank; function; total DL; rel. probability;
    # negloglike; parameter codelength; function codelength; params...
    fname = os.path.join(likelihood.out_dir, f'final_{comp}.dat')
    with open(fname, 'r') as f:
        best = f.readline().split(';')
    # Same published Table-1 values as test_cc, now via the old-method settings.
    assert best[1] == 'a0*x'
    assert np.isclose(float(best[2]), 29.959725666004328, atol=2e-2)  # total DL
    assert np.isclose(float(best[4]), 24.138886, atol=2e-2)           # negloglike
    assert np.isclose(float(best[5]), 2.5250028, atol=2e-2)           # param codelen
    assert np.isclose(float(best[6]), 3.295836866004329, atol=2e-2)   # func codelen
    assert np.isclose(float(best[7]), 5638.4157, atol=10)             # best-fit a0
    assert np.all(np.array(best[8:], dtype=float) == 0)
    # DL is the sum of its three published components.
    assert np.isclose(
        float(best[2]), float(best[4]) + float(best[5]) + float(best[6]),
        atol=2e-2)

    # Range check: the old-method scoring must produce finite, correctly-ranked
    # description lengths across all complexity-3 functions, not merely the best
    # row. (Complexity 3 only contains 0- and 1-parameter functions, for which
    # the diagonal and det(I) codelengths coincide; the genuinely multi-parameter
    # old-vs-new comparison is covered by
    # ``test_old_method_codelen_matches_diagonal_formula_from_data``.)
    with open(fname, 'r') as f:
        rows = [line.rstrip('\n').split(';') for line in f if line.strip()]
    assert len(rows) > 5
    dl = np.array([float(r[2]) for r in rows])   # column 2 is already the total DL
    assert np.all(np.isfinite(dl))
    assert np.all(np.diff(dl) >= -1e-6)          # combine_DL ranks by ascending DL
    old_dl = {r[1]: float(r[2]) for r in rows}

    # Cross-check the whole range against the current method: complexity 3 has
    # only 0/1-parameter functions, where the diagonal and det(I) codelengths
    # coincide, so re-scoring the same fits with the new settings must reproduce
    # the old-method DL of *every* function -- not just the best row. This is an
    # internal consistency check between two current code paths (old-method vs
    # det(I) settings), not a comparison against independently fixed historical
    # values: only the best row (above) is tied to published Table-1 numbers.
    esr.fitting.test_all_Fisher.main(
        comp, likelihood, use_det_I=True, snap_choice=1)
    esr.fitting.match.main(comp, likelihood)
    esr.fitting.combine_DL.main(comp, likelihood)
    with open(fname, 'r') as f:
        new_rows = [line.rstrip('\n').split(';') for line in f if line.strip()]
    new_dl = {r[1]: float(r[2]) for r in new_rows}
    assert set(new_dl) == set(old_dl)
    for fcn, value in old_dl.items():
        assert np.isclose(value, new_dl[fcn], atol=2e-2), fcn


def test_projected_eigenbasis_codelen_is_eigenbasis_consistent():
    """snap_choice=2 scores in the Hessian eigenbasis.

    For a well-constrained (unsnapped) fit the codelength equals
    ``_compute_codelen`` evaluated with the diagonalised Hessian and the rotated
    parameters, and it differs from the original-basis (snap_choice=0/1) score
    because the precision floor is taken in the eigenbasis rather than against
    the original H_ii -- the point of Deaglan's README question.
    """
    from esr.fitting.test_all_Fisher import (
        _score_projected_eigenbasis, _compute_codelen)

    H = np.array([[100.0, 40.0], [40.0, 60.0]])
    theta = np.array([20.0, 20.0])
    eigvals, V = np.linalg.eigh(H)
    b = V.T @ theta

    theta_f, nll_f, k, cl = _score_projected_eigenbasis(
        H, theta, 7.0, True, lambda t: 7.0)

    assert k == 2                       # nothing snapped
    np.testing.assert_allclose(theta_f, theta)
    assert nll_f == 7.0

    expected_eig = _compute_codelen(
        np.diag(eigvals), eigvals, b, np.ones(2, dtype=bool), True)
    assert np.isclose(cl, expected_eig)

    # The determinant/volume term is basis-invariant, but the floor term is not,
    # so the eigenbasis score differs from the original-basis score.
    orig = _compute_codelen(H, np.diag(H), theta, np.ones(2, dtype=bool), True)
    assert not np.isclose(cl, orig)


def test_projected_eigenbasis_snaps_weak_direction():
    """A weakly-constrained projected coordinate is zeroed and the codelength is
    scored over the retained eigendirections only."""
    from esr.fitting.test_all_Fisher import _score_projected_eigenbasis

    c = 1.0 / np.sqrt(2.0)
    V = np.array([[c, -c], [c, c]])            # 45-degree rotation
    eigvals = np.array([1.0e6, 1.0])           # second direction weak but resolved
    H = V @ np.diag(eigvals) @ V.T
    b = np.array([3.0, 1.0e-3])                # tiny projection on the weak axis
    theta = V @ b

    calls = []

    def eval_nll(t):
        calls.append(np.asarray(t))
        return 12.0

    theta_f, nll_f, k, cl = _score_projected_eigenbasis(
        H, theta, 12.0, True, eval_nll)

    assert k == 1                              # weak eigendirection dropped
    assert len(calls) == 1                     # likelihood re-evaluated once
    assert nll_f == 12.0
    assert np.isfinite(cl)
    np.testing.assert_allclose(theta_f, V @ np.array([b[0], 0.0]))


def test_projected_eigenbasis_handles_exact_flat_direction():
    """An exact flat direction (zero eigenvalue / zero Hessian diagonal) is
    projected out by snap_choice=2 with a finite codelength.

    The diagonal ``Fisher_diag <= 0`` rejection used by modes 0/1 would discard
    such a fit outright; the projected-eigenbasis branch runs before that check
    so it can legitimately remove the flat coordinate instead.
    """
    from esr.fitting.test_all_Fisher import _score_projected_eigenbasis

    # Zero diagonal element (flat direction aligned with the second parameter).
    H = np.diag([100.0, 0.0])
    theta = np.array([2.0, 3.0])
    theta_f, nll_f, k, cl = _score_projected_eigenbasis(
        H, theta, 8.0, True, lambda t: 8.0)
    assert k == 1                              # the flat direction is removed
    assert np.isfinite(cl)
    np.testing.assert_allclose(theta_f, [2.0, 0.0])


def test_snap_choice_2_requires_determinant(tmp_path):
    """snap_choice=2 is rejected without det(I) scoring, and persists otherwise."""
    from esr.fitting import test_all_Fisher as F

    class Likelihood:
        out_dir = str(tmp_path)

    with pytest.raises(ValueError, match="requires use_det_I=True"):
        F.save_scoring_settings(3, Likelihood, False, 2)

    F.save_scoring_settings(3, Likelihood, True, 2)
    assert F.load_scoring_settings(3, Likelihood) == {
        'use_det_I': True, 'snap_choice': 2}


def test_snap_choice_2_end_to_end(monkeypatch, tmp_path):
    """The projected-eigenbasis mode runs through the whole pipeline, including
    the matching step, and reproduces the (1-parameter) best function exactly
    -- for a single parameter the eigenbasis and original-basis scores agree."""
    if monkeypatch is not None:
        monkeypatch.setattr(plt, 'show', lambda: None)

    comp = 3
    # Isolated catalogue + output directories (safe under concurrent runs).
    likelihood = CCLikelihood(fn_dir=str(tmp_path / 'functions'),
                              base_out_dir=str(tmp_path / 'output'))
    esr.generation.duplicate_checker.main(
        'core_maths', comp, fn_dir=likelihood.fn_dir)

    esr.fitting.test_all.main(comp, likelihood)
    esr.fitting.test_all_Fisher.main(
        comp, likelihood, use_det_I=True, snap_choice=2)
    esr.fitting.match.main(comp, likelihood)
    assert esr.fitting.match.check_match_results(comp, likelihood) == 0
    esr.fitting.combine_DL.main(comp, likelihood)

    assert esr.fitting.test_all_Fisher.load_scoring_settings(comp, likelihood) == {
        'use_det_I': True, 'snap_choice': 2}

    fname = os.path.join(likelihood.out_dir, f'final_{comp}.dat')
    with open(fname, 'r') as f:
        best = f.readline().split(';')
    assert best[1] == 'a0*x'
    assert np.isclose(float(best[2]), 29.959725666004328, atol=2e-2)   # total DL
    # 1-parameter: eigenbasis and original-basis codelengths coincide.
    assert np.isclose(float(best[6]), 3.295836866004329, atol=2e-2)    # func codelen


def test_snap_choice_2_match_multiparam_end_to_end(monkeypatch, tmp_path):
    """Genuine end-to-end exercise of snap_choice=2 through ``match.main`` on
    multi-parameter (rotatable) functions.

    Complexity 5 core_maths contains 2-parameter functions, whose Hessians are
    genuinely rotated relative to the parameter axes -- unlike the 1-parameter
    complexity-3 best function. This generates that catalogue into an isolated
    directory, fits a Gaussian likelihood, scores with the projected eigenbasis,
    matches, and checks that multi-parameter functions receive a finite mode-2
    codelength and that every stored likelihood re-evaluates correctly.
    """
    if monkeypatch is not None:
        monkeypatch.setattr(plt, 'show', lambda: None)

    np.random.seed(321)
    xvar = np.random.uniform(0.5, 3.0, 60)
    yvar = 1.0 + 2.0 * xvar + 0.5 * xvar ** 2 + np.random.normal(0, 0.3, xvar.size)
    yerr = np.full_like(xvar, 0.3)
    np.savetxt(tmp_path / 'data.txt', np.column_stack([xvar, yvar, yerr]))

    comp = 5
    likelihood = GaussLikelihood(
        'data.txt', 'snap2_mp', data_dir=str(tmp_path),
        fn_dir=str(tmp_path / 'functions'),
        base_out_dir=str(tmp_path / 'output'))
    esr.generation.duplicate_checker.main(
        'core_maths', comp, fn_dir=likelihood.fn_dir)
    esr.fitting.test_all.main(
        comp, likelihood, Niter_params=[4], Nconv_params=[2])

    # Spy on the snap-callback so we can assert the projected snap path (which
    # zeros a coordinate and re-evaluates the likelihood) actually ran on a
    # genuinely rotatable (multi-parameter) function. Counting any call is not
    # enough: 1-parameter functions also re-evaluate, but their Hessian is not
    # rotated relative to the parameter axis, so they do not exercise the
    # projection. We record the largest parameter-vector length seen.
    calls = {'n': 0, 'multi': 0}
    original = esr.fitting.match._variant_negloglike

    def counting(likelihood, fcn_i, theta_vec, *args, **kwargs):
        calls['n'] += 1
        if np.size(theta_vec) >= 2:
            calls['multi'] += 1
        return original(likelihood, fcn_i, theta_vec, *args, **kwargs)
    monkeypatch.setattr(esr.fitting.match, '_variant_negloglike', counting)

    esr.fitting.test_all_Fisher.main(
        comp, likelihood, use_det_I=True, snap_choice=2)
    esr.fitting.match.main(comp, likelihood)
    assert esr.fitting.match.check_match_results(comp, likelihood) == 0
    esr.fitting.combine_DL.main(comp, likelihood)

    assert calls['multi'] > 0  # a multi-parameter function's projected coordinate was re-evaluated

    # codelen_matches columns: negloglike; codelen; index; params...
    matched = np.atleast_2d(np.loadtxt(
        os.path.join(likelihood.out_dir, f'codelen_matches_comp{comp}.dat')))
    nparam = np.sum(matched[:, 3:] != 0, axis=1)
    finite = np.isfinite(matched[:, 1])
    assert np.any(nparam >= 2)                 # 2-parameter functions are present
    assert np.any(finite & (nparam >= 2))      # and got a finite mode-2 codelen


def test_projected_eigenbasis_correlated_scoring(tmp_path):
    """snap_choice=2 on a genuinely correlated two-parameter model scores in the
    rotated eigenbasis and differs from the original-basis snap_choice=1 score.

    This is a focused ``convert_params``-level check (not an end-to-end run; the
    multi-parameter match path is covered by
    ``test_snap_choice_2_match_multiparam_end_to_end``): with an off-diagonal
    Hessian the eigenbasis is genuinely rotated relative to the parameter axes,
    the case the projected scorer must handle. It fits from data in a temporary
    directory.
    """
    import sympy
    from esr.fitting import test_all_Fisher
    from esr.fitting.sympy_symbols import x as xsym

    rng = np.random.default_rng(7)
    xvar = np.linspace(0.5, 3.0, 60)
    yerr = np.full_like(xvar, 0.3)
    # x and x**2 are strongly correlated over this range, so the Hessian has
    # large off-diagonal terms and the eigenbasis differs from the (a0, a1) axes.
    yvar = 1.2 * xvar + 0.7 * xvar ** 2 + rng.normal(0.0, 0.3, xvar.size)
    np.savetxt(tmp_path / 'data.txt', np.column_stack([xvar, yvar, yerr]))
    likelihood = GaussLikelihood('data.txt', 'proj_corr', data_dir=str(tmp_path))

    a0s, a1s = sympy.symbols('a0 a1', real=True)
    eq = a0s * xsym + a1s * xsym ** 2
    basis = [xvar, xvar ** 2]
    Gram = _gaussian_design_hessian(basis, yerr)
    w = 1.0 / yerr ** 2
    theta_mle = np.linalg.solve(
        Gram, (np.column_stack(basis) * w[:, None]).T @ yvar)
    eq_numpy = sympy.lambdify([xsym, a0s, a1s], eq, 'numpy')
    nll = likelihood.negloglike(theta_mle, eq_numpy)

    common = ('a0*x + a1*x**2', eq, False, np.pad(theta_mle, (0, 2)),
              likelihood, nll)
    _, _, _, cl2 = test_all_Fisher.convert_params(
        *common, max_param=4, use_det_I=True, snap_choice=2)
    _, _, _, cl1 = test_all_Fisher.convert_params(
        *common, max_param=4, use_det_I=True, snap_choice=1)

    eigvals, V = np.linalg.eigh(Gram)
    b = V.T @ theta_mle
    expected2 = test_all_Fisher._compute_codelen(
        np.diag(eigvals), eigvals, b, np.ones(2, dtype=bool), True)
    assert np.isfinite(cl2)
    assert np.isclose(cl2, expected2, rtol=1e-2)
    # The rotation genuinely matters: the eigenbasis floor differs from the
    # original-basis floor, so mode 2 and mode 1 disagree here.
    assert not np.isclose(cl2, cl1, rtol=1e-3)


def test_old_method_codelen_matches_diagonal_formula_from_data(tmp_path):
    """The pre-det(I) settings reproduce the published diagonal codelength,
    computed from data, for a range of genuinely multi-parameter functions --
    and differ from the det(I) score when the Hessian is correlated.

    Complexity-3 functions have at most one parameter, where the two scores
    coincide, so this isolated test covers the 2- and 3-parameter case that
    ``test_legacy_diagonal_settings_reproduce_published_values`` cannot.
    """
    import sympy
    from esr.fitting import test_all_Fisher
    from esr.fitting.sympy_symbols import x as xsym

    rng = np.random.default_rng(11)
    xvar = np.linspace(0.4, 3.0, 80)
    yerr = np.full_like(xvar, 0.3)

    cases = [
        (sympy.symbols('a0 a1', real=True),
         [xsym, xsym ** 2], [xvar, xvar ** 2], [1.0, 0.5]),
        (sympy.symbols('a0 a1 a2', real=True),
         [sympy.Integer(1), xsym, xsym ** 2],
         [np.ones_like(xvar), xvar, xvar ** 2], [1.0, 2.0, 0.5]),
    ]
    for syms, basis_expr, basis_cols, coeffs in cases:
        nparam = len(syms)
        eq = sum(s * be for s, be in zip(syms, basis_expr))
        G = np.column_stack(basis_cols)
        yvar = G @ np.array(coeffs) + rng.normal(0.0, 0.3, xvar.size)
        np.savetxt(tmp_path / 'data.txt', np.column_stack([xvar, yvar, yerr]))
        likelihood = GaussLikelihood(
            'data.txt', f'old_method_{nparam}', data_dir=str(tmp_path))

        Gram = _gaussian_design_hessian(basis_cols, yerr)
        w = 1.0 / yerr ** 2
        theta = np.linalg.solve(Gram, (G * w[:, None]).T @ yvar)
        eqn = sympy.lambdify([xsym, *syms], eq, 'numpy')
        nll = likelihood.negloglike(theta, eqn)

        common = ('f', eq, False, np.pad(theta, (0, 4 - nparam)), likelihood, nll)
        _, _, _, cl_old = test_all_Fisher.convert_params(
            *common, max_param=4, use_det_I=False, snap_choice=0)
        _, _, _, cl_new = test_all_Fisher.convert_params(
            *common, max_param=4, use_det_I=True, snap_choice=1)

        expected_diag = test_all_Fisher._compute_codelen(
            Gram, np.diag(Gram), theta, np.ones(nparam, dtype=bool), False)
        assert np.isfinite(cl_old)
        assert np.isclose(cl_old, expected_diag, rtol=1e-2)
        # det(H) < prod(diag) for a correlated Hessian, so the det(I) codelength
        # is strictly smaller -- the two methods genuinely differ here.
        assert cl_new < cl_old


def test_likelihood_catalogue_cache_invalidates_on_equation_change(tmp_path):
    """Editing all_equations must rebuild the catalogue, not reuse a stale
    matches file (which could associate equations with the wrong representative).

    Runs isolated in temporary directories.
    """
    import sympy
    from esr.fitting.sympy_symbols import x
    from esr.fitting import test_all

    class NormalisingLikelihood:
        use_likelihood_catalogue = True
        catalogue_transform_version = 'v1'
        is_mse = False
        base_out_dir = str(tmp_path / 'out_base')
        out_dir = str(tmp_path / 'out')
        temp_dir = str(tmp_path / 'tmp')
        fn_dir = str(tmp_path / 'functions')

        def run_sympify(self, fcn_i, **kwargs):
            a0, a1, a2 = sympy.symbols('a0 a1 a2', real=True)
            eq = sympy.sympify(
                fcn_i, locals={'x': x, 'a0': a0, 'a1': a1, 'a2': a2})
            return fcn_i, sympy.cancel(eq / eq.subs(x, 1)), True

    comp = 1
    compl_dir = tmp_path / 'functions' / f'compl_{comp}'
    compl_dir.mkdir(parents=True)

    def write(funcs):
        (compl_dir / f'all_equations_{comp}.txt').write_text(
            '\n'.join(funcs) + '\n')
        (compl_dir / f'unique_equations_{comp}.txt').write_text(
            '\n'.join(funcs) + '\n')
        (compl_dir / f'matches_{comp}.txt').write_text(
            '\n'.join(str(i) for i in range(len(funcs))) + '\n')

    likelihood = NormalisingLikelihood()
    write(['a0*(a1 + x)', 'a2 + x'])
    assert test_all.ensure_likelihood_catalogue(comp, likelihood, tmax=5)
    paths = likelihood_catalogue_paths(comp, likelihood)
    assert len(open(paths['matches']).read().splitlines()) == 2

    # Append an equation. Without a content-aware cache key the old two-line
    # matches file would be reused; it must rebuild instead.
    write(['a0*(a1 + x)', 'a2 + x', 'a0*x'])
    assert test_all.ensure_likelihood_catalogue(comp, likelihood, tmax=5)
    assert len(open(paths['matches']).read().splitlines()) == 3
    metadata = test_all._read_likelihood_catalogue_metadata(comp, likelihood)
    assert metadata['n_all'] == 3
    assert metadata['settings']['all_equations_hash'] is not None


def test_likelihood_catalogue_versioned_cache_hit_skips_rebuild(tmp_path, monkeypatch):
    """An unchanged, explicitly versioned catalogue is reused, not rebuilt.

    The invalidation tests prove the cache is *discarded* when its inputs change.
    This proves the complementary property: a second call with identical
    equations, transform and ``catalogue_transform_version`` takes the cache-hit
    path and does not repeat the expensive per-equation transform pass. Without
    this, an implementation that always rebuilt would still pass every other
    catalogue test.
    """
    import sympy
    from esr.fitting.sympy_symbols import x
    from esr.fitting import test_all

    class NormalisingLikelihood:
        use_likelihood_catalogue = True
        catalogue_transform_version = 'v1'
        is_mse = False
        base_out_dir = str(tmp_path / 'out_base')
        out_dir = str(tmp_path / 'out')
        temp_dir = str(tmp_path / 'tmp')
        fn_dir = str(tmp_path / 'functions')

        def run_sympify(self, fcn_i, **kwargs):
            a0, a1, a2 = sympy.symbols('a0 a1 a2', real=True)
            eq = sympy.sympify(
                fcn_i, locals={'x': x, 'a0': a0, 'a1': a1, 'a2': a2})
            return fcn_i, sympy.cancel(eq / eq.subs(x, 1)), True

    comp = 1
    compl_dir = tmp_path / 'functions' / f'compl_{comp}'
    compl_dir.mkdir(parents=True)
    funcs = ['a0*(a1 + x)', 'a2 + x']
    for name in (f'all_equations_{comp}.txt', f'unique_equations_{comp}.txt'):
        (compl_dir / name).write_text('\n'.join(funcs) + '\n')
    (compl_dir / f'matches_{comp}.txt').write_text(
        '\n'.join(str(i) for i in range(len(funcs))) + '\n')

    likelihood = NormalisingLikelihood()

    # _transformed_keys_for_slice runs only on a genuine (re)build, never on a
    # cache hit, so counting its calls distinguishes the two paths.
    builds = {'n': 0}
    original = test_all._transformed_keys_for_slice

    def counting(*args, **kwargs):
        builds['n'] += 1
        return original(*args, **kwargs)
    monkeypatch.setattr(test_all, '_transformed_keys_for_slice', counting)

    active_first = test_all.ensure_likelihood_catalogue(comp, likelihood, tmax=5)
    assert active_first is True          # normalising removes a0 -> active catalogue
    assert builds['n'] == 1              # first call built it
    matches_before = open(
        likelihood_catalogue_paths(comp, likelihood)['matches']).read()

    # Nothing changed: identical equations, transform and version.
    active_second = test_all.ensure_likelihood_catalogue(comp, likelihood, tmax=5)
    assert active_second is True         # same active state returned
    assert builds['n'] == 1              # second call reused the cache, no rebuild

    matches_after = open(
        likelihood_catalogue_paths(comp, likelihood)['matches']).read()
    assert matches_after == matches_before


def test_likelihood_catalogue_warns_on_failed_transforms(tmp_path):
    """Transformation failures during the build are surfaced with a warning
    rather than silently cached (a programming error -- e.g. an invalid tmax --
    would otherwise produce a quietly incomplete catalogue)."""
    from esr.fitting import test_all

    class BrokenLikelihood:
        use_likelihood_catalogue = True
        is_mse = False
        base_out_dir = str(tmp_path / 'out_base')
        out_dir = str(tmp_path / 'out')
        temp_dir = str(tmp_path / 'tmp')
        fn_dir = str(tmp_path / 'functions')

        def run_sympify(self, fcn_i, **kwargs):
            raise RuntimeError('deliberate transform failure')

    comp = 1
    compl_dir = tmp_path / 'functions' / f'compl_{comp}'
    compl_dir.mkdir(parents=True)
    funcs = ['a0*x', 'a0 + x']
    (compl_dir / f'all_equations_{comp}.txt').write_text('\n'.join(funcs) + '\n')
    (compl_dir / f'unique_equations_{comp}.txt').write_text(
        '\n'.join(funcs) + '\n')
    (compl_dir / f'matches_{comp}.txt').write_text(
        '\n'.join(str(i) for i in range(len(funcs))) + '\n')

    likelihood = BrokenLikelihood()
    with pytest.warns(test_all.LikelihoodCatalogueWarning, match='failed to'):
        test_all.ensure_likelihood_catalogue(comp, likelihood, tmax=5)
    metadata = test_all._read_likelihood_catalogue_metadata(comp, likelihood)
    assert metadata['failed_count'] == 2


def test_duplicate_checker_and_likelihood_share_custom_fn_dir(tmp_path):
    """duplicate_checker.main writes into a caller-supplied fn_dir, and a
    likelihood built with the same fn_dir reads that catalogue -- so generation
    and fitting can run in a private, isolated location rather than the shared
    package function_library (the fix for concurrent/xdist-safe runs)."""
    from esr.fitting.likelihood import CCLikelihood

    fn_dir = str(tmp_path / 'lib')
    likelihood = CCLikelihood(fn_dir=fn_dir, base_out_dir=str(tmp_path / 'out'))
    # Both the catalogue and the output now live under tmp_path.
    assert os.path.abspath(fn_dir) in likelihood.fn_dir
    assert str(tmp_path) in likelihood.out_dir

    esr.generation.duplicate_checker.main(
        'core_maths', 3, fn_dir=likelihood.fn_dir)

    raw = raw_catalogue_paths(3, likelihood)
    assert os.path.exists(raw['all'])
    assert os.path.exists(raw['unique'])
    # The files really landed under the custom fn_dir, not the package library.
    assert os.path.abspath(fn_dir) in os.path.abspath(raw['all'])
    assert sum(1 for _ in open(raw['all'])) == 24   # core_maths complexity 3


def test_convert_params_snap2_flat_direction_from_pipeline(monkeypatch, tmp_path):
    """An exact flat (zero-diagonal) direction is projected out by snap_choice=2
    through the real ``convert_params`` entry point, not just the private scorer.

    The diagonal retry/rejection loop would otherwise discard a zero-diagonal
    Hessian and return nan (the auditor's H=diag(1,0) case); the mode-2 branch
    must run before it.
    """
    import sympy
    from esr.fitting import test_all_Fisher
    from esr.fitting.sympy_symbols import x as xsym

    xvar = np.linspace(0.5, 3.0, 40)
    yerr = np.full_like(xvar, 0.5)
    yvar = 2.0 * xvar + 0.5
    np.savetxt(tmp_path / 'data.txt', np.column_stack([xvar, yvar, yerr]))
    likelihood = GaussLikelihood('data.txt', 'flat', data_dir=str(tmp_path))

    a0s, a1s = sympy.symbols('a0 a1', real=True)
    eq = a0s * xsym + a1s

    # Force the numerically-computed Hessian to be exactly diag(100, 0): a0
    # constrained, a1 an exact flat direction.
    monkeypatch.setattr(
        test_all_Fisher.nd, 'Hessian',
        lambda *a, **k: (lambda th: np.array([[100.0, 0.0], [0.0, 0.0]])))

    params, nll, deriv, codelen = test_all_Fisher.convert_params(
        'a0*x + a1', eq, False, np.array([2.0, 0.5, 0.0, 0.0]), likelihood, 5.0,
        max_param=4, use_det_I=True, snap_choice=2)
    assert np.isfinite(codelen)          # was nan before the fix
    assert params[1] == 0.0              # flat direction a1 projected out


def test_likelihood_catalogue_cache_invalidates_on_transform_change(tmp_path):
    """Changing the run_sympify transform (same equations and output directory)
    must invalidate the cache rather than reuse the previous transform's
    catalogue -- caught by the probe-based transform fingerprint."""
    import sympy
    from esr.fitting.sympy_symbols import x
    from esr.fitting import test_all

    comp = 1
    compl_dir = tmp_path / 'functions' / f'compl_{comp}'
    compl_dir.mkdir(parents=True)
    funcs = ['a0*(a1 + x)', 'a2 + x']
    for name in (f'all_equations_{comp}.txt', f'unique_equations_{comp}.txt'):
        (compl_dir / name).write_text('\n'.join(funcs) + '\n')
    (compl_dir / f'matches_{comp}.txt').write_text(
        '\n'.join(str(i) for i in range(len(funcs))) + '\n')

    class Base:
        is_mse = False
        use_likelihood_catalogue = True
        catalogue_transform_version = 'v1'
        fn_dir = str(tmp_path / 'functions')
        base_out_dir = str(tmp_path / 'out_base')
        out_dir = str(tmp_path / 'out')
        temp_dir = str(tmp_path / 'tmp')

    class Normalising(Base):
        def run_sympify(self, fcn_i, **kwargs):
            a0, a1, a2 = sympy.symbols('a0 a1 a2', real=True)
            eq = sympy.sympify(
                fcn_i, locals={'x': x, 'a0': a0, 'a1': a1, 'a2': a2})
            return fcn_i, sympy.cancel(eq / eq.subs(x, 1)), True

    class Identity(Base):
        def run_sympify(self, fcn_i, **kwargs):
            a0, a1, a2 = sympy.symbols('a0 a1 a2', real=True)
            return fcn_i, sympy.sympify(
                fcn_i, locals={'x': x, 'a0': a0, 'a1': a1, 'a2': a2}), False

    test_all.ensure_likelihood_catalogue(comp, Normalising(), tmax=5)
    md_norm = test_all._read_likelihood_catalogue_metadata(comp, Normalising())

    # Same equations + output directory, different transform: must rebuild.
    test_all.ensure_likelihood_catalogue(comp, Identity(), tmax=5)
    md_id = test_all._read_likelihood_catalogue_metadata(comp, Identity())

    fp_norm = md_norm['settings']['transform_fingerprint']
    fp_id = md_id['settings']['transform_fingerprint']
    assert fp_norm is not None and fp_id is not None
    assert fp_norm != fp_id                       # different transform -> different key
    assert md_id['n_unique'] != md_norm['n_unique']   # genuinely rebuilt


def test_variant_negloglike_reevaluates_correctly(tmp_path):
    """match._variant_negloglike rebuilds a variant's numpy function and returns
    the correct -log(L) -- the callback the snap_choice=2 match path invokes when
    a projected coordinate is snapped."""
    import sympy
    from esr.fitting import match
    from esr.fitting.sympy_symbols import x as xsym

    xvar = np.linspace(0.5, 3.0, 40)
    yerr = np.full_like(xvar, 0.5)
    np.savetxt(tmp_path / 'data.txt',
               np.column_stack([xvar, 1.0 + 2.0 * xvar, yerr]))
    likelihood = GaussLikelihood('data.txt', 'variant', data_dir=str(tmp_path))

    theta = np.array([1.5, 2.5])
    got = match._variant_negloglike(
        likelihood, 'a0 + a1*x', theta, tmax=5, try_integration=False)
    a0s, a1s = sympy.symbols('a0 a1', real=True)
    eqn = sympy.lambdify([xsym, a0s, a1s], a0s + a1s * xsym, 'numpy')
    assert np.isclose(got, likelihood.negloglike(theta, eqn))
    # single-parameter form also works
    got1 = match._variant_negloglike(
        likelihood, 'a0*x', np.array([2.0]), tmax=5, try_integration=False)
    eqn1 = sympy.lambdify([xsym, a0s], a0s * xsym, 'numpy')
    assert np.isclose(got1, likelihood.negloglike([2.0], eqn1))


def test_likelihood_catalogue_retries_after_failed_transforms(tmp_path):
    """A cached catalogue with failed_count > 0 is not reused: the build (and its
    warning) is repeated so failures are not silently cached."""
    import sympy
    from esr.fitting.sympy_symbols import x
    from esr.fitting import test_all

    comp = 1
    compl_dir = tmp_path / 'functions' / f'compl_{comp}'
    compl_dir.mkdir(parents=True)
    funcs = ['a0*(a1 + x)', 'a2 + x']
    for name in (f'all_equations_{comp}.txt', f'unique_equations_{comp}.txt'):
        (compl_dir / name).write_text('\n'.join(funcs) + '\n')
    (compl_dir / f'matches_{comp}.txt').write_text(
        '\n'.join(str(i) for i in range(len(funcs))) + '\n')

    class SometimesFail:
        is_mse = False
        use_likelihood_catalogue = True
        catalogue_transform_version = 'v1'
        fn_dir = str(tmp_path / 'functions')
        base_out_dir = str(tmp_path / 'ob')
        out_dir = str(tmp_path / 'out')
        temp_dir = str(tmp_path / 'tmp')

        def run_sympify(self, fcn_i, **kwargs):
            if fcn_i.startswith('a2'):
                raise RuntimeError('deliberate failure')
            a0, a1, a2 = sympy.symbols('a0 a1 a2', real=True)
            eq = sympy.sympify(
                fcn_i, locals={'x': x, 'a0': a0, 'a1': a1, 'a2': a2})
            return fcn_i, sympy.cancel(eq / eq.subs(x, 1)), True

    lik = SometimesFail()
    with pytest.warns(test_all.LikelihoodCatalogueWarning, match='failed to'):
        test_all.ensure_likelihood_catalogue(comp, lik, tmax=5)
    assert test_all._read_likelihood_catalogue_metadata(
        comp, lik)['failed_count'] == 1
    # Second call must not reuse the failed cache: it warns again (rebuilds).
    with pytest.warns(test_all.LikelihoodCatalogueWarning, match='failed to'):
        test_all.ensure_likelihood_catalogue(comp, lik, tmax=5)


def test_projected_eigenbasis_warns_on_degenerate_hessian():
    """snap_choice=2 warns for (near-)degenerate Hessian eigenvalues, where the
    eigenbasis -- and hence the codelength -- is ambiguous."""
    from esr.fitting.test_all_Fisher import (
        _score_projected_eigenbasis, ProjectedEigenbasisWarning)

    with pytest.warns(ProjectedEigenbasisWarning, match='degenerate'):
        _score_projected_eigenbasis(
            np.diag([100.0, 100.0]), np.array([3.0, 5.0]), 7.0, True,
            lambda t: 7.0)


def test_transform_version_tuple_does_not_force_rebuild():
    """A tuple catalogue_transform_version survives the JSON round-trip, so the
    freshly-built settings still compare equal to the stored copy (otherwise the
    cache would be rebuilt on every call)."""
    import json
    from esr.fitting import test_all

    s = test_all._likelihood_catalogue_settings(5, False, 'h', ('a', 1), 'fp')
    assert s['cache_schema_version'] == 1  # bumped when the build moved
    #  from all_equations to the simplifier's unique equations
    assert s == json.loads(json.dumps(s))            # stable across load
    assert isinstance(s['transform_version'], list)  # tuple normalised to list


def test_likelihood_catalogue_activates_on_transformed_collision(tmp_path):
    """The catalogue activates when a transform collapses raw-distinct
    expressions onto the same transformed model (a dedup benefit), even with no
    parameter-layout change -- not only on layout changes."""
    import sympy
    from esr.fitting.sympy_symbols import x
    from esr.fitting import test_all

    comp = 1
    compl_dir = tmp_path / 'functions' / f'compl_{comp}'
    compl_dir.mkdir(parents=True)
    funcs = ['a0 + x', 'a0 - x', 'a0*x']
    for name in (f'all_equations_{comp}.txt', f'unique_equations_{comp}.txt'):
        (compl_dir / name).write_text('\n'.join(funcs) + '\n')
    (compl_dir / f'matches_{comp}.txt').write_text(
        '\n'.join(str(i) for i in range(len(funcs))) + '\n')

    class Merging:
        is_mse = False
        use_likelihood_catalogue = True
        catalogue_transform_version = 'v1'
        fn_dir = str(tmp_path / 'functions')
        base_out_dir = str(tmp_path / 'ob')
        out_dir = str(tmp_path / 'out')
        temp_dir = str(tmp_path / 'tmp')

        def run_sympify(self, fcn_i, **kwargs):
            a0, a1, a2 = sympy.symbols('a0 a1 a2', real=True)
            eq = sympy.sympify(
                fcn_i, locals={'x': x, 'a0': a0, 'a1': a1, 'a2': a2})
            # eq * eq(x -> -x) maps a0+x and a0-x both onto a0**2 - x**2.
            return fcn_i, sympy.expand(eq * eq.subs(x, -x)), False

    lik = Merging()
    assert test_all.ensure_likelihood_catalogue(comp, lik, tmax=5)
    md = test_all._read_likelihood_catalogue_metadata(comp, lik)
    assert md['changed_layout_count'] == 0            # no parameter removed/relabelled
    assert md['n_unique'] < md['raw_unique_count']    # but a transformed-model collision


def test_versionless_transform_is_not_cached(tmp_path, monkeypatch):
    """A transforming likelihood without catalogue_transform_version is rebuilt on
    every call (its cache is never reused) and warns, so a stale mapping cannot
    survive a transform change the probes miss."""
    import sympy
    from esr.fitting.sympy_symbols import x
    from esr.fitting import test_all

    comp = 1
    compl_dir = tmp_path / 'functions' / f'compl_{comp}'
    compl_dir.mkdir(parents=True)
    funcs = ['a0*(a1 + x)', 'a2 + x']
    for name in (f'all_equations_{comp}.txt', f'unique_equations_{comp}.txt'):
        (compl_dir / name).write_text('\n'.join(funcs) + '\n')
    (compl_dir / f'matches_{comp}.txt').write_text(
        '\n'.join(str(i) for i in range(len(funcs))) + '\n')

    class Versionless:
        is_mse = False
        use_likelihood_catalogue = True   # opt in, but no catalogue_transform_version
        fn_dir = str(tmp_path / 'functions')
        base_out_dir = str(tmp_path / 'ob')
        out_dir = str(tmp_path / 'out')
        temp_dir = str(tmp_path / 'tmp')

        def run_sympify(self, fcn_i, **kwargs):
            a0, a1, a2 = sympy.symbols('a0 a1 a2', real=True)
            eq = sympy.sympify(
                fcn_i, locals={'x': x, 'a0': a0, 'a1': a1, 'a2': a2})
            return fcn_i, sympy.cancel(eq / eq.subs(x, 1)), True

    builds = {'n': 0}
    original = test_all._transformed_keys_for_slice

    def counting(*args, **kwargs):
        builds['n'] += 1
        return original(*args, **kwargs)
    monkeypatch.setattr(test_all, '_transformed_keys_for_slice', counting)

    lik = Versionless()
    with pytest.warns(test_all.LikelihoodCatalogueWarning,
                      match='catalogue_transform_version'):
        test_all.ensure_likelihood_catalogue(comp, lik, tmax=5)
    test_all.ensure_likelihood_catalogue(comp, lik, tmax=5)  # nothing changed
    assert builds['n'] == 2   # rebuilt both times (cache not reused without a version)
