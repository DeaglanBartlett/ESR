"""Tests for running ESR on functions of several input variables."""

import os

import numpy as np
import pytest
import sympy

import esr.fitting.combine_dl
import esr.fitting.match
import esr.fitting.test_all
import esr.fitting.test_all_fisher
import esr.generation.duplicate_checker
from esr.fitting import sympy_symbols
from esr.fitting.fit_single import fit_from_string
from esr.fitting.likelihood import GaussLikelihood, PoissonLikelihood
from esr.generation import generator, simplifier
from esr.generation.duplicate_checker import _apply_variables


def test_validate_variables():
    assert sympy_symbols.validate_variables(None) == ("x",)
    assert sympy_symbols.validate_variables("t") == ("t",)
    assert sympy_symbols.validate_variables(["x", "y", "z1"]) == ("x", "y", "z1")
    for bad in [[], ["x", "x"], ["a1"], ["xa0"], ["alpha"], ["pow"], ["x y"], [1]]:
        with pytest.raises(ValueError):
            sympy_symbols.validate_variables(bad)


def test_variable_helpers():
    basis = [["x", "y", "a", "2"], ["inv"], ["+", "*"]]
    assert sympy_symbols.get_variables(basis) == ("x", "y")
    assert sympy_symbols.get_variables([["a"], [], []]) == ("x",)

    x, y = sympy_symbols.make_variables(["x", "y"])
    assert x == sympy_symbols.x
    assert y.is_positive

    assert sympy_symbols.variable_arg(None) == sympy_symbols.x
    assert sympy_symbols.variable_arg(["x", "y"]) == (x, y)

    assert sympy_symbols.function_set_dirname("core_maths") == "core_maths"
    assert sympy_symbols.function_set_dirname("core_maths", ["x"]) == "core_maths"
    assert (
        sympy_symbols.function_set_dirname("core_maths", ["x", "y"]) == "core_maths_x_y"
    )


def test_lambdify_unpacks_variables():
    x, y = sympy_symbols.make_variables(["x", "y"])
    a0 = sympy_symbols.a0
    f = sympy.lambdify(
        [sympy_symbols.variable_arg(["x", "y"]), a0], a0 * x + y, modules=["numpy"]
    )
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert np.allclose(f(X, 2.0), [5.0, 8.0])


def test_apply_variables():
    builtin = [["x", "a"], ["inv"], ["+"]]
    basis, variables = _apply_variables(builtin, ["x", "y"])
    assert basis[0] == ["x", "y", "a"]
    assert variables == ("x", "y")
    # Input is not modified
    assert builtin[0] == ["x", "a"]

    basis, variables = _apply_variables(builtin, None)
    assert basis[0] == ["x", "a"] and variables == ("x",)

    custom = [["u", "v", "a"], ["inv"], ["+"]]
    basis, variables = _apply_variables(custom, None, custom=True)
    assert variables == ("u", "v")
    with pytest.raises(ValueError):
        _apply_variables(custom, ["x", "y"], custom=True)


def test_check_operators_keeps_variables():
    basis = [["x", "y", "a"], ["inv"], ["+", "*", "-", "/", "pow"]]
    sympy_symbols.register_variables(["x", "y"])
    _, nodes, c = generator.string_to_node("a0*x + y", basis, check_ops=True)
    assert c == 5
    assert generator.check_operators(nodes, basis)
    labels = generator.labels_to_shape(["+", "*", "a0", "x", "y"], basis)
    assert labels == [2, 2, 0, 0, 0]


def test_numerical_fingerprint_with_several_variables():
    x, y = sympy_symbols.make_variables(["x", "y"])
    a0 = sympy_symbols.a0
    fp = simplifier.numerical_fingerprint
    assert fp(x * y + a0) == fp(y * x + a0)
    assert fp(x - y) is not None
    assert fp(x - y) != fp(sympy.Integer(0))
    assert fp(x + 2 * y) != fp(2 * x + y)
    # Single-variable fingerprints are unchanged by adding variable support
    assert np.allclose(
        fp(x * a0), simplifier._FPRINT_X_POINTS * simplifier._FPRINT_PARAM_POINTS["a0"]
    )


def _write_data(path, nvar_cols, extra_cols):
    rng = np.random.default_rng(1)
    cols = [rng.uniform(0.5, 3.0, 40) for _ in range(nvar_cols)]
    np.savetxt(path, np.column_stack(cols + extra_cols(cols)))


def test_likelihood_loads_variable_columns(tmp_path):
    _write_data(
        tmp_path / "data.txt",
        2,
        lambda c: [c[0] * c[1], 0.1 * np.ones_like(c[0])],
    )
    like = GaussLikelihood(
        "data.txt",
        "mv",
        data_dir=str(tmp_path),
        base_out_dir=str(tmp_path / "out"),
        variables=["x", "y"],
    )
    assert like.variables == ("x", "y")
    assert like.xvar.shape == (2, 40)
    assert like.yvar.shape == (40,)
    assert like.fn_dir.rstrip("/").endswith("function_library/core_maths_x_y")

    # The single-variable default is unchanged
    with pytest.raises(ValueError, match="columns"):
        GaussLikelihood("data.txt", "sv", data_dir=str(tmp_path))
    with pytest.raises(ValueError, match="columns"):
        PoissonLikelihood("data.txt", "pv", data_dir=str(tmp_path), variables="x")

    x, y = sympy_symbols.make_variables(["x", "y"])
    _, eq2, _ = like.run_sympify("a0*x*y")
    assert eq2 == sympy_symbols.a0 * x * y
    f = sympy.lambdify(
        [sympy_symbols.likelihood_xarg(like), sympy_symbols.a0], eq2, modules=["numpy"]
    )
    assert np.isfinite(like.negloglike([1.0], f))
    assert like.negloglike([1.0], f) < like.negloglike([2.0], f)


def test_fit_from_string_multivariable(tmp_path):
    _write_data(
        tmp_path / "data.txt",
        2,
        lambda c: [2.0 * c[0] * c[1], 0.1 * np.ones_like(c[0])],
    )
    like = GaussLikelihood(
        "data.txt",
        "mv",
        data_dir=str(tmp_path),
        base_out_dir=str(tmp_path / "out"),
        variables=["x", "y"],
    )
    basis = [["x", "y", "a"], ["inv"], ["+", "*", "-", "/", "pow"]]
    _nll, DL, labels, params = fit_from_string("a0*x*y", basis, like, return_params=True)
    assert "y" in labels and "x" in labels
    assert np.isclose(params[0], 2.0, rtol=1e-3)
    assert np.isfinite(DL)


def test_multivariable_pipeline(tmp_path, monkeypatch):
    """Generate a two-variable catalogue and recover y = x*y from mock data."""
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(0)
    xv = rng.uniform(0.5, 3.0, 50)
    yv = rng.uniform(0.5, 3.0, 50)
    err = 0.05 * np.ones(50)
    np.savetxt(
        tmp_path / "data.txt",
        np.c_[xv, yv, xv * yv + err * rng.standard_normal(50), err],
    )

    fn_dir = str(tmp_path / "lib")
    like = GaussLikelihood(
        "data.txt",
        "mv",
        data_dir=str(tmp_path),
        fn_dir=fn_dir,
        base_out_dir=str(tmp_path / "out"),
        variables=["x", "y"],
    )
    for comp in range(1, 4):
        esr.generation.duplicate_checker.main(
            "core_maths", comp, fn_dir=fn_dir, variables=["x", "y"]
        )
        esr.fitting.test_all.main(comp, like)
        esr.fitting.test_all_fisher.main(comp, like)
        esr.fitting.match.main(comp, like)
        esr.fitting.combine_dl.main(comp, like)

    with open(os.path.join(fn_dir, "compl_1", "unique_equations_1.txt")) as f:
        assert sorted(f.read().split()) == ["a0", "x", "y"]

    with open(os.path.join(like.out_dir, "final_3.dat")) as f:
        best = f.readline().split(";")[1]
    assert best == "x*y"
