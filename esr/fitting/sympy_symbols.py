"""Script defining functions and symbols which can be used to interpret functions used in fitting functions"""

import sympy

# The variables, which are always +ve here
x, y = sympy.symbols("x y", positive=True)
a, b = sympy.symbols("a b", real=True)
# The constants, which can be -ve
a0, a1, a2, a3 = sympy.symbols("a0 a1 a2 a3", real=True)
sympy.init_printing(use_unicode=True)
inv = sympy.Lambda(a, 1 / a)
square = sympy.Lambda(a, a * a)
cube = sympy.Lambda(a, a * a * a)

sqrt = sympy.Lambda(a, sympy.sqrt(sympy.Abs(a, evaluate=False)))
log = sympy.Lambda(a, sympy.log(sympy.Abs(a, evaluate=False)))
pow = sympy.Lambda((a, b), sympy.Pow(sympy.Abs(a, evaluate=False), b))

pow_abs = sympy.Lambda((a, b), sympy.Pow(sympy.Abs(a, evaluate=False), b))
sqrt_abs = sympy.Lambda(a, sympy.sqrt(sympy.Abs(a, evaluate=False)))
log_abs = sympy.Lambda(a, sympy.log(sympy.Abs(a, evaluate=False)))

log10_abs = sympy.Lambda(a, sympy.log(sympy.Abs(a, evaluate=False), 10))
tenexp = sympy.Lambda(a, sympy.Pow(10, a))


sympy_locs = {
    "inv": inv,
    "square": square,
    "cube": cube,
    "pow": pow_abs,
    "Abs": sympy.Abs,
    "x": x,
    "sqrt_abs": sqrt_abs,
    "log_abs": log_abs,
    "log10_abs": log10_abs,
    "tenexp": tenexp,
}


# ---------------------------------------------------------------------------
# Input variables
# ---------------------------------------------------------------------------
#
# ESR was originally written for functions of a single variable ``x``. To
# search over functions of several variables, list every variable name in the
# nullary basis (``basis_functions[0]``), e.g. ``["x", "y", "a"]``, and pass
# the same names to the likelihood (``variables=["x", "y"]``). The data are
# then stored as ``likelihood.xvar`` with shape ``(nvar, ndata)`` and each
# compiled function is called as ``eq_numpy(xvar, *params)``, the first
# argument being unpacked into the individual variables. With the default
# ``["x"]`` everything behaves exactly as in the single-variable code.
#
# Like ``x``, every variable is a positive real sympy symbol. Symbols compare
# equal by (name, assumptions), so ``sympy.Symbol("y", positive=True)`` made
# anywhere matches the one used elsewhere (it need not be the same object).

DEFAULT_VARIABLES = ("x",)

# Names that would be mis-read as parameters, numbers or functions
_RESERVED_NAMES = set(sympy_locs.keys()) - {"x"}
_RESERVED_NAMES |= {
    "a",
    "e",
    "E",
    "I",
    "S",
    "N",
    "O",
    "Q",
    "pi",
    "oo",
    "zoo",
    "nan",
    "exp",
    "log",
    "sin",
    "cos",
    "tan",
    "pow",
    "sqrt",
    "lambda",
}


def validate_variables(variables):
    """Check a list of variable names is usable and return it as a tuple.

    Args:
        :variables (list or None): names of the input variables. None gives
            ``("x",)``

    Returns:
        :variables (tuple): the validated names

    Raises:
        ValueError: if a name is not a valid identifier, is reserved, could be
            confused with a parameter ``a<i>``, or is repeated
    """
    if variables is None:
        return DEFAULT_VARIABLES
    if isinstance(variables, str):
        variables = [variables]
    variables = tuple(variables)
    if len(variables) == 0:
        raise ValueError("At least one input variable is required")
    if len(set(variables)) != len(variables):
        raise ValueError(f"Repeated variable names in {variables}")
    for v in variables:
        if not isinstance(v, str) or not v.isidentifier():
            raise ValueError(f"Variable name {v!r} is not a valid identifier")
        if v in _RESERVED_NAMES:
            raise ValueError(f"Variable name {v!r} is reserved")
        # Parameters are found by searching function strings for "a<i>", and
        # labels starting with "a" followed by digits are treated as parameters
        if v.startswith("a") or any(
            v[i] == "a" and v[i + 1].isdigit() for i in range(len(v) - 1)
        ):
            raise ValueError(
                f"Variable name {v!r} could be confused with a parameter a<i>"
            )
    return variables


def get_variables(basis_functions):
    """Extract the input variable names from the nullary basis functions.

    Every nullary basis function other than the parameter ``"a"`` and numeric
    constants is an input variable.

    Args:
        :basis_functions (list): list of lists basis functions

    Returns:
        :variables (tuple): the variable names, in the order listed
    """
    out = []
    for f in basis_functions[0]:
        if f == "a":
            continue
        try:
            float(f)
            continue
        except (TypeError, ValueError):
            out.append(f)
    if len(out) == 0:
        return DEFAULT_VARIABLES
    return validate_variables(out)


def make_variables(variables=None):
    """Return the positive sympy symbols for a list of variable names.

    Args:
        :variables (list or None): names of the input variables

    Returns:
        :symbols (list): sympy symbols, one per variable
    """
    variables = validate_variables(variables)
    return [sympy.Symbol(v, positive=True) for v in variables]


def register_variables(variables=None, locs=None):
    """Add the variable symbols to a dictionary of sympify locals.

    Args:
        :variables (list or None): names of the input variables
        :locs (dict or None): dictionary to update. If None, the module-level
            ``sympy_locs`` (used throughout generation) is updated

    Returns:
        :locs (dict): the updated dictionary
    """
    if locs is None:
        locs = sympy_locs
    for v, s in zip(validate_variables(variables), make_variables(variables)):
        locs[v] = s
    return locs


def variable_arg(variables=None):
    """First argument to pass to ``sympy.lambdify`` for these variables.

    For a single variable this is just its symbol, so the compiled function is
    called as ``f(xvar, *params)`` with a 1D ``xvar``. For several variables it
    is a tuple of symbols, so the compiled function unpacks a first argument of
    shape ``(nvar, ndata)``.

    Args:
        :variables (list or None): names of the input variables

    Returns:
        :arg (sympy.Symbol or tuple): the lambdify argument
    """
    syms = make_variables(variables)
    if len(syms) == 1:
        return syms[0]
    return tuple(syms)


def function_set_dirname(fn_set, variables=None):
    """Name of the catalogue directory for a function set and variables.

    Single-variable catalogues keep their historical name ``fn_set`` so
    existing (and downloaded) libraries are still found. Other choices of
    variables get ``fn_set`` followed by the variable names, e.g.
    ``core_maths_x_y``, so catalogues for different variables never clash.

    Args:
        :fn_set (str): name of the function set
        :variables (list or None): names of the input variables

    Returns:
        :name (str): the directory name
    """
    variables = validate_variables(variables)
    if variables == DEFAULT_VARIABLES:
        return fn_set
    return fn_set + "_" + "_".join(variables)


def likelihood_xarg(likelihood):
    """The ``sympy.lambdify`` variable argument for a likelihood object.

    Uses ``likelihood.variables`` if set (likelihoods written before
    multivariable support, which do not call ``Likelihood.__init__``, fall
    back to the single variable ``x``).

    Args:
        :likelihood (fitting.likelihood object): the likelihood

    Returns:
        :arg (sympy.Symbol or tuple): see ``variable_arg``
    """
    return variable_arg(getattr(likelihood, "variables", None))
