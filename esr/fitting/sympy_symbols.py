"""Script defining functions and symbols which can be used to interpret functions used in fitting functions
"""

import sympy

x, y = sympy.symbols('x y', positive=True)                  # The variables, which are always +ve here
a, b = sympy.symbols('a b', real=True)
a0, a1, a2, a3 = sympy.symbols('a0 a1 a2 a3', real=True)    # The constants, which can be -ve
sympy.init_printing(use_unicode=True)
inv = sympy.Lambda(a, 1/a)
square = sympy.Lambda(a, a*a)
cube = sympy.Lambda(a, a*a*a)

sqrt = sympy.Lambda(a, sympy.sqrt(sympy.Abs(a, evaluate=False)))
log = sympy.Lambda(a, sympy.log(sympy.Abs(a, evaluate=False)))
pow = sympy.Lambda((a,b), sympy.Pow(sympy.Abs(a, evaluate=False), b))

pow_abs = sympy.Lambda((a,b), sympy.Pow(sympy.Abs(a, evaluate=False), b))
sqrt_abs = sympy.Lambda(a, sympy.sqrt(sympy.Abs(a, evaluate=False)))
log_abs = sympy.Lambda(a, sympy.log(sympy.Abs(a, evaluate=False)))

log10_abs = sympy.Lambda(a, sympy.log(sympy.Abs(a, evaluate=False), 10))
tenexp = sympy.Lambda(a, sympy.Pow(10, a))


sympy_locs = {"inv": inv,
            "square": square,
            "cube": cube,
            "pow": pow_abs,
            "Abs": sympy.Abs,
            "x":x,
            "sqrt_abs":sqrt_abs,
            "log_abs":log_abs,
            "log10_abs":log10_abs,
            "tenexp":tenexp
            }
