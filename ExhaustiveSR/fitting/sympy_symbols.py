"""Script defining functions and symbols which can be used to interpret functions used in fitting functions
"""

import sympy

x, y = sympy.symbols('x y', positive=True)                  # The variables, which are always +ve here
a0, a1, a2, a3 = sympy.symbols('a0 a1 a2 a3', real=True)    # The constants, which can be -ve
sympy.init_printing(use_unicode=True)
inv = sympy.Lambda(x, 1/x)
square = sympy.Lambda(x, x*x)
cube = sympy.Lambda(x, x*x*x)
sqrt = sympy.Lambda(x, sympy.sqrt(sympy.Abs(x, evaluate=False)))
log = sympy.Lambda(x, sympy.log(sympy.Abs(x, evaluate=False)))
pow = sympy.Lambda((x,y), sympy.Pow(sympy.Abs(x, evaluate=False), y))
