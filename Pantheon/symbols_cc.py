import sympy
import sys
from filenames import *
sys.path.insert(0, esr_dir)
from simplifier import time_limit

x, y = sympy.symbols('x y', positive=True)                  # The variables, which are always +ve here
a0, a1, a2, a3 = sympy.symbols('a0 a1 a2 a3', real=True)    # The constants, which can be -ve
sympy.init_printing(use_unicode=True)
inv = sympy.Lambda(x, 1/x)
square = sympy.Lambda(x, x*x)
cube = sympy.Lambda(x, x*x*x)
sqrt = sympy.Lambda(x, sympy.sqrt(sympy.Abs(x, evaluate=False)))
log = sympy.Lambda(x, sympy.log(sympy.Abs(x, evaluate=False)))
pow = sympy.Lambda((x,y), sympy.Pow(sympy.Abs(x, evaluate=False), y))

def run_sympify(fcn_i, **kwargs):
    fcn_i = fcn_i.replace('\n', '')
    fcn_i = fcn_i.replace('\'', '')
    
    eq = sympy.sympify(fcn_i,
                    locals={"inv": inv,
                            "square": square,
                            "cube": cube,
                            "sqrt": sqrt,
                            "log": log,
                            "pow": pow,
                            "x": x,
                            "a0": a0,
                            "a1": a1,
                            "a2": a2})
    return fcn_i, eq, False
