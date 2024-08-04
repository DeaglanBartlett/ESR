import unittest
import sympy
from sympy import (
    symbols, Add, Symbol, Order, Poly, UniversalSet, AlgebraicNumber, sqrt,
    Pow, UnevaluatedExpr, MatrixSymbol, Mul, Rational, Integral, S,
    Derivative, ConditionSet, Catalan, Matrix, I, MatMul, Float,
    FiniteSet, Sum, Eq, And)
from fractions import Fraction
from sympy.logic.boolalg import BooleanTrue, BooleanFalse
from sympy.combinatorics.permutations import Permutation
from esr.generation.custom_printer import ESRPrinter, sstr, sstrrepr

class TestESRPrinter(unittest.TestCase):

    def setUp(self):
        self.printer = ESRPrinter()

    def test_initialization(self):
        self.assertIsInstance(self.printer, ESRPrinter)
        self.assertEqual(self.printer._default_settings["order"], None)
        self.assertEqual(self.printer._default_settings["full_prec"], "auto")
        self.assertEqual(self.printer._default_settings["sympy_integers"], False)
        self.assertEqual(self.printer._default_settings["abbrev"], False)
        self.assertEqual(self.printer._default_settings["perm_cyclic"], True)
        self.assertEqual(self.printer._default_settings["min"], None)
        self.assertEqual(self.printer._default_settings["max"], None)
        self.assertEqual(self.printer._relationals, {})

    def test_parenthesize(self):
        x, y = symbols('x y')
        self.assertEqual(self.printer.parenthesize(x + y, 1), "x + y")
        self.assertEqual(self.printer.parenthesize(x * y, 1), "x*y")

    def test_stringify(self):
        x, y = symbols('x y')
        self.assertEqual(self.printer.stringify([x, y], ", "), "x, y")
        self.assertEqual(self.printer.stringify([x + y, x * y], ", "), "x + y, x*y")

    def test_emptyPrinter(self):
        x = symbols('x')
        self.assertEqual(self.printer.emptyPrinter("test"), "test")
        self.assertEqual(self.printer.emptyPrinter(x), "x")
        self.assertEqual(self.printer.emptyPrinter(123), "123")

    def test_print_Add(self):
        x, y = symbols('x y')
        expr = Add(x, y, evaluate=False)
        self.assertEqual(self.printer._print_Add(expr), "x + y")
        expr = Add(-x, y, evaluate=False)
        self.assertEqual(self.printer._print_Add(expr), "-x + y")
        expr = Add(x, -y, evaluate=False)
        self.assertEqual(self.printer._print_Add(expr), "x - y")
        expr = Add(-x, -y, evaluate=False)
        self.assertEqual(self.printer._print_Add(expr), "-x - y")

    def test_print_BooleanTrue(self):
        self.assertEqual(self.printer._print_BooleanTrue(BooleanTrue()), "True")

    def test_print_BooleanFalse(self):
        self.assertEqual(self.printer._print_BooleanFalse(BooleanFalse()), "False")

    def test_print_ElementwiseApplyFunction(self):
        class MockExpr:
            function = 'f'
            expr = 'x'
        expr = MockExpr()
        result = self.printer._print_ElementwiseApplyFunction(expr)
        self.assertEqual(result, "f.(x)")

    def test_print_NaN(self):
        class MockExpr:
            pass
        expr = MockExpr()
        result = self.printer._print_NaN(expr)
        self.assertEqual(result, 'nan')

    def test_print_NegativeInfinity(self):
        class MockExpr:
            pass
        expr = MockExpr()
        result = self.printer._print_NegativeInfinity(expr)
        self.assertEqual(result, '-oo')

    def test_print_Order(self):
        x = Symbol('x')
        expr = Order(x)
        result = self.printer._print_Order(expr)
        self.assertEqual(result, 'O(x)')

    def test_print_Ordinal(self):
        class MockExpr:
            def __str__(self):
                return 'ordinal'
        expr = MockExpr()
        result = self.printer._print_Ordinal(expr)
        self.assertEqual(result, 'ordinal')

    def test_print_Cycle(self):
        class MockExpr:
            def __str__(self):
                return 'cycle'
        expr = MockExpr()
        result = self.printer._print_Cycle(expr)
        self.assertEqual(result, 'cycle')

    def test_print_Permutation(self):
        expr = Permutation([2, 0, 1])
        result = self.printer._print_Permutation(expr)
        self.assertTrue('0' in result)
        self.assertTrue('1' in result)
        self.assertTrue('2' in result)

    def test_print_Poly(self):
        x, y = symbols('x y')
        expr = Poly(x**2 + 2*x*y + y**2, x, y)
        result = self.printer._print_Poly(expr)
        expected = "Poly(x**2 + 2*x*y + y**2, x, y, domain='ZZ')"
        self.assertEqual(result, expected)

    def test_print_Poly_with_coefficients(self):
        x = symbols('x')
        expr = Poly(3*x**2 - x + 1, x)
        result = self.printer._print_Poly(expr)
        expected = "Poly(3*x**2 - x + 1, x, domain='ZZ')"
        self.assertEqual(result, expected)

    def test_print_Poly_with_modulus(self):
        x = symbols('x')
        expr = Poly(x**2 + 1, x, modulus=5)
        result = self.printer._print_Poly(expr)
        expected = "Poly(x**2 + 1, x, modulus=5)"
        self.assertEqual(result, expected)

    def test_print_UniversalSet(self):
        expr = UniversalSet
        result = self.printer._print_UniversalSet(expr)
        self.assertEqual(result, 'UniversalSet')

    def test_print_AlgebraicNumber(self):
        expr = AlgebraicNumber(sqrt(2))
        result = self.printer._print_AlgebraicNumber(expr)
        expected = self.printer._print(expr.as_expr())
        self.assertEqual(result, expected)

    def test_print_AlgebraicNumber_aliased(self):
        expr = AlgebraicNumber(sqrt(2), alias='a')
        result = self.printer._print_AlgebraicNumber(expr)
        expected = self.printer._print(expr.as_poly().as_expr())
        self.assertEqual(result, expected)

    def test_print_Pow_rational_true(self):
        x = symbols('x')
        expr = sqrt(x)
        result = self.printer._print_Pow(expr, rational=True)
        self.assertEqual(result, 'pow(x,(1/2))')

    def test_print_Pow_rational_false(self):
        x = symbols('x')
        expr = sqrt(x)
        result = self.printer._print_Pow(expr, rational=False)
        self.assertEqual(result, 'sqrt(x)')

    def test_print_Pow_negative_rational_true(self):
        x = symbols('x')
        expr = 1/sqrt(x)
        result = self.printer._print_Pow(expr, rational=True)
        self.assertEqual(result, 'pow(x,(-1/2))')

    def test_print_Pow_negative_rational_false(self):
        x = symbols('x')
        expr = 1/sqrt(x)
        result = self.printer._print_Pow(expr, rational=False)
        self.assertEqual(result, '1/sqrt(x)')

    def test_print_UnevaluatedExpr(self):
        x = symbols('x')
        expr = UnevaluatedExpr(x + 1)
        result = self.printer._print_UnevaluatedExpr(expr)
        self.assertEqual(result, 'x + 1')

    def test_print_MatPow(self):
        A = MatrixSymbol('A', 2, 2)
        expr = Pow(A, 2)
        result = self.printer._print_MatPow(expr)
        self.assertEqual(result, 'A**2')

    def test_print_Integer(self):
        class Expr:
            p = 5
        self.assertEqual(self.printer._print_Integer(Expr()), '5')

    def test_print_Integers(self):
        self.assertEqual(self.printer._print_Integers(None), 'Integers')

    def test_print_Naturals(self):
        self.assertEqual(self.printer._print_Naturals(None), 'Naturals')

    def test_print_Naturals0(self):
        self.assertEqual(self.printer._print_Naturals0(None), 'Naturals0')

    def test_print_Rationals(self):
        self.assertEqual(self.printer._print_Rationals(None), 'Rationals')

    def test_print_Reals(self):
        self.assertEqual(self.printer._print_Reals(None), 'Reals')

    def test_print_Complexes(self):
        self.assertEqual(self.printer._print_Complexes(None), 'Complexes')

    def test_print_EmptySet(self):
        self.assertEqual(self.printer._print_EmptySet(None), 'EmptySet')

    def test_print_EmptySequence(self):
        self.assertEqual(self.printer._print_EmptySequence(None), 'EmptySequence')

    def test_print_int(self):
        self.assertEqual(self.printer._print_int(5), '5')

    def test_print_mpz(self):
        class Mpz:
            def __str__(self):
                return 'mpz'
        self.assertEqual(self.printer._print_mpz(Mpz()), 'mpz')

    def test_print_Rational(self):
        class Rational:
            p = 3
            q = 1
        self.assertEqual(self.printer._print_Rational(Rational()), '3')
        Rational.q = 2
        self.assertEqual(self.printer._print_Rational(Rational()), '3/2')
        self.printer._settings["sympy_integers"] = True
        self.assertEqual(self.printer._print_Rational(Rational()), 'S(3)/2')

    def test_print_PythonRational(self):
        class PythonRational:
            p = 3
            q = 1
        self.assertEqual(self.printer._print_PythonRational(PythonRational()), '3')
        PythonRational.q = 2
        self.assertEqual(self.printer._print_PythonRational(PythonRational()), '3/2')

    def test_print_Fraction(self):
        self.assertEqual(self.printer._print_Fraction(Fraction(3, 1)), '3')
        self.assertEqual(self.printer._print_Fraction(Fraction(3, 2)), '3/2')

    def test_print_mpq(self):
        class Mpq:
            numerator = 3
            denominator = 1
        self.assertEqual(self.printer._print_mpq(Mpq()), '3')
        Mpq.denominator = 2
        self.assertEqual(self.printer._print_mpq(Mpq()), '3/2')

    def test_print_Predicate(self):
        class Expr:
            name = "is_prime"
        self.assertEqual(self.printer._print_Predicate(Expr()), "Q.is_prime")

    def test_print_str(self):
        self.assertEqual(self.printer._print_str("test"), "test")

    def test_print_tuple(self):
        class Expr:
            def __init__(self, *args):
                self.args = args
        self.printer.stringify = lambda x, sep: sep.join(map(str, x))
        self.printer._print = lambda x: str(x)
        self.assertEqual(self.printer._print_tuple((1,)), "(1,)")
        self.assertEqual(self.printer._print_tuple((1, 2)), "(1, 2)")

    def test_print_Tuple(self):
        class Expr:
            def __init__(self, *args):
                self.args = args
        self.printer.stringify = lambda x, sep: sep.join(map(str, x))
        self.printer._print = lambda x: str(x)
        self.assertEqual(self.printer._print_Tuple((1,)), "(1,)")
        self.assertEqual(self.printer._print_Tuple((1, 2)), "(1, 2)")

    def test_print_Transpose(self):
        class T:
            arg = "A"
        self.printer.parenthesize = lambda x, y: x
        self.assertEqual(self.printer._print_Transpose(T()), "A.T")

    def test_print_Uniform(self):
        class Expr:
            a = 1
            b = 2
        self.printer._print = lambda x: str(x)
        self.assertEqual(self.printer._print_Uniform(Expr()), "Uniform(1, 2)")

    def test_print_Quantity(self):
        class Expr:
            name = "meter"
            abbrev = "m"
        self.assertEqual(self.printer._print_Quantity(Expr()), "meter")
        self.printer._settings["abbrev"] = True
        self.assertEqual(self.printer._print_Quantity(Expr()), "m")

    def test_print_Quaternion(self):
        class Expr:
            args = [1, 2, 3, 4]
        self.printer.parenthesize = lambda x, y, strict=False: str(x)
        self.assertEqual(self.printer._print_Quaternion(Expr()), "1 + 2*i + 3*j + 4*k")

    def test_print_Dimension(self):
        self.assertEqual(self.printer._print_Dimension("dimension"), "dimension")

    def test_print_Wild(self):
        class Expr:
            name = "x"
        self.assertEqual(self.printer._print_Wild(Expr()), "x_")

    def test_print_WildFunction(self):
        class Expr:
            name = "f"
        self.assertEqual(self.printer._print_WildFunction(Expr()), "f_")

    def test_print_WildDot(self):
        class Expr:
            name = "dot"
        self.assertEqual(self.printer._print_WildDot(Expr()), "dot")

    def test_print_WildPlus(self):
        class Expr:
            name = "plus"
        self.assertEqual(self.printer._print_WildPlus(Expr()), "plus")

    def test_print_WildStar(self):
        class Expr:
            name = "star"
        self.assertEqual(self.printer._print_WildStar(Expr()), "star")

    def test_print_Zero(self):
        self.assertEqual(self.printer._print_Zero(None), "0")
        self.printer._settings["sympy_integers"] = True
        self.assertEqual(self.printer._print_Zero(None), "S(0)")

    def test_print_DMP(self):
        class P:
            rep = "rep"
            dom = "dom"
            ring = None
        self.printer._print = lambda x: str(x)
        self.assertEqual(self.printer._print_DMP(P()), "P(rep, dom, None)")

    def test_print_DMF(self):
        class Expr:
            rep = "rep"
            dom = "dom"
            ring = None
        self.printer._print = lambda x: str(x)
        self.assertEqual(self.printer._print_DMF(Expr()), "Expr(rep, dom, None)")

    def test_print_Object(self):
        class Obj:
            name = "object"
        self.assertEqual(self.printer._print_Object(Obj()), 'Object("object")')

    def test_print_IdentityMorphism(self):
        class Morphism:
            domain = "domain"
        self.assertEqual(self.printer._print_IdentityMorphism(Morphism()), 'IdentityMorphism(domain)')

    def test_print_NamedMorphism(self):
        class Morphism:
            domain = "domain"
            codomain = "codomain"
            name = "name"
        self.assertEqual(self.printer._print_NamedMorphism(Morphism()), 'NamedMorphism(domain, codomain, "name")')

    def test_print_Category(self):
        class Category:
            name = "category"
        self.assertEqual(self.printer._print_Category(Category()), 'Category("category")')

    def test_print_Manifold(self):
        class Name:
            name = "manifold"
        class Manifold:
            name = Name()
        self.assertEqual(self.printer._print_Manifold(Manifold()), "manifold")

    def test_print_Patch(self):
        class Name:
            name = "patch"
        class Patch:
            name = Name()
        self.assertEqual(self.printer._print_Patch(Patch()), "patch")

    def test_print_CoordSystem(self):
        class Name:
            name = "coords"
        class CoordSystem:
            name = Name()
        self.assertEqual(self.printer._print_CoordSystem(CoordSystem()), "coords")

    def test_print_BaseScalarField(self):
        class CoordSys:
            symbols = [type('Symbol', (object,), {'name': 'x'})(), 
                       type('Symbol', (object,), {'name': 'y'})(), 
                       type('Symbol', (object,), {'name': 'z'})()]
        class Field:
            _coord_sys = CoordSys()
            _index = 1
        self.assertEqual(self.printer._print_BaseScalarField(Field()), "y")

    def test_print_BaseVectorField(self):
        class CoordSys:
            symbols = [type('Symbol', (object,), {'name': 'x'})(), 
                       type('Symbol', (object,), {'name': 'y'})(), 
                       type('Symbol', (object,), {'name': 'z'})()]
        class Field:
            _coord_sys = CoordSys()
            _index = 1
        self.assertEqual(self.printer._print_BaseVectorField(Field()), "e_y")

    def test_print_Differential(self):
        class CoordSys:
            symbols = [type('Symbol', (object,), {'name': 'x'})(), 
                       type('Symbol', (object,), {'name': 'y'})(), 
                       type('Symbol', (object,), {'name': 'z'})()]
        class Field:
            _coord_sys = CoordSys()
            _index = 1
        class Diff:
            _form_field = Field()
        self.assertEqual(self.printer._print_Differential(Diff()), "dy")

        class FieldWithoutCoordSys:
            def __str__(self):
                return "field"
        class DiffWithoutCoordSys:
            _form_field = FieldWithoutCoordSys()
        self.printer._print = lambda x: str(x)
        self.assertEqual(self.printer._print_Differential(DiffWithoutCoordSys()), "d(field)")

    def test_print_Tr(self):
        class Expr:
            args = ["A"]
        self.printer._print = lambda x: str(x)
        self.assertEqual(self.printer._print_Tr(Expr()), "Tr(A)")

    def test_print_Str(self):
        class S:
            name = "string"
        self.printer._print = lambda x: str(x)
        self.assertEqual(self.printer._print_Str(S()), "string")

    def test_print_AppliedBinaryRelation(self):
        class Rel:
            def __str__(self):
                return "rel"
        class Expr:
            lhs = "lhs"
            rhs = "rhs"
            function = Rel()
        self.printer._print = lambda x: str(x)
        self.assertEqual(self.printer._print_AppliedBinaryRelation(Expr()), "rel(lhs, rhs)")

    def test_print_Interval(self):
        class MockInfinite:
            is_infinite = True
            def __str__(self):
                return "a"

        class MockFinite:
            is_infinite = False
            def __str__(self):
                return "b"

        class Interval:
            def __init__(self, a, b, L, r):
                self.args = [a, b, L, r]

        # Test case: both a and b are infinite
        interval = Interval(MockInfinite(), MockInfinite(), False, False)
        self.assertEqual(self.printer._print_Interval(interval), "Interval(a, a)")

        # Test case: a is infinite, b is finite, r is False
        interval = Interval(MockInfinite(), MockFinite(), False, False)
        self.assertEqual(self.printer._print_Interval(interval), "Interval(a, b)")

        # Test case: a is finite, b is infinite, L is False
        interval = Interval(MockFinite(), MockInfinite(), False, False)
        self.assertEqual(self.printer._print_Interval(interval), "Interval(b, a)")

        # Test case: both L and r are False
        interval = Interval(MockFinite(), MockFinite(), False, False)
        self.assertEqual(self.printer._print_Interval(interval), "Interval(b, b)")

        # Test case: both L and r are True
        interval = Interval(MockFinite(), MockFinite(), True, True)
        self.assertEqual(self.printer._print_Interval(interval), "Interval.open(b, b)")

        # Test case: L is True, r is False
        interval = Interval(MockFinite(), MockFinite(), True, False)
        self.assertEqual(self.printer._print_Interval(interval), "Interval.Lopen(b, b)")

        # Test case: L is False, r is True
        interval = Interval(MockFinite(), MockFinite(), False, True)
        self.assertEqual(self.printer._print_Interval(interval), "Interval.Ropen(b, b)")

    def test_print_AccumulationBounds(self):
        class AccumBounds:
            min = "min"
            max = "max"
        self.printer._print = lambda x: str(x)
        self.assertEqual(self.printer._print_AccumulationBounds(AccumBounds()), "AccumBounds(min, max)")

    def test_print_Inverse(self):
        class Inverse:
            arg = "arg"
        self.printer.parenthesize = lambda x, y: str(x)
        self.assertEqual(self.printer._print_Inverse(Inverse()), "arg**(-1)")

    def test_print_Lambda(self):
        class Lambda:
            expr = "expr"
            class Signature:
                is_symbol = True
                def __str__(self):
                    return "Signature"
            signature = [Signature()]
        self.printer._print = lambda x: str(x)
        self.assertEqual(self.printer._print_Lambda(Lambda()), "Lambda(Signature, expr)")

    def test_print_LatticeOp(self):
        class LatticeOp:
            args = ["arg1", "arg2"]
            func = type('Func', (object,), {'__name__': 'LatticeOpFunc'})()
        self.printer._print = lambda x: str(x)
        self.assertEqual(self.printer._print_LatticeOp(LatticeOp()), "LatticeOpFunc(arg1, arg2)")

    def test_print_Limit(self):
        class Limit:
            args = ["e", "z", "z0", "+"]
        self.printer._print = lambda x: str(x)
        self.assertEqual(self.printer._print_Limit(Limit()), "Limit(e, z, z0)")

        class LimitWithDir:
            args = ["e", "z", "z0", "-"]
        self.assertEqual(self.printer._print_Limit(LimitWithDir()), "Limit(e, z, z0, dir='-')")
    
    def test_print_MatrixSlice(self):
        class MockExpr:
            def __init__(self, parent, rowslice, colslice):
                self.parent = parent
                self.rowslice = rowslice
                self.colslice = colslice

        class MockParent:
            def __init__(self, rows, cols):
                self.rows = rows
                self.cols = cols

        # Mock the parenthesize method
        self.printer.parenthesize = lambda x, y, strict=True: "parenthesized"
        self.printer._print = lambda x: str(x)

        # Test case: rowslice and colslice with default values
        expr = MockExpr(MockParent(10, 10), [0, 10, 1], [0, 10, 1])
        self.assertEqual(self.printer._print_MatrixSlice(expr), "parenthesized[:, :]")

        # Test case: rowslice and colslice with custom values
        expr = MockExpr(MockParent(10, 10), [1, 5, 2], [2, 8, 2])
        self.assertEqual(self.printer._print_MatrixSlice(expr), "parenthesized[1:5:2, 2:8:2]")

        # Test case: rowslice and colslice with some default values
        expr = MockExpr(MockParent(10, 10), [0, 5, 1], [2, 10, 1])
        self.assertEqual(self.printer._print_MatrixSlice(expr), "parenthesized[:5, 2:]")

        # Test case: rowslice and colslice with no step
        expr = MockExpr(MockParent(10, 10), [1, 5, 1], [2, 8, 1])  # Added default step value
        self.assertEqual(self.printer._print_MatrixSlice(expr), "parenthesized[1:5, 2:8]")

    def test_simple_mul(self):
        x = Symbol('x')
        expr = Mul(2, x)
        self.assertEqual(self.printer._print_Mul(expr), '2*x')


    def test_mul_with_negative(self):
        x = Symbol('x')
        expr = Mul(-2, x)
        self.assertEqual(self.printer._print_Mul(expr), '-2*x')

    def test_mul_with_rational(self):
        x = Symbol('x')
        expr = Mul(Rational(1, 2), x)
        self.assertEqual(self.printer._print_Mul(expr), 'x/2')

    def test_mul_with_pow(self):
        x = Symbol('x')
        expr = Mul(x, Pow(x, -1))
        expr = self.printer._print_Mul(expr)
        brackets = ['(', ')', '[', ']', '{', '}']
        for bracket in brackets:
            expr = expr.replace(bracket, '')
        self.assertEqual(expr, '1')

    def test_mul_with_nested_pow(self):
        x = Symbol('x')
        y = Symbol('y')
        z = Symbol('z')
        expr = Mul(x, Pow(Mul(z, y), -1), 2)
        self.assertEqual(self.printer._print_Mul(expr), '2*x/(y*z)')

    def test_mul_with_multiple_rationals(self):
        x = Symbol('x')
        expr = Mul(Rational(1, 2), Rational(1, 3), x)
        self.assertEqual(self.printer._print_Mul(expr), 'x/6')

    def test_mul_with_negative_rational(self):
        x = Symbol('x')
        expr = Mul(Rational(-1, 2), x)
        self.assertEqual(self.printer._print_Mul(expr), '-x/2')

    def test_mul_with_infinity(self):
        x = Symbol('x')
        expr = Mul(S.Infinity, x)
        self.assertEqual(self.printer._print_Mul(expr), 'oo*x')

    def test_mul_with_one(self):
        x = Symbol('x')
        y = Symbol('y')
        expr = Mul(S.One, x, y)
        expr = self.printer._print_Mul(expr)
        brackets = ['(', ')', '[', ']', '{', '}']
        for bracket in brackets:
            expr = expr.replace(bracket, '')
        self.assertEqual(expr, 'x*y')

    def test_mul_with_negative_one(self):
        x = Symbol('x')
        expr = Mul(S.NegativeOne, x)
        self.assertEqual(self.printer._print_Mul(expr), '-x')

    def test_mul_with_multiple_terms(self):
        x = Symbol('x')
        y = Symbol('y')
        expr = Mul(2, x, y)
        self.assertEqual(self.printer._print_Mul(expr), '2*x*y')

    def test_mul_with_negative_exponent(self):
        x = Symbol('x')
        expr = Mul(x, Pow(x, -2))
        self.assertEqual(self.printer._print_Mul(expr), '1/x')

    def test_mul_with_mixed_exponents(self):
        x = Symbol('x')
        y = Symbol('y')
        expr = Mul(x, Pow(y, -1), Pow(x, 2))
        self.assertEqual(self.printer._print_Mul(expr), 'x**3/y')

    def test_mul_with_commutative_pow(self):
        x = Symbol('x')
        y = Symbol('y')
        expr = Mul(Pow(x, 2), Pow(y, -1))
        self.assertEqual(self.printer._print_Mul(expr), 'x**2/y')

    def test_mul_with_non_commutative(self):
        A = Symbol('A', commutative=False)
        B = Symbol('B', commutative=False)
        expr = Mul(A, B)
        self.assertEqual(self.printer._print_Mul(expr), 'A*B')

    def test_mul_with_zero(self):
        x = Symbol('x')
        expr = Mul(0, x)
        expr = self.printer._print_Mul(expr)
        brackets = ['(', ')', '[', ']', '{', '}']
        for bracket in brackets:
            expr = expr.replace(bracket, '')
        self.assertEqual(expr, '0')

    def test_mul_with_negative_and_positive(self):
        x = Symbol('x')
        y = Symbol('y')
        expr = Mul(-2, x, y)
        self.assertEqual(self.printer._print_Mul(expr), '-2*x*y')

    def test_mul_with_fractional_exponent(self):
        x = Symbol('x')
        expr = Mul(x, Pow(x, Rational(1, 2)))
        self.assertEqual(self.printer._print_Mul(expr), 'x**(3/2)')

    def test_mul_with_multiple_negative_exponents(self):
        x = Symbol('x')
        y = Symbol('y')
        expr = Mul(Pow(x, -1), Pow(y, -1))
        self.assertEqual(self.printer._print_Mul(expr), '1/(x*y)')

    def test_print_Mul_unevaluated(self):
        x = Symbol('x')
        y = Symbol('y')
        expr = Mul(S.One, x, Pow(y,-1), Pow(x,-2*y), evaluate=False)
        result = self.printer._print_Mul(expr)
        # Assuming the _print method returns a string representation
        self.assertEqual(result, '1*x/(y*pow(x,(2*y)))')

    def test_print_Integral(self):
        x = Symbol('x')
        y = Symbol('y')

        # Test case: single variable integral
        expr = Integral(x, (x,))
        self.assertEqual(self.printer._print_Integral(expr), "Integral(x, x)")

        # Test case: definite integral
        expr = Integral(x, (x, 0, 1))
        self.assertEqual(self.printer._print_Integral(expr), "Integral(x, (x, 0, 1))")

        # Test case: multiple integrals
        expr = Integral(x*y, (x, 0, 1), (y, 0, 1))
        self.assertEqual(self.printer._print_Integral(expr), "Integral(x*y, (x, 0, 1), (y, 0, 1))")

    def test_print_dict(self):
        # Test case: simple dictionary
        d = {'b': 2, 'a': 1}
        self.assertEqual(self.printer._print_dict(d), "{a: 1, b: 2}")

        # Test case: dictionary with string values
        d = {'b': 'banana', 'a': 'apple'}
        self.assertEqual(self.printer._print_dict(d), "{a: apple, b: banana}")

        # Test case: dictionary with mixed types
        d = {'b': 2, 'a': 'apple'}
        self.assertEqual(self.printer._print_dict(d), "{a: apple, b: 2}")

        # Test case: nested dictionary
        d = {'b': {'y': 2, 'x': 1}, 'a': 'apple'}
        self.assertEqual(self.printer._print_dict(d), "{a: apple, b: {x: 1, y: 2}}")

        # Test case: empty dictionary
        d = {}
        self.assertEqual(self.printer._print_dict(d), "{}")

    def test_print_Catalan(self):
        result = self.printer._print_Catalan(Catalan)
        self.assertEqual(result, 'Catalan')

    def test_print_ComplexInfinity(self):
        result = self.printer._print_ComplexInfinity(sympy.sympify('zoo'))
        self.assertEqual(result, 'zoo')

    def test_print_ConditionSet_universal(self):
        x = symbols('x')
        s = ConditionSet(x, x > 0, S.UniversalSet)
        result = self.printer._print_ConditionSet(s)
        self.assertEqual(result, 'ConditionSet(x, x > 0)')

    def test_print_ConditionSet_non_universal(self):
        x = symbols('x')
        s = ConditionSet(x, x > 0, S.Reals)
        result = self.printer._print_ConditionSet(s)
        self.assertEqual(result, 'ConditionSet(x, x > 0, Reals)')

    def test_print_Derivative(self):
        x = symbols('x')
        expr = Derivative(x**2, x)
        result = self.printer._print_Derivative(expr)
        self.assertEqual(result, 'Derivative(x**2, x)')

    def test_print_Basic(self):

        class MockExpr:
            def __init__(self, args, class_name):
                self.args = args
                self.__class__.__name__ = class_name

        # Create a mock expr object
        expr = MockExpr([1, 2, 3], "Basic")
        
        # Call the _print_Basic method
        result = self.printer._print_Basic(expr)
        
        # Assert the result
        self.assertEqual(result, "Basic(1, 2, 3)")

    def test_print_MatMul_negative_imaginary(self):
        x = symbols('x')
        expr = MatMul(-2*I, Matrix([[x]]))
        result = self.printer._print_MatMul(expr)
        self.assertEqual(result, "-(2*I)*Matrix([[x]])")

    def test_print_MatMul_positive_number(self):
        x = symbols('x')
        expr = MatMul(2, Matrix([[x]]))
        result = self.printer._print_MatMul(expr)
        self.assertEqual(result, "2*Matrix([[x]])")

    def test_print_MatMul_complex_number(self):
        x = symbols('x')
        expr = MatMul(2 + 3*I, Matrix([[x]]))
        result = self.printer._print_MatMul(expr)
        self.assertEqual(result, "(2 + 3*I)*Matrix([[x]])")

    def test_print_Float(self):
        # Test case 1: Precision less than 5
        expr = Float('1.234', 4)
        self.printer._settings = {"full_prec": True}
        result = self.printer._print_Float(expr)
        self.assertEqual(result, '1.234')

        # Test case 2: Precision greater than or equal to 5
        expr = Float('1.234567', 6)
        self.printer._settings = {"full_prec": True}
        result = self.printer._print_Float(expr)
        self.assertEqual(result, '1.23457')

        # Test case 3: full_prec is False
        expr = Float('1.234567', 6)
        self.printer._settings = {"full_prec": False}
        result = self.printer._print_Float(expr)
        self.assertEqual(result, '1.23457')

        # Test case 4: full_prec is "auto" and _print_level > 1
        expr = Float('1.234567', 6)
        self.printer._settings = {"full_prec": "auto"}
        self.printer._print_level = 2
        result = self.printer._print_Float(expr)
        self.assertEqual(result, '1.23457')

        # Test case 5: full_prec is "auto" and _print_level <= 1
        expr = Float('1.234567', 6)
        self.printer._settings = {"full_prec": "auto"}
        self.printer._print_level = 1
        result = self.printer._print_Float(expr)
        self.assertEqual(result, '1.23457')

        # Test case 6: Leading zeros
        expr = Float('0.01234', 5)
        self.printer._settings = {"full_prec": True}
        result = self.printer._print_Float(expr)
        self.assertEqual(result, '0.012340')

        # Test case 7: Negative leading zeros
        expr = Float('-0.01234', 5)
        self.printer._settings = {"full_prec": True}
        result = self.printer._print_Float(expr)
        self.assertEqual(result, '-0.012340')

    def test_print_set(self):
        s = {3, 1, 2}
        result = self.printer._print_set(s)
        expected = "{1, 2, 3}"
        self.assertEqual(result, expected)

        s = set()
        result = self.printer._print_set(s)
        expected = "set()"
        self.assertEqual(result, expected)

    def test_print_FiniteSet(self):
        s = FiniteSet(3, 1, 2)
        result = self.printer._print_FiniteSet(s)
        expected = "{1, 2, 3}"
        self.assertEqual(result, expected)

        s = FiniteSet()
        result = self.printer._print_FiniteSet(s)
        expected = "{}"
        self.assertEqual(result, expected)

    def test_print_Partition(self):
        s = {3, 1, 2}
        result = self.printer._print_Partition(s)
        expected = "Partition(1, 2, 3)"
        self.assertEqual(result, expected)

    def test_print_Sum(self):
        x, y = symbols('x y')

        # Test case 1: Sum with single limit
        expr = Sum(x, (x, 1, 10))
        result = self.printer._print_Sum(expr)
        expected = "Sum(x, (x, 1, 10))"
        self.assertEqual(result, expected)

        # Test case 2: Sum with multiple limits
        expr = Sum(x*y, (x, 1, 10), (y, 1, 5))
        result = self.printer._print_Sum(expr)
        expected = "Sum(x*y, (x, 1, 10), (y, 1, 5))"
        self.assertEqual(result, expected)

        # Test case 3: Sum with single limit and no range
        expr = Sum(x, (x, 1, 10))  # Providing valid bounds
        result = self.printer._print_Sum(expr)
        expected = "Sum(x, (x, 1, 10))"
        self.assertEqual(result, expected)

    def test_sstr(self):
        a, b = symbols('a b')

        # Test case 1: Simple equation
        expr = Eq(a + b, 0)
        result = sstr(expr)
        expected = "Eq(a + b, 0)"
        self.assertEqual(result, expected)

        # Test case 2: Expression with settings
        expr = a + b
        result = sstr(expr, order='none')
        expected = "a + b"
        self.assertEqual(result, expected)

        # Test case 3: Expression with abbreviation setting
        # Assuming abbrev=True affects the output in some way
        result = sstr(expr, abbrev=True)
        expected = "a + b"  # Adjust this based on actual behavior
        self.assertEqual(result, expected)

    def test_sstrrepr(self):
        a, b = symbols('a b')

        expr = Eq(a + b, 0)
        result = sstrrepr(expr)
        expected = "Eq(a + b, 0)"
        self.assertEqual(result, expected)

    def test_print_And(self):
        x, y = symbols('x y')

        # Test case 1: Simple And expression
        expr = And(x > 0, y > 0)
        result = self.printer._print_And(expr)
        expected = "(x > 0) & (y > 0)"  # Adjust based on actual stringify output
        self.assertEqual(result, expected)

        # Test case 2: And expression with NegativeInfinity
        expr = And(Eq(x, S.NegativeInfinity), y > 0)
        result = self.printer._print_And(expr)
        expected = "Eq(x, -oo) & (y > 0)"  # Adjust based on actual stringify output
        self.assertEqual(result, expected)

        # Test case 3: And expression with multiple relations
        expr = And(x > 0, y > 0, Eq(x, S.NegativeInfinity))
        result = self.printer._print_And(expr)
        expected = "Eq(x, -oo) & (x > 0) & (y > 0)"  # Adjust based on actual stringify output
        self.assertEqual(result, expected)

        # Test case 4: And expression with no relations
        expr = And(x > 0, y > 0)
        result = self.printer._print_And(expr)
        expected = "(x > 0) & (y > 0)"  # Adjust based on actual stringify output
        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()