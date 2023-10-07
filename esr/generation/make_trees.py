import itertools
import sympy
from esr.fitting.sympy_symbols import *
from esr.generation.custom_printer import ESRPrinter

#from mpi4py import MPI
#comm = MPI.COMM_WORLD
#rank = comm.Get_rank()
#size = comm.Get_size()

import time

import numpy as np
import matplotlib.pyplot as plt

class Operator:
    """
    Class describing the characteristics of an operator:
    its symbol, arity and whether its arguments can be commuted
    """

    def __init__(self, symbol, arity, commutative):
        self.symbol = symbol
        self.arity = arity
        self.commutative = commutative
        
    def __repr__(self,):
        return self.symbol
        
    def copy(self):
        return Operator(self.symbol, self.arity, self.commutative)

class Function:
    """
    Class describing a function as a list of operators.
    """

    def __init__(self, ops):
        self.ops = ops
        
    def __repr__(self):
        return str([a.symbol for a in self.ops])
        
    def print_op(self, idx):
        if self.ops[idx].arity == 0:
            return self.ops[idx].symbol
        elif self.ops[idx].arity == 1:
            return self.ops[idx].symbol + '(' + self.print_op(self.left[idx]) + ')'
        elif self.ops[idx].symbol in ['*', '/', '-', '+']:
            s = ''
            if self.ops[self.left[idx]].arity < 2:
                s += self.print_op(self.left[idx])
            else:
                s += '(' + self.print_op(self.left[idx]) + ')'
            s += self.ops[idx].symbol
            if self.ops[self.right[idx]].arity < 2:
                s += self.print_op(self.right[idx])
            else:
                s += '(' + self.print_op(self.right[idx]) + ')'
            return s
        else:
            return self.ops[idx].symbol + '(' + self.print_op(self.left[idx]) + ',' + self.print_op(self.right[idx]) + ')'
            
    
    def find_parents(self):
        self.left = [None] * len(self.ops)
        self.right = [None] * len(self.ops)
        self.parent = [None] * len(self.ops)
        if self.complexity() == 1:
            return
        for i in range(len(self.ops)):
            success = False
            if self.ops[i].arity > 0:
                self.left[i] = i+1
                self.parent[i+1] = i
                success = True
            else:
                # try to go up the tree
                j = self.parent[i]
                while not success:
                    if (self.ops[j].arity == 2) and (self.right[j] is None):
                        # Add to right node if possible
                        self.right[j] = i+1
                        self.parent[i+1] = j
                        success = True
                    elif (self.parent[j] is None):
                    # Check if can't move up the tree any higher
                        break
                    
                    # Go up the tree to this node's parent
                    j = self.parent[j]
                    
    def count_par(self,):
        par = [a for a in self.ops if a.symbol == 'a']
        return len(par)
        
    def count_leaves(self,):
        leaves = [a for a in self.ops if a.arity == 0]
        return len(leaves)
    
    def count_var(self,):
        return self.count_leaves() - self.count_par()
        
    def __str__(self):
    
        # Change parameters so they have different symbols
        self.ops = [a.copy() for a in self.ops]
        isym = [i for i, a in enumerate(self.ops) if a.symbol == "a"]
        for i, j in enumerate(isym):
            self.ops[j].symbol = 'a%i'%i
    
        # Find parent and child labels
        self.find_parents()
        
        p = self.print_op(0)
        
        # Change parameter symbols back
        for i in isym:
            self.ops[i].symbol = 'a'
        
        return p


    def to_sympy(self, locs):
        s = self.__str__()
        npar = self.count_par()
        if npar > 0:
            param_list = ['a%i'%i for i in range(npar)]
            all_a = sympy.symbols(" ".join(param_list), real=True)
            if npar == 1:
                locs["a0"] = all_a
            else:
                for i in range(len(all_a)):
                    locs["a%i"%i] = all_a[i]
        s = sympy.sympify(s, locals=locs)
        return s
        
        
    def to_string(self, locs):
        p = ESRPrinter()
        s = self.to_sympy(locs)
        return p.doprint(s)
        
    def complexity(self,):
        return len(self.ops)
        

def split_list(Ntotal):
    Neach_section, extras = divmod(Ntotal, size)
    section_sizes = ([0] +
                         extras * [Neach_section+1] +
                         (size-extras) * [Neach_section])
    div_points = np.array(section_sizes, dtype=np.intp).cumsum()
    return div_points[rank], div_points[rank+1]
        

def make_fun(max_comp, basis_functions, commutative_dict, unary_inv, locs):
    
    basis_ops = [[Operator(a, 0, None) for a in basis_functions[0]],
                [Operator(a, 1, None) for a in basis_functions[1]],
                [Operator(a, 2, commutative_dict[a]) for a in basis_functions[2]],
                ]
    
    # Make complexity 1 functions
    all_fun = [None] * (max_comp + 1)
    all_fun[0] = []
    all_fun[1] = [Function([op]) for op in basis_ops[0]]
    print(1, len(all_fun[1]))
    
#    # Paramteter mappings to make non-unique variants
#    uniq_maps = [None] * (max_comp + 1)
#    uniq_maps[0] = {}  # complexity 0 does not exist
#    uniq_maps[1] = {}  # complexity 1 functions are all unique
    
    # Index mappings from all_fun -> unique variants
    uniq_idx = [None] * (max_comp + 1)
    uniq_idx[0] = []  # complexity 0 does not exist
    uniq_idx[1] = list(np.arange(len(all_fun[1])))  # complexity 1 functions are all unique
    
    print(uniq_idx)
        
    for comp in range(2, max_comp+1):
        
        # Root is unary
        # If child of root is inverse of root we do not need to use this function
        all_fun[comp] = [Function([b] + f.ops) for (b, f) in itertools.product(basis_ops[1], all_fun[comp-1]) if unary_inv[b.symbol] != f.ops[0].symbol]
        
        # Root is binary so need at least complexity 3
        if comp > 2:
            for b in basis_ops[2]:
                
                if b.commutative:
                    # b(left, right) = b(right, left) so can restrict comp(right) <= comp(left)
                    for cl in range(1, comp - 1):
                        cr = comp - 1 - cl
                        if cr < cl:
                            all_fun[comp] = all_fun[comp] + [Function([b] + l.ops + r.ops) for (l,r) in itertools.product(all_fun[cl], all_fun[cr])]
                        elif cr == cl:
                            # If same complexity, can remove duplicates
                            all_fun[comp] = all_fun[comp] + [Function([b] + l.ops + r.ops) for (l,r) in itertools.combinations_with_replacement(all_fun[cl], 2)]
                else:
                    # Have to have all combinations of left and right nodes
                    for cl in range(1, comp - 1):
                        cr = comp - 1 - cl
                        all_fun[comp] = all_fun[comp] + [Function([b] + l.ops + r.ops) for (l,r) in itertools.product(all_fun[cl], all_fun[cr])]
        
        # We don't want functions involving binary operators acting on two parameters, since this has
        # a simpler representation with a single parameter. These will be removed at all complexities
        # if we remove them all at complexity 3
        if comp == 3:
            all_fun[comp] = [f for f in all_fun[comp] if not (f.ops[1].symbol == "a" and f.ops[2].symbol == "a")]
            
        # Similarly, if we have a function without any x's, it cannot contain more than one constant
        all_fun[comp] = [f for f in all_fun[comp] if not (f.count_var() == 0 and f.count_par() > 1)]
        
        # Now look for unique functions
            
#        # We don't want functions involving unary operators acting on two parameters
#        # These will be removed at all complexities if we remove them all at complexity 2
#        if comp == 2:
#            all_fun[comp] = [f for f in all_fun[comp] if not (f.ops[1].symbol == "a")]

        # Now convert to strings with sympy and see if we have any duplicates
        new_eq = [f.to_string(locs) for f in all_fun[comp]]
        new_eq, index, inverse = np.unique(new_eq, return_index=True, return_inverse=True)
        uniq_idx[comp] = [index[i] for i in inverse]

    return all_fun, uniq_idx
