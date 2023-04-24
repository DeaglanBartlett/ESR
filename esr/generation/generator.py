import numpy as np
import itertools
import sys
from mpi4py import MPI
import sympy
from sympy.core.sympify import kernS
import gc
import os
import pprint

import esr.generation.simplifier as simplifier
import esr.generation.utils as utils
from esr.fitting.sympy_symbols import *

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

class Node:
    def __init__(self, t):
        self.type = t
        self.parent = None
        self.left = None
        self.right = None
        self.op = None
        self.val = None
        self.node_name = None
        self.tree = None
    
    def copy(self):
        new_node = Node(self.type)
        new_node.parent = self.parent
        new_node.left = self.left
        new_node.right = self.right
        return new_node
        
    def is_used(self):
        if (self.type == 0) and (self.parent is None):
            return False
        elif (self.type == 1) and (self.left is None):
            return False
        elif (self.type == 2) and (self.left is None) and (self.right is None):
            return False
        return True

    def assign_op(self, op):
        self.op = op
        v = op
        if v.lstrip("-").isdigit():
            self.val = int(v)
        elif v.lstrip("-").lstrip("/").isnumeric():
            self.val = float(v)
            
class DecoratedNode:

    def __init__(self, fun, basis_functions, parent_op=None, parent=None):
        
        self.expr = fun
        self.type = type(fun)
        self.constant = fun.is_number
        self.degree = len(fun.args)
        self.op = fun.__class__.__name__
        self.parent_op = parent_op
        self.parent = parent
        self.tree = None
        
        if self.constant:
            self.val = str(fun)
        elif fun.is_symbol:
            self.val = fun.name
        else:
            self.val = None
            
        if self.op == 'Pow' and fun.args[1] == 2.0 and 'square' in basis_functions[1]:
            self.op = 'Square'
            self.children = [DecoratedNode(fun.args[0], basis_functions, parent_op=self.op, parent=self)]
        elif self.op == 'Pow' and fun.args[1] == 3.0 and 'cube' in basis_functions[1]:
            self.op = 'Cube'
            self.children = [DecoratedNode(fun.args[0], basis_functions, parent_op=self.op, parent=self)]
        elif self.op == 'Pow' and fun.args[1] == 1/2 and ('sqrt' in basis_functions[1]) or ('sqrt_abs' in basis_functions[1]):
            self.op = 'Sqrt'
            self.children = [DecoratedNode(fun.args[0], basis_functions, parent_op=self.op, parent=self)]
        elif self.op == 'Mul' and len(fun.args) == 2 and fun.args[1].__class__.__name__ == 'Pow' and fun.args[1].args[1] == -1:
            self.op = 'Div'
            self.children = [DecoratedNode(fun.args[0], basis_functions, parent_op=self.op, parent=self),
                            DecoratedNode(fun.args[1].args[0], basis_functions, parent_op=self.op, parent=self)]
        elif self.op == 'Pow' and fun.args[1] == -1 and 'inv' in basis_functions[1]:
            self.op = 'Inv'
            self.children = [DecoratedNode(fun.args[0], basis_functions, parent_op=self.op, parent=self)]
        else:
            if (len(fun.args) > 2):
                f = fun.as_two_terms()
                self.children = [DecoratedNode(f[0], basis_functions, parent_op=self.op, parent=self),
                                DecoratedNode(f[1], basis_functions, parent_op=self.op, parent=self)]
            else:
                self.children = [DecoratedNode(a, basis_functions, parent_op=self.op, parent=self) for a in fun.args]
                
    def is_unity(self):
        try:
            f = float(self.val)
            return f == float(1)
        except:
            return False
            
    def count_nodes(self, basis_functions):
        """
        :basis_functions (list): list of lists basis functions. basis_functions[0] are nullary, basis_functions[1] are unary and basis_functions[2] are binary operators
        """
        
        if self.degree == 0:
            return 1
        
        # Sqrt(x) instead of pow(x, 1/2)
        elif self.op == "Pow" and (self.children[1].type==sympy.core.numbers.Half) and (("sqrt" in basis_functions[1]) or ("sqrt_abs" in basis_functions[1])):
            v = 0

        # Square(x) instead of pow(x, 2)
        elif self.op == "Pow" and (self.children[1].val == str(2)) and "square" in basis_functions[1]:
            v = 0
            
        # Cube(x) instead of pow(x, 3)
        elif self.op == "Pow" and (self.children[1].val == str(3)) and "cube" in basis_functions[1]:
            v = 0

        # Inv(x) instead of pow(x, -1)
        elif self.op == "Pow" and (self.children[1].type==sympy.core.numbers.NegativeOne) and ("inv" in basis_functions[1]):
            if self.parent_op == "Mul":
                v = 0
            else:
                v = 1
                
        # Multiply or divide by one doesn't do anything
        elif self.op == "Mul" and (self.children[0].is_unity() or self.children[1].is_unity()):
            v = 0
        elif self.op == "Div" and (self.children[0] == 1 or self.children[1] == 1):
            v = 0

        # Treat sqrt_abs and pow_abs as a single function
        elif self.op == "Abs" and self.parent_op == "Pow":
            v = 0
                
        else:
            v = 1
            
        carr = np.array([c.count_nodes(basis_functions) for c in self.children])
        
        return v + carr.sum()
        
    def to_list(self, basis_functions):
        """
        
        """
    
        if self.degree == 0:
            return [str(self.val)]
        elif self.degree == 1:
            return [self.op] + self.children[0].to_list(basis_functions)
        # Sqrt(x) instead of pow(x, 1/2)
        elif self.op == "Pow" and (self.children[1].type==sympy.core.numbers.Half) and (("sqrt" in basis_functions[1]) or ("sqrt_abs" in basis_functions[1])):
            if ("sqrt" in basis_functions[1]):
                return ["sqrt"] + self.children[0].to_list(basis_functions)
            else:
                return ["sqrt_abs"] + self.children[0].to_list(basis_functions)
        # Square(x) instead of pow(x, 2) if possible
        elif self.op == "Pow" and (self.children[1].val == str(2)) and "square" in basis_functions[1]:
            return ["square"] + self.children[0].to_list(basis_functions)
        # pow(x,2) instead of Square(x) if necessary
        elif self.op == "Square" and "sqaure" not in basis_functions[1]:
            return ["pow"] + self.children[0].to_list(basis_functions) + ["2"]
        # Cube(x) instead of pow(x, 3)
        elif self.op == "Pow" and (self.children[1].val == str(3)) and "cube" in basis_functions[1]:
            return ["cube"] + self.children[0].to_list(basis_functions)
        # pow(x,2) instead of Square(x) if necessary
        elif self.op == "Cube" and "cube" not in basis_functions[1]:
            return ["pow"] + self.children[0].to_list(basis_functions) + ["3"]
        # Inv(x) instead of pow(x, -1)
        elif self.op == "Pow" and (self.children[1].type==sympy.core.numbers.NegativeOne) and ("inv" in basis_functions[1]):
            return ["Inv"] + self.children[0].to_list(basis_functions)
        # Deal with * inv = /
        elif self.op == "Mul" and self.children[0].op == "Pow" and (self.children[1].type==sympy.core.numbers.NegativeOne) and ("/" in basis_functions[2]):
            return ["Mul"] + self.children[1].to_list(basis_functions)
        # Deal with / inv = *
        elif self.op == "Div" and self.children[0].op == "Pow" and (self.children[1].type==sympy.core.numbers.NegativeOne) and ("*" in basis_functions[2]):
            return ["Mul"] + self.children[1].to_list(basis_functions)
        # Multiply or divide by one doesn't do anything
        elif self.op == "Mul" and (self.children[0].is_unity() or self.children[1].is_unity()):
            if self.children[0].is_unity():
                return self.children[1].to_list(basis_functions)
            else:
                return self.children[0].to_list(basis_functions)
        # Don't keep abs after pow or sqrt
        elif self.op == "Abs" and self.parent.op in ["Sqrt", "Pow"]:
            return self.children[0].to_list(basis_functions)
        elif self.op == "Div" and (self.children[0] == 1 or self.children[1] == 1):
            v = 0
        else:
            r = [self.op]
            for c in self.children:
                r = r + c.to_list(basis_functions)
            return r
            
    def get_lineage(self):
        p = [self.op, self.parent_op]
        q = self.parent
        while q is not None:
            p += [q.parent_op]
            q = q.parent
        p.reverse()
        p = [tuple(p)]
        for c in self.children:
            p += c.get_lineage()
        return p

    def get_sibling_lineage(self):
        # First get direct lineage of nodes
        p = [self.op, self.parent_op]
        v = [self.val]
        if self.parent_op is None:
            v += [None]
        else:
            v += [self.parent.val]
        q = self.parent
        while q is not None:
            p += [q.parent_op]
            if q.parent_op is None:
                v += [None]
            else:
                v += [q.parent.val]
            q = q.parent
        p.reverse()
        v.reverse()
        # Now add the siblings
        if len(self.children) == 1:
            p += [(self.children[0].op, None)]
            v += [(self.children[0].val, None)]
        elif len(self.children) > 1:
            p += [tuple([c.op for c in self.children])]
            v += [tuple([c.val for c in self.children])]
        
        p = [tuple(p)]
        v = [tuple(v)]
        for c in self.children:
            if len(c.children) > 0:
                pp, vv = c.get_sibling_lineage()
                p += pp
                v += vv
        return p, v
        
    def get_siblings(self):
        if self.parent is not None and len(self.parent.children) > 1:
            p = [tuple([c.op for c in self.parent.children])]
        else:
            p = [(self.op, 'None')]
        for c in self.children:
            p += c.get_siblings()
        return p

        
def check_tree(s):
    """ Given a candidate string of 0, 1 and 2s, see whether one can make a function out of this
    
    Args:
        :s (str): string comprised of 0, 1 and 2 representing tree of nullary, unary and binary nodes
        
    Returns:
        :success (bool): whether candidate string can form a valid tree (True) or not (False)
        :part_considered (str): string of length <= s, where s[:len(part_considered)] = part_considered
        :tree (list): list of Node objects corresponding to string s
    """

    tree = [Node(t) for t in s]
    
    for i in range(len(s)-1):
        
        success = False

        if (tree[i].type == 2) or (tree[i].type == 1):
            # Add to the left if possible
            tree[i].left = i+1
            tree[i+1].parent = i
            success = True
        else:
            # try to go up the tree
            j = tree[i].parent
            
            while not success:
            
                if (tree[j].type == 2) and (tree[j].right is None):
                    # Add to right of node if possible
                    tree[j].right = i+1
                    tree[i+1].parent = j
                    success = True
                elif (tree[j].parent is None):
                    # Check if can't move up the tree any higher
                    break
                
                # Go up the tree to this node's parent
                j = tree[j].parent
                
        if not success:
            break
            
    if len(s) > 1:
            
        # Need to check for parents without left nodes which should have them
        if success:
            lefts = [t.left for t in tree if t.type == 1 or t.type == 2]
            if None in lefts:
                success = False
        
        # Need to check for parents without right nodes which should have them
        if success:
            rights = [t.right for t in tree if t.type == 2]
            if None in rights:
                success = False
                
        # This will allow us to delete any trees which start with this
        part_considered = s[:i+2]
        
    else:
        success = True
        part_considered = None
        
    return success, part_considered, tree
            
        
def get_allowed_shapes(compl):
    """ Find the shapes of all allowed trees containing compl nodes
    
    Args:
        :compl (int): complexity of tree = number of nodes
        
    Returns:
        :cand (list): list of strings comprised of 0, 1 and 2 representing valid trees of nullary, unary and binary nodes
    """

    if rank == 0:
        # Make all graphs with this complexity
        cand = np.array([list(t) for t in itertools.product('012', repeat=compl)], dtype=int)

        # Graph cannot start with a type0 node
        if compl > 1:
            cand = cand[cand[:,0] != 0]

        # Graph must end at a type0 node
        cand = cand[cand[:,-1] == 0]

        # The penultimate node cannot be of type2
        if cand.shape[1] > 1:
            cand = cand[cand[:,-2] != 2]

        msk = np.ones(cand.shape[0], dtype=bool)

        for i in range(cand.shape[0]):
            if not msk[i]:
                pass
            success, part_considered, tree = check_tree(cand[i,:])
            if not success:
                msk[i] = False
                
                # Remove other candidates where this string appears at the start
                m = cand[:,:len(part_considered)] == part_considered[None,:]
                m = np.prod(m, axis=1)
                msk[np.where(m)] = False
        
        cand = cand[msk,:]
    else:
        cand = None

    cand = comm.bcast(cand, root=0)
            
    return cand
    
    
def node_to_string(idx, tree, labels):
    """Convert a tree with labels into a string giving function
    
    Args:
        :idx (int): index of tree to consider
        :tree (list): list of Node objects corresponding to the tree
        :labels (list): list of strings giving node labels of tree
        
    Returns:
        Function as a string
    """
    
    if len(tree) == 0:
        return '0'
    elif tree[idx].type == 0:
        return labels[idx]
    elif tree[idx].type == 1:
        return labels[idx] + '(' + node_to_string(tree[idx].left, tree, labels) + ')'
    elif tree[idx].type == 2:
        if labels[idx] in ['*', '/', '-', '+']:
            return '(' + node_to_string(tree[idx].left, tree, labels) + ')' + labels[idx] + \
                    '(' + node_to_string(tree[idx].right, tree, labels) + ')'
        else:
            return labels[idx] + '(' + node_to_string(tree[idx].left, tree, labels) + \
                ',' + node_to_string(tree[idx].right, tree, labels) + ')'
    return
    
    
def string_to_expr(s, kern=False, evaluate=False, locs=None):
    """Convert a string giving function into a sympy object
    
    Args:
        :s (str): string representation of the function considered
        :kern (bool): whether to use sympy's kernS function or sympify
        :evaluate (bool): whether to use powsimp, factor and subs
        :locs (dict): dictionary of string:sympy objects. If None, will create here
        
    Returns:
        :expr (sympy object): expression corresponding to s
    
    """
    
    s = s.replace('[', '(')
    s = s.replace(']', ')')
    s = s.replace('Sqrt', 'sqrt')
    s = s.replace('*^', '*10^')
    
    if locs is None:
        locs = {"inv": inv,
                "square": square,
                "cube": cube,
                "sqrt": sqrt,
                "log": log,
                "pow": pow,
                "x": x
                }
    
    if kern:
        expr = kernS(s)
    else:
#        expr = sympy.sympify(s, evaluate=evaluate,
        expr = sympy.sympify(s,
                            locals=locs)
        if evaluate:
            expr = expr.powsimp(expr)
            expr = expr.factor()
            expr = expr.subs(1.0, 1)
    
    return expr
    
    
def string_to_node(s, basis_functions, locs=None, evalf=False):
    """Convert a string giving function into a tree with labels
    
    Args:
        :s (str): string representation of the function considered
        :basis_functions (list): list of lists basis functions. basis_functions[0] are nullary, basis_functions[1] are unary and basis_functions[2] are binary operators
        :locs (dict): dictionary of string:sympy objects. If None, will create here
        :evalf (bool): whether to run evalf() on function (default=False)
        
    Returns:
        :tree (list): list of Node objects corresponding to the tree
        :labels (list): list of strings giving node labels of tree
    
    """
    
    expr = [None] * 4
    nodes = [None] * 4
    c = np.ones(4)

    i = 0
    try:
        expr[i] = string_to_expr(s, kern=False, evaluate=True, locs=locs)
        if evalf:
            expr[i] = expr[i].evalf()
        nodes[i] = DecoratedNode(expr[i], basis_functions)
        c[i] = nodes[i].count_nodes(basis_functions)
    except:
        c[i] = np.nan

    
    i = 1
    try:
        expr[i] = string_to_expr(s, kern=False, evaluate=False, locs=locs)
        if evalf:
            expr[i] = expr[i].evalf()
        nodes[i] = DecoratedNode(expr[i], basis_functions)
        c[i] = nodes[i].count_nodes(basis_functions)
    except:
        c[i] = np.nan

    i = 2
    try:
        expr[i] = string_to_expr(s, kern=True, evaluate=True, locs=locs)
        if evalf:
            expr[i] = expr[i].evalf()
        nodes[i] = DecoratedNode(expr[i], basis_functions)
        c[i] = nodes[i].count_nodes(basis_functions)
    except:
        c[i] = np.nan
    
    i = 3
    try:
        expr[i] = string_to_expr(s, kern=True, evaluate=False, locs=locs)
        if evalf:
            expr[i] = expr[i].evalf()
        nodes[i] = DecoratedNode(expr[i], basis_functions)
        c[i] = nodes[i].count_nodes(basis_functions)
    except:
        c[i] = np.nan
    
    i = np.nanargmin(c)
    
    # FINISH THIS OFF

    return expr[i], nodes[i], int(c[i])
    
    
def update_tree(tree, labels, try_idx, basis_functions):
    """Try to combine exponentials and powers to make simpler representations of functions
    
    Args:
        :tree (list): list of Node objects corresponding to tree of function
        :labels (list): list of strings giving node labels of tree
        :try_idx (int): when we have multiple substituions we can attempt, this indicates which one to try
        :basis_functions (list): list of lists basis functions. basis_functions[0] are nullary, basis_functions[1] are unary and basis_functions[2] are binary operators
    
    Returns:
        :new_labels (list): list of strings giving node labels of new tree
        :new_shape (list): list of 0, 1 and 2 representing whether nodes in new tree are nullary, unary or binary
        :nadded (int): number of new functions added
    
    """

    pow_set = ["square", "cube", "sqrt_abs", "inv"]
    pow_num = {"square":'*2', "cube":'*3', "sqrt_abs":'/2', "inv":'*-1'}
    exp_set = ["log_abs", "exp", "pow_abs"]
    # log_abs comes first, exp comes second, pow_abs can go in either order
    exp_ord = {"log_abs":1, "exp":2, "pow_abs":3}
    
    common_pow = list(set(labels) & set(pow_set))
    common_exp = list(set(labels) & set(exp_set))
    
    special_idx = []
    diff1_idx = []
    diff2_idx = []
    num1 = []
    num2 = []
    
    # See if we have any of the correct patterns in the labels list
    if len(common_pow) != 0 and len(common_exp) != 0:
        for i in range(len(labels)):
            if labels[i] in common_exp:
                if (exp_ord[labels[i]] == 1) or (exp_ord[labels[i]] == 3):
                    success = False
                    if (i < len(labels) - 1) and (labels[i+1] in common_pow):
                        special_idx.append(i)
                        j = 0
                        success = False
                        s = '*1'
                        while not success:
                            j += 1
                            if (i >= len(labels) - j):
                                success = True
                            elif labels[i+j] in common_pow:
                                n = sympy.sympify((s + pow_num[labels[i+j]][0] + '(' + pow_num[labels[i+j]][1:] + ')')[1:])
                                if n.is_integer or (1/n).is_integer:
                                    s += pow_num[labels[i+j]][0] + '(' + pow_num[labels[i+j]][1:] + ')'
                                else:
                                    success = True
                            else:
                                success = True
                        diff1_idx.append(j-1)
                        diff2_idx.append(0)
                        
                        n = sympy.sympify(s[1:])
                        if n.is_integer:
                            s = '*' + str(n)
                        else:
                            s = '/' + str(1/n)
                        num1.append(s)
                        num2.append(None)
                            
                if (exp_ord[labels[i]] == 2) or (exp_ord[labels[i]] == 3):
                    success = False
                    if (i > 0) and (labels[i-1] in common_pow):
                        if i not in special_idx:
                            special_idx.append(i)
                        j = 0
                        success = False
                        s = '*1'
                        while not success:
                            j += 1
                            if (i - j) < 0:
                                success = True
                            elif labels[i-j] in common_pow:
                                n = sympy.sympify((s + pow_num[labels[i-j]][0] + '(' + pow_num[labels[i-j]][1:] + ')')[1:])
                                if n.is_integer or (1/n).is_integer:
                                    s += pow_num[labels[i-j]][0] + '(' + pow_num[labels[i-j]][1:] + ')'
                                else:
                                    success = True
                            else:
                                success = True
                        if len(diff2_idx) != len(special_idx):
                            diff1_idx.append(0)
                            diff2_idx.append(j-1)
                        else:
                            diff2_idx[-1] = j-1
                        n = sympy.sympify(s[1:])
                        if n.is_integer:
                            s = '*' + str(n)
                        else:
                            s = '/' + str(1/n)
                        if len(num2) != len(special_idx):
                            num1.append(None)
                            num2.append(s)
                        else:
                            num2[-1] = s
                               
    new_shape = None
    new_labels = None
    nadded = 0
        
    if len(special_idx) > try_idx:

        i = special_idx[try_idx]
        if (exp_ord[labels[i]] == 1):
            n = num1[try_idx]
            d = diff1_idx[try_idx]
        elif (exp_ord[labels[i]] == 2):
            n = num2[try_idx]
            d = diff2_idx[try_idx]
        elif (exp_ord[labels[i]] == 3):
            n1 = num1[try_idx]
            n2 = num2[try_idx]
            d1 = diff1_idx[try_idx]
            d2 = diff2_idx[try_idx]
            
        if (exp_ord[labels[i]] == 3) and (n1 is None or n1[0] in basis_functions[2]) and (n2 is None or n2[0] in basis_functions[2]):
            
            # Combine the two numbers
            s = '*1'
            if n1 is not None:
                s += n1
            if n2 is not None:
                s += n2
            s = sympy.sympify(s[1:])
            if s.is_integer:
                n = '*' + str(s)
            else:
                n = '/' + str(1/s)
            
            orig_parents = np.array([t.parent for t in tree])
            orig_shape = [t.type for t in tree]
            
            # Start of exponent
            j = np.argwhere(orig_parents[i+2:] == i)
            j = j[0,0] + i + 2
            
            # End of exponent
            if tree[i].parent is None:
                k = len(labels)
            else:
                k = np.argwhere(orig_parents[i+2:] <= tree[i].parent)
                if len(k) == 0:
                    k = len(labels)
                else:
                    k = k[0,0] + i + 2
                
            if int(n[1:]) == 1:
                new_labels = (
                    # First part of tree (up to the d2 operators which make number)
                    labels[:i-d2] +
                    # The pow comes next
                    [labels[i]] +
                    # Skip the d1 operators which give the number
                    labels[i+d1+1:]
                    )
                new_shape = orig_shape[:i-d2] + \
                    [orig_shape[i]] + \
                    orig_shape[i+d1+1:]
            else:
                new_labels = (
                        # First part of tree (up to the d2 operators which make number)
                        labels[:i-d2] +
                        # The pow comes next
                        [labels[i]] +
                        # Skip the d2 operators which give the number
                        labels[i+d1+1:j] +
                        # Add in a * or /
                        [n[0]] +
                        # The original exponent on left of * or /
                        labels[j:k] +
                        # Put number at right of * or /
                        [n[1:]] +
                        # Rest of tree
                        labels[k:]
                        )
                new_shape = orig_shape[:i-d2] + \
                        [orig_shape[i]] + \
                        orig_shape[i+d1+1:j] + \
                        [2] + \
                        orig_shape[j:k] + \
                        [0] + \
                        orig_shape[k:]
            
            nadded += 1

        elif (exp_ord[labels[i]] != 3) and (n[0] in basis_functions[2]):

            orig_parents = np.array([t.parent for t in tree])
            orig_shape = [t.type for t in tree]
            if i > 0:
                j = np.argwhere(orig_parents[i+1:] <= tree[i].parent)
                if len(j) == 0:
                    j = len(labels)
                else:
                    j = j[0,0] + i + 1
            else:
                j = len(labels)

            if (i > 0) and (labels[orig_parents[i]] in ["+", "-"]) and n.startswith('*-') and exp_ord[labels[i]] == 1:
                inv_op = "-" if (labels[orig_parents[i]] == "+") else "+"
                if (tree[orig_parents[i]].right == i) and (inv_op in basis_functions[2]):
                        
                    if int(n[2:]) == 1:
                        new_labels = (
                                # First part of tree
                                labels[:orig_parents[i]] + [inv_op] +
                                # Left part of + unchanged
                                labels[orig_parents[i]+1:i] +
                                # Skip the d operators which give the number
                                [labels[i]] + labels[i+d+1:]
                                )
                        new_shape = orig_shape[:i] + [orig_shape[i]] + \
                                orig_shape[i+d+1:j] + orig_shape[j:]
                    else:
                        new_labels = (
                                # First part of tree
                                labels[:orig_parents[i]] + [inv_op] +
                                # Left part of + unchanged
                                labels[orig_parents[i]+1:i] +
                                # Add * or / to right of +
                                [n[0]] +
                                # Put "log_abs" at left of * or /
                                [labels[i]] +
                                # Skip the d operators which give the number
                                labels[i+d+1:j] +
                                # Put number at right of * or / and add rest of tree
                                [n[2:]] + labels[j:]
                                )
                        new_shape = orig_shape[:orig_parents[i]] + [2] + \
                                orig_shape[orig_parents[i]+1:i] + \
                                [2] + \
                                [orig_shape[i]] + \
                                orig_shape[i+d+1:j] + \
                                [0] + orig_shape[j:]
                    
                    nadded += 1
                    
                elif (labels[orig_parents[i]] == "+") and (inv_op in basis_functions[2]):
                
                    # Index of where right side of + ends
                    k = np.argwhere(orig_parents[tree[orig_parents[i]].right+1:] <= tree[i].parent)
                    if len(k) == 0:
                        k = len(labels)
                    else:
                        k = k[0,0] + tree[orig_parents[i]].right + 1
                    
                    if int(n[2:]) == 1:
                        new_labels = (
                            # First part of tree
                            labels[:orig_parents[i]] + [inv_op] +
                            # Move right part of + to the left
                            labels[tree[orig_parents[i]].right:k] +
                            # Put "log_abs" at start of right of +
                            labels[orig_parents[i]+1:i+1] +
                            # Skip the d operators which give the number
                            labels[i+d+1:tree[orig_parents[i]].right] +
                            # Rest of tree
                            labels[k:]
                            )
                        new_shape = orig_shape[:orig_parents[i]] + [2] + \
                            orig_shape[tree[orig_parents[i]].right:k] + \
                            orig_shape[orig_parents[i]+1:i+1] + \
                            orig_shape[i+d+1:tree[orig_parents[i]].right] + \
                            orig_shape[k:]
                    else:
                        new_labels = (
                            # First part of tree
                            labels[:orig_parents[i]] + [inv_op] +
                            # Move right part of + to the left
                            labels[tree[orig_parents[i]].right:k] +
                            # Add * or / to right of +
                            [n[0]] +
                            # Put "log_abs" at left of * or /
                            labels[orig_parents[i]+1:i+1] +
                            # Skip the d operators which give the number
                            labels[i+d+1:tree[orig_parents[i]].right] +
                            # Put number at right of * or / and add rest of tree
                            [n[2:]] + labels[k:]
                            )
                        new_shape = orig_shape[:orig_parents[i]] + [2] + \
                            orig_shape[tree[orig_parents[i]].right:k] + \
                            [2] + \
                            orig_shape[orig_parents[i]+1:i+1] + \
                            orig_shape[i+d+1:tree[orig_parents[i]].right] + \
                            [0] + orig_shape[k:]
                      
                    nadded += 1
                else:
                    # TWO CASES:
                    # (1) F - G -> (-1)*(n * H + G) /
                    # (2) F - G -> (-n)*H - G
                    new_labels = []
                    new_shape = []
                    if (inv_op in basis_functions[2]):
                        new_labels.append(
                                    # First part of tree
                                    labels[:orig_parents[i]] +
                                    # Add *(-1) before + or -
                                    ['*', '-1', inv_op] +
                                    # * or / the first term
                                    [n[0], n[2:]] +
                                    # Put "log_abs" at right of * or /
                                    [labels[i]] +
                                    # Skip the d operators which give the number
                                    labels[i+d+1:]
                                    )
                        new_shape.append(
                                    orig_shape[:orig_parents[i]] + \
                                    [2, 0, 2] + \
                                    [2, 0] + \
                                    [orig_shape[i]] + \
                                    orig_shape[i+d+1:]
                                    )
                        nadded += 1

                    new_labels.append(
                                # First part of tree
                                labels[:orig_parents[i]+1] +
                                # * or / the first term
                                [n[0], n[1:]] +
                                # Put "log_abs" at right of * or /
                                [labels[i]] +
                                # Skip the d operators which give the number
                                labels[i+d+1:]
                                )
                    new_shape.append(
                                orig_shape[:orig_parents[i]+1] +
                                [2, 0] +
                                [orig_shape[i]] +
                                orig_shape[i+d+1:]
                                )
                    nadded += 1

            else:
                if exp_ord[labels[i]] == 1:
                    if int(n[1:]) == 1:
                        new_labels = (
                                    # First part of tree
                                    labels[:i+1] +
                                    # Skip the d operators which give the number
                                    labels[i+d+1:]
                                    )
                        new_shape = orig_shape[:i+1] + \
                                    orig_shape[i+d+1:]
                    else:
                        new_labels = (
                                    # First part of tree
                                    labels[:i] +
                                    # * or / the "log_abs"
                                    [n[0]] +
                                    # Put "log_abs" at left of tree
                                    [labels[i]] +
                                    # Skip the d operators which give the number
                                    labels[i+d+1:j] +
                                    # Put number at right of * or /
                                    [n[1:]] +
                                    # Rest of tree
                                    labels[j:]
                                    )
                        new_shape = orig_shape[:i] + \
                                    [2] + \
                                    [orig_shape[i]] + \
                                    orig_shape[i+d+1:j] + \
                                    [0] + \
                                    orig_shape[j:]
                elif exp_ord[labels[i]] == 2:
                    if int(n[1:]) == 1:
                        new_labels = (
                                    # First part of tree (up to the d operators which make number)
                                    labels[:i-d] +
                                    # The rest of the tree
                                    labels[i:]
                                    )
                        new_shape = orig_shape[:i-d] + \
                                    orig_shape[i:]
                    else:
                        new_labels = (
                                    # First part of tree (up to the d operators which make number)
                                    labels[:i-d] +
                                    # The exp comes next
                                    [labels[i]] +
                                    # * or / the argument of "exp"
                                    [n[0]] +
                                    # First part of argument of "exp" on left of * or /
                                    labels[i+1:j] +
                                    # Put number at right of * or /
                                    [n[1:]] +
                                    # Rest of tree
                                    labels[j:]
                                    )
                        new_shape = orig_shape[:i-d] + \
                                    [orig_shape[i]] + \
                                    [2] + \
                                    orig_shape[i+1:j] + \
                                    [0] + \
                                    orig_shape[j:]
                
                nadded += 1

    return new_labels, new_shape, nadded


def update_sums(tree, labels, try_idx, basis_functions):
    """Try to combine sums to make simpler representations of functions
    
    Args:
        :tree (list): list of Node objects corresponding to tree of function
        :labels (list): list of strings giving node labels of tree
        :try_idx (int): when we have multiple substituions we can attempt, this indicates which one to try
        :basis_functions (list): list of lists basis functions. basis_functions[0] are nullary, basis_functions[1] are unary and basis_functions[2] are binary operators
    
    Returns:
        :new_labels (list): list of strings giving node labels of new tree
        :new_shape (list): list of 0, 1 and 2 representing whether nodes in new tree are nullary, unary or binary
        :nadded (int): number of new functions added
    
    """

    new_shape = None
    new_labels = None
    nadded = 0

    if ("+" not in labels) and ("-" not in labels):
        return new_labels, new_shape, nadded
        
    # Find all the +s or -s which aren't children of other +s or -s
    plus_idx = [i for i in range(len(labels)) if labels[i] == "+" or labels[i] == "-"]
    plus_idx = [i for i in plus_idx if (tree[i].parent is None) or (labels[tree[i].parent] not in ["+", "-"])]
    
    if try_idx >= len(plus_idx):
        return new_labels, new_shape, nadded
        
    i = plus_idx[try_idx]
    orig_parents = np.array([t.parent for t in tree])
    orig_parents[0] = -1
    
    # Find all terms in sum (sometimes this happens over multiple layers of tree if nested +s or -s)
    def get_sum(j, run_anyway):
        r = run_anyway
        if labels[j] in ["+", "-"]:
            sum_list = []
            idx_list = []
            s, idx, r2 = get_sum(tree[j].left, r)
            if r2:
                r = True
            if len(s) > 0:
                if type(s[0]) is str:
                    sum_list.append(s)
                    idx_list.append(idx)
                else:
                    for k in range(len(s)):
                        sum_list.append(s[k])
                        idx_list.append(idx[k])
            s, idx, r2 = get_sum(tree[j].right, r)
            if r2:
                r = True
            if len(s) > 0:
                if type(s[0]) is str:
                    sum_list.append(s)
                    idx_list.append(idx)
                else:
                    for k in range(len(s)):
                        sum_list.append(s[k])
                        idx_list.append(idx[k])
            else:
                r = True
        elif labels[j] == "*" and labels[tree[j].left].lstrip("-").isdigit():
            temp_list, temp_idx, r2 = get_sum(tree[j].right, r)
            if r2:
                r = True
            sum_list = []
            idx_list = []
            for k in range(len(temp_list)):
                if type(temp_list[k][0]) in [str, np.str_]:
                    sum_list += [temp_list[k]] * int(labels[tree[j].left].lstrip("-"))
                    idx_list += [temp_idx[k]] * int(labels[tree[j].left].lstrip("-"))
                else:
                    for s in temp_list[k]:
                        sum_list += s * int(labels[tree[j].left].lstrip("-"))
                        idx_list += temp_idx[k] * int(labels[tree[j].left].lstrip("-"))
        elif labels[j] == "*" and labels[tree[j].right].lstrip("-").isdigit():
            temp_list, temp_idx, r2 = get_sum(tree[j].left, r)
            if r2:
                r = True
            sum_list = []
            idx_list = []
            for k in range(len(temp_list)):
                temp_idx[k][-1] = tree[j].right + 1
                if type(temp_list[k][0]) in [str, np.str_]:
                    sum_list += [temp_list[k]] * int(labels[tree[j].right].lstrip("-"))
                    idx_list += [temp_idx[k]] * int(labels[tree[j].right].lstrip("-"))
                else:
                    for s in temp_list[k]:
                        sum_list += s * int(labels[tree[j].right].lstrip("-"))
                        idx_list += temp_idx[k] * int(labels[tree[j].right].lstrip("-"))
        else:
            k = np.argwhere(orig_parents[j+1:] <= tree[j].parent) + j + 1
            if len(k) == 0:
                k = len(labels)
            else:
                k = k[0,0]
            sum_list = [labels[j:k]]
            idx_list = [[j,k,k]]
        return sum_list, idx_list, r
    
    # run_anyway checks to see if we have any 0* in the sum
    # this won't show up in all_s, but means we will want to
    # try to rewrite this sum
    all_s, all_idx, run_anyway = get_sum(i, False)
    all_start = [tree[j[0]].parent for j in all_idx]
    orig_parents = [tt.parent for tt in tree]
    children = [j for j in range(len(tree)) if tree[j].parent == i] #p
    last_child = children[-1]
    if last_child == len(tree) - 1:
        end_idx = len(tree)
    else:
        end_idx = np.argwhere(np.array(orig_parents[last_child+1:]) < i) #p
        if len(end_idx) == 0:
            end_idx = len(tree)
        else:
            end_idx = end_idx[0,0] + last_child + 1
    
    # Work out the signs of each term in the sum
    all_sign = []
    neg_const = []
    orig_parents = np.array([t.parent for t in tree])
    for j in range(len(all_start)):
        n = 1
        k = all_start[j]
        neg_const.append(False)
        if k is not None and labels[k] in ["*", "/"]:
            if labels[tree[k].left].lstrip("-").isdigit() and labels[tree[k].left].startswith("-"):
                n *= -1
                neg_const[-1] = True
            elif labels[tree[k].right].lstrip("-").isdigit() and labels[tree[k].right].startswith("-"):
                n *= -1
                neg_const[-1] = True

        if (labels[k] == "-") and (tree[k].right == all_idx[j][0]):
            n *= -1
        
        while (tree[k].parent is not None) and (tree[k].parent >= i):
            if (labels[tree[k].parent] == "-") and (tree[tree[k].parent].right == k):
                n *= -1
            if labels[tree[k].parent] in ["*", "/"]:
                if labels[tree[tree[k].parent].left].lstrip("-").isdigit() and labels[tree[tree[k].parent].left].startswith("-"):
                    n *= -1
                elif labels[tree[tree[k].parent].right].lstrip("-").isdigit() and labels[tree[tree[k].parent].right].startswith("-"):
                    n *= -1
                    neg_const[-1] = True
            k = tree[k].parent
        all_sign.append(n)

    # Get unique terms in sum
    s = []
    for ss in all_s:
        if len(ss) > 0 and type(ss[0]) == list:
            s.append(tuple(list(**s)))
        else:
            s.append(tuple(list(ss)))
    s = sorted(set(s), key=s.index)
    s = [list(ss) for ss in s]
    
    new_labels = []
    new_shape = []
    
    if (len(s) != len(all_s)) or run_anyway:
        for j in range(len(s)):
                
            l = labels[:i]
            t = [tt.type for tt in tree[:i]]

            rep = [a for a in range(len(all_s)) if all_s[a] == s[j]]
            rep_val = np.array([all_sign[a] for a in rep], dtype=int).sum()
            uni = [all_s.index(s[a]) for a in range(len(s)) if s[a] != s[j]]

            # Add the unique stuff
            # Always adding to the right
            l_uni = []
            n_uni = []
            t_uni = []
            for a in uni:
                nrep = [all_sign[b] for b in range(len(all_sign)) if (all_s[a] == all_s[b])]
                len_nrep = len(nrep)
                nrep = sum(nrep)
                
                if (neg_const[a] == True) and (len_nrep == 1):
                    # If neg_const try to get the version with the - instead of + (or vice versa)
                    left_idx = tree[tree[all_idx[a][0]].parent].left
                    right_idx = tree[tree[all_idx[a][0]].parent].right
                    if tree[tree[all_idx[a][0]].parent].left == all_idx[a][0]:
                        if nrep == 1:
                            l_uni = labels[all_idx[a][0]:all_idx[a][1]] + l_uni
                            t_uni = [tt.type for tt in tree[all_idx[a][0]:all_idx[a][1]]] + t_uni
                            n_uni = n_uni + ['+']
                        elif nrep != 0:
                            l_uni = ['*', str(nrep)] + labels[all_idx[a][0]:all_idx[a][1]] + l_uni
                            t_uni = [2, 0] + [tt.type for tt in tree[all_idx[a][0]:all_idx[a][1]]] + t_uni
                            n_uni = n_uni + ['+']
                    else:
                        if labels[left_idx].lstrip("-").isdigit() and labels[left_idx].startswith("-"):
                            x = labels[left_idx].lstrip("-")
                            if x == str(1) and nrep == 1:
                                l_uni = labels[left_idx+1:all_idx[a][1]] + l_uni
                                t_uni = [tt.type for tt in tree[left_idx+1:all_idx[a][1]]] + t_uni
                            else:
                                l_uni = labels[all_idx[a][0]:left_idx] + \
                                        ["*", str(abs(nrep))] + \
                                        labels[left_idx+1:all_idx[a][1]] + l_uni
                                t_uni = [tt.type for tt in tree[all_idx[a][0]:left_idx]] + \
                                        [2, 0] + \
                                        [tt.type for tt in tree[left_idx+1:all_idx[a][1]]] + t_uni
                        elif right_idx is None:
                            if nrep == 1:
                                l_uni = labels[left_idx-1:all_idx[a][1]] + l_uni
                                t_uni = [tt.type for tt in tree[left_idx-1:all_idx[a][1]]] + t_uni
                            else:
                                l_uni = labels[all_idx[a][0]:left_idx-1] + \
                                        ["*", str(abs(nrep))] + \
                                        labels[left_idx-1:all_idx[a][1]] + l_uni
                                t_uni = [tt.type for tt in tree[all_idx[a][0]:left_idx-1]] + \
                                        [2, 0] + \
                                        [tt.type for tt in tree[left_idx-1:all_idx[a][1]]] + t_uni
                        else:
                            x = labels[right_idx].lstrip("-")
                            if x == str(1) and nrep == 1:
                                l_uni = labels[all_idx[a][0]+1:right_idx] + labels[right_idx+1:all_idx[a][1]] + l_uni
                                t_uni = [tt.type for tt in tree[all_idx[a][0]+1:right_idx]] + \
                                        [tt.type for tt in tree[right_idx+1:all_idx[a][1]]] + t_uni
                            elif nrep != 0:
                                l_uni = labels[all_idx[a][0]:right_idx] + \
                                    ["*", str(abs(nrep))] + \
                                    labels[right_idx+1:all_idx[a][1]] + l_uni
                                t_uni = [tt.type for tt in tree[all_idx[a][0]:right_idx]] + \
                                        [2, 0] + \
                                        [tt.type for tt in tree[right_idx+1:all_idx[a][1]] ] + t_uni
                        if all_sign[a] == 1:
                            n_uni = n_uni + ['+']
                        else:
                            n_uni = n_uni + ['-']
                elif all_sign[a] == 1:
                    if nrep == 1:
                        l_uni = labels[all_idx[a][0]:all_idx[a][1]] + l_uni
                        t_uni = [tt.type for tt in tree[all_idx[a][0]:all_idx[a][1]]] + t_uni
                        n_uni = n_uni + ['+']
                    elif nrep != 0:
                        l_uni = ['*', str(nrep)] + \
                                labels[all_idx[a][0]:all_idx[a][1]] + l_uni
                        t_uni = [2, 0] + [tt.type for tt in tree[all_idx[a][0]:all_idx[a][1]]] + t_uni
                        n_uni = n_uni + ['+']
                else:
                    if abs(nrep) == 1:
                        l_uni = labels[all_idx[a][0]:all_idx[a][1]] + l_uni
                        t_uni = [tt.type for tt in tree[all_idx[a][0]:all_idx[a][1]]] + t_uni

                    elif nrep != 0:
                        l_uni = ['*', str(abs(nrep))] + \
                                labels[all_idx[a][0]:all_idx[a][1]] + l_uni
                        t_uni = [2, 0] + [tt.type for tt in tree[all_idx[a][0]:all_idx[a][1]]] + t_uni
                    if nrep > 0:
                        n_uni = n_uni + ['+']
                    elif nrep < 0:
                        n_uni = n_uni + ['-']
                        
            l_rep = []
            t_rep = []
            if ((len(rep) == 1) or ("*" in basis_functions[2])) and (rep_val != 0):
                
                if rep_val != 1:
                    l_rep = ['*', str(rep_val)]
                    t_rep = [2, 0]
            
                for a in range(len(s[j])):
                    l_rep.append(s[j][a])
                    t_rep.append(tree[labels.index(s[j][a])].type)

                if len(l_uni) == 0:
                    l = l + l_uni + l_rep
                    t = t + t_uni + t_rep
                else:
                    l = l + n_uni + l_rep + l_uni
                    t = t + [2] * len(n_uni) + t_rep + t_uni
                    
            else:
                
                # Now remove the +/- 0
                if len(n_uni) == 0:
                    l = l + ['0']
                    t = t + [0]
                elif n_uni[-1] == '+':
                    l = l + n_uni[:-1] + l_uni
                    t = t + [2] * (len(n_uni)-1) + t_uni
                else:
                    if n_uni == ["-"] * len(n_uni):
                        #If all the things added are negative, we can change the top node
                        # and make them all +'s provided they are on the right of that node
                        l = l + ["*", "-1"] + ["+"] * (len(n_uni)-1) + l_uni
                        t = t + [2, 0] + [2] * (len(n_uni) - 1) + t_uni
                    else:
                        # Otherwise we can move the right hand side of one of the + nodes
                        # down to bottom left to terminate the tree
                        plus_idx = n_uni.index("+")
                        l_uni = []
                        n_uni = []
                        t_uni = []
                        for k in range(len(uni)):
                            if k != plus_idx:
                                a = uni[k]
                                nrep = [all_sign[b] for b in range(len(all_sign)) if (all_s[a] == all_s[b])]
                                nrep = sum(nrep)
                                if neg_const[a] == True:
                                    # If neg_const try to get the version with the - instead of + (or vice versa)
                                    left_idx = tree[all_idx[a][0]].left
                                    right_idx = tree[all_idx[a][0]].right
                                    if tree[tree[all_idx[a][0]].parent].left == all_idx[a][0]:
                                        if nrep == 1:
                                            l_uni = labels[all_idx[a][0]:all_idx[a][1]] + l_uni
                                            t_uni = [tt.type for tt in tree[all_idx[a][0]:all_idx[a][1]]] + t_uni
                                            n_uni = n_uni + ['+']
                                        elif nrep != 0:
                                            l_uni = ['*', str(nrep)] + labels[all_idx[a][0]:all_idx[a][1]] + l_uni
                                            t_uni = [2, 0] + [tt.type for tt in tree[all_idx[a][0]:all_idx[a][1]]] + t_uni
                                            n_uni = n_uni + ['+']
                                    else:
                                        if (labels[left_idx].lstrip("-").isdigit() and labels[left_idx].startswith("-")):
                                            x = labels[left_idx].lstrip("-")
                                            if x == str(1) and nrep == 1:
                                                l_uni = labels[left_idx+1:all_idx[a][1]] + l_uni
                                                t_uni = [tt.type for tt in tree[left_idx+1:all_idx[a][1]]] + t_uni
                                            elif nrep != 0:
                                                l_uni = labels[all_idx[a][0]:left_idx] + \
                                                        ["*", str(abs(nrep))] + \
                                                        labels[left_idx+1:all_idx[a][1]] + l_uni
                                                t_uni = [tt.type for tt in tree[all_idx[a][0]:left_idx]] + \
                                                        [2, 0] + \
                                                        [tt.type for tt in tree[left_idx+1:all_idx[a][1]]] + t_uni
                                        elif right_idx is None:
                                            if nrep == 1:
                                                l_uni = labels[left_idx-1:all_idx[a][1]] + l_uni
                                                t_uni = [tt.type for tt in tree[left_idx-1:all_idx[a][1]]] + t_uni
                                            else:
                                                l_uni = labels[all_idx[a][0]:left_idx-1] + \
                                                        ["*", str(abs(nrep))] + \
                                                        labels[left_idx-1:all_idx[a][1]] + l_uni
                                                t_uni = [tt.type for tt in tree[all_idx[a][0]:left_idx-1]] + \
                                                        [2, 0] + \
                                                        [tt.type for tt in tree[left_idx-1:all_idx[a][1]]] + t_uni
                                        else:
                                            x = labels[right_idx].lstrip("-")
                                            if x == str(1) and nrep == 1:
                                                l_uni = labels[all_idx[a][0]+1:right_idx] + labels[right_idx+1:all_idx[a][1]] + l_uni
                                                t_uni = [tt.type for tt in tree[all_idx[a][0]+1:right_idx]] + \
                                                        [tt.type for tt in tree[right_idx+1:all_idx[a][1]]] + t_uni
                                            elif nrep != 0:
                                                l_uni = labels[all_idx[a][0]:right_idx] + \
                                                        ["*", str(abs(nrep))] + \
                                                        labels[right_idx+1:all_idx[a][1]] + l_uni
                                                t_uni = [tt.type for tt in tree[all_idx[a][0]:right_idx]] + \
                                                        [2, 0] + \
                                                        [tt.type for tt in tree[right_idx+1:all_idx[a][1]] ] + t_uni
                                        if all_sign[a] == 1:
                                            n_uni = n_uni + ['+']
                                        else:
                                            n_uni = n_uni + ['-']
                                elif all_sign[a] == 1:
                                    if nrep == 1:
                                        l_uni = labels[all_idx[a][0]:all_idx[a][1]] + l_uni
                                        t_uni = [tt.type for tt in tree[all_idx[a][0]:all_idx[a][1]]] + t_uni
                                        n_uni = n_uni + ['+']
                                    elif nrep != 0:
                                        l_uni = ['*', str(nrep)] + \
                                                labels[all_idx[a][0]:all_idx[a][1]] + l_uni
                                        t_uni = [2, 0] + [tt.type for tt in tree[all_idx[a][0]:all_idx[a][1]]] + t_uni
                                        n_uni = n_uni + ['+']
                                else:
                                    if abs(nrep) == 1:
                                        l_uni = labels[all_idx[a][0]:all_idx[a][1]] + l_uni
                                        t_uni = [tt.type for tt in tree[all_idx[a][0]:all_idx[a][1]]] + t_uni
                                        n_uni = n_uni
                                    elif nrep != 0:
                                        l_uni = ['*', str(abs(nrep))] + \
                                                labels[all_idx[a][0]:all_idx[a][1]] + l_uni
                                        t_uni = [2, 0] + [tt.type for tt in tree[all_idx[a][0]:all_idx[a][1]]] + t_uni
                                        n_uni = n_uni

                                    if nrep > 0:
                                        n_uni = n_uni + ['+']
                                    elif nrep < 0:
                                        n_uni = n_uni + ['-']
                                        
                        a = uni[plus_idx]
                        nrep = [all_sign[b] for b in range(len(all_sign)) if (all_s[a] == all_s[b])]
                        nrep = sum(nrep)
                        if nrep == 1:
                            l = l + n_uni + labels[all_idx[a][0]:all_idx[a][1]] + l_uni
                            t = t + [2] * len(n_uni) + \
                                [tt.type for tt in tree[all_idx[a][0]:all_idx[a][1]]] + t_uni
                        else:
                            l = l + n_uni + ['*', str(nrep)] + \
                                labels[all_idx[a][0]:all_idx[a][1]] + l_uni
                            t = t + [2] * len(n_uni) + [2, 0] + \
                                [tt.type for tt in tree[all_idx[a][0]:all_idx[a][1]]] + t_uni
            
            l += labels[end_idx:]
            t += [tt.type for tt in tree[end_idx:]]
            new_labels.append(l)
            new_shape.append(t)

            nadded += 1
    
    if nadded == 1:
        new_labels = new_labels[0]
        new_shape = new_shape[0]

    return new_labels, new_shape, nadded
    
    
def find_additional_trees(tree, labels, basis_functions):
    """For a given tree, try to find all simpler representations of the function by combining sums, exponentials and powers
    
    Args:
        :tree (list): list of Node objects corresponding to tree of function
        :labels (list): list of strings giving node labels of tree
        :basis_functions (list): list of lists basis functions. basis_functions[0] are nullary, basis_functions[1] are unary and basis_functions[2] are binary operators
        
    Returns:
        :new_tree (list): list of equivalent trees, given as lists of Node objects
        :new_labels (list): list of lists of strings giving node labels of new_tree
    """

    new_tree = [tree]
    new_labels = [labels]
    try_idx = [0]
    old_len = 0
    
    # Try log, exp and sqrt changes
    while len(new_tree) != old_len:
        old_len = len(new_tree)
        for i in range(old_len):
            l, s, n = update_tree(new_tree[i],
                        new_labels[i],
                        try_idx[i],
                        basis_functions)

            if (s is not None) and (l not in new_labels):
                if n == 1:
                    _, _, t = check_tree(s)
                    new_tree.append(t)
                    new_labels.append(l)
                    try_idx.append(0)
                else:
                    for j in range(n):
                        if (l[j] not in new_labels):
                            _, _, t = check_tree(s[j])
                            new_tree.append(t)
                            new_labels.append(l[j])
                            try_idx.append(0)
            try_idx[i] += 1
        
    # Try sum changes
    try_idx = [0] * len(try_idx)
    old_len = 0
    while len(new_tree) != old_len:
        old_len = len(new_tree)
        for i in range(old_len):
            l, s, n = update_sums(new_tree[i],
                        new_labels[i],
                        try_idx[i],
                        basis_functions)

            if (s is not None) and (l not in new_labels):
                if n == 1:
                    _, _, t = check_tree(s)
                    max_param = max(1, len([a for a in new_labels[i] if a.startswith('a')]))
                    f = [node_to_string(0, new_tree[i], new_labels[i]), node_to_string(0, t, l)]
                    try:
                        _, sym = simplifier.initial_sympify(f, max_param, verbose=False, parallel=False)
                        if len(sym) != 1:
                            print('Maybe bad (not keeping):', new_labels[i], '\t', sym[0], '\t', sym[1])
                        else:
                            new_tree.append(t)
                            new_labels.append(l)
                            try_idx.append(0)
                    except:
                        print('Failed sympy (not keeping):', new_labels[i], '\t', l)
                        
                else:
                    for j in range(n):
                        if (l[j] not in new_labels):
                            _, _, t = check_tree(s[j])
                            max_param = max(1, len([a for a in new_labels[i] if a.startswith('a')]))
                            f = [node_to_string(0, new_tree[i], new_labels[i]), node_to_string(0, t, l[j])]
                            try:
                                _, sym = simplifier.initial_sympify(f, max_param, verbose=False, parallel=False)
                                #if not sym[0].equals(sym[1]):
                                if len(sym) != 1:
                                    print('Maybe bad (not keeping):', new_labels[i], '\t', sym[0], '\t', sym[1])
                                else:
                                    new_tree.append(t)
                                    new_labels.append(l[j])
                                    try_idx.append(0)
                            except:
                                print('Failed sympy (not keeping):', new_labels[i], '\t', l[j])
                                
            if n <= 1:
                try_idx[i] += 1

    return new_tree, new_labels
    
    
def shape_to_functions(s, basis_functions):
    """Find all possible functions formed from the given list of 0s, 1s and 2s defining a tree and basis functions
    
    Args:
        :s (str): string comprised of 0, 1 and 2 representing tree of nullary, unary and binary nodes
        :basis_functions (list): list of lists basis functions. basis_functions[0] are nullary, basis_functions[1] are unary and basis_functions[2] are binary operators
    
    Returns:
        :all_fun (list): list of strings containing all functions generated directly from tree
        :all_tree (list): list of lists of Node objects corresponding to the trees of functions in all_fun
        :extra_fun (list): list of strings containing functions generated by combining sums, exponentials and powers of the functions in all_fun
        :extra_tree (list): list of lists of Node objects corresponding to the trees of functions in extra_fun
        :extra_orig (list): list of strings corresponding to original versions of extra_fun, as found in all_fun
    
    """

    n0 = np.sum(s == 0)
    n1 = np.sum(s == 1)
    n2 = np.sum(s == 2)
    
    t0 = [list(t) for t in itertools.product(basis_functions[0], repeat = n0)]
    t1 = [list(t) for t in itertools.product(basis_functions[1], repeat = n1)]
    t2 = [list(t) for t in itertools.product(basis_functions[2], repeat = n2)]
    
    # Rename parameters so appear in order
    for i in range(len(t0)):
        indices = [j for j, x in enumerate(t0[i]) if x == 'a']
        for j in range(len(indices)):
            t0[i][indices[j]] = 'a%i'%j
    
    success, part_considered, tree = check_tree(s)
    
    all_fun = [None] * (len(t0) * len(t1) * len(t2))
    if rank == 0:
        all_tree = [None] * (len(t0) * len(t1) * len(t2))
    else:
        all_tree = None
    
    pos = 0
    labels = np.empty(len(s), dtype='U100')
    m0 = (s == 0)
    m1 = (s == 1)
    m2 = (s == 2)
    
    t0 = np.array(t0)
    t1 = np.array(t1)
    t2 = np.array(t2)
    
    extra_tree = []
    extra_fun = []
    extra_orig = []

    i = utils.split_idx(len(t0) * len(t1) * len(t2), rank, size)
    if len(i) == 0:
        imin = 0
        imax = 0
    else:
        imin = i[0]
        imax = i[-1] + 1 
    
    for i in range(len(t0)):
        for j in range(len(t1)):
            for k in range(len(t2)):
            
                labels[:] = None
                labels[m0] = t0[i,:]
                labels[m1] = t1[j,:]
                labels[m2] = t2[k,:]

                if rank == 0:
                    all_tree[pos] = labels.copy()
                all_fun[pos] = node_to_string(0, tree, labels)

                if (pos >= imin) and (pos < imax):
                    new_tree, new_labels = find_additional_trees(tree, list(labels), basis_functions)
                    if len(new_tree) > 1:
                        for n in range(1, len(new_tree)):
                            extra_tree.append(new_labels[n].copy())
                            extra_fun.append(node_to_string(0, new_tree[n], new_labels[n]))
                            extra_orig.append(all_fun[pos])
                pos += 1

    extra_tree = comm.gather(extra_tree, root=0)
    extra_fun = comm.gather(extra_fun, root=0)
    extra_orig = comm.gather(extra_orig, root=0)
    if rank == 0:
        extra_tree = list(itertools.chain(*extra_tree))
        extra_fun = list(itertools.chain(*extra_fun))
        extra_orig = list(itertools.chain(*extra_orig))
        print('Number of extra trees for', s, len(extra_fun))
    extra_fun = comm.bcast(extra_fun, root=0)
    extra_orig = comm.bcast(extra_orig, root=0)
    
    comm.Barrier()

    return all_fun, all_tree, extra_fun, extra_tree, extra_orig


def labels_to_shape(labels, basis_functions):
    """Find the representation of the shape of a tree given its labels
    
    Args:
        :labels (list): list of strings giving node labels of tree
        :basis_functions (list): list of lists basis functions. basis_functions[0] are nullary, basis_functions[1] are unary and basis_functions[2] are binary operators
    
    Returns:
        :s (str): string comprised of 0, 1 and 2 representing tree of nullary, unary and binary nodes
        
    """

    basis_dict = {}
    for i in range(len(basis_functions)):
        for f in basis_functions[i]:
            basis_dict[f] = i

    s = [None] * len(labels)
    for i, t in enumerate(labels):
        try:
            s[i] = basis_dict[t]
        except:
            if (t.startswith('a') and t[1:].isdigit()) or (t.isdigit()):
                s[i] = 0
            else:
                raise ValueError
    return s


def aifeyn_complexity(tree, param_list):
    """Compute contribution to description length from describing tree
    
    Args:
        :tree (list): list of strings giving node labels of tree
        :param_list (list): list of strings of all possible parameter names
    
    Returns:
        :aifeyn (float): the contribution to description length from describing tree
    
    """

    t = [tt for tt in tree if (tt not in param_list) and (not tt.lstrip("-").isdigit())]  # Operators
    n = np.array([int(tt) for tt in tree if tt.lstrip("-").isdigit()])  # Integers
    n[n==0] = 1  # So we have log(1) for 0 instead of log(0)
    has_param = int(len(t) != len(tree))  # Has either an a0 or an integer
    nop = len(set(t)) + has_param
    return len(tree) * np.log(nop) + np.sum(np.log(np.abs(n)))


def generate_equations(compl, basis_functions, dirname):
    """Generate all equations at a given complexity for a set of basis functions and save results to file
    
    Args:
        :compl (int): complexity of functions to consider
        :basis_functions (list): list of lists basis functions. basis_functions[0] are nullary, basis_functions[1] are unary and basis_functions[2] are binary operators
        :dirname (str): directory path to save results in
    
    Returns:
        :all_fun (list): list of strings containing all functions generated
        :extra_orig (list): list of strings containing functions generated by combining sums, exponentials and powers of the functions in all_fun as they appear in all_fun
    
    """

    shapes = get_allowed_shapes(compl)

    nfun = np.empty((shapes.shape[0], 3))
    for i in range(3):
        nfun[:,i] = len(basis_functions[i]) ** np.sum(shapes == i, axis=1)
    nfun = np.prod(nfun, axis=1)
    
    if rank == 0:
        print('\nNumber of topologies:', shapes.shape[0])
        for i in range(shapes.shape[0]):
            print(shapes[i,:], int(nfun[i]))
        sys.stdout.flush()
        
    nfun = np.sum(nfun)
    if rank == 0:
        print('\nOriginal number of trees:', int(nfun))
    sys.stdout.flush()

    all_fun = [None] * len(shapes)
    extra_fun = [None] * len(shapes)
    extra_orig = [None] * len(shapes)
    sys.stdout.flush()
    comm.Barrier()

    # Clear the files
    if rank == 0:
        for fname in ['orig_trees', 'extra_trees', 'orig_aifeyn', 'extra_aifeyn']:
            with open(dirname + '/%s_%i.txt'%(fname,compl), 'w') as f:
                pass

    ntree = 0
    nextratree = 0

    for i in range(len(shapes)):
        if rank == 0:
            print('\n%i of %i'%(i+1, len(shapes)))
            print(shapes[i])
            sys.stdout.flush()
        all_fun[i], all_tree, extra_fun[i], extra_tree, extra_orig[i] = shape_to_functions(shapes[i], basis_functions)

        if rank == 0:
            ntree += len(all_tree)
            nextratree += len(extra_tree)

        max_param = simplifier.get_max_param(all_fun[i], verbose=False)
        param_list = ['a%i'%j for j in range(max_param)]

        if rank == 0:

            print(i, len(all_fun[i]), len(all_tree), len(extra_fun[i]), len(extra_tree), len(extra_orig[i]))
            print(len(all_fun[i]) == len(all_tree), len(extra_fun[i]) == len(extra_tree))

            with open(dirname + '/orig_trees_%i.txt'%compl, 'a') as f:
                w = 80
                pp = pprint.PrettyPrinter(width=w, stream=f)
                for t in all_tree:
                    s = str(t)
                    if len(s + '\n') > w / 2:
                        w = 2 * len(s)
                        pp = pprint.PrettyPrinter(width=w, stream=f)
                    pp.pprint(s)

            with open(dirname + '/extra_trees_%i.txt'%compl, 'a') as f:
                w = 80
                pp = pprint.PrettyPrinter(width=w, stream=f)
                for t in extra_tree:
                    s = str(t)
                    if len(s + '\n') > w / 2:
                        w = 2 * len(s)
                        pp = pprint.PrettyPrinter(width=w, stream=f)
                    pp.pprint(s)

            with open(dirname + '/orig_aifeyn_%i.txt'%compl, 'a') as f:
                for tree in all_tree:
                    print(aifeyn_complexity(tree, param_list), file=f)

            with open(dirname + '/extra_aifeyn_%i.txt'%compl, 'a') as f:
                for tree in extra_tree:
                    print(aifeyn_complexity(tree, param_list), file=f)

    if rank == 0:
        s = 'cat %s/orig_trees_%i.txt %s/extra_trees_%i.txt > %s/trees_%i.txt'%(dirname,compl,dirname,compl,dirname,compl)
        print('\n%s'%s)
        sys.stdout.flush()
        os.system(s)
        s = 'cat %s/orig_aifeyn_%i.txt %s/extra_aifeyn_%i.txt > %s/aifeyn_%i.txt'%(dirname,compl,dirname,compl,dirname,compl)
        print('\n%s'%s)
        sys.stdout.flush()
        os.system(s)

        print('\nntree:', ntree)
        print('nextratree:', nextratree)
        print('sum:', ntree + nextratree)

    all_fun = list(itertools.chain(*all_fun))
    extra_fun = list(itertools.chain(*extra_fun))
    extra_orig = list(itertools.chain(*extra_orig))

    if rank == 0:
        print('\nall_fun', len(all_fun))
        print('extra_fun', len(extra_fun))
        print('extra_orig', len(extra_orig))
        sys.stdout.flush()

    all_fun = all_fun + extra_fun
    
    if rank == 0:
        print('\nNew number of trees:', len(all_fun))
    sys.stdout.flush()
    
    return all_fun, extra_orig
