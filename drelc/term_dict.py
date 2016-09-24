# -*- coding: utf-8 -*-
"""
Created on Mon Jun 01 20:03:59 2015

@author: Richard
"""


from collections import defaultdict, MutableMapping
import operator
from functools import total_ordering
from abc import ABCMeta, abstractmethod, abstractproperty
from sympy_helper_fns import is_monic, expressions_to_variables, num_add_terms
import sympy

from sympy.core.exprtools import Term

PRUNING = True

SYMPY_VAR_TEMPLATE = 'z_{}'

@total_ordering
class MonomialVariables(object):
    ''' An immutable collection of variables, which form the non-constant part of a monomial '''

    @abstractmethod
    def __init__(self, variables, exponents):
        pass

    def __iter__(self):
        return iter(self.variables)

    def __hash__(self):
        return hash(self.variables)

    def degree(self):
        ''' Return the degree of the monomial variables '''
        return sum(self.exponents)

    @abstractproperty
    def variables(self):
        ''' Return an iterable of variables '''
        pass

    @abstractproperty
    def exponents(self):
        ''' Return the exponents of the variables, ordered as in self.variables '''
        pass

    def __eq__(self, other):
        ''' Define equality '''
        if isinstance(other, MonomialVariables):
            return self.variables == other.variables
        return False

    def __lt__(self, other):
        ''' Useful for defining a monomial ordering '''
        if isinstance(other, MonomialVariables):
            if self.degree != other.degree:
                return self.degree < other.degree
            return self.variables == other.variables
        return False

    def __str__(self):
        return ''.join(map(str, self.variables))

    def to_coef_line(self):
        ''' Return the representation of these variables, as required by a coef_str '''
        return ' '.join(map(str, self.variables))

    @abstractmethod
    def from_coef_line(self):
        ''' Inverse of the to_coef_line '''
        pass


    @abstractmethod
    def as_sympy_expr(self):
        ''' Express as a sympy expression '''
        pass

class SympyVariables(MonomialVariables):
    ''' Use sympy '''

    def __init__(self, sympy_expr):
        assert is_monic(sympy_expr)
        assert all([isinstance(atom, sympy.symbol.Symbol) for atom in sympy_expr.atoms()]) or (sympy_expr is sympy.S.One)
        assert '+' not in str(sympy_expr)
        self._sympy_expr = sympy_expr
        self._variables = tuple(sorted(sympy_expr.atoms(), key=str))

    @property
    def variables(self):
        ''' Return the variables themselves '''
        return self._variables

    @property
    def exponents(self):
        ''' Return all 1s '''
        return tuple([1 for _ in xrange(len(self._variables))])

    @classmethod
    def from_coef_line(cls, string):
        ''' Create a class instance from a string where variables are separated by a ' '. '''
        ids = string.split(' ')
        variables = [SYMPY_VAR_TEMPLATE.format(_id) for _id in ids]
        sympy_expr = '*'.join(variables)
        return cls(sympy.sympify(sympy_expr))

    def __str__(self):
        return str(self._sympy_expr)

    def as_sympy_expr(self):
        '''' Easy '''
        return self._sympy_expr

class TupleMonomialVariables(MonomialVariables):
    ''' Use ids sorted in a tuple '''
    def __init__(self, variables):
        self._variables = tuple(sorted(variables))

    @property
    def variables(self):
        ''' Return the variables themselves '''
        return self._variables

    @property
    def exponents(self):
        ''' Return all 1s '''
        return tuple([1 for _ in xrange(len(self._variables))])

    @classmethod
    def from_coef_line(cls, string):
        ''' Create a class instance from a string where variables are separated by a ' '. '''
        return cls(' '.split(string))


    def as_sympy_expr(self):
        '''' Return a sympy expression '''
        variables = [SYMPY_VAR_TEMPLATE.format(_id) for _id in self._variables]
        sympy_expr = '*'.join(variables)
        return cls(sympy.sympify(sympy_expr))


class TermDict(MutableMapping):
    ''' TermDict is a dictionary designed to represent a polynomial, allowing
        for constant lookup time of terms
    '''
    
    def __init__(self, default_type=int):
        self.default_type = default_type
        self._term_dict = defaultdict(default_type)
    
    def __delitem__(self, key):
        del self._term_dict[key]
    
    def __getitem__(self, key):
        value = self._term_dict[key]
        if PRUNING and (value == self.default_type()):
            self._term_dict.pop(key, None)
        return value

    def __setitem__(self, key, value):
        if not isinstance(key, MonomialVariables):
            raise ValueError('{} is not a MonomialVariables instance'.format(key))
        self._term_dict[key] = value
 
        if (PRUNING and (value == self.default_type())):
            self._term_dict.pop(key, None)
    
    def __iter__(self):
        return self._term_dict.__iter__()
    
    def __len__(self):
        return len(self._term_dict)
    
    def __repr__(self):
        #return self._term_dict.__repr__()
        sorted_ = sorted(self.items(), key=lambda x: len(str(x[0])))
        return '\n'.join(['{}: {}'.format(k, v) for k, v in sorted_])

    def copy(self):
        copy = type(self)(self.default_type)
        copy._term_dict = self._term_dict.copy()
        return copy


    ## Arithmetic functions
    def __additive_op__(self, other, operation):
        ''' Support addition 
        
            >>> td = TermDict()
            >>> x, y = sympy.var('x y')
            >>> td[SympyVariables(x)] = 1
            >>> print td
            x: 1
            >>> td[SympyVariables(x)] -= 1
            >>> print td
            <BLANKLINE>
            
            >>> td[SympyVariables(x)] = td[SympyVariables(y)] = td[SympyVariables(x*y)] = 1
            >>> print td
            x: 1
            y: 1
            x*y: 1

            >>> td = td + 1
            >>> print td
            x: 2
            y: 2
            x*y: 2

            >>> print td - 2
            <BLANKLINE>
            
            >>> print td + td
            x: 4
            y: 4
            x*y: 4
        '''
        result = self.copy()

        if isinstance(other, (int, float)):
            for _key, _value in self.iteritems():
                result[_key] = operation(_value, other)
            return result

        if isinstance(other, TermDict):
            for _key, _value in other.iteritems():
                result[_key] = operation(self[_key], _value)
            return result

        raise ValueError('Unknown type of other {}: {}'.format(other, type(other)))
    
    def __add__(self, other):
        return self.__additive_op__(other, operator.add)
    
    def __sub__(self, other):
        return self.__additive_op__(other, operator.sub)

    @classmethod
    def from_coef_str(cls, coef_str, variable_holder_type=SympyVariables, default_type=int):
        ''' Read a string of coefficients and return a TermDict '''
        term_dict = cls()
        for line in coef_str.split('\n'):
            line = line.replace('\r', '')  # Remove silly windows newline characters
            if not line:
                break
            atoms = line.split(' ')
            variables = variable_holder_type.from_coef_line(' '.join(atoms[:-1]))
            coef = default_type(atoms[-1])
            # Add on the coefficients in case the same variables crop up twice on two different lines
            term_dict[variables] += coef
        return term_dict

    def to_coef_str(self):
        ''' Print out the coefficient string '''
        lines = []
        for variables in sorted(self):
            lines.append(' '.join([variables.to_coef_line, self[variables]]))
        return '\n'.join(lines)

    @property
    def variables(self):
        ''' Return a set of all of the variables this term dict contains

            >>> td = TermDict()
            >>> x, y = sympy.var('x y')
            >>> td[SympyVariables(x)] = td[SympyVariables(y)] = td[SympyVariables(x*y)] = 1
            >>> print td.variables()
            set([x, y])
         '''
        return set.union(*map(set, self))

    def variables_to_terms(self, variables):
        ''' Given an iterable of variables, return a TermDict which contains every term that has an intersection with the variables

            >>> td = TermDict()
            >>> x, y = sympy.var('x y')
            >>> td[SympyVariables(x)] = td[SympyVariables(y)] = td[SympyVariables(x*y)] = 1
            >>> print td.variables_to_terms([x])
            x: 1
            x*y: 1
         '''
        variables = set(variables)

        term_dict = type(self)(default_type=self.default_type)

        for _var, _coef in self.iteritems():
            if variables.intersection(set(_var)):
                term_dict[_var] = _coef
        return term_dict


    def as_sympy_expr(self):
        ''' Return a sympy expression of the TermDict '''
        if len(self) > 40:
            raise ValueError('TermDict too long to convert into sympy!')
        terms = [_var.as_sympy_expr() * _coef for _var, _coef in self.iteritems()]
        return sum(terms)

if __name__ == '__main__':
    import doctest
    doctest.testmod()