import numpy as np

from regreg.atoms import atom, _work_out_conjugate, affine_atom
from regreg.objdoctemplates import objective_doc_templater
from regreg.doctemplates import doc_template_user

from .core import (dual_cost_bound,
                   dual_cost_lagrange,
                   primal_cost_bound,
                   primal_cost_lagrange,
                   lagrange_prox,
                   bound_prox)

### regreg objects for proximal gradient solver

class perspective_bound_atom(atom):

    tol = 1.0e-05

    objective_vars = atom.objective_vars.copy()
    objective_vars['klass'] = 'perspective_bound_atom'
    objective_vars['dualklass'] = 'perspective_bound_atom_conjugate'
    objective_vars['initargs'] = '(4,)'

    def __init__(self,
                 shape,
                 lam_2,
                 M,
                 C,
                 offset=None, 
                 quadratic=None,
                 initial=None):

        atom.__init__(self,
                      shape,
                      offset=offset,
                      quadratic=quadratic,
                      initial=initial)

        self.M = M
        self.lam_2 = lam_2
        self.C = C

    def __eq__(self, other):
        if self.__class__ == other.__class__:
            return ((self.shape == other.shape) and
                    (self.M == other.M) and
                    (self.lam_2 == other.lam_2) and
                    (self.C == lam.C))
        return False

    def __copy__(self):
        return self.__class__(copy(self.shape),
                              M=self.M,
                              lam_2=self.lam_2,
                              C=self.C,
                              offset=copy(self.offset),
                              quadratic=self.quadratic)
    
    def __repr__(self):
        if self.quadratic.iszero:
            return "%s(%s, %s, %s, %s, offset=%s)" % \
                (self.__class__.__name__,
                 repr(self.shape), 
                 repr(self.M),
                 repr(self.lam_2),
                 repr(self.C),
                 repr(self.offset))
        else:
            return "%s(%s, %s, %s, %s, offset=%s, quadratic=%s)" % \
                (self.__class__.__name__,
                 repr(self.shape), 
                 repr(self.M),
                 repr(self.lam_2),
                 repr(self.C),
                 repr(self.offset),
                 self.quadratic)

    @doc_template_user
    def get_conjugate(self):
        if self.quadratic.coef == 0:

            offset, outq = _work_out_conjugate(self.offset, self.quadratic)

            cls = conjugate_pairs[self.__class__]
            atom = cls(self.shape, 
                       M=self.M,
                       lam_2=self.lam_2,
                       C=self.C,
                       offset=offset,
                       quadratic=outq)
        else:
            atom = smooth_conjugate(self)
        self._conjugate = atom
        self._conjugate._conjugate = self
        return self._conjugate
    conjugate = property(get_conjugate)

    @doc_template_user
    def proximal(self, proxq, prox_control=None):
        r"""
        Projection onto the simplex.

        """
        offset, totalq = (self.quadratic + proxq).recenter(self.offset)
        if totalq.coef == 0:
            raise ValueError('lipschitz + quadratic coef must be positive')

        eta = self._basic_prox(-totalq.linear_term, totalq.coef)
        if offset is None:
            return eta
        else:
            return eta + offset

    def _basic_prox(self, prox_arg, lipschitz):
        # lipschitz is ignored because it is a constraint
                    
        if not hasattr(self, 'delta_star_prox_'):
            self.delta_star_prox_ = 1

        # bound_prox is written with term -v^T\beta
        # to put it "inside" a quadratic with

        (self.z_star_prox_,
         prox,
         self.delta_star_prox_) = bound_prox(prox_arg,
                                             lipschitz,
                                             self.lam_2,
                                             self.M,
                                             self.C,
                                             delta_guess=self.delta_star_prox_)                                  
        return prox

    @doc_template_user
    def nonsmooth_objective(self, arg, check_feasibility=False):
        """
        Value of the simplex constraint.

        Always 0 unless `check_feasibility` is True.

        Parameters
        ----------

        arg : `np.ndarray(np.float)`
            Argument of the seminorm.

        check_feasibility : `bool`
            If `True`, then return `np.inf` if appropriate.

        Returns
        -------

        value : `np.float`
            The seminorm of `arg`.

        """
        arg = np.asarray(arg)
        x_offset = self.apply_offset(arg)

        if not hasattr(self, 'delta_star_nonsmooth_'):
            self.delta_star_nonsmooth_ = 1

        (self.z_star_nonsmooth_,
         self.delta_star_nonsmooth_,
         value) = primal_cost_bound(x_offset,
                                   self.lam_2,
                                   self.M,
                                   self.C,
                                   delta_guess=self.delta_star_nonsmooth_)
        
        value += self.quadratic.objective(arg, 'func')
        return value

class perspective_bound_atom_conjugate(atom):

    tol = 1.0e-05

    objective_vars = atom.objective_vars.copy()
    objective_vars['klass'] = 'perspective_bound_atom_conjugate'
    objective_vars['dualklass'] = 'perspective_bound_atom'
    objective_vars['initargs'] = '(4,)'

    def __init__(self,
                 shape,
                 lam_2,
                 M,
                 C,
                 offset=None, 
                 quadratic=None,
                 initial=None):

        atom.__init__(self,
                      shape,
                      offset=offset,
                      quadratic=quadratic,
                      initial=initial)

        self.M = M
        self.lam_2 = lam_2
        self.C = C

    def __eq__(self, other):
        if self.__class__ == other.__class__:
            return ((self.shape == other.shape) and
                    (self.M == other.M) and
                    (self.lam_2 == other.lam_2) and
                    (self.C == lam.C))
        return False

    def __copy__(self):
        return self.__class__(copy(self.shape),
                              M=self.M,
                              lam_2=self.lam_2,
                              C=self.C,
                              offset=copy(self.offset),
                              quadratic=self.quadratic)
    
    def __repr__(self):
        if self.quadratic.iszero:
            return "%s(%s, %s, %s, %s, offset=%s)" % \
                (self.__class__.__name__,
                 repr(self.shape), 
                 repr(self.M),
                 repr(self.lam_2),
                 repr(self.C),
                 repr(self.offset))
        else:
            return "%s(%s, %s, %s, %s, offset=%s, quadratic=%s)" % \
                (self.__class__.__name__,
                 repr(self.shape), 
                 repr(self.M),
                 repr(self.lam_2),
                 repr(self.C),
                 repr(self.offset),
                 self.quadratic)

    @doc_template_user
    def get_conjugate(self):
        if self.quadratic.coef == 0:

            offset, outq = _work_out_conjugate(self.offset, self.quadratic)

            cls = conjugate_pairs[self.__class__]
            atom = cls(self.shape, 
                       M=self.M,
                       lam_2=self.lam_2,
                       C=self.C,
                       offset=offset,
                       quadratic=outq)
        else:
            atom = smooth_conjugate(self)
        self._conjugate = atom
        self._conjugate._conjugate = self
        return self._conjugate
    conjugate = property(get_conjugate)

    @doc_template_user
    def proximal(self, proxq, prox_control=None):
        r"""
        Projection onto the simplex.

        """
        offset, totalq = (self.quadratic + proxq).recenter(self.offset)
        if totalq.coef == 0:
            raise ValueError('lipschitz + quadratic coef must be positive')

        eta = self._basic_prox(-totalq.linear_term, totalq.coef)
        if offset is None:
            return eta
        else:
            return eta + offset

    def _basic_prox(self, prox_arg, lipschitz):
        # lipschitz is ignored because it is a constraint
                    
        if not hasattr(self, 'delta_star_prox_'):
            self.delta_star_prox_ = 1

        # bound_prox is written with term -v^T\beta
        # to put it "inside" a quadratic with

        (self.z_star_prox_,
         dual_prox,
         self.delta_star_prox_) = bound_prox(prox_arg / lipschitz,
                                             1 / lipschitz,
                                             self.lam_2,
                                             self.M,
                                             self.C,
                                             delta_guess=self.delta_star_prox_)                                  
        return (prox_arg - dual_prox) / lipschitz

    @doc_template_user
    def nonsmooth_objective(self, arg, check_feasibility=False):
        """
        Value of the simplex constraint.

        Always 0 unless `check_feasibility` is True.

        Parameters
        ----------

        arg : `np.ndarray(np.float)`
            Argument of the seminorm.

        check_feasibility : `bool`
            If `True`, then return `np.inf` if appropriate.

        Returns
        -------

        value : `np.float`
            The seminorm of `arg`.

        """
        arg = np.asarray(arg)
        x_offset = self.apply_offset(arg)

        value = dual_cost_bound(x_offset,
                                self.lam_2,
                                self.M,
                                self.C)
        
        value += self.quadratic.objective(arg, 'func')
        return value
    
class perspective_lagrange_atom(atom):

    tol = 1.0e-05

    objective_vars = atom.objective_vars.copy()
    objective_vars['klass'] = 'perspective_lagrange_atom'
    objective_vars['dualklass'] = 'perspective_lagrange_atom_conjugate'
    objective_vars['initargs'] = '(4,)'

    def __init__(self,
                 shape,
                 lam_2,
                 M,
                 lam_0,
                 offset=None, 
                 quadratic=None,
                 initial=None):

        atom.__init__(self,
                      shape,
                      offset=offset,
                      quadratic=quadratic,
                      initial=initial)

        self.M = M
        self.lam_2 = lam_2
        self.lam_0 = lam_0

    def __eq__(self, other):
        if self.__class__ == other.__class__:
            return ((self.shape == other.shape) and
                    (self.M == other.M) and
                    (self.lam_2 == other.lam_2) and
                    (self.lam_0 == lam.lam_0))
        return False

    def __copy__(self):
        return self.__class__(copy(self.shape),
                              M=self.M,
                              lam_2=self.lam_2,
                              lam_0=self.lam_0,
                              offset=copy(self.offset),
                              quadratic=self.quadratic)
    
    def __repr__(self):
        if self.quadratic.iszero:
            return "%s(%s, %s, %s, %s, offset=%s)" % \
                (self.__class__.__name__,
                 repr(self.shape), 
                 repr(self.M),
                 repr(self.lam_2),
                 repr(self.lam_0),
                 repr(self.offset))
        else:
            return "%s(%s, %s, %s, %s, offset=%s, quadratic=%s)" % \
                (self.__class__.__name__,
                 repr(self.shape), 
                 repr(self.M),
                 repr(self.lam_2),
                 repr(self.lam_0),
                 repr(self.offset),
                 self.quadratic)

    @doc_template_user
    def get_conjugate(self):
        if self.quadratic.coef == 0:

            offset, outq = _work_out_conjugate(self.offset, self.quadratic)

            cls = conjugate_pairs[self.__class__]
            atom = cls(self.shape, 
                       M=self.M,
                       lam_2=self.lam_2,
                       lam_0=self.lam_0,
                       offset=offset,
                       quadratic=outq)
        else:
            atom = smooth_conjugate(self)
        self._conjugate = atom
        self._conjugate._conjugate = self
        return self._conjugate
    conjugate = property(get_conjugate)

    @doc_template_user
    def proximal(self, proxq, prox_control=None):
        r"""
        Projection onto the simplex.

        """
        offset, totalq = (self.quadratic + proxq).recenter(self.offset)
        if totalq.coef == 0:
            raise ValueError('lipschitz + quadratic coef must be positive')

        eta = self._basic_prox(-totalq.linear_term, totalq.coef)
        if offset is None:
            return eta
        else:
            return eta + offset

    def _basic_prox(self, prox_arg, lipschitz):
        # lipschitz is ignored because it is a constraint
                    
        # lagrange_prox is written with term -v^T\beta
        # to put it "inside" a quadratic with
        # \beta we divide by lipschitz

        # therefore to pull it out we multiply
        # bt lipschitz

        (self.z_star_prox_,
         prox) = lagrange_prox(prox_arg,
                               lipschitz,
                               self.lam_2,
                               self.M,
                               self.lam_0)                                  
        return prox

    @doc_template_user
    def nonsmooth_objective(self, arg, check_feasibility=False):
        """
        Value of the simplex constraint.

        Always 0 unless `check_feasibility` is True.

        Parameters
        ----------

        arg : `np.ndarray(np.float)`
            Argument of the seminorm.

        check_feasibility : `bool`
            If `True`, then return `np.inf` if appropriate.

        Returns
        -------

        value : `np.float`
            The seminorm of `arg`.

        """
        arg = np.asarray(arg)
        x_offset = self.apply_offset(arg)

        (self.z_star_nonsmooth_,
         value) = primal_cost_lagrange(x_offset,
                                       self.lam_2,
                                       self.M,
                                       self.lam_0)
        
        value += self.quadratic.objective(arg, 'func')
        return value

class perspective_lagrange_atom_conjugate(atom):

    tol = 1.0e-05

    objective_vars = atom.objective_vars.copy()
    objective_vars['klass'] = 'perspective_lagrange_atom_conjugate'
    objective_vars['dualklass'] = 'perspective_lagrange_atom'
    objective_vars['initargs'] = '(4,)'

    def __init__(self,
                 shape,
                 lam_2,
                 M,
                 lam_0,
                 offset=None, 
                 quadratic=None,
                 initial=None):

        atom.__init__(self,
                      shape,
                      offset=offset,
                      quadratic=quadratic,
                      initial=initial)

        self.M = M
        self.lam_2 = lam_2
        self.lam_0 = lam_0

    def __eq__(self, other):
        if self.__class__ == other.__class__:
            return ((self.shape == other.shape) and
                    (self.M == other.M) and
                    (self.lam_2 == other.lam_2) and
                    (self.lam_0 == lam.lam_0))
        return False

    def __copy__(self):
        return self.__class__(copy(self.shape),
                              M=self.M,
                              lam_2=self.lam_2,
                              lam_0=self.lam_0,
                              offset=copy(self.offset),
                              quadratic=self.quadratic)
    
    def __repr__(self):
        if self.quadratic.iszero:
            return "%s(%s, %s, %s, %s, offset=%s)" % \
                (self.__class__.__name__,
                 repr(self.shape), 
                 repr(self.M),
                 repr(self.lam_2),
                 repr(self.lam_0),
                 repr(self.offset))
        else:
            return "%s(%s, %s, %s, %s, offset=%s, quadratic=%s)" % \
                (self.__class__.__name__,
                 repr(self.shape), 
                 repr(self.M),
                 repr(self.lam_2),
                 repr(self.lam_0),
                 repr(self.offset),
                 self.quadratic)

    @doc_template_user
    def get_conjugate(self):
        if self.quadratic.coef == 0:

            offset, outq = _work_out_conjugate(self.offset, self.quadratic)

            cls = conjugate_pairs[self.__class__]
            atom = cls(self.shape, 
                       M=self.M,
                       lam_2=self.lam_2,
                       lam_0=self.lam_0,
                       offset=offset,
                       quadratic=outq)
        else:
            atom = smooth_conjugate(self)
        self._conjugate = atom
        self._conjugate._conjugate = self
        return self._conjugate
    conjugate = property(get_conjugate)

    @doc_template_user
    def proximal(self, proxq, prox_control=None):
        r"""
        Projection onto the simplex.

        """
        offset, totalq = (self.quadratic + proxq).recenter(self.offset)
        if totalq.coef == 0:
            raise ValueError('lipschitz + quadratic coef must be positive')

        eta = self._basic_prox(-totalq.linear_term, totalq.coef)
        if offset is None:
            return eta
        else:
            return eta + offset

    def _basic_prox(self, prox_arg, lipschitz):
        # lipschitz is ignored because it is a constraint
                    
        # lagrange_prox is written with term -v^T\beta
        # to put it "inside" a quadratic with
        # \beta we divide by lipschitz

        # therefore to pull it out we multiply
        # bt lipschitz

        (self.z_star_prox_,
         dual_prox) = lagrange_prox(prox_arg / lipschitz,
                                    1 / lipschitz,
                                    self.lam_2,
                                    self.M,
                                    self.lam_0)                                  
        return (prox_arg - dual_prox) / lipschitz

    @doc_template_user
    def nonsmooth_objective(self, arg, check_feasibility=False):
        """
        Value of the simplex constraint.

        Always 0 unless `check_feasibility` is True.

        Parameters
        ----------

        arg : `np.ndarray(np.float)`
            Argument of the seminorm.

        check_feasibility : `bool`
            If `True`, then return `np.inf` if appropriate.

        Returns
        -------

        value : `np.float`
            The seminorm of `arg`.

        """
        arg = np.asarray(arg)
        x_offset = self.apply_offset(arg)

        value = dual_cost_lagrange(x_offset,
                                   self.lam_2,
                                   self.M,
                                   self.lam_0)
        
        value += self.quadratic.objective(arg, 'func')
        return value


conjugate_pairs = {}
for n1, n2 in [(perspective_bound_atom, perspective_bound_atom_conjugate),
               (perspective_lagrange_atom, perspective_lagrange_atom_conjugate)]:
    conjugate_pairs[n1] = n2
    conjugate_pairs[n2] = n1
