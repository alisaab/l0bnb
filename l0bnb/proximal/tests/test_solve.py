import numpy as np
import regreg.api as rr

from l0bnb.proximal import (perspective_bound_atom,
                            perspective_lagrange_atom,
                            perspective_bound_atom_conjugate,
                            perspective_lagrange_atom_conjugate)

def test_bound_solve():

    n, p = 100, 50
    v = np.random.standard_normal(100)

    X = np.random.standard_normal((n, p))
    Y = np.random.standard_normal(n)

    lips, lam_2, M, C = 1.5, 0.02, 2, 5
    atom = perspective_bound_atom((p,),
                                  lam_2,
                                  M,
                                  C)
    loss = rr.squared_error(X, Y)

    problem = rr.simple_problem(loss, atom)
    problem.solve(debug=True, tol=1e-7, min_its=50)

def test_lagrange_solve():

    n, p = 100, 50
    v = np.random.standard_normal(100)

    X = np.random.standard_normal((n, p))
    Y = np.random.standard_normal(n)

    lips, lam_2, M, lam_0 = 1.5, 0.02, 2, 0.2
    atom = perspective_lagrange_atom((p,),
                                     lam_2,
                                     M,
                                     lam_0)
    loss = rr.squared_error(X, Y)

    problem = rr.simple_problem(loss, atom)
    problem.solve(debug=True, tol=1e-7, min_its=50)

def test_bound_conjugate_solve():

    n, p = 100, 50
    v = np.random.standard_normal(100)

    X = np.random.standard_normal((n, p))
    Y = np.random.standard_normal(n)

    lips, lam_2, M, C = 1.5, 0.02, 2, 5
    atom = perspective_bound_atom_conjugate((p,),
                                            lam_2,
                                            M,
                                            C)
    loss = rr.squared_error(X, Y)

    problem = rr.simple_problem(loss, atom)
    problem.solve(debug=True, tol=1e-7, min_its=50)

def test_lagrange_conjugate_solve():

    n, p = 100, 50
    v = np.random.standard_normal(100)

    X = np.random.standard_normal((n, p))
    Y = np.random.standard_normal(n)

    lips, lam_2, M, lam_0 = 1.5, 0.02, 2, 0.2
    atom = perspective_lagrange_atom_conjugate((p,),
                                               lam_2,
                                               M,
                                               lam_0)
    loss = rr.squared_error(X, Y)

    problem = rr.simple_problem(loss, atom)
    problem.solve(debug=True, tol=1e-7, min_its=50)
    
