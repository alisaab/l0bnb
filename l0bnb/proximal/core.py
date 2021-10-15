r"""
Proximal operators for the maps

$$
\varphi_{B,M,\lambda_2,C}(\beta) = \inf_z  \sum_i \lambda_2 \frac{\beta_i^2}{2  z_i}
$$
s.t. $$
\begin{aligned}
z_i &\in [0,1] \\
-Mz_i &\leq \beta_i \leq Mz_i \\
\sum_i z_i &= C
\end{aligned}
$$

and 

$$
\varphi_{L,M,\lambda_2,\lambda_0}(\beta) = \inf_z \sum_i \left[\lambda_2 \frac{\beta_i^2}{2  z_i} + \lambda_0 z_i \right]
$$
s.t. $$
\begin{aligned}
z_i &\in [0,1] \\
-Mz_i &\leq \beta_i \leq Mz_i \\
\end{aligned}
$$

"""

from copy import copy
import functools

import numpy as np
from scipy.optimize import root_scalar

def dual_cost_bound(conjugate_arg, lam_2, M, C):
    r"""
    Conjugate of $\varphi_{B,M,\lambda_2,C}$.

    $$
    \sup_{\beta} \beta^Tv - \varphi_{B,M,\lambda_2,C}(\beta) 
    $$
    
    with $v$ set to `conjugate_arg`.

    Parameters
    ----------

    conjugate_arg : np.ndarray
        Argument to conjugate function.

    lam_2 : float
        Non-negative float.

    M : float
        Positive float

    C : float
        Non-negative float

    Notes
    -----

    A small increment of 1e-10 is added to `lam_2` to allow it to be set to exactly 0 
    in calling this function.

    """
    v = conjugate_arg

    if not lam_2 >= 0:
        raise ValueError('lam_2 must be non-negative')
    lam_2 += 1e-10 

    idx = np.fabs(v) > M * lam_2
    vals = np.zeros_like(v)
    vals[idx] = M * np.fabs(v[idx]) - M**2 * lam_2 / 2
    vals[~idx] = v[~idx]**2 / (2 * lam_2)
    vals = np.sort(vals)[::-1]
    
    Cf = int(np.floor(C))
    delta = C - Cf
    
    return vals[:Cf].sum() + delta * vals[Cf]

def dual_cost_lagrange(conjugate_arg, lam_2, M, delta):
    r"""
    Conjugate of $\varphi_{B,M,\lambda_2,C}$.

    $$
    \sup_{\beta} \beta^Tv - \varphi_{L,M,\lambda_2,\delta}(\beta) 
    $$
    
    with $v$ set to `conjugate_arg`.

    Parameters
    ----------

    conjugate_arg : np.ndarray
        Argument to conjugate function.

    lam_2 : float
        Non-negative float.

    M : float
        Positive float

    delta : float

    Notes
    -----

    A small increment of 1e-10 is added to `lam_2` to allow it to be set to exactly 0 
    in calling this function.

    """
    v = conjugate_arg
    if not lam_2 >= 0:
        raise ValueError('lam_2 must be non-negative')
    lam_2 += 1e-10 

    idx = np.fabs(v) > M * lam_2
    vals = np.zeros_like(v)
    vals[idx] = M * np.fabs(v[idx]) - M**2 * lam_2 / 2
    vals[~idx] = v[~idx]**2 / (2 * lam_2)
    vals += delta
    return np.maximum(vals, 0).sum()


def primal_cost_lagrange(arg, lam_2, M, delta):
    r"""
    Computes of $\varphi_{L,M,\lambda_2,\delta}(\beta)$.

    with $\beta$ set to `arg`.

    Parameters
    ----------

    conjugate_arg : np.ndarray
        Argument.

    lam_2 : float
        Non-negative float.

    M : float
        Positive float

    delta : float
        Float

    Notes
    -----

    A small increment of 1e-10 is added to `lam_2` to allow it to be set to exactly 0 
    in calling this function.

    """
    HUGE = np.inf

    beta = arg
    if not lam_2 >= 0:
        raise ValueError('lam_2 must be non-negative')
    lam_2 += 1e-10 

    # bounds on z
    
    U = np.ones_like(beta)
    L = np.fabs(beta) / M

    if np.any(L > U * (1 + 1e-7)):
        return np.zeros_like(arg), HUGE

    if delta <= 0:  # z_star=1
        soln = np.ones_like(arg)
        return soln, (0.5 * lam_2 * arg**2 + delta * soln).sum()
    else:
        roots = arg**2 * lam_2 / delta
        soln = np.zeros_like(arg)

        idx1 = roots >= U
        soln[idx1] = U[idx1]

        idx2 = (roots < U) * (roots > L)
        soln[idx2] = roots[idx2]

        idx3 = roots <= L
        soln[idx3] = L[idx3]

        nz = soln > 0
        soln_nz = soln[soln > 0]
        return soln, (lam_2 * 0.5 * arg[nz]**2 / soln_nz + delta * soln_nz).sum()

def primal_cost_bound(arg, lam_2, M, C, delta_guess=1):
    r"""
    Computes of $\varphi_{B,M,\lambda_2,C}(\beta)$.

    with $\beta$ set to `arg`.

    Parameters
    ----------

    conjugate_arg : np.ndarray
        Argument.

    lam_2 : float
        Non-negative float.

    M : float
        Positive float

    delta : float
        Float

    Notes
    -----

    A small increment of 1e-10 is added to `lam_2` to allow it to be set to exactly 0 
    in calling this function.

    """

    # z_star is first return value
    val = primal_cost_lagrange(arg, lam_2, M, delta_guess)[0].sum() 
    
    if val > C:
        # want to increase delta
        L = delta_guess
        
        if delta_guess > 0:
            U = 10 * delta_guess
        else:
            U = 1
            
        niter = 0
        while True:
            val = primal_cost_lagrange(arg, lam_2, M, U)[0].sum()
            if val < C or niter >= 20:
                break
            U *= 10
            niter += 1
    else:
        # want to decrease delta
        U = delta_guess
        if delta_guess < 0:
            L = 10 * delta_guess
        else:
            L = -1
            
        niter = 0
        while True:
            val = primal_cost_lagrange(arg, lam_2, M, L)[0].sum()
            if val > C or niter >= 20:
                break
            L *= 10
            niter += 1
            
    def fsum(arg, lam_2, M, C, delta):
        return primal_cost_lagrange(arg, lam_2, M, delta)[0].sum() - C
    f = functools.partial(fsum, arg, lam_2, M, C)

    if np.fabs(f(U)) < 1e-7:
        delta_star = U
    elif np.fabs(f(L)) < 1e-7:
        delta_star = L
    else:
        delta_star = root_scalar(f, method='bisect', bracket=(L, U)).root

    z_star, value =  primal_cost_lagrange(arg, lam_2, M, delta_star)
    return z_star, delta_star, value

def lagrange_prox(prox_arg, lips, lam_2, M, delta):
    r"""
    This is the prox map of $\varphi_L$ the Lagrange form.

    Solves

    $$
    \text{minimize}_{\beta} \frac{{\cal L}}{2} \|\beta\|^2_2 - v^T\beta + \varphi_{L,M,\lambda_2,\delta}(\beta)
    $$

    with ${\cal L}$ set to `lips` and $v$ set to `prox_arg`.

    Parameters
    ----------

    prox_arg : np.ndarray
        Argument to proximal map.

    lips : float
        A positive Lipschitz constant / inverse stepsize.

    lam_2 : float
        Non-negative float.

    M : float
        Positive float

    delta : float

    Returns
    -------

    z_star : np.ndarray
        Optimal z values

    beta_star : np.ndarray
        Optimal beta values

    Notes
    -----

    A small increment of 1e-10 is added to `lam_2` to allow it to be set to exactly 0 
    in calling this function.

    """
    
    L = lips
    v = prox_arg
    
    if not lam_2 >= 0:
        raise ValueError('lam_2 must be non-negative')
    lam_2 += 1e-10 

    z_star = np.zeros_like(v)
    if not np.isinf(M):
        bdry = (np.fabs(v) - lam_2 * M) / (L*M) # \tau
    else:
        bdry = - lam_2 * np.ones_like(v)
    
    root_quadratic = bdry - delta / (M**2 * L)
    if delta > 0:
        root_rational = np.fabs(v) * np.sqrt(lam_2) / (np.sqrt(2 * L * delta) * L)
    
    ## Case 1
    
    idx1 = bdry >= 1
    z_star[idx1] = np.clip(root_quadratic[idx1], 0, 1)
    
    ## Case 2
    
    idx2 = bdry <= 0
    if delta > 0:
        z_star[idx2] = np.clip(root_rational[idx2], 0, 1)
    else:
        z_star[idx2] = 1
        
    ## Case 3
    
    idx3 = (bdry > 0) * (bdry < 1)
    if delta > 0:
        z_star[idx3] = np.clip(root_rational[idx3], 0, bdry[idx3])
    else:
        z_star[idx3] = np.clip(root_quadratic[idx3], bdry[idx3], 1)
        
    beta_star = np.sign(v) * np.minimum(M * z_star, np.fabs(v) * z_star / (z_star * L + lam_2))

    return z_star, beta_star

def bound_prox(prox_arg, lips, lam_2, M, C, delta_guess=1):
    r"""
    This is the prox map of the bound form.

    Solves

    $$
    \text{minimize}_{\beta} \frac{{\cal L}}{2} \|\beta\|^2_2 - v^T\beta + \varphi_{B,M,\lambda_2,C}(\beta)
    $$

    with ${\cal L}$ set to `lips` and $v$ set to `prox_arg`.

    Parameters
    ----------

    prox_arg : np.ndarray
        Argument to proximal map.

    lips : float
        A positive Lipschitz constant / inverse stepsize.

    lam_2 : float
        Non-negative float.

    M : float
        Positive float

    C : float
        Non-negative float

    delta_guess : float
        Guess at Lagrange parameter, could be set with a 
        warm start.

    Returns
    -------

    z_star : np.ndarray
        Optimal z values

    beta_star : np.ndarray
        Optimal beta values

    delta_star : float
        Lagrange multiplier

    Notes
    -----

    - Uses a grid search over $\delta$ using `lagrange_prox`.

    - A small increment of 1e-10 is added to `lam_2` within `lagrange_prox` to allow it to be set to exactly 0 
    in calling this function.
    """
    
    L = lips
    v = prox_arg
    
    if not lam_2 >= 0:
        raise ValueError('lam_2 must be non-negative')

    if C <= 0 or C >= v.shape[0]:
        raise ValueError('C must be between 0 and %d' % v.shape[0])
    
    # z_star is first return value
    val = lagrange_prox(v, lips, lam_2, M, delta_guess)[0].sum() 
    
    if val > C:
        # want to increase delta
        L = delta_guess
        
        if delta_guess > 0:
            U = 10 * delta_guess
        else:
            U = 1
            
        niter = 0
        while True:
            val = lagrange_prox(v, lips, lam_2, M, U)[0].sum()
            if val < C or niter >= 20:
                break
            U *= 10
            niter += 1
    else:
        # want to decrease delta
        U = delta_guess
        if delta_guess < 0:
            L = 10 * delta_guess
        else:
            L = -1
            
        niter = 0
        while True:
            val = lagrange_prox(v, lips, lam_2, M, L)[0].sum()
            if val > C or niter >= 20:
                break
            L *= 10
            niter += 1
            
    def fsum(v, lips, lam_2, M, C, delta):
        return lagrange_prox(v, lips, lam_2, M, delta)[0].sum() - C

    f = functools.partial(fsum, v, lips, lam_2, M, C)
    if np.fabs(f(U)) < 1e-7:
        delta_star = U
    elif np.fabs(f(L)) < 1e-7:
        delta_star = L
    else:
        delta_star = root_scalar(f, method='bisect', bracket=(L, U)).root

    z_star, beta_star =  lagrange_prox(v, lips, lam_2, M, delta_star)
    return z_star, beta_star, delta_star

