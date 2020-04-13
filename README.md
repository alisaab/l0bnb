# L0BnB: Sparse Regression at Scale
### Hussein Hazimeh, Rahul Mazumder, and Ali Saab
### Massachusetts Institute of Technology

## Introduction
L0BnB is a scalable global optimization framework for solving linear regression problems penalized with a combination of the L0 and L2 norms. More concretely, given a data matrix X (with n samples and p features) and a response vector y, L0BnB solves the following problem to  optimality:

<img src="formulation.png" width = 350>

where the L0 norm counts the number of nonzeros in the coefficients vector B. Here the L0 norm performs variable selection, while the L2 norm adds shrinkage which can be effective in low-signal settings. L0BnB implements a custom branch-and-bound (BnB) framework that leverages a  highly specialized first-order method to solve the node subproblems. It achieves over 3600x speed-ups compared to the state-of-the-art mixed integer programming (MIP) solvers, and can scale to problems where p ~ 10^7. For more details, check out our paper *Sparse Regression at Scale: Branch-and-Bound rooted in First Order Optimization*.

The toolkit is implemented in Python, with critical code sections accelerated using Numba. See below for details on installation and usage.

## Installation
To install in Python 3, run the following command:
```bash
pip install L0BnB
```

## Quick Start
More details to be added soon.


## References
More details to be added soon.
