import copy
import numpy as np


def coordinate_descent_loop(x, beta, index_map, m, lambda_value, zlb, zub, support, r):
    zub_active_is_zero = np.where(zub == 0)[0]
    zlb_active_is_zero = np.where(np.logical_and(zlb == 0, zub > 0))[0]
    zlb_active_is_one = np.where(zlb > 0)[0]
    set_add = support.add
    set_discard = support.discard
    dot_product = np.dot

    for i in zub_active_is_zero:
        r = r + dot_product(beta[i], x[:, i])
        set_discard(index_map[i])
        beta[i] = 0

    for i in zlb_active_is_zero:
        x_i = x[:, i]
        r = r + dot_product(beta[i], x_i)
        ri_xi = dot_product(r, x_i)
        criteria = abs(ri_xi) - lambda_value
        if criteria > 0:
            if index_map[i] not in support:
                set_add(index_map[i])
            beta[i] = (criteria if criteria < m else m) * np.sign(ri_xi)
            r = r - dot_product(beta[i], x_i)
        else:
            set_discard(index_map[i])
            beta[i] = 0

    for i in zlb_active_is_one:
        x_i = x[:, i]
        r = r + dot_product(beta[i], x_i)
        ri_xi = dot_product(r, x_i)
        criteria = abs(ri_xi)
        if index_map[i] not in support:
            set_add(index_map[i])
        beta[i] = (criteria if criteria < m else m) * np.sign(ri_xi)
        r = r - dot_product(beta[i], x_i)

    return beta, r


def coordinate_descent(x, beta, cost, m, lambda_value, zlb, zub, support, r, reltol):
    tol = 1
    while tol > reltol:
        old_cost = cost
        active_set = list(support)
        zlb_active = zlb[active_set]
        zub_active = zub[active_set]
        beta_active = beta[active_set]
        x_active = x[:, active_set]
        beta_active, r = coordinate_descent_loop(x_active, beta_active, active_set, m, lambda_value, zlb_active,
                                                 zub_active, support, r)
        cost = (np.dot(r, r) / 2 + lambda_value * np.dot(abs(beta_active), (1 - zlb_active)) +
                lambda_value * m * sum(zlb[zlb == 1]))
        beta[active_set] = beta_active
        tol = abs(1 - old_cost / cost)
    return beta, cost, r


def initial_active_set(x, beta, m, lambda_value, zlb, zub, support, r):
    num_of_similar_supports = 0
    active_set = list(range(len(beta)))
    while num_of_similar_supports < 3:
        old_support = copy.deepcopy(support)
        beta, r = coordinate_descent_loop(x, beta, active_set, m, lambda_value, zlb, zub, support, r)
        if old_support == support:
            num_of_similar_supports += 1
        else:
            num_of_similar_supports = 0
    return support, r


def relaxation_solve(x, y, m, lambda_value, zlb, zub, beta, r, reltol=1e-12):
    p = x.shape[1]
    if beta is None:
        beta = np.zeros(p)
        r = y - np.matmul(x, beta)
        support, r = initial_active_set(x, beta, m, lambda_value, zlb, zub, set(), r)
    else:
        support = set(abs(beta).nonzero()[0])
    cost = np.dot(r, r) / 2 + lambda_value * np.dot(abs(beta), (1 - zlb)) + lambda_value * m * sum(zlb[zlb == 1])
    while True:
        beta, cost, r = coordinate_descent(x, beta, cost, m, lambda_value, zlb, zub, support, r, reltol)
        above_threshold = np.where(zub * abs(np.matmul(r, x)) - lambda_value > 0)[0]
        outliers = [i for i in above_threshold if i not in support]
        if not outliers:
            break
        support = support | set(outliers)
    return beta, r, np.minimum(np.maximum(zlb, abs(beta) / m), zub), cost
