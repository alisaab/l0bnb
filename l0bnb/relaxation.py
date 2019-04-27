import copy
import numpy as np


def _case_one_update(criteria, ri_xi, l2, golden_ratio, xi_xi):
    if criteria >= golden_ratio:
        return ri_xi / (xi_xi + 2 * l2)
    return criteria


def _case_two_update(criteria, *args):
    return criteria


def coordinate_descent_loop(x, beta, index_map, l0, l2, m, zlb, zub, support, r):
    zub_active_is_zero = np.where(zub == 0)[0]
    zlb_active_normal = np.where(np.logical_and(zlb == 0, zub > 0))[0]
    zlb_active_is_one = np.where(zlb > 0)[0]
    set_add = support.add
    set_discard = support.discard
    dot_product = np.dot
    golden_ratio = np.sqrt(l0/l2) if l2 != 0 else np.Inf
    if golden_ratio <= m:
        threshold = 2 * np.sqrt(l0 * l2)
        update_criteria = _case_one_update
    else:
        threshold = l0/m + l2 * m
        update_criteria = _case_two_update

    for i in zub_active_is_zero:
        r = r + dot_product(beta[i], x[:, i])
        set_discard(index_map[i])
        beta[i] = 0

    for i in zlb_active_normal:
        x_i = x[:, i]
        xi_xi = np.dot(x_i, x_i)
        r = r + dot_product(beta[i], x_i)
        ri_xi = dot_product(r, x_i)
        ri_xi_abs = abs(ri_xi)
        if ri_xi_abs <= threshold:
            set_discard(index_map[i])
            beta[i] = 0
        else:
            if index_map[i] not in support:
                set_add(index_map[i])
            criteria = (ri_xi_abs - threshold) / xi_xi
            criteria = update_criteria(criteria, ri_xi, l2, golden_ratio, xi_xi)
            beta[i] = (criteria if criteria < m else m) * np.sign(ri_xi)
            r = r - dot_product(beta[i], x_i)

    for i in zlb_active_is_one:
        x_i = x[:, i]
        r = r + dot_product(beta[i], x_i)
        ri_xi = dot_product(r, x_i)
        criteria = abs(ri_xi)/(2 * l2 + np.dot(x_i, x_i))
        if index_map[i] not in support:
            set_add(index_map[i])
        beta[i] = (criteria if criteria < m else m) * np.sign(ri_xi)
        r = r - dot_product(beta[i], x_i)

    return beta, r


def coordinate_descent(x, beta, cost, l0, l2, m, zlb, zub, support, r, reltol):
    tol = 1
    while tol > reltol:
        old_cost = cost
        active_set = list(support)
        zlb_active = zlb[active_set]
        zub_active = zub[active_set]
        beta_active = beta[active_set]
        x_active = x[:, active_set]
        beta_active, r = coordinate_descent_loop(x_active, beta_active, active_set, l0, l2, m, zlb_active,
                                                 zub_active, support, r)
        beta[active_set] = beta_active
        if l2 != 0:
            s = beta ** 2 * np.logical_and(abs(beta) > np.sqrt(l0 / l2), np.sqrt(l0 / l2) <= m) + \
                abs(beta) * np.sqrt(l0 / l2) * np.logical_and(abs(beta) <= np.sqrt(l0 / l2), np.sqrt(l0 / l2) <= m) + \
                abs(beta) * m * (np.sqrt(l0 / l2) > m)
            s = s * (zlb < 1) + beta ** 2 * (zlb == 1)
            z = abs(beta) / m
            z[z > 0] = np.maximum(z[z > 0], beta[z > 0] ** 2 / s[z>0])
        else:
            s = np.zeros(len(beta))
            z = abs(beta) / m
        z = np.minimum(np.maximum(zlb, z), zub)
        cost = np.dot(r, r) / 2 + l0 * sum(z) + l2 * sum(s)
        tol = abs(1 - old_cost / cost)
    return beta, cost, r


def initial_active_set(x, beta, l0, l2, m, zlb, zub, support, r):
    num_of_similar_supports = 0
    active_set = list(range(len(beta)))
    while num_of_similar_supports < 3:
        old_support = copy.deepcopy(support)
        beta, r = coordinate_descent_loop(x, beta, active_set, l0, l2, m, zlb, zub, support, r)
        if old_support == support:
            num_of_similar_supports += 1
        else:
            num_of_similar_supports = 0
    return support, r


def relaxation_solve(x, y, l0, l2, m, zlb, zub, beta, r, reltol=1e-12):
    p = x.shape[1]
    if beta is None:
        beta = np.zeros(p)
        r = y - np.matmul(x, beta)
        support, r = initial_active_set(x, beta, l0, l2, m, zlb, zub, set(), r)
    else:
        support = set(abs(beta).nonzero()[0])
    if l2 != 0:
        s = beta**2 * np.logical_and(abs(beta) > np.sqrt(l0/l2), np.sqrt(l0/l2) <= m) + \
            abs(beta) * np.sqrt(l0 / l2) * np.logical_and(abs(beta) <= np.sqrt(l0/l2), np.sqrt(l0/l2) <= m) + \
            abs(beta)*m * (np.sqrt(l0/l2) > m)
        s = s * (zlb < 1) + beta ** 2 * (zlb == 1)
        z = abs(beta)/m
        z[z > 0] = np.maximum(z[z > 0], beta[z > 0] ** 2 / s[z>0])
    else:
        s = np.zeros(p)
        z = abs(beta) / m
    z = np.minimum(np.maximum(zlb, z), zub)
    cost = np.dot(r, r) / 2 + l0 * sum(z) + l2 * sum(s)
    while True:
        beta, cost, r = coordinate_descent(x, beta, cost, l0, l2, m, zlb, zub, support, r, reltol)
        if l2 != 0 and l0/l2 <= m**2:
            above_threshold = np.where(zub * abs(np.matmul(r, x)) - 2*np.sqrt(l0*l2) > 0)[0]
        elif l2 == 0:
            above_threshold = np.where(zub * abs(np.matmul(r, x)) - l0/m > 0)[0]
        else:
            above_threshold = np.where(zub * abs(np.matmul(r, x)) - l0 / m - l2 * m > 0)[0]
        outliers = [i for i in above_threshold if i not in support]
        if not outliers:
            break
        support = support | set(outliers)
    if l2 != 0:
        s = beta**2 * np.logical_and(abs(beta) > np.sqrt(l0/l2), np.sqrt(l0/l2) <= m) + \
            abs(beta) * np.sqrt(l0 / l2) * np.logical_and(abs(beta) <= np.sqrt(l0/l2), np.sqrt(l0/l2) <= m) + \
            abs(beta)*m * (np.sqrt(l0/l2) > m)
        s = s * (zlb < 1) + beta ** 2 * (zlb == 1)
        z = abs(beta)/m
        z[z > 0] = np.maximum(z[z > 0], beta[z > 0]**2/s[z>0])
    else:
        s = np.zeros(p)
        z = abs(beta) / m
    return beta, r, np.minimum(np.maximum(zlb, z), zub), cost
