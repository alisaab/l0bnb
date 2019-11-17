import copy
import numpy as np


def _calculate_cost(beta, r, l0, l2, golden_ratio, m, zlb, zub):
    s = beta ** 2 * (abs(beta) > golden_ratio) + abs(beta) * golden_ratio * (abs(beta) <= golden_ratio)
    s = s * (golden_ratio <= m) + abs(beta) * m * (golden_ratio > m)
    s = s * (zlb < 1) + beta ** 2 * (zlb == 1)
    z = abs(beta) / m
    z[z > 0] = np.maximum(z[z > 0], beta[z > 0] ** 2 / s[z > 0])
    z = np.minimum(np.maximum(zlb, z), zub)
    return np.dot(r, r) / 2 + l0 * sum(z[z > 0]) + l2 * sum(s[s > 0]), z


def _coordinate_descent_loop(x, beta, index_map, l2, golden_ratio, threshold, m, xi_xi, zlb, zub, support, r):
    zub_active_is_zero = np.where(zub == 0)[0]
    zlb_active_normal = np.where(np.logical_and(zlb == 0, zub > 0))[0]
    zlb_active_is_one = np.where(zlb > 0)[0]
    set_add = support.add
    set_discard = support.discard
    dot_product = np.dot
    if golden_ratio <= m:
        def update_criteria(*args): return args[0] if args[0] < golden_ratio else args[1] / (args[2] + 2 * l2)
    else:
        def update_criteria(*args): return args[0]

    for i in zub_active_is_zero:
        r = r + dot_product(beta[i], x[:, i])
        # set_discard(index_map[i])
        beta[i] = 0

    for i in zlb_active_normal:
        x_i = x[:, i]
        r = r + dot_product(beta[i], x_i)
        ri_xi = dot_product(r, x_i)
        abs_ri_xi = abs(ri_xi)
        if abs_ri_xi <= threshold:
            # set_discard(index_map[i])
            beta[i] = 0
        else:
            if index_map[i] not in support:
                set_add(index_map[i])
            criteria = (abs_ri_xi - threshold) / xi_xi[i]
            criteria = update_criteria(criteria, abs_ri_xi, xi_xi[i])
            beta[i] = (criteria if criteria < m else m) * np.sign(ri_xi)
            r = r - dot_product(beta[i], x_i)

    for i in zlb_active_is_one:
        x_i = x[:, i]
        r = r + dot_product(beta[i], x_i)
        ri_xi = dot_product(r, x_i)
        criteria = abs(ri_xi) / (2 * l2 + xi_xi[i])
        if index_map[i] not in support:
            set_add(index_map[i])
        beta[i] = (criteria if criteria < m else m) * np.sign(ri_xi)
        r = r - dot_product(beta[i], x_i)

    return beta, r


def coordinate_descent(x, beta, cost, l0, l2, golden_ratio, threshold, m, xi_xi, zlb, zub, support, r, reltol):
    tol = 1
    while tol > reltol:
        old_cost = cost
        active_set = list(support)
        zlb_active = zlb[active_set]
        zub_active = zub[active_set]
        beta_active = beta[active_set]
        xi_xi_active = xi_xi[active_set]
        x_active = x[:, active_set]
        beta_active, r = _coordinate_descent_loop(x_active, beta_active, active_set, l2, golden_ratio, threshold, m,
                                                  xi_xi_active, zlb_active, zub_active, support, r)
        beta[active_set] = beta_active
        cost, _ = _calculate_cost(beta[active_set], r, l0, l2, golden_ratio, m, zlb[active_set], zub[active_set])
        tol = abs(1 - old_cost / cost)
        support = set(abs(beta).nonzero()[0])
        print('new', cost, tol, len(support))
    return beta, cost, r


def initial_active_set(y, x, beta, l2, golden_ratio, threshold, m, xi_xi, zlb, zub, support, r):
    num_of_similar_supports = 0
    correlations = np.matmul(y, x)/xi_xi
    partition = np.argpartition(-correlations, min(2000, len(beta)))
    active_set = list(partition[0:min(2000, len(beta))])
    zlb_active = zlb[active_set]
    zub_active = zub[active_set]
    beta_active = beta[active_set]
    xi_xi_active = xi_xi[active_set]
    x_active = x[:, active_set]
    while num_of_similar_supports < 3:
        old_support = copy.deepcopy(support)
        beta_active, r = _coordinate_descent_loop(x_active, beta_active, active_set, l2, golden_ratio, threshold, m,
                                                  xi_xi_active, zlb_active, zub_active, support, r)
        if old_support == support:
            num_of_similar_supports += 1
        else:
            num_of_similar_supports = 0
        kes, _ = _calculate_cost(beta_active, r, 70, l2, golden_ratio, m, zlb_active, zub_active)
        print(kes, num_of_similar_supports)
    beta[active_set] = beta_active
    return support, r


def relaxation_solve(x, y, l0, l2, m, xi_xi, zlb, zub, beta, r, reltol=1e-12):
    p = x.shape[1]
    golden_ratio = np.sqrt(l0 / l2) if l2 != 0 else np.Inf
    threshold = 2 * np.sqrt(l0 * l2) if golden_ratio <= m else l0 / m + l2 * m
    if beta is None:
        beta = np.zeros(p)
        r = y - np.matmul(x, beta)
        support, r = initial_active_set(y, x, beta, l2, golden_ratio, threshold, m, xi_xi, zlb, zub, set(), r)
    else:
        support = set(abs(beta).nonzero()[0])
    cost, _ = _calculate_cost(beta, r, l0, l2, golden_ratio, m, zlb, zub)
    while True:
        print(len(support))
        beta, cost, r = coordinate_descent(x, beta, cost, l0, l2, golden_ratio, threshold, m, xi_xi, zlb, zub, support,
                                           r, reltol)
        above_threshold = np.where(zub * abs(np.matmul(r, x)) - threshold > 0)[0]
        outliers = [i for i in above_threshold if i not in support]
        if not outliers:
            break
        support = support | set(outliers)
    cost, z = _calculate_cost(beta, r, l0, l2, golden_ratio, m, zlb, zub)
    return beta, r, np.minimum(np.maximum(zlb, z), zub), cost, support
