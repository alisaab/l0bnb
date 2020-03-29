import os
from time import time, sleep
import subprocess
import _pickle as pickle

import numpy as np

from scripts.generate_data import GenData as gen_data


P_VALUES = [1e2, 1e3, 1e4, 1e5, 1e6]
N = int(1e3)
SNR = 10.0
SUPPORT_SIZE = 10
RHO = 0
# L0 = 0.000242168
# L2 = 5.994843e-02
CORR_MATRIX = 'I'


if __name__ == '__main__':
    result = {}
    for p in P_VALUES:
        p = int(p)
        print(f"Solving for p = {p}")
        # x, y, features, covariance = \
        #     gen_data(CORR_MATRIX, RHO, N, p, SUPPORT_SIZE, SNR)
        # for i in range(x.shape[1]):
        #     x[:, i] = (x[:, i] - np.mean(x[:, i])) / np.linalg.norm(x[:, i])
        # y = (y - np.mean(y))/np.linalg.norm(y)
        # print('generated data')
        # np.save(f'data/x_vary_p_{p}.npy', x)
        # np.save(f'data/y_vary_p_{p}.npy', y)
        # np.save(f'data/x.npy', x)
        # np.save(f'data/y.npy', y)
        # print('saved data')
        # with open('data\config.txt', 'w') as f:
        #     f.write(f'p={p}\nn={N}\nk={SUPPORT_SIZE}\n')
        # subprocess.Popen([r"C:\Program Files\R\R-3.6.3\bin\Rscript.exe",
        #                   r"C:\Users\aks14\Documents\try.R"])

        while True:
            if os.path.exists(r'C:/Users/aks14/Documents/l0bnb/data/result.txt'):
                sleep(5)
                break
        print('process ran')
        with open(r'C:/Users/aks14/Documents/l0bnb/data/result.txt') as f:
            lines = f.readlines()[1:]
        os.remove(r"C:/Users/aks14/Documents/l0bnb/data/result.txt")
        beta = np.array([float(i.split(',')[1]) for i in lines])
        l0 = beta[0]
        l2 = beta[1]
        features = beta[2:]
        m = max(abs(features)) * 2
        print(l0, l2, m, len(features[features != 0]))
        result[p] = {'l0': l0, 'l2': l2, 'm': m, 'warm_start': features}
        with open(f'data/result_vary_p_l0learn_{p}.pkl', 'wb') as f:
            pickle.dump(result, f, protocol=4)
