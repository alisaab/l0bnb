import scipy as sc
from math import fabs
from sklearn import preprocessing
import itertools
from numpy.linalg import cholesky
from numpy.random import normal
import numpy as np


def NormalizeMatrix(X):
    '''
    Sets u=0 and ||Xi||_2 = 1
    '''
    scaler = preprocessing.StandardScaler(with_mean=False).fit(
        X)  # CAREFUL! Remove with mean.. included for sparse case
    X = scaler.transform(X)
    X = X / np.sqrt(X.shape[0])
    # X = preprocessing.normalize(X,axis=0)
    # print('sum= '+str(np.sum(X[:,0])))
    # print('sum2= '+str(X[:,0].T.dot(X[:,0])))
    # return X
    return (X, scaler, np.sqrt(X.shape[0]))


def MSE(X, y, B):
    diff = y - X.dot(B)
    return (1 / X.shape[0]) * (diff.T.dot(diff))


def CreateGrid(P1, P2):
    return itertools.product(P1, P2)


def OrderbyCorrelation(X, y):
    '''
    Can be implemented using a simple matrix multiplication
    Used this in case it's applied on a non-normalized matrix
    '''
    i = 0
    order = []
    for column in X.T:
        order.append((fabs(sc.stats.pearsonr(column, y)[0]), i))
        i += 1
    order = sorted(order, key=lambda x: x[0], reverse=True)
    order = [elem[1] for elem in order]
    return np.array(order)


def GenGaussianData(n, p, Covariance, SNR, B, I=False):
    # Generate X
    if I == False:
        A = cholesky(Covariance)
        Z = normal(size=(n, p))
        X = Z.dot(A.T)
        var_XB = np.dot(np.dot(B, Covariance), B)
    else:
        X = normal(size=(n, p))
        var_XB = np.dot(B, B)
    mu = X.dot(B)

    # var_XB = (np.std(mu,ddof=1))**2
    # var_XB = np.dot(np.dot(B,Covariance),B)

    # Generate epsilon
    sd_epsilon = np.sqrt(var_XB / SNR)
    epsilon = normal(size=n, scale=sd_epsilon)

    # Generate y
    y = mu + epsilon

    return X, y, B


def GenGaussianDataFixed(n, p, Covariance, SNR, B, D=""):
    # Generate X

    if D == "I":
        X = normal(size=(n, p))
        var_XB = np.dot(B, B)
        mu = X.dot(B)

    elif D == "CLarge":
        rho = Covariance
        X = normal(size=(n, p)) + np.sqrt(rho / (1 - rho)) * normal(size=(n, 1))
        mu = X.dot(B)
        var_XB = (np.std(mu, ddof=1)) ** 2

    else:
        A = cholesky(Covariance)
        Z = normal(size=(n, p))
        X = Z.dot(A.T)
        var_XB = np.dot(np.dot(B, Covariance), B)
        mu = X.dot(B)

    # var_XB = (np.std(mu,ddof=1))**2
    # var_XB = np.dot(np.dot(B,Covariance),B)

    # Generate epsilon
    sd_epsilon = np.sqrt(var_XB / SNR)

    '''
    # Generate X
    A = cholesky(Covariance)
    Z = normal(size=(n,p))
    X = Z.dot(A.T)
    mu = X.dot(B)


    #var_XB = (np.std(mu,ddof=1))**2
    var_XB = np.dot(np.dot(B,Covariance),B)

    # Generate epsilon
    sd_epsilon = np.sqrt(var_XB/SNR)
    '''
    epsilon = normal(size=n, scale=sd_epsilon)
    epsilondev = normal(size=n, scale=sd_epsilon)

    # Generate y
    y = mu + epsilon
    ydev = mu + epsilondev

    return X, y, ydev, B


def GenData(dataset, parameter, n, p, SuppSize, SNR):
    """
    dataset: correlation matrix: I, E, C
    parameter: rho
    SuppSize: num of nnz in beta
    SNR:  signal to noise ratio
    """
    np.random.seed(1)

    B = np.zeros(p)
    support = [int(i * (p / SuppSize)) for i in range(SuppSize)]
    Covariance = None

    if dataset == "I":
        Covariance = 'I'

        ############### Remove

        # B = np.zeros(p)

        # support = np.random.choice(range(p), SuppSize, False)
        # B[support] = np.random.uniform(-1, 1, size=SuppSize)
        B[support] = np.ones(SuppSize)
        ##############

    elif dataset == "E":
        B[support] = np.ones(len(support))
        ps = list(range(int(p / SuppSize), p, int(p / SuppSize))) + [p]
        X_training = np.zeros((n, p))
        y_training = np.zeros(n)
        prev_p = 0
        for cur_p in ps:
            print(cur_p)
            Covariance = np.array([[i - j for j in range(cur_p - prev_p)] for i in range(cur_p - prev_p)])
            Covariance = np.abs(Covariance)
            Covariance = parameter ** Covariance
            X, y, _, _ = GenGaussianDataFixed(n, cur_p - prev_p, Covariance,
                                              SNR, B[prev_p: cur_p], dataset)

            X_training[:, prev_p: cur_p] = X[0:n, :]
            y_training += y[0:n]
            prev_p = cur_p

            # X_training, scaler, factor = NormalizeMatrix(X_training)

        return X_training, y_training, B, Covariance

    elif dataset == "C":
        Covariance = np.zeros((p, p))
        Covariance = Covariance + parameter
        np.fill_diagonal(Covariance, 1)
        B[support] = np.ones(len(support))

    elif dataset == "CLarge":
        Covariance = parameter
        B[support] = np.ones(len(support))

    # X, y, B = GenGaussianData(n + 1000, p, Covariance, SNR, B)
    X, y, ydev, B = GenGaussianDataFixed(n, p, Covariance, SNR, B, dataset)

    X_training = X[0:n, :]
    y_training = y[0:n]

    # X_training, scaler, factor = NormalizeMatrix(X_training)

    return X_training, y_training, B, Covariance
