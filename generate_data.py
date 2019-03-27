import scipy as sc
from math import fabs
from sklearn import preprocessing
import itertools
from numpy.linalg import cholesky
from numpy.random import normal
import numpy as np
from scipy.linalg import block_diag
import pandas as pd
from sklearn.feature_selection import VarianceThreshold


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


def GenSynthetic(dataset, num_instances, SNR=3.0, corr=0):
    np.random.seed(1)
    n = num_instances
    if dataset == "S0":
        p = 30
        Covariance = np.eye(30)
        B = np.array([0 if i % 3 != 0 else 1 for i in range(30)])
    elif dataset == "S1":
        CovS1 = np.zeros((30, 30))
        CovS1 = CovS1 + 0.4  # 0.7
        np.fill_diagonal(CovS1, 1)
        p = 30
        Covariance = CovS1
        B = np.array([0.03, 0.07, 0.1, 0.9, 0.93, 0.97] + [0 for i in range(24)])
    elif dataset == "Dev":
        p = 100
        CovS1 = np.zeros((p, p))
        CovS1 = CovS1 + corr
        np.fill_diagonal(CovS1, 1)
        Covariance = CovS1
        B = np.array([0 if i % 10 != 0 else np.random.standard_normal() for i in range(p)])
    elif dataset == "M0":
        p = 2000
        Covariance = np.eye(2000)
        B = np.array([0 if i % 100 != 0 else 1 for i in range(2000)])
    elif dataset == "M1B5":
        CovM1 = np.array([[0.7 ** np.abs(i - j) for j in range(200)] for i in range(200)])
        BM1 = np.array([0 if i % 20 != 0 else 1 for i in range(200)])
        p = 1000
        Covariance = block_diag(CovM1, CovM1, CovM1, CovM1, CovM1)
        B = np.tile(BM1, 5)
    elif dataset == "M1B10":
        CovM1 = np.array([[0.7 ** np.abs(i - j) for j in range(200)] for i in range(200)])
        BM1 = np.array([0 if i % 20 != 0 else 1 for i in range(200)])
        p = 2000
        Covariance = block_diag(CovM1, CovM1, CovM1, CovM1, CovM1, CovM1, CovM1, CovM1, CovM1, CovM1)
        B = np.tile(BM1, 10)
    elif dataset == "M2B5":
        CovM2 = np.zeros((200, 200))
        CovM2 = CovM2 + 0.5
        np.fill_diagonal(CovM2, 1)
        BM2 = np.append(np.linspace(0.0, 0.5, num=10), np.zeros(190))
        p = 1000
        Covariance = block_diag(CovM2, CovM2, CovM2, CovM2, CovM2)
        B = np.tile(BM2, 5)
    elif dataset == "M2B10":
        CovM2 = np.zeros((200, 200))
        CovM2 = CovM2 + 0.5
        np.fill_diagonal(CovM2, 1)
        BM2 = np.append(np.linspace(0.0, 0.5, num=10), np.zeros(190))
        p = 2000
        Covariance = block_diag(CovM2, CovM2, CovM2, CovM2, CovM2, CovM2, CovM2, CovM2, CovM2, CovM2)
        B = np.tile(BM2, 10)
    elif dataset == "L1B5":
        CovM1 = np.array([[0.4 ** np.abs(i - j) for j in range(2000)] for i in range(2000)])
        BM1 = np.array([0 if i % 200 != 0 else 1 for i in range(2000)])
        p = 10000
        Covariance = block_diag(CovM1, CovM1, CovM1, CovM1, CovM1)
        B = np.tile(BM1, 5)
    elif dataset == "MIP":
        p = 500
        Covariance = np.eye(p)
        B = np.array([0 for i in range(p)])
        B[[0, 50, 100, 150, 200]] = np.ones(5)
        # B = np.array([1, 1, 1, 1, 1]+[0 for i in range(p-5)])
    elif dataset == "MIP2":
        p = 500
        CovS1 = np.zeros((p, p))
        CovS1 = CovS1 + 0.5
        np.fill_diagonal(CovS1, 1)
        Covariance = CovS1
        B = np.array([0 for i in range(p)])
        B[[0, 50, 100, 150, 200]] = np.ones(5)

    return GenGaussianData(n, p, Covariance, SNR, B)


def GenHousingDataset():
    from sklearn.feature_selection import SelectKBest
    import pandas as pd
    import numpy as np
    from sklearn import preprocessing
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.feature_selection import f_regression

    from scipy.stats import skew
    from scipy.stats.stats import pearsonr
    train = pd.read_csv("train.csv")
    all_data = train.loc[:, 'MSSubClass':'SaleCondition']
    # log transform the target:
    # train["SalePrice"] = np.log1p(train["SalePrice"])

    # log transform skewed numeric features:
    # numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

    # skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
    # skewed_feats = skewed_feats[skewed_feats > 0.75]
    # skewed_feats = skewed_feats.index

    # all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

    all_data = pd.get_dummies(all_data)

    # filling NA's with the mean of the column:
    all_data = all_data.fillna(all_data.mean())

    # creating matrices for sklearn:
    X = all_data
    y = train.SalePrice
    y = y / 1000
    poly = preprocessing.PolynomialFeatures(2, interaction_only=True)
    X = poly.fit_transform(X)
    sel = VarianceThreshold()
    X = sel.fit_transform(X)
    X = SelectKBest(f_regression, k=1000).fit_transform(X, y)
    return X, y


def LoadDataset(Name, SNR):
    if Name == "Arcene":
        '''
        n = 100
        p = 9920
        Lambda = 0.05
        '''
        X = np.loadtxt('arcene_train.data.txt')
    elif Name == "RCV1":
        '''
        n = 100
        p = 2776
        Lambda = 0.05
        '''
        num_features = 100000
        from sklearn.datasets import fetch_rcv1
        rcv1 = fetch_rcv1()
        X_original = rcv1.data[0:100, :].todense()
        X = X_original[:, 0:num_features]
    elif Name == "BostonAdvanced":
        '''
        n = 1000
        p = 23787
        Lambda = 0.1
        '''
        train = pd.read_csv("BostonAdvanced.csv")
        all_data = train.loc[:, 'MSSubClass':'SaleCondition']
        all_data = pd.get_dummies(all_data)
        # filling NA's with the mean of the column:
        all_data = all_data.fillna(all_data.mean())
        # creating matrices for sklearn:
        X = all_data
        poly = preprocessing.PolynomialFeatures(2, interaction_only=True)
        X = poly.fit_transform(X)
        X = X[0:10000, :]

    elif Name == "GaussianI":
        '''
        lambda = 0.05
        '''
        n = 10000
        p = 200

        Cov = np.eye(p)
        A = cholesky(Cov)
        Z = normal(size=(n, p))
        X = Z.dot(A.T)


    elif Name == "GaussianIL":
        '''
        lambda = 0.05
        '''
        # n = 10000
        # p = 10000

        # Cov = np.eye(p)
        # A = cholesky(Cov)
        # Z = normal(size=(n,p))
        # X = Z.dot(A.T)
        X = np.random.normal(size=(1000, 100000))
    elif Name == "GaussianCorr":
        '''
        lambda = 0.03
        '''
        n = 500  # Fix AGAIN
        p = 1000

        Cov = np.zeros((p, p))
        Cov = Cov + 0.5
        np.fill_diagonal(Cov, 1)
        A = cholesky(Cov)
        Z = normal(size=(n, p))
        X = Z.dot(A.T)

    elif Name == "GaussianExp":
        '''
        lambda = 0.05
        '''
        n = 500
        p = 1000

        Cov = np.array([[0.7 ** np.abs(i - j) for j in range(p)] for i in range(p)])
        A = cholesky(Cov)
        Z = normal(size=(n, p))
        X = Z.dot(A.T)


    elif Name == "GaussianExpL":
        '''
        lambda = 0.05
        '''
        n = 1000
        p = 100000

        Cov = np.array([[0.7 ** np.abs(i - j) for j in range(p)] for i in range(p)])
        A = cholesky(Cov)
        Z = normal(size=(n, p))
        X = Z.dot(A.T)

    selector = VarianceThreshold()
    X = selector.fit_transform(X)  # Remove constant features
    X, scaler, factor = NormalizeMatrix(X)

    Beta = np.zeros(X.shape[1])
    support = np.random.choice(X.shape[1], 100, replace=False)
    Beta[support] = np.random.uniform(-1, 1, size=100)  # np.ones(100)
    mu = X.dot(Beta)
    var_XB = (np.std(mu, ddof=1)) ** 2
    # Generate epsilon
    sd_epsilon = np.sqrt(var_XB / SNR)
    epsilon = normal(size=X.shape[0], scale=sd_epsilon)
    # print(mu)
    # print(epsilon)
    y = mu + epsilon

    X, scaler, factor = NormalizeMatrix(X)
    return X, y, Beta


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
    B[support] = np.random.uniform(-1, 1, size=SuppSize)
    Covariance = None

    if dataset == "I":
        Covariance = 'I'

        ############### Remove

        B = np.zeros(p)

        # support = random.sample(range(p), suppSize)
        # B[support] = np.random.uniform(-1, 1, size=SuppSize)
        support = list(range(SuppSize))
        B[support] = np.ones(SuppSize)
        ##############

    elif dataset == "E":
        Covariance = np.array([[parameter ** np.abs(i - j) for j in range(p)] for i in range(p)])
        B[support] = np.ones(len(support))

    elif dataset == "C":
        Covariance = np.zeros((p, p))
        Covariance = Covariance + parameter
        np.fill_diagonal(Covariance, 1)
        B[support] = np.ones(len(support))

    # X, y, B = GenGaussianData(n + 1000, p, Covariance, SNR, B)
    X, y, ydev, B = GenGaussianDataFixed(n, p, Covariance, SNR, B, dataset)

    X_training = X[0:n, :]
    y_training = y[0:n]

    # X_training, scaler, factor = NormalizeMatrix(X_training)

    return X_training, y_training, B, Covariance
