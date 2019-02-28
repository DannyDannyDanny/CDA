from IPython.display import Markdown

import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
from sklearn import preprocessing as preproc
import scipy.linalg as lng
import matplotlib.pyplot as plt
import os

# Setting right path
os.listdir('exercises/week1/')
path = 'exercises/week1/python/'

# %% Ex.1
# a)
# b)
# c)

# %%

if __name__ == "__main__":
    diabetPath = path + 'DiabetesDataNormalized.txt'
    T = np.loadtxt(diabetPath, delimiter = ' ', skiprows = 1)

    y = T[:, 10]
    X = T[:,:10]

    K = 5 # Nr of neighbors

    [n, p] = np.shape(X)
    yhat = np.zeros(n)

    X = preproc.scale(X) # Normalize to zero mean and unit variance
    distances = np.zeros(n)
    # For each obs, compare distance to all other points in X
    for i in range(n):
        for j in range(n):
            distances[j] = distance.euclidean(X[i,:], X[j, :])

        # Sort all the distances
        idx = np.argsort(distances)[1:(K + 1)] # Skip first, as distance to "itself" does not make sense
        Wt = sum(distances[idx]) # Weight of k nearest neighbors
        W = distances[idx] / Wt # Weighing average

        yhat[i] = np.matmul(W, y[idx]) # Final value

    MSE = np.mean((y-yhat) ** 2)

    plt.scatter(y, yhat, marker = "*")
    plt.xlabel("y")
    plt.ylabel("yhat")
    plt.title("KNN on diabetes data")
    plt.show()

    print('MSE:',MSE)

# %% Ex.2
import scipy.linalg as lng

if __name__ == "__main__":
    n = 10 # nr of obs
    p = 3 # features / vars
    beta_true = np.array([1, 2, 3])
    X = np.random.randn(n, p) # random drawing, feature matrix

    m = 100 # nr of experiments

    betas = np.zeros((p, m)) # all variable estimates

    sigma = 0.1

    for i in range(m):
        # Measures response - true value plus noise level
        y = np.matmul(X, beta_true) + sigma * np.random.randn(n)
        betas[:, i] = lng.lstsq(X, y)[0] # Estimates

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title("All estimated betas")
    lines = ax.scatter([range(m)] * 3, betas.T, marker = ".")

    ax2 = fig.add_subplot(1,2,2)
    ax2.set_title("Estimated betas as boxplot")
    ax2.boxplot(betas.T)

# %% Ex.3


if __name__ == "__main__":
    def ridgeMulti(X, _lambda, p, y):
        inner_prod = np.linalg.inv(np.matmul(X.T, X) + _lambda * np.eye(p,p))
        outer_prod = np.matmul(X.T, y)
        betas = np.matmul(inner_prod, outer_prod)
        return betas

    diabetPath = path + 'DiabetesDataNormalized.txt'
    T = np.loadtxt(diabetPath, delimiter = ' ', skiprows = 1)
    y = T[:, 10]
    X = T[:,:10]

    [n, p] = np.shape(X)

    off = np.ones(n)
    M = np.c_[off, X] # Include offset / intercept

    # Linear solver
    beta_ols, res, rnk, s = lng.lstsq(M, y)

    k = 100; # try k values of lambda
    lambdas = np.logspace(-4, 3, k)

    betas = np.zeros((p,k))

    for i in range(k):
        betas[:, i] = ridgeMulti(X, lambdas[i], p, y)

    plt.figure()
    plt.semilogx(lambdas, betas.T )
    plt.xlabel("Lambdas")
    plt.ylabel("Betas")

    # Bias and variance of the ridge regression, same as exercise 2 - just for ridge
    n = 10 # nr of obs
    p = 3 # features / vars
    beta_true = np.array([1, 2, 3])
    X = np.random.randn(n, p) # random drawing, feature matrix

    m = 100 # nr of experiments

    betas2 = np.zeros((k, p, m)) # all variable estimates

    sigma = 0.2

    for i in range(k):
        for j in range(m):
            y = np.matmul(X, beta_true) + sigma * np.random.randn(n) # Measures response - true value plus noise level
            betas2[i, :, j] = ridgeMulti(X, lambdas[i], p, y)

    betas_mean = np.mean(betas2, axis = 2)

    plt.figure()
    plt.semilogx(lambdas, betas_mean)
    plt.xlabel("Lambdas")
    plt.ylabel("Betas")

# %% Ex.4
if __name__ == "__main__":
    def ridgeMulti(X, _lambda, p, y):
        inner_prod = np.linalg.inv(np.matmul(X.T, X) + _lambda * np.eye(p,p))
        outer_prod = np.matmul(X.T, y)
        betas = np.matmul(inner_prod, outer_prod)
        return betas

    diabetPath = path + 'DiabetesDataNormalized.txt'
    T = np.loadtxt(diabetPath, delimiter = ' ', skiprows = 1)
    y = T[:, 10]
    X = T[:,:10]

    [n, p] = np.shape(X)

    off = np.ones(n)
    M = np.c_[off, X] # Include offset / intercept

    # Linear solver
    beta_ols, res, rnk, s = lng.lstsq(M, y)

    k = 100; # try k values of lambda
    lambdas = np.logspace(-4, 3, k)

    betas = np.zeros((p,k))

    for i in range(k):
        betas[:, i] = ridgeMulti(X, lambdas[i], p, y)

    plt.figure()
    plt.semilogx(lambdas, betas.T )
    plt.xlabel("Lambdas")
    plt.ylabel("Betas")

    # Bias and variance of the ridge regression, same as exercise 2 - just for ridge
    n = 10 # nr of obs
    p = 3 # features / vars
    beta_true = np.array([1, 2, 3])
    X = np.random.randn(n, p) # random drawing, feature matrix

    m = 100 # nr of experiments

    betas2 = np.zeros((k, p, m)) # all variable estimates

    sigma = 0.2

    for i in range(k):
        for j in range(m):
            y = np.matmul(X, beta_true) + sigma * np.random.randn(n) # Measures response - true value plus noise level
            betas2[i, :, j] = ridgeMulti(X, lambdas[i], p, y)

    betas_mean = np.mean(betas2, axis = 2)

    plt.figure()
    plt.semilogx(lambdas, betas_mean)
    plt.xlabel("Lambdas")
    plt.ylabel("Betas")
