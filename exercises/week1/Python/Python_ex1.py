import numpy as np
import scipy.linalg as lng
import matplotlib.pyplot as plt

# If using as script and not from console;
path = "C:\\Users\\dnor\\Desktop\\02582\\Lecture1\\S1"

if __name__ == "__main__":
    diabetPath = path + '\\DiabetesDataNormalized.txt'
    T = np.loadtxt(diabetPath, delimiter = ' ', skiprows = 1)
    y = T[:, 10]
    X = T[:,:10]

    [n, p] = np.shape(X)

    off = np.ones(n)
    M = np.c_[off, X] # Include offset / intercept

    # Linear solver
    beta, res, rnk, s = lng.lstsq(M, y)

    yhat = np.matmul(M, beta)

    # Same residuals as above
    res = (y - yhat) ** 2

    rss = np.sum(res)
    mse = np.mean(res)
    tss = np.sum((y - np.mean(y))** 2)
    r2 = (1 - rss / tss) * 100
