import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
from sklearn import preprocessing as preproc
import os

# If using as script and not from console;
path = "C:\\Users\\dnor\\Desktop\\02582\\Lecture1\\S1"

os.listdir('exercises/week1/')

if __name__ == "__main__":
    diabetPath = path + '\\DiabetesDataNormalized.txt'
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
