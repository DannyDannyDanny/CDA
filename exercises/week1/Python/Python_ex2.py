# -*- coding: utf-8 -*-
"""
Computer intensive datahandling exercise 2

@author: dnor
"""
import numpy as np
import scipy.linalg as lng
import matplotlib.pyplot as plt

if __name__ == "__main__":
    n = 10 # nr of obs
    p = 3 # features / vars
    beta_true = np.array([1, 2, 3])
    X = np.random.randn(n, p) # random drawing, feature matrix
    
    m = 100 # nr of experiments
    
    betas = np.zeros((p, m)) # all variable estimates
    
    sigma = 0.1
    
    for i in range(m):
        y = np.matmul(X, beta_true) + sigma * np.random.randn(n) # Measures response - true value plus noise level
        betas[:, i] = lng.lstsq(X, y)[0] # Estimates
        
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title("All estimated betas")
    lines = ax.scatter([range(m)] * 3, betas.T, marker = ".")
    
    ax2 = fig.add_subplot(1,2,2)
    ax2.set_title("Estimated betas as boxplot")
    ax2.boxplot(betas.T)