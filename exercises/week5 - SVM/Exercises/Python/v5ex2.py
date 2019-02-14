#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

# #############################################################################
# Load overlapping 2D data
T = pd.read_csv('Data/Ex2Data.csv', header=None)

X = np.array(T.loc[:, T.columns != 2])
Y = np.array(T.loc[:, T.columns == 2]).T.reshape(136,1)

# #############################################################################
# Try different values for the Support Vector Machine
#

kernelType = 'rbf'  # <----- YOUR CHOICE. Specifies the kernel type to be used in the algorithm. 
                    # It must be one of ‘linear’, ‘poly’ or ‘rbf’. 
                    # If none is given, ‘rbf’ will be used. If a callable is given it is used to pre-compute
                    # the kernel matrix from data matrices; that matrix should be an array of shape 
                    # (n_samples, n_samples).

degreeVal = 0.001   # <----- YOUR CHOICE. Degree of the polynomial kernel function ('poly'). Ignored by all other kernels.

gammaVal = 1    # <---- YOUR CHOICE. Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. 
                # If gamma is ‘auto’ then 1/n_features will be used instead.
    
Cval = 20000      # <----- YOUR CHOICE. Penalty parameter C of the error term.
                # The C parameter trades off misclassification of training examples
                # against simplicity of the decision surface. A low C makes the decision 
                # surface smooth, while a high C aims at classifying all training examples
                # correctly by giving the model freedom to select more samples as support vectors.
    

# Estimation models
clf0 = SVC(C=Cval, gamma = gammaVal, kernel=kernelType, degree=degreeVal)
clf1 = SVC(C=Cval, gamma = gammaVal, kernel = "sigmoid", degree = degreeVal)
clfs = [clf0, clf1] # List of all classifiers you want to see
     
# Set-up 2x2 grid for plotting.
X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

fig, ax = plt.subplots(1,len(clfs))
# Go through all subplots
for i, clf in enumerate(clfs):
    plot_contours(ax[i], clf.fit(X,Y), xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    # plot support vectors, they determine the margin
    ax[i].scatter(X0, X1, s=20, c = clf.predict(X), marker = 'o')
    ax[i].scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s = 20, marker = '+')
    ax[i].set_xlim(xx.min(), xx.max())
    ax[i].set_ylim(yy.min(), yy.max())
    ax[i].set_title('SVC with %s kernel' % clf.kernel)
plt.show()