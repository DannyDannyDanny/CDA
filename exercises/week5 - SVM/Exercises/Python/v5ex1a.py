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
# Load and prepare data set

T = pd.read_csv('Data/Synthetic2DNoOverlapp.csv', header=None)

X = np.array(T.loc[:, T.columns != 2])
Y = np.array(T.loc[:, T.columns == 2]).T.reshape(40,1)

# #############################################################################
# Try different values for the Support Vector Machine
#

kernelType = 'sigmoid'  # <----- YOUR CHOICE. Specifies the kernel type to be used in the algorithm. 
                    # It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable. 
                    # If none is given, ‘rbf’ will be used. If a callable is given it is used to pre-compute
                    # the kernel matrix from data matrices; that matrix should be an array of shape 
                    # (n_samples, n_samples).

degreeVal = 5   # <----- YOUR CHOICE. Degree of the polynomial kernel function ('poly'). Ignored by all other kernels.

cVal = 2000   #  # <----- YOUR CHOICE. Penalty parameter C of the error term.
                # The C parameter trades off misclassification of training examples
                # against simplicity of the decision surface. A low C makes the decision 
                # surface smooth, while a high C aims at classifying all training examples
                # correctly by giving the model freedom to select more samples as support vectors.
    

clf = SVC(degree=degreeVal, C=cVal, kernel=kernelType)
# title for the plots
title = 'SVC with %s kernel' % kernelType
        
# Set-up 2x2 grid for plotting.

X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

ax = plt.subplot(111)
plot_contours(ax, clf.fit(X,Y), xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X0, X1, c=Y.T.tolist()[0], cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())
ax.set_title(title)
plt.show()