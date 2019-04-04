import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# %% Ex.2a

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
datapath = 'exercises/datasets/'
T = pd.read_csv(datapath+'Synthetic2DNoOverlapp.csv', header=None)

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
# %% Ex.2.b
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

T = pd.read_csv(datapath + 'Synthetic2DOverlap.csv', header=None)

X = np.array(T.loc[:, T.columns != 2])
Y = np.array(T.loc[:, T.columns == 2]).T.reshape(200,1)

# #############################################################################
# Try different values for the kernel parameter
#

cVal = 2000   # <----- YOUR CHOICE. Penalty parameter C of the error term.
                # The C parameter trades off misclassification of training examples
                # against simplicity of the decision surface. A low C makes the decision
                # surface smooth, while a high C aims at classifying all training examples
                # correctly by giving the model freedom to select more samples as support vectors.


# Estimate model
clf = SVC(C=cVal, kernel='rbf')

# title for the plots
title = 'SVC with rbf kernel'

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

# %% 2b
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

T = pd.read_csv(datapath+'Synthetic2DOverlap.csv', header=None)

X = np.array(T.loc[:, T.columns != 2])
Y = np.array(T.loc[:, T.columns == 2]).T.reshape(200,1)

# #############################################################################
# Try different values for the kernel parameter
#

cVal = 2000   # <----- YOUR CHOICE. Penalty parameter C of the error term.
                # The C parameter trades off misclassification of training examples
                # against simplicity of the decision surface. A low C makes the decision
                # surface smooth, while a high C aims at classifying all training examples
                # correctly by giving the model freedom to select more samples as support vectors.


# Estimate model
clf = SVC(C=cVal, kernel='rbf')

# title for the plots
title = 'SVC with rbf kernel'

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
# %% ex.5
import numpy as np
import pandas as pd
from sklearn.cross_validation import KFold
    from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import matplotlib.pyplot as plt

### Exercise b ###
# read in the data to pandas dataframes and convert to numpy arrays
dataFrame = pd.read_csv(datapath + 'ACS.csv')

dataTrain = np.asarray(dataFrame)[dataFrame['Train'] == 1,:]
dataTest = np.asarray(dataFrame)[dataFrame['Train'] == 0,:]

# split data -> some data is completely outside CV-optimization to reduce overfitting

# normal way of doing it;
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)
# But we already have indexes in the dataFrame;
X_train = dataTrain[:,:-1]
y_train = dataTrain[:,-2:-1].ravel() # Second to last row in dataframe is class

X_test = dataTest[:,:-1]
y_test = dataTest[:,-2:-1].ravel() # Second to last row in dataframe is class

# First Logistic regression <- below is extremely naiv way to use it, rampant overfit is expected
LogModel = LogisticRegression()
yhat_log = LogModel.fit(X_train, y_train).predict(X_test)
LogModelAcc = abs(sum(yhat_log == y_test))/len(y_test) # abit better then the simple imple. in matlab

# Which decision functions / kernels to test
decisionFunc = ["rbf", "poly", "sigmoid"]# , "linear"]

K = 5
n = np.shape(X_train)[0]
CV = KFold(n,K,shuffle=True)

error = np.zeros((K, len(decisionFunc)))
errorTest = np.zeros((K, len(decisionFunc)))
# Performing CV for SVM
for i, (train_index, test_index) in enumerate(CV):
    for k, dec in enumerate(decisionFunc):
        # Remember to look up the function to see how everything fits with the theory
        # http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
        model = svm.SVC(kernel = dec, shrinking = False, C = 1)
        model = model.fit(X_train[train_index, :], y_train[train_index])
        y_est = model.predict(X_train[test_index,:])
        y_est_test = model.predict(X_test) # Data not actually visible, just for visualization
        # the error is the difference between the estimate and the test)
        error[i,k] = np.abs(sum(y_est == y_train[test_index])/len(y_train[test_index]))
        errorTest[i,k] = np.abs(sum(y_est_test == y_test))/len(y_test)

# In below plot rbf test seems steady, why is this?
plt.plot(error)
plt.plot(errorTest)
legend = [decisionFunc, [decFunc + " test" for decFunc in decisionFunc]]
plt.legend([item for sublist in legend for item in sublist])
plt.ylabel("%% Accuracy")
plt.xlabel("CV fold")

modelOpt = svm.SVC(kernel = "rbf", shrinking = False, C = 1)
modelOpt = modelOpt.fit(X_train, y_train)
y_est_test = modelOpt.predict(X_test)
acc_rbf = np.abs(sum(y_est_test == y_test))/len(y_test)
print("Logistic Regression accuracy of %0.2f, and svm accuracy of %0.2f" % (LogModelAcc, acc_rbf))
