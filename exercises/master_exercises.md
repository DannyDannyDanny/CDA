
# Week One - DiabetesData

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Specify folder path where dataset is etc.

path = "exercises/datasets/"
data = pd.read_csv(path + "DiabetesData.txt", sep = "\s+", header = 0)

# Linear mixed effects model, probably not what one wants here
lme = smf.mixedlm('Y ~ AGE+SEX+BMI+BP+S1+S2+S3+S4+S5+S6', data, groups = data['SEX'])

# Ordinary linear regression model
X = data[["AGE", "SEX","BMI","BP", "S1", "S2", "S3", "S4", "S5", "S6"]]
y = data["Y"]
lm = sm.OLS(y, X).fit()

lm.summary()
```

# Week Two

## Exercise 1

```python
#%% Exercise a
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as lng
import scipy.io
from sklearn import preprocessing

def ridgeMulti(X, _lambda, p, y):
    inner_prod = np.linalg.inv(np.matmul(X.T, X) + _lambda * np.eye(p,p))
    outer_prod = np.matmul(X.T, y)
    betas = np.matmul(inner_prod, outer_prod)
    return betas

path = "exercises/datasets/"
prostatePath = path + 'Prostate.txt'

T = np.loadtxt(prostatePath, delimiter = ' ', skiprows = 1, usecols=[1,2,3,4,5,6,7,8,9])

y = T[:, 8]
X = T[:,:8]

[n, p] = np.shape(X)

k = 100; # try k values of lambda
lambdas = np.logspace(-4, 3, k)

betas = np.zeros((p,k))

for i in range(k):
    betas[:, i] = ridgeMulti(X, lambdas[i], p, y)

plt.figure()
plt.semilogx(lambdas, betas.T )
plt.xlabel("Lambdas")
plt.ylabel("Betas")
plt.title("Regularized beta estimates")

```

```python
#%% Exercise b
K = 10
N = len(X)

I = np.asarray([0] * N)
for i in range(N):
    I[i] = (i + 1) % K + 1

I = I[np.random.permutation(N)]
lambdas = np.logspace(-4, 3, k)
MSE = np.zeros((10, 100))

for i in range(1, K+1):
    XTrain = X[i != I, :]
    yTrain = y[i != I]
    Xtest = X[i == I, :]
    yTest = y[i == I]

    # centralize and normalize data
    XTrain = preprocessing.scale(XTrain)
    yTrain = preprocessing.scale(XTrain)

    for j in range(100):
        Beta = ridgeMulti(XTrain, lambdas[j], 8, yTrain)
        _zz = (yTest - np.matmul(Xtest, Beta))
        MSE[(i - 1), j] = np.mean(_zz ** 2)

meanMSE = np.mean(MSE, axis = 0)
jOpt = np.argsort(meanMSE)[0]

lambda_OP = lambdas[jOpt]

# Remember excact solution depends on a random indexing, so results may vary
plt.semilogx([lambda_OP, lambda_OP], [np.min(betas), np.max(betas)], marker = ".")
```

```python
#%% Exercise c
seMSE = np.std(MSE, axis = 0) / np.sqrt(K)

J = np.where(meanMSE[jOpt] + seMSE[jOpt] > meanMSE)[0]
j = int(J[-1:])
Lambda_CV_1StdRule = lambdas[j]

print("CV lambda 1 std rule %0.2f" % Lambda_CV_1StdRule)

#%% Exercise 1 d
N = len(y)
[n, p] = np.shape(X)

off = np.ones(n)
M = np.c_[off, X] # Include offset / intercept

# Linear solver
beta, _, rnk, s = lng.lstsq(M, y)

yhat = np.matmul(M, beta)

e = y - np.matmul(X, Beta) # Low bias std
s = np.std(e)
D = np.zeros(100)
AIC = np.zeros(100)
BIC = np.zeros(100)

for j in range(100):
    Beta = ridgeMulti(XTrain, lambdas[j], 8, yTrain)
    inner = np.linalg.inv(np.matmul(X.T, X) + lambdas[j] *np.eye(8))
    outer = np.matmul(np.matmul(X, inner), X.T)
    D[j] = np.trace(outer)
    e = y - np.matmul(X, Beta)
    err = np.sum(e ** 2) / N
    AIC[j] = err + 2 * D[j] / N * s ** 2
    BIC[j] = N / (s ** 2) * (err + np.log(N) * D[j] / N * s ** 2)

jAIC = np.min(AIC)
jBIC = np.min(BIC)

print("AIC at %0.2f" % jAIC)
print("BIC at %0.2f" % jBIC)
```

```python
#%% Exercise 1

NBoot = 100
[N, p] = np.shape(X)
Beta = np.zeros((p, len(lambdas), NBoot))

for i in range(NBoot):
    I = np.random.randint(0, N, N)
    XBoot = X[I, :]
    yBoot = y[I]
    for j in range(100):
        Beta[:, j, i] = ridgeMulti(XBoot, lambdas[j], p, yBoot)

stdBeta = np.std(Beta, axis = 2)
plt.figure()
for i in range(8):
    plt.semilogx(lambdas, stdBeta[i,:])
plt.title("Bootstrapped standard error")
plt.ylabel("Sigma of beta")
plt.xlabel("lambda")
```

## Exercise 1

```python3

```

***

# Week Two -

## ex1
```python3
# -*- coding: utf-8 -*-
"""
Computer intensive datahandling exercise 1 week 2

@author: dnor
"""
#%% Exercise a
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as lng
import scipy.io
from sklearn import preprocessing

def ridgeMulti(X, _lambda, p, y):
    inner_prod = np.linalg.inv(np.matmul(X.T, X) + _lambda * np.eye(p,p))
    outer_prod = np.matmul(X.T, y)
    betas = np.matmul(inner_prod, outer_prod)
    return betas

#path = "C:\\Users\\dnor\\Desktop\\02582\\Lecture2\\S2"
prostatePath = 'Prostate.txt'

T = np.loadtxt(prostatePath, delimiter = ' ', skiprows = 1, usecols=[1,2,3,4,5,6,7,8,9])

y = T[:, 8]
X = T[:,:8]

[n, p] = np.shape(X)

k = 100; # try k values of lambda
lambdas = np.logspace(-4, 3, k)

betas = np.zeros((p,k))

for i in range(k):
    betas[:, i] = ridgeMulti(X, lambdas[i], p, y)

plt.figure()
plt.semilogx(lambdas, betas.T )
plt.xlabel("Lambdas")
plt.ylabel("Betas")
plt.title("Regularized beta estimates")

#%% Exercise b
K = 10
N = len(X)

I = np.asarray([0] * N)
for i in range(N):
    I[i] = (i + 1) % K + 1

I = I[np.random.permutation(N)]
lambdas = np.logspace(-4, 3, k)
MSE = np.zeros((10, 100))

for i in range(1, K+1):
    XTrain = X[i != I, :]
    yTrain = y[i != I]
    Xtest = X[i == I, :]
    yTest = y[i == I]

    # centralize and normalize data
    XTrain = preprocessing.scale(XTrain)
    yTrain = preprocessing.scale(XTrain)

    for j in range(100):
        Beta = ridgeMulti(XTrain, lambdas[j], 8, yTrain)
        MSE[(i - 1), j] = np.mean((yTest - np.matmul(Xtest, Beta)) ** 2)

meanMSE = np.mean(MSE, axis = 0)
jOpt = np.argsort(meanMSE)[0]

lambda_OP = lambdas[jOpt]

# Remember excact solution depends on a random indexing, so results may vary
plt.semilogx([lambda_OP, lambda_OP], [np.min(betas), np.max(betas)], marker = ".")

#%% Exercise c
seMSE = np.std(MSE, axis = 0) / np.sqrt(K)

J = np.where(meanMSE[jOpt] + seMSE[jOpt] > meanMSE)[0]
j = int(J[-1:])
Lambda_CV_1StdRule = lambdas[j]

print("CV lambda 1 std rule %0.2f" % Lambda_CV_1StdRule)

#%% Exercise 1 d
N = len(y)
[n, p] = np.shape(X)

off = np.ones(n)
M = np.c_[off, X] # Include offset / intercept

# Linear solver
beta, _, rnk, s = lng.lstsq(M, y)

yhat = np.matmul(M, beta)

e = y - np.matmul(X, Beta) # Low bias std
s = np.std(e)
D = np.zeros(100)
AIC = np.zeros(100)
BIC = np.zeros(100)

for j in range(100):
    Beta = ridgeMulti(XTrain, lambdas[j], 8, yTrain)
    inner = np.linalg.inv(np.matmul(X.T, X) + lambdas[j] *np.eye(8))
    outer = np.matmul(np.matmul(X, inner), X.T)
    D[j] = np.trace(outer)
    e = y - np.matmul(X, Beta)
    err = np.sum(e ** 2) / N
    AIC[j] = err + 2 * D[j] / N * s ** 2
    BIC[j] = N / (s ** 2) * (err + np.log(N) * D[j] / N * s ** 2)

jAIC = np.min(AIC)
jBIC = np.min(BIC)

print("AIC at %0.2f" % jAIC)
print("BIC at %0.2f" % jBIC)


#%% Exercise 1

NBoot = 100
[N, p] = np.shape(X)
Beta = np.zeros((p, len(lambdas), NBoot))

for i in range(NBoot):
    I = np.random.randint(0, N, N)
    XBoot = X[I, :]
    yBoot = y[I]
    for j in range(100):
        Beta[:, j, i] = ridgeMulti(XBoot, lambdas[j], p, yBoot)

stdBeta = np.std(Beta, axis = 2)
plt.figure()
for i in range(8):
    plt.semilogx(lambdas, stdBeta[i,:])
plt.title("Bootstrapped standard error")
plt.ylabel("Sigma of beta")
plt.xlabel("lambda")
```

## ex2
```python
# -*- coding: utf-8 -*-
"""
Computer intensive datahandling exercise 2 week 2

@author: dnor
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

path = "C:\\Users\\dnor\\Desktop\\02582\\Lecture2\\S2"

mat = scipy.io.loadmat('Silhouettes.mat')
Fem = mat['Fem'].ravel() - 1 # Get rid of outer dim, -1 due to stupid matlab indexing
Male = mat['Male'].ravel() - 1
Xa = mat['Xa']
#%% Exercise 2
fig, axis = plt.subplots(2, 2)
axis[0,0].plot(Xa[Fem,:65].T, Xa[Fem, 65:].T)
axis[0,0].set_title("Female Silhouettes")

axis[0,1].plot(Xa[Male, :65].T, Xa[Male, 65:].T)
axis[0,1].set_title("Male Silhouttes")

for i in range(2):
    axis[0,i].axis('equal')
    axis[0,i].axis([-0.25, 0.25, -0.25, 0.25])

N = np.shape(Xa)[0]
y = np.zeros(N)
y[Fem] = 1
n_classes = 2

MCrep = 10
K = 5

Error = np.zeros((K, 10))

I = np.asarray([0] * N)
for j in range(MCrep):
    for n in range(N):
        I[n] = (n + 1) % K + 1
    I = I[np.random.permutation(N)]
    for i in range(1, K+1):
        X_train = Xa[i != I, :]
        y_train = y[i != I]
        X_test = Xa[i == I, :]
        y_test = y[i == I]
        '''
        Can also make test training split as;
        X_train, X_test, y_train, y_test = train_test_split(...
                Xa, y, test_size=0.33, random_state=42)
        '''
        for k in range(1,11):
            # Use Scikit KNN classifier, as you have already tried implementing it youself
            neigh = KNeighborsClassifier(n_neighbors=k, weights = 'uniform', metric = 'euclidean')
            neigh.fit(X_train, y_train)
            yhat = neigh.predict(X_test)

            Error[i-1, k-1] = sum(np.abs(y_test - yhat)) / len(yhat)

E = np.mean(Error, axis = 0)
axis[1,0].scatter(list(range(1,11)), E, marker = '*')
axis[1,0].axis([0, 11, 0.2, 0.6])
axis[1,0].set_title("CV test error")
axis[1,0].set_xlabel("K")
axis[1,0].set_ylabel("Error")

# ROC curve
# Compute ROC curve and ROC area for each class, here based on last cross-validation fold

fpr = dict()
tpr = dict()
roc_auc = dict()

y_score = neigh.predict_proba(X_test) # find probabilities for a specific fold case

# Make y 1-hot encoded
y_1hot = np.zeros((8, n_classes))
y_1hot[np.arange(8), y_test.astype(int)] = 1    
for i in range(n_classes): # A bit redundant here, as there only is 2 classes
    fpr[i], tpr[i], threshold = roc_curve(y_1hot[:, i], y_score[:, i])
    print (threshold)
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area, class 1 (females)
lw = 2
axis[1,1].plot(fpr[1], tpr[1], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
axis[1,1].plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
axis[1,1].set_xlim([0.0, 1.0])
axis[1,1].set_ylim([0.0, 1.05])
axis[1,1].set_xlabel('False Positive Rate')
axis[1,1].set_ylabel('True Positive Rate')
axis[1,1].set_title('ROC')
axis[1,1].legend(loc="upper left")

#%% Exercise 3
# use the same data as above from the last cv loop

def roc_data(y_hat, y_true, cut):

    sensitivity = np.zeros(len(cut))
    specificity = np.zeros(len(cut))
    for index, threshold in enumerate(cut):
        num_positive = np.sum(y_true == 1)
        num_negative = np.sum(y_true == 0)
        true_positive = np.sum(y_hat[y_true==1] >= threshold)
        true_negative = np.sum(y_hat[y_true==0]  < threshold)

        sensitivity[index] = true_positive / num_positive
        specificity[index] = true_negative / num_negative

    return sensitivity, specificity

fpr = dict()
tpr = dict()
roc_auc = dict()
cutoff = np.linspace(1.0, 0.0, 10)
for i in range(n_classes): # A bit redundant here, as there only is 2 classes
    tpr[i], specificity = roc_data(y_score[:, i], y_1hot[:, i], cutoff)
    # FPR is 1 - TNR(specificity)
    fpr[i] = 1 - specificity
    roc_auc[i] = auc(fpr[i], tpr[i])



plt.figure()
plt.plot(fpr[1], tpr[1], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="upper left")
```

# Week 3

## Ex1

```python
import scipy.io
import numpy as np
from sklearn import linear_model
from scipy import linalg
from sklearn import preprocessing
import matplotlib.pyplot as plt
# Exercises for lecture 3 in 02582, lars

# Helper functions for data handling
def center(X):
    """ Center the columns (variables) of a data matrix to zero mean.

        X, MU = center(X) centers the observations of a data matrix such that each variable
        (column) has zero mean and also returns a vector MU of mean values for each variable.
     """
    n = X.shape[0]
    mu = np.mean(X,0)
    X = X - np.ones((n,1)) * mu
    return X, mu

def normalize(X):
    """Normalize the columns (variables) of a data matrix to unit Euclidean length.
    X, MU, D = normalize(X)
    i) centers and scales the observations of a data matrix such
    that each variable (column) has unit Euclidean length. For a normalized matrix X,
    X'*X is equivalent to the correlation matrix of X.
    ii) returns a vector MU of mean values for each variable.
    iii) returns a vector D containing the Euclidean lengths for each original variable.

    See also CENTER
    """

    n = np.size(X, 0)
    X, mu = center(X)
    d = np.linalg.norm(X, ord = 2, axis = 0)
    d[np.where(d==0)] = 1
    X = np.divide(X, np.ones((n,1)) * d)
    return X, mu, d

def normalizetest(X,mx,d):
    """Normalize the observations of a test data matrix given the mean mx and variance varx of the training.
       X = normalizetest(X,mx,varx) centers and scales the observations of a data matrix such that each variable
       (column) has unit length.
       Returns X: the normalized test data"""

    n = X.shape[0]
    X = np.divide(np.subtract(X, np.ones((n,1))*mx), np.ones((n,1)) * d)
    return X

# if your file is elsewhere:
path = ""
mat = scipy.io.loadmat(path + 'sand.mat')
X = mat['X']
y = mat['Y']

[n,p] = X.shape

CV = 5 # if CV = n leave-one-out, you may try different numbers
# this corresponds to crossvalind in matlab
# permutes observations - useful when n != 0
I = np.asarray([0] * n)
for i in range(n):
    I[i] = (i + 1) % CV + 1

I = I[np.random.permutation(n)]
K = range(53)

Err_tr = np.zeros((CV,len(K)))
Err_tst = np.zeros((CV, len(K)))

# Lars
for i in range(1, CV+1):
    # Split data according to the earlier random permutation
    Ytr = y[I != i].ravel() # ravel collapses the array, ie dim(x,1) to (x,)
    Ytst = y[I == i].ravel()
    Xtr = X[I != i, :]
    Xtst = X[I == i, :]

    my = np.mean(Ytr)
    Ytr, my = center(Ytr) # center training response
    Ytr = Ytr[0,:] # Indexing in python thingy, no time to solve it
    Ytst = Ytst-my # use the mean value of the training response to center the test response
    mx =np.mean(Xtr,0)
    varx = np.var(Xtr, 0)
    Xtr, mx, varx = normalize(Xtr) # normalize training data
    Xtst = normalizetest(Xtst, mx, varx)
           #np.ones((np.size(Xtst, 0), p)) *
    # NOTE: If you normalize outside the CV loop the data implicitly carry information of the test data
    # We should perform CV "the right way" and keep test data unseen.
    # compute all LARS solutions
    Betas = np.zeros((len(K), p))
    for j in K:
        reg = linear_model.Lars(n_nonzero_coefs=j, fit_path = False, fit_intercept = False, verbose = True)
        reg.fit(Xtr,Ytr)
        beta = reg.coef_.ravel()
        Betas[i-1, :] = beta

        # Predict with this model, and find error
        YhatTr = np.matmul(Xtr, beta)
        YhatTest = np.matmul(Xtst, beta)
        Err_tr[i-1, j] = np.matmul((YhatTr-Ytr).T,(YhatTr-Ytr))/len(Ytr) # training error
        Err_tst[i-1, j] = np.matmul((YhatTest-Ytst).T,(YhatTest-Ytst))/len(Ytst) # test error

err_tr = np.mean(Err_tr, axis=0) # mean training error over the CV folds
err_tst = np.mean(Err_tst, axis=0) # mean test error over the CV folds
err_ste = np.std(err_tst, axis=0)/np.sqrt(CV) # Note: we divide with sqrt(n) to get the standard error as opposed to the standard deviation

# Compute Cp-statistic, assumption n > p

Y_OLS = np.matmul(X, (linalg.lstsq(X,y))[0]) # NOTE: OLS solution doesn't make sense for p>>n
s2 = ((Y_OLS - y)**2).sum(axis=0) # add n for variance estimate
# NOTE: Check the value of s2 - our estimate of the variance of the noise in data is numerically zero
# so what we are saying with the Cp is: we believe data has no noise. Go ahead and make as complicated a model
# as possible. Cp works when n > p.
inner_prod = np.linalg.inv(np.matmul(X.T, X) + 0.001*np.eye(p,p))
outer_prod = np.matmul(X.T, y)
Y_ridge = np.matmul(inner_prod, outer_prod)

Betas = np.zeros((p, n))
for j in range(n):
    reg = linear_model.Lars(n_nonzero_coefs=j)
    reg.fit(Xtr,Ytr)
    Betas[:,i] = reg.coef_

Cp = np.zeros((len(K)))
for j in range(len(K)):
    Yhat = np.matmul(X, Betas[:,K[j]])
    Cp[j] = 1/((y-Yhat)**2).sum()-n+2*K[j]

fig, ax = plt.subplots(1,2)    
ax[0].plot(K, err_tr, 'b', label='train')
ax[0].plot(K, err_tst, 'r', label='test')
ax[0].plot(K, Cp/5e24, 'g', label= 'C_p') # scale to put in same plot
ax[0].legend()
ax[0].set_xlabel('k')
ax[0].set_ylabel('error estimate')
ax[0].set_title("error estimate")

ax[1].plot(K, np.log(err_tr), 'b', label='train')
ax[1].plot(K, np.log(err_tst), 'r', label='test')
ax[1].plot(K, Cp/5e24, 'g', label= 'C_p') # scale to put in same plot
ax[1].legend()
ax[1].set_xlabel('k')
ax[1].set_ylabel('error estimate')
ax[1].set_title("Log error estimate")
plt.show()

# NOTE: Curves are flat, and minimum varies with split of data (running K-fold CV several times)
# We can use the "one-standard-error rule" to choose the optimal value of K and avoid these effects
# You may run the diabetes data set (see ex_lars_solution_diabetes.m) to check how it works when n>p.
```

## Ex2

```python
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 14:43:23 2018

@author: dnor
"""

import scipy.io
import numpy as np
from sklearn import linear_model
from scipy import linalg
from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib.colors as colors
# Exercises for lecture 3 in 02582, lars
# Helga Svala Sigurðardóttir

# Helper functions for data handling
def center(X):
    """ Center the columns (variables) of a data matrix to zero mean.

        X, MU = center(X) centers the observations of a data matrix such that each variable
        (column) has zero mean and also returns a vector MU of mean values for each variable.
     """
    n = X.shape[0]
    mu = np.mean(X,0)
    X = X - np.ones((n,1)) * mu
    return X, mu

def normalize(X):
    """Normalize the columns (variables) of a data matrix to unit Euclidean length.
    X, MU, D = normalize(X)
    i) centers and scales the observations of a data matrix such
    that each variable (column) has unit Euclidean length. For a normalized matrix X,
    X'*X is equivalent to the correlation matrix of X.
    ii) returns a vector MU of mean values for each variable.
    iii) returns a vector D containing the Euclidean lengths for each original variable.

    See also CENTER
    """

    n = np.size(X, 0)
    X, mu = center(X)
    d = np.linalg.norm(X, ord = 2, axis = 0)
    d[np.where(d==0)] = 1
    X = np.divide(X, np.ones((n,1)) * d)
    return X, mu, d

def normalizetest(X,mx,d):
    """Normalize the observations of a test data matrix given the mean mx and variance varx of the training.
       X = normalizetest(X,mx,varx) centers and scales the observations of a data matrix such that each variable
       (column) has unit length.
       Returns X: the normalized test data"""

    n = X.shape[0]
    X = np.divide(np.subtract(X, np.ones((n,1))*mx), np.ones((n,1)) * d)
    return X

# if your file is elsewhere:
path = "C:\\Users\\dnor\\Desktop\\02582\\Lecture3\\M5\\"
mat = scipy.io.loadmat(path + 'sand.mat')
X = mat['X']
y = mat['Y']

[n,p] = X.shape
lambdas = np.array([1e-6, 1e-3, 1e0, 1e3])
K = range(0,11,1) # Ratio between L1 and L2 norms

Err_tr = np.zeros((100,len(lambdas), len(K)))
Err_tst = np.zeros((100, len(lambdas), len(K)))
K_elasticNet = np.empty((100, len(lambdas), len(K)))

# Elastic net, run time ~ 5 min on my laptop anyhow
for i in range(1, 101): # Bootstrap samples
    I = np.random.randint(0, n, n)
    OOBag = list(set(range(n)) - set(I)) # Out of bag indexes
    Xtrain = X[I, :]
    ytrain= y[I].ravel()
    Xtest = X[OOBag, :]
    ytest = y[OOBag].ravel()

    # Normalize and center as in last exercise
    my = np.mean(ytrain)
    ytrain, my = center(ytrain) # center training response
    ytrain = ytrain[0,:] # Indexing in python thingy, no time to solve it
    ytest = ytest-my # use the mean value of the training response to center the test response
    mx =np.mean(Xtrain,0)
    varx = np.var(Xtrain, 0)
    Xtrain, mx, varx = normalize(Xtrain) # normalize training data
    Xtest = normalizetest(Xtest, mx, varx)

    for k, _lambda in enumerate(lambdas):
        for j, ratio in enumerate(K):
            # Note that the elastic net in sklearn automatically cycles through all the parameters to find best fit
            reg_elastic = linear_model.ElasticNet(alpha = _lambda, l1_ratio = ratio/10, fit_intercept = False) # L1-ratio, how much ridge or lasso, l1_ratio = 1 is the lasso
            reg_elastic.fit(Xtrain, ytrain)

            beta = reg_elastic.coef_

            # Find errors for this bootstrap with the ratio set
            YhatTrain = np.matmul(Xtrain, beta)
            YhatTest = np.matmul(Xtest, beta)
            Err_tr[i-1, k, j] = np.matmul((YhatTrain-ytrain).T,(YhatTrain-ytrain))/len(ytrain) # training error
            Err_tst[i-1, k, j] = np.matmul((YhatTest-ytest).T,(YhatTest-ytest))/len(ytest) # test error
            K_elasticNet[i-1, k, j] = len(np.where(beta != 0)[0])

err_trEN = np.mean(Err_tr,0) # Mean across bootstrap samples
err_tstEN = np.mean(Err_tst,0)
K_nonzEN = np.mean(K_elasticNet,0)


#%% error development when going from ridge to lasso, training, 0.5 ratio corrosponse to elastic net
lambda_leg = ["Lambda %0.3f" % lambdas[0], "Lambda %0.3f" % lambdas[1], "Lambda %i" % lambdas[2], "Lambda %i" % lambdas[3]]
titles = ["Train", "Test", "Nr of vars"]
fig, ax = plt.subplots(1,3)
ax[0].plot(err_trEN.T)
ax[1].plot(err_tstEN.T)
ax[2].plot(K_nonzEN.T)

for i in range(3):
    ax[i].set_xlabel("L1/L2 Ratio (Lasso to Ridge)")
    ax[i].set_ylabel("Error")
    ax[i].legend(lambda_leg)
    ax[i].xaxis.set_ticks(np.arange(0,11))
    labels = [item.get_text() for item in ax[0].get_xticklabels()]
    for k in K:
        labels[k] = str(k/10)
    ax[i].set_xticklabels(labels)
    ax[i].set_title(titles[i])
ax[2].set_ylabel("Nr of variables")

#%% Plots as in matlab, doesn't make as much sense here, since no control of nr of parameters

# Color norm
colNorm = colors.LogNorm(vmin=err_trEN.min(), vmax=err_trEN.max())
fig, ax = plt.subplots()
pcm = ax.pcolor(K, lambdas, np.log(err_trEN), norm= colNorm)
ax.set_title("Log of mean training error")
ax.set_xlabel("K")
ax.set_ylabel("lambda")

fig, ax = plt.subplots()
pcm = ax.pcolor(K, lambdas, np.log(err_trEN), norm= colNorm)
ax.set_title("Log of mean test error")
ax.set_xlabel("K")
ax.set_ylabel("lambda")

fig, ax = plt.subplots()
pcm = ax.pcolor(K, lambdas, np.log(K_nonzEN), norm= colNorm)
ax.set_title("Nr. of none-zero parameters")
ax.set_xlabel("K")
ax.set_ylabel("lambda")
```

## Ex3

```python
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 14:43:23 2018

@author: dnor
"""

import scipy.io
import numpy as np
from scipy.stats import linregress
from statsmodels.sandbox.stats.multicomp import multipletests

# if your file is elsewhere:
path = "C:\\Users\\dnor\\Desktop\\02582\\Lecture3\\M5\\"

# Using pandas frame, for more "Matlab / R'esq" formulation
mat = scipy.io.loadmat(path + 'sand.mat')
X = mat['X']
y = mat['Y'].ravel()

[n, p] = X.shape
off = np.ones(n)
M = np.c_[off, X] # Include offset / intercept

PValues = np.zeros((p))
Xsub = np.zeros((p))


for j in range(p):
    Xsub = X[:,j]
    # Use the stats models linear regression, since p value already is included
    # Otherwise check https://stackoverflow.com/questions/27928275/find-p-value-significance-in-scikit-learn-linearregression
    # Which explains how to expand the class in sklearn to calculate it
    slope, intercept, r_value, PValues[j], std_err = linregress(Xsub, y)

idx1 = np.argsort(PValues)
p = PValues[idx1]

a = len(np.where(p < (0.05 / 2016))[0]) # Amount af features included
FDR = multipletests(PValues, alpha = 0.05, method = "fdr_bh")[1] # Computing Benjamini Hochberg's FDR

idx2 = np.argsort(FDR)
fdr = FDR[idx2]

b = len(np.where(fdr < 0.15)[0]) # How many values are below 0.15?
```


# Week 4

## Ex Fisher Iris

```python
# -*- coding: utf-8 -*-
"""
Fisher discriminant line classification for use in 02582

@author: dnor
"""

import numpy as np
from sklearn.metrics import confusion_matrix

def produceDiscriminantLine(X, S, mu, pi):
    Sinv = np.linalg.inv(S)
    first = np.matmul(np.matmul(X, Sinv), mu.T)
    second = 0.5 * np.matmul(np.matmul(mu, Sinv), mu.T)
    return first - second + np.log(pi)

path = "C:\\Users\\dnor\\Desktop\\02582\\Lecture4\\Exercises 4\Data"
dataPath = path + '\\FisherIris.csv'
attributeNames = []
T = []
# Dump data file into an array
with open(dataPath, "r") as ins:
    listArray = []
    for line in ins:
        # Remove junk, irritating formating stuff
        listArray.append(line.replace('\n', '').split('\t'))

# Encode data in desired format
n = len(listArray) - 1
p = len(listArray[0][0].split(',')) - 1
X = np.zeros((n, p))
y = np.zeros(n)
for i, data in enumerate(listArray):
    dataTemp = data[0].split(',')
    if i == 0: # first row is attribute names
        attributeNames = dataTemp[0:4]
    else:
        X[i - 1,:] = dataTemp[0:4]
        flowerInd = dataTemp[4]
        if flowerInd == 'Setosa':
            y[i-1] = 0
        elif flowerInd == "Versicolor":
            y[i-1] = 1
        else:
            y[i-1] = 2

# Actual Fisher discriminant done after here
pi = np.zeros(3)
mu = np.zeros((3, p))
S = np.zeros((p,p))
for i in range(3):
    XSubset = X[np.where(y == i)[0], :]
    pi[i]  = len(np.where(y == i)[0]) / n
    mu[i,:] = np.mean(XSubset, axis = 0)
    S += np.matmul((XSubset - mu[i, :]).T, XSubset - mu[i, :])
S = S / (n-3)

# Discriminants
d = np.zeros((3, n))
for i in range(3):
    d[i,:] = produceDiscriminantLine(X, S, mu[i,:], pi[i])

# Classify according to discriminant
yhat = np.unravel_index(np.argmax(d, axis=0), d.shape)[1] # index on "1" to get indexes
confusion_matrix(y, yhat) # can also do manually by just checking if y == yhat and counting
```

## Ex2

```python
#%%

from __future__ import division
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
import matplotlib.pyplot as plt

### Exercise b ###
path = "C:\\Users\\dnor\\Desktop\\02582\\Lecture4\\Exercises 4\\Data\\"
# read in the data to pandas dataframes and convert to numpy arrays
GXtrain = pd.read_csv(path + 'GolubGXtrain.csv', header=None)
GXtest = pd.read_csv(path + 'GolubGXtest.csv', header=None)

Xtrain = np.array(GXtrain.loc[:, GXtrain.columns != 0])
Ytrain = np.array(GXtrain.loc[:, GXtrain.columns == 0]).ravel()

Xtest = np.array(GXtest.loc[:, GXtest.columns != 0])
Ytest = np.array(GXtest.loc[:, GXtest.columns == 0]).ravel()

# choose regularization strength
nr_regs = 60
Cvals =np.linspace(1e-6, 1, nr_regs)
n = len(Xtrain)
# perform a 5-fold cross-validation - there are so few samples that K = 10 gives worse results
K = 5
CV = KFold(n,K,shuffle=True)
errorList = []

error = np.zeros((K, nr_regs))
# perform a logistic regression with L1 (Lasso) penalty and iterate C over the lambda values
# we train on the training sets and estimate on the testing set
for i, (train_index, test_index) in enumerate(CV):
    for k, Cval in enumerate(Cvals):
        model = LogisticRegression(penalty = 'l1', C = Cval, tol = 1e-6)
        model = model.fit(Xtrain[train_index, :], Ytrain[train_index])
        y_est = model.predict(Xtrain[test_index])
        # the error is the difference between the estimate and the test
        error[i,k] = np.abs(sum(y_est-Ytrain[test_index])/len(Ytrain[test_index]))

#%%
# we take the mean error from every lambda value
meanError = list(np.mean(error, axis=0))
# we take the standard deviation from every lambda value
std = np.std(error, axis=0)
# this is the index of the smallest error
minError = meanError.index(min(meanError))

# We want to find the simplest model that is only one standard error away from the smallest error
# We start by finding all indices that are less than one standard error away from the minimum error
J = np.where(meanError[minError] + std[minError] > meanError)[0]
# then we take the simplest model (furthest to the right)
j = int(J[-1::])
Lambda_CV_1StdRule = Cvals[j]

print("CV lambda 1 std rule %0.2f" % Lambda_CV_1StdRule)

### Exercise c ###

# After we know our optimal lambda we can create our model with our training set
modelOpt = LogisticRegression(penalty = 'l1', C = Lambda_CV_1StdRule)
modelOpt = modelOpt.fit(Xtrain, Ytrain)
y_estOpt = modelOpt.predict(Xtest)
coef = modelOpt.coef_
nrCoefs = sum(float(num) > 0 for num in coef.tolist()[0])

print("The number of coefficients in our optimal model is: %d" % nrCoefs)

### Exercise d ###
modelOptTest = modelOpt.fit(Xtest, Ytest)
accuracy = 1- abs(sum(y_estOpt-list(Ytest.ravel()))/len(Ytest))

print("The accuracy for our optimal model is: %0.2f" % accuracy)

### plot ###

# we plot the mean errors with their std values
#plt.errorbar(Cvals, meanError, std, marker='.', color='orange', markersize=10)
plt.plot(Cvals, meanError)
# and the locations of the smallest error and the simplest model according to the one standard error rule
#plt.plot(Cvals[minError], meanError[minError], marker='o', markersize=8, color="red")
plt.plot(Cvals[j], meanError[j], marker='o', markersize=8, color="blue")
xposition = [Cvals[minError], Cvals[j]]
for xc in xposition:
    plt.axvline(x=xc, color='k', linestyle='--')
#plt.xscale('log')
plt.xlabel("Lambda")
plt.ylabel("Deviance")
plt.title("Cross-validated deviance of Lasso fit")
plt.show()
```

## Ex3

```python
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 06:08:48 2019

@author: Mark
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io


mat = scipy.io.loadmat(r'..\..\..\S2\Silhouettes.mat')
Fem = mat['Fem'].ravel() - 1 # Get rid of outer dim, -1 due to stupid matlab indexing
Male = mat['Male'].ravel() - 1
num_female = Fem.size
num_male = Male.size
Xa = mat['Xa']
gamma = np.linspace(0.01, 0.99, 12) #  try in the range [0.01,0.99]
iterations = 100

cl_tr = np.zeros(iterations)
cl_tst = np.zeros(iterations)
train = np.zeros(len(gamma))
test = np.zeros(len(gamma))
s_test = np.zeros(len(gamma))

#%% Exercise 3
plt.figure(1)
plt.plot(Xa[Fem,:65].T, Xa[Fem, 65:].T)
plt.title("Female Silhouettes")
plt.figure(2)
plt.plot(Xa[Male, :65].T, Xa[Male, 65:].T)
plt.title("Male Silhouttes")


fig, axis = plt.subplots(3,4)
plt_col = 0
plt_row = -1
for j in range(len(gamma)):
    for i in range(iterations):
        femP = np.random.choice(Fem, size=num_female, replace=True) # bootstrap sample
        maleP = np.random.choice(Male, size=num_male, replace=True) # bootstrap sample

        femOOB = [x for x in Fem.tolist() if x not in femP.tolist() ]  # Out of bag samples
        maleOOB = [x for x in Male.tolist() if x not in maleP.tolist() ]  # Out of bag samples

        train_sample = np.concatenate([femP, maleP]).tolist()
        test_sample = femOOB + maleOOB

        # get means of the two classes
        f_mean = np.mean(Xa[femP], axis=0)
        m_mean = np.mean(Xa[maleP], axis=0)

        # Calculate the pooled within class covariance matrix
        Sfem = np.cov(Xa[femP], rowvar=0)
        Smale = np.cov(Xa[maleP], rowvar=0)
        Sw = (Sfem+Smale)/2

        # Calculate the regularized discriminant analysis estimate of the covariance matrix
        srda = gamma[j] * Sw + (1-gamma[j]) * np.diag(np.diag(Sw))

        # predict train and test
        const_fem = 0.5 * f_mean @ np.linalg.solve(srda, f_mean)
        const_male = 0.5 * m_mean @ np.linalg.solve(srda, m_mean)
        score_fem = f_mean @ np.linalg.solve(srda, Xa.T) - const_fem
        score_male = m_mean @ np.linalg.solve(srda, Xa.T) - const_male

        class_true = np.ones((Xa.shape[0]))
        class_true[Male] = 2
        class_pred = (score_male>score_fem)+1

        cl_tr[i] = np.mean(class_true[train_sample]==class_pred[train_sample]) #classification rate/prediction accuracy train
        cl_tst[i] = np.mean(class_true[test_sample]==class_pred[test_sample]) #classification rate/prediction accuracy test

    train[j] = np.mean(cl_tr)
    test[j] = np.mean(cl_tst)
    s_test[j] = np.std(cl_tst)

    plt_col = j % 4
    if plt_col == 0:
        plt_row +=1

    #plt.subplot(3,4,j), imagesc(Srda), axis image
    axis[plt_row, plt_col].imshow(srda)
    plt.title('RDA Covariance matrix')



    #NOTE: Look at how the covariance matrix is shrunken towards the
    # diagonal as alpha gets smaller

plt.figure(4)
plt.plot(gamma,train)
plt.xlabel('gamma')
plt.ylabel('Classification rate')
plt.plot(gamma,test)
plt.plot(gamma,test-s_test)
plt.plot(gamma,test+s_test)
plt.legend(('train','test'))
plt.title('Prediciton accuracy')


# NOTE: Classification rates are low when gamma is too small and we assume
#       that covariates are independent (underfitting - both train and test are low), but also if we use the full
#       covariance matrix without regularizing (overfitting - note
#       difference in train and test). We need to find a trade off.                
```


# Week 5

## Ex2a

```python
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
```

## Ex2b

```python
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

T = pd.read_csv('Data/Synthetic2DOverlap.csv', header=None)

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
```

## Ex5

```python
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 15:08:11 2018

@author: dnor
"""
#%%
import numpy as np
import pandas as pd
from sklearn.cross_validation import KFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import matplotlib.pyplot as plt

### Exercise b ###
path = "C:\\Users\\dnor\\Desktop\\02582\\Lecture5\\Data\\"
# read in the data to pandas dataframes and convert to numpy arrays
dataFrame = pd.read_csv(path + 'ACS.csv')

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
```


# Week 6

## Ex1

```python

```
