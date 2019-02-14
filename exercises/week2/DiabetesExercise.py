import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import scipy.linalg as lng
import matplotlib.pyplot as plt

# Specify folder path where dataset is etc.
path = "/Users/dth/Documents/"
path += "DTU/02582 - Computational Data Analysis/CDA/exercises/week2/"

data = pd.read_csv(path + "DiabetesData.txt", sep = "\s+", header = 0)

# Linear mixed effects model, probably not what one wants here
form = 'Y ~ AGE+SEX+BMI+BP+S1+S2+S3+S4+S5+S6'
lme = smf.mixedlm(form, data, groups = data['SEX'])

# Ordinary linear regression model
X = data[["AGE", "SEX","BMI","BP", "S1", "S2", "S3", "S4", "S5", "S6"]]
y = data["Y"]
lm = sm.OLS(y, X).fit()

lm.summary()

#%% Ex 1
diabetPath = path + 'DiabetesDataNormalized.txt'
T = np.loadtxt(diabetPath, delimiter = ' ', skiprows = 1)
y = T[:, 10]
X = T[:,:10]

[n, p] = np.shape(X)

off = np.ones(n)
M = np.c_[off, X] # Include offset / intercept

# ----> Invert and calculate


# ----> Linear solver
beta, res, rnk, s = lng.lstsq(M, y)
yhat = np.matmul(M, beta)

# Same residuals as above
res = (y - yhat) ** 2

rss = np.sum(res)
mse = np.mean(res)
tss = np.sum((y - np.mean(y))** 2)
r2 = (1 - rss / tss) * 100


#%% Ex 2


#%% Ex 3

#%% Ex 4
