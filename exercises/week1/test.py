import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import os

# Specify folder path where dataset is etc.
path = r"/Users/dth/Documents/DTU/02582 - Computational Data Analysis/CDA/exercises/week1/"

data = pd.read_csv(path + "DiabetesData.txt", sep = "\s+", header = 0)

# Linear mixed effects model, probably not what one wants here
form = 'Y ~ AGE+SEX+BMI+BP+S1+S2+S3+S4+S5+S6'
lme = smf.mixedlm(form, data, groups = data['SEX'])

# Ordinary linear regression model
X = data[["AGE", "SEX","BMI","BP", "S1", "S2", "S3", "S4", "S5", "S6"]]
y = data["Y"]
lm = sm.OLS(y, X).fit()

lm.summary()

#%%
X[1:5]
#%%
y[1:5]
#%%
# 1.a Solving the linear system of equations with `linang.lnstsq`:
np.linalg.lstsq(X,y)[0]
#%%

#%%
#%%
#%%
#%%
#%%
