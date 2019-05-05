# %% IMPORTS AND FUNCTIONS
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def RMSE(y_true, y_pred):
    """
        Relative mean squared error (RMSE)
    """
    numerator = np.sqrt(np.mean(np.power((y_true-y_pred),2)))
    denominator = np.sqrt(np.mean(np.power(y_true-np.mean(y_true), 2)))
    rmse = numerator / denominator
    return rmse

def CV(model, X, y, K, plot, **kwargs):
    """
        Cross-validation
    """

    kf = KFold(K, shuffle=True)

    # OPTIONAL PLOT
    if plot:
        fig, axarr = plt.subplots(2,5, figsize=(13,8))
        axarr = axarr.flatten()

    train_error = []
    test_error = []
    k = 0
    for train_idx, test_idx in kf.split(X):
        # Create k'th model for k'th fold
        modelk = model(**kwargs)

        # Split into training set and test set
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Fit model on training data
        modelk.fit(X_train, y_train)

        # Calculate RMSE
        train_pred = modelk.predict(X_train)
        test_pred = modelk.predict(X_test)
        train_error.append(RMSE(y_train, train_pred))
        test_error.append(RMSE(y_test, test_pred))

        # OPTIONAL PLOT
        if plot:
            axarr[k].plot(y_test, label="True")
            axarr[k].plot(test_pred, label="Prediction")
            axarr[k].set_title("RMSE: " + '{0:.6f}'.format(test_error[k]))

        k += 1

    # OPTIONAL PLOT
    if plot:
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()

    return np.mean(train_error), np.mean(test_error)

kft = KFold(100, shuffle=True)

# %% LOADING DATA
data = pd.read_csv("case1_master/case1/dataCase1.csv")

delme = set()
for col_name in cat_cols:
    delme = delme.union(set(data[col_name].unique()))

# %% FILL CATEGORICAL NANs VALUES WITH MOST POPULAR VALUE
cat_cols = ['X96','X97','X98','X99','X100']
for col_name in cat_cols:
    filler = data[col_name].value_counts().idxmax()
    data[col_name].fillna(filler,inplace=True)

# ONE HOT ENCODING
for col_name in cat_cols:
    # print(col_name)
    for unique_val in sorted(data[col_name].unique()):
        # print(unique_val)
        data[col_name + unique_val] = (data[col_name].values==unique_val)*1
    del data[col_name]

# %% REPLACE NUMERICAL NAN-VALUES WITH MEAN VALUES
data_clean = data
data_clean.iloc[:,1:] = data_clean.iloc[:,1:].fillna(data_clean.iloc[:,1:].mean())

X = data_clean[data_clean["Y"].notnull()].iloc[:,1:].values
y = data_clean[data_clean["Y"].notnull()]["Y"]
y = y.values

Xn = data_clean[data_clean["Y"].isnull()].iloc[:,1:].values
yn = data_clean["Y"][data_clean["Y"].isnull()].values

# %% X STANDARDIZATION
X = (X - X.mean(axis=0)) / X.std(axis=0)
Xn = (Xn - Xn.mean(axis=0)) / Xn.std(axis=0)

# %% MODELLING
model_scores = []
K = 10

# %% OLS
reg_ols = linear_model.LinearRegression
train_error, test_error = CV(reg_ols, X, y, K, fit_intercept=True, plot=False)
model_scores.append({"model":"OLS", "lambda":0, "train_error":train_error, "test_error":test_error})
print("train RMSE={:.3f}\ntest RMSE={:.3f}".format(train_error, test_error))

# %% LARS
reg_lars = linear_model.Lars

lambdas = range(1,50,1)
train_errors, test_errors = [], []
for lambda_ in lambdas:
    train_error, test_error = CV(reg_lars, X, y, K, n_nonzero_coefs=lambda_, fit_intercept=True, plot=False)
    train_errors.append(train_error)
    test_errors.append(test_error)
    model_scores.append({"model":"LARS", "lambda":lambda_, "train_error":train_error, "test_error":test_error})

plt.figure(figsize=(9,5))
plt.plot(lambdas, train_errors,label='train errors')
plt.plot(lambdas, test_errors,label='test errors')
plt.legend()
plt.title("LARS Cross-validation")
plt.xlabel("Lambda (non-zero coefficients)")
plt.ylabel("RMSE")
plt.savefig("lars_cv.png")
plt.show()

# %% RIDGE
reg_ridge = linear_model.Ridge

lambdas = np.arange(0.1,30,0.05)
train_errors, test_errors = [], []
for lambda_ in lambdas:
    train_error, test_error = CV(reg_ridge, X, y, K, alpha=lambda_, fit_intercept=True, plot=False)
    train_errors.append(train_error)
    test_errors.append(test_error)
    model_scores.append({"model":"RIDGE", "lambda":lambda_, "train_error":train_error, "test_error":test_error})

plt.figure(figsize=(9,5))
plt.plot(lambdas, train_errors, label = 'train error')
plt.plot(lambdas, test_errors, label = 'test error')
plt.title("Ridge Regression Cross-validation")
plt.xlabel("Lambda (L2-norm)")
plt.ylabel("RMSE")
plt.legend()
plt.savefig("ridge_cv.png")
plt.show()

# %% ELASTIC NET
manual_elastic = linear_model.ElasticNet
CV(manual_elastic, X, y, 10, alpha=0.1, l1_ratio=0.9, fit_intercept=True, max_iter=700, plot=True)
# %%
## ELASTIC NET ##
reg_elastic = linear_model.ElasticNet

lambdas = np.arange(0.1, 1.5, 0.05)
kappas = np.arange(0.6, 1.5, 0.05)
for lambda_ in lambdas:
    for kappa in kappas:
        train_error, test_error = CV(reg_elastic, X, y, K, alpha=lambda_, l1_ratio=kappa, max_iter=1000, fit_intercept=True, plot=False)
        model_scores.append({"model":"ELASTIC", "lambda":lambda_, "kappa":kappa, "train_error":train_error, "test_error":test_error})
# %%
elastics = sorted(list(filter(lambda x: x["model"]=="ELASTIC", model_scores)), key=lambda x:x["test_error"])
# %%
lambdas = list(map(lambda x: x["lambda"], elastics))
kappas = list(map(lambda x: x["kappa"], elastics))
tste = list(map(lambda x: x["test_error"], elastics))

plt.figure(figsize=(9,5))
plt.scatter(lambdas, kappas,s=80, c=tste,cmap='RdBu')
plt.title("ElasticNet Cross-validation")
plt.xlabel("Lambda 1 (L2-norm)")
plt.ylabel("Lambda 2 (L1-norm)")
colorbar = 'RdBu'
cbar = plt.colorbar()
cbar.ax.set_ylabel("RMSE")
plt.savefig("elastic_cv.png")
plt.show()
# %% TOP 5 MODELS
top5 = sorted(model_scores, key=lambda x: x["test_error"])[:5]
print("Top 5 models:")
top5

# %%
for i in top5:
    i0 = 'ElasticNet'
    i1 = round(i['lambda'],4)
    i2 = round(i['kappa'],4)
    i3 = round(i['train_error'],4)
    i4 = round(i['test_error'],4)
    print(i0,'&',i1,'&',i2,'&',i3,'&',i4,'\\\\')

# %% TOP MODEL
top1 = sorted(model_scores, key=lambda x: x["test_error"])[0]
print("Best model:")
top1

# %% ONE STANDARD ERROR RULE
cv_std = np.std(list(map(lambda x: x["test_error"], elastics)))
print("Test error standard deviation:", cv_std)

onestd = elastics[0]["test_error"] + cv_std
print("Choose least complex model closest to a performance of:", onestd)

top1std = list(filter(lambda x: x["test_error"] > onestd, elastics))[0]
top1std

# %%

for i in [top1std]:
    i0 = 'ElasticNet'
    i1 = round(i['lambda'],4)
    i2 = round(i['kappa'],4)
    i3 = round(i['train_error'],4)
    i4 = round(i['test_error'],4)
    print(i0,'&',i1,'&',i2,'&',i3,'&',i4,'\\\\')

# %%
model1std = linear_model.ElasticNet(alpha=top1std["lambda"], l1_ratio=top1std["kappa"])
model1std.fit(X, y)
# %%
for i in model1std.sparse_coef_:
    print(i)
# %%
s = """
  (0, 2)	1.4121946510707377
  (0, 42)	0.6602090684585923
  (0, 56)	3.039021800143413
  (0, 100)	-0.3415358118525393
  (0, 103)	0.28130981785499043
  (0, 107)	0.038981939776535496
  (0, 112)	0.09307844239561533
"""

# data.columns

for i in s.split('\n')[1:-1]:
    i = str(i.replace('\t','').replace(' ',''))
    i = i.split('(')[1].split(')')
    i[0] = int(i[0].split(',')[1])
    i[0] = data.columns[i[0]].replace('X','$X_{')+'}$'

    i[1] = round(float(i[1]),3)
    print(i[0],'&',i[1],'\\\\')
    # i[0] = '$x_{'+i[0]+'}$'
    # print(i[0],'\t',i[1])
# %%
predn = model1std.predict(Xn)
# %%
path = 'case1_master/case1/results/'
pred_path = path + 'prediction.csv'
predn.tofile(pred_path,'\n')

# %%
fig = plt.figure(figsize=(13,8))
plt.plot(predn)
plt.title("Predictions of the final model for the test data.")
plt.ylabel("Estimate $\hat{y}$")
plt.xlabel("Observation")
plt.savefig("test_estimates.png")
plt.show()
