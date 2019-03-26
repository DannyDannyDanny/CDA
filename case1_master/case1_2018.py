# %%
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
# %% markdown
# # Function definitions
# %%
def RMSE(y_true, y_pred):
    """
        Relative mean squared error (RMSE)
    """
    numerator = np.sqrt(np.mean(np.power((y_true-y_pred),2)))
    denominator = np.sqrt(np.mean(np.power(y_true-np.mean(y_true), 2)))
    rmse = numerator / denominator
    return rmse
# %%
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
# %%
kft = KFold(100, shuffle=True)

i = 0
for train, test in kft.split(X):
    print(i, train, test)
    i += 1
# %% LOADING DATA
data = pd.read_csv("case1_master/case1/dataCase1.csv")
# data18 = pd.read_csv('case1_master/2018/Case1_Data.csv')
data.shape
data18.shape

# %% TRANSFORMING CATEGORICAL COLUMNS
# data['X96'].unique()
# data['X97'].unique()
# data['X98'].unique()
# data['X99'].unique()
# data['X100'].unique()

# FILL CATEGORICAL VALUES WITH MOST POPULAR VALUE
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

# %% 1.2 Cleaning data
data.describe()
# %% Some of the columns are missing values, hence not all columns have 100 observations.


# #### Checking for NaN values
# %%
## FIND NAN VALUES
data[data.iloc[:,1:].isnull().any(axis=1)]
# %% markdown
# #### Replacing NaN values with mean value of corresponding column.
# %%
data_clean = data
data_clean.iloc[:,1:] = data_clean.iloc[:,1:].fillna(data_clean.iloc[:,1:].mean())#.values;

X = data_clean[data_clean["Y"].notnull()].iloc[:,1:].values
y = data_clean[data_clean["Y"].notnull()]["Y"]
y = y.values

Xn = data_clean[data_clean["Y"].isnull()].iloc[:,1:].values
yn = data_clean["Y"][data_clean["Y"].isnull()].values

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Xn shape:", Xn.shape)
print("yn shape:", yn.shape)
# %% markdown
# ## 1.3 Standardization of X
# %%
X = (X - X.mean(axis=0)) / X.std(axis=0)
Xn = (Xn - Xn.mean(axis=0)) / Xn.std(axis=0)
# %% markdown
# # 2 Data Modelling
# %%
model_scores = []
K = 10
# %% markdown
# ## 2.1 OLS
# %%
## OLS ##
reg_ols = linear_model.LinearRegression

train_error, test_error = CV(reg_ols, X, y, K, fit_intercept=True, plot=False)
model_scores.append({"model":"OLS", "lambda":0, "train_error":train_error, "test_error":test_error})
print("train RMSE={:.3f}\t test RMSE={:.3f}".format(train_error, test_error))
# %% markdown
# ## 2.2 LARS
# %%
## LARS ##
reg_lars = linear_model.Lars

lambdas = range(1,40,1)
train_errors, test_errors = [], []
for lambda_ in lambdas:
    train_error, test_error = CV(reg_lars, X, y, K, n_nonzero_coefs=lambda_, fit_intercept=True, plot=False)
    train_errors.append(train_error)
    test_errors.append(test_error)
    model_scores.append({"model":"LARS", "lambda":lambda_, "train_error":train_error, "test_error":test_error})

plt.figure(figsize=(9,5))
plt.plot(lambdas, train_errors)
plt.plot(lambdas, test_errors)
plt.title("LARS Cross-validation")
plt.xlabel("Lambda (non-zero coefficients)")
plt.ylabel("RMSE")
plt.savefig("lars_cv.png")
plt.show()
# %% markdown
# ## 2.3 RIDGE
# %%
## RIDGE ##
reg_ridge = linear_model.Ridge

lambdas = np.arange(0.1,4,0.05)
train_errors, test_errors = [], []
for lambda_ in lambdas:
    train_error, test_error = CV(reg_ridge, X, y, K, alpha=lambda_, fit_intercept=True, plot=False)
    train_errors.append(train_error)
    test_errors.append(test_error)
    model_scores.append({"model":"RIDGE", "lambda":lambda_, "train_error":train_error, "test_error":test_error})

plt.figure(figsize=(9,5))
plt.plot(lambdas, train_errors)
plt.plot(lambdas, test_errors)
plt.title("Ridge Regression Cross-validation")
plt.xlabel("Lambda (L2-norm)")
plt.ylabel("RMSE")
plt.savefig("ridge_cv.png")
plt.show()
# %% markdown
# ## 2.4 ELASTIC NET
# %%
manual_elastic = linear_model.ElasticNet
CV(manual_elastic, X, y, 10, alpha=0.1, l1_ratio=0.9, fit_intercept=True, max_iter=700, plot=True)
# %%
## ELASTIC NET ##
reg_elastic = linear_model.ElasticNet

lambdas = np.arange(0.1, 1.5, 0.05)
kappas = np.arange(0.1, 1, 0.05)
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
plt.scatter(lambdas, kappas, c=tste)
plt.title("ElasticNet Cross-validation")
plt.xlabel("Lambda 1 (L2-norm)")
plt.ylabel("Lambda 2 (L1-norm)")
plt.gray()
cbar= plt.colorbar()
cbar.ax.set_ylabel("RMSE")
plt.savefig("elastic_cv.png")
plt.show()
# %% markdown
# ## 2.5 Best model
# %% markdown
# ### top 5
# %%
top5 = sorted(model_scores, key=lambda x: x["test_error"])[:5]
print("Top 5 models:")
top5
# %% markdown
# ### top 1
# %%
top1 = sorted(model_scores, key=lambda x: x["test_error"])[0]
print("Best model:")
top1
# %% markdown
# ### Choose with $+1\sigma$
# %%
cv_std = np.std(list(map(lambda x: x["test_error"], elastics)))
print("Test error standard deviation:", cv_std)

onestd = elastics[0]["test_error"] + cv_std
print("Choose least complex model closest to a performance of:", onestd)

top1std = list(filter(lambda x: x["test_error"] > onestd, elastics))[0]
top1std
# %%
model1std = linear_model.ElasticNet(alpha=top1std["lambda"], l1_ratio=top1std["kappa"])
model1std.fit(X, y)
# %%
predn = model1std.predict(Xn)
# %%
fig = plt.figure(figsize=(13,8))
plt.plot(predn)
plt.title("Predictions of the final model for the test data.")
plt.ylabel("Estimate $\hat{y}$")
plt.xlabel("Observation")
plt.savefig("test_estimates.png")
plt.show()
