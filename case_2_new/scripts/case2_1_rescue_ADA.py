# %% IMPORTS AND FUNCTIONS
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
import os
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def eval_confusion(conf):
    tn, fp, fn, tp = conf
    sens = tp/(tp+fn)*100
    spec = tn/(fp+tn)*100
    ppv = tp/(tp+fp)*100
    npv = tn/(fn+tn)*100
    print('TP:',tp)
    print('FP:',fp)
    print('PPV:',('%.1f' % ppv),'%')
    print('FN:',fn)
    print('TN:',tn)
    print('NPV:',('%.1f' % npv),'%')
    print('Sens:',('%.1f' % sens),'%')
    print('Spec:',('%.1f' % spec),'%')

if 'visitor_aug.csv' not in os.listdir(os.curdir):
    path = './case_2_new/data/'
    os.chdir(path)

# %% Tree Based
e = "ISO-8859-1"
s = 7000
df = pd.read_csv("page_aug.csv",
                 encoding =e,
                 nrows=s,
                 error_bad_lines = False)

print('cleaning dataset')
df.referringpageinstanceid.fillna(0,inplace=True)
df.drop(columns=['eventtimestamp','sessionnumber'],inplace=True)
n0 = df.iscustomer.value_counts()[0]
n1 = df.iscustomer.value_counts()[1]

print('balancing dataset',df.shape)
if n0 > n1:
    df.drop(df[df.iscustomer==0][:n0-n1].index,inplace=True)

if n1 > n0:
    df.drop(df[df.iscustomer==1][:n1-n0].index,inplace=True)

this_results = (df.shape[0],df.pagelocation.nunique(),{})

print('label - feature split')
y = df.iscustomer
# X = pd.get_dummies(df.pagelocation)
X = pd.get_dummies(df[['pagesequenceinsession','pagelocation']])

print('test - train split')
tt_split = train_test_split(X, y, test_size=.25)
X_train, X_test, y_train, y_test = tt_split

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=.25, random_state=42)

#%%
print('\n--ADA training')
ada = AdaBoostClassifier()
ada.fit(X_train, y_train)
ada_score = np.mean(cross_val_score(ada, X, y, cv=5))
print('ADA score:',ada_score)
y_pred = ada.predict(X_test)
ada_confusion = confusion_matrix(y_test,y_pred).ravel()
print('ADA conf')
eval_confusion(ada_confusion)
# tn, fp, fn, tp = ada_confusion
#print('accuracy:',(tp+tn)/sum([tn, fp, fn, tp]))

print('\n--RF training')
rf = RandomForestClassifier(n_estimators=200)
rf.fit(X_train, y_train)
rf_score = np.mean(cross_val_score(rf, X, y, cv=5))
print('RF score',rf_score)
y_pred = rf.predict(X_test)
rf_confusion = confusion_matrix(y_test,y_pred).ravel()
print('RF conf')
eval_confusion(rf_confusion)

print('\n--DT training')
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt_score = np.mean(cross_val_score(dt, X, y, cv=5))
print('DT score',dt_score)
y_pred = dt.predict(X_test)
dt_confusion = confusion_matrix(y_test,y_pred).ravel()
print('DT conf')
eval_confusion(dt_confusion)

#%%
rf.feature_importances_

# feat_imp = ada.feature_importances_
feat_imp = rf.feature_importances_
zeros = []
heros = []

for i,f_i in enumerate(np.argsort(feat_imp)[::-1]):
    if feat_imp[f_i] > 0:
        print(i,'\t',f_i,'\t',feat_imp[f_i]*100,'%\t',X.columns[f_i],'<--')
        heros.append((feat_imp[f_i],X.columns[f_i]))
    else:
        print(i,'\t',f_i,'\t',feat_imp[f_i]*100,'%\t',X.columns[f_i],'-->')
        zeros.append(X.columns[f_i])

# %%
for p,u in heros[1:20]:
    print(p,',',u[31:])
