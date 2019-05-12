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
print('loading data')
s = 3000 #desired sample size
e = "ISO-8859-1"
skip = 1
if skip:
    print('randomized',s,'rows')
    #number of records in file (excludes header)
    n = sum(1 for line in open("page_aug.csv",encoding=e)) - 1
     #the 0-indexed header will not be included in the skip list
    skip = sorted(random.sample(range(1,n+1),n-s))
    df_1 = pd.read_csv("page_aug.csv",
                           skiprows=skip,
                           encoding =e,
                           error_bad_lines = False)
else:
    print('serialized',s,'rows')
    df_1 = pd.read_csv("page_aug.csv",
                     nrows=s,
                     encoding =e,
                     error_bad_lines = False)

print('cleaning dataset')
df_1.referringpageinstanceid.fillna(0,inplace=True)
# df_1.drop(columns=['eventtimestamp','sessionnumber'],inplace=True)
n0 = df_1.iscustomer.value_counts()[0]
n1 = df_1.iscustomer.value_counts()[1]

print('balancing dataset')
if n0 > n1:
    df_1.drop(df_1[df_1.iscustomer==0][:n0-n1].index,inplace=True)

if n1 > n0:
    df_1.drop(df_1[df_1.iscustomer==1][:n1-n0].index,inplace=True)


print('label - feature split')
X = df_1[['pageinstanceid','referringpageinstanceid','pagesequenceinattribution','pagesequenceinsession']]
# X = pd.get_dummies(df_1).drop(columns=['sessionnumber','iscustomer']).values
y = pd.get_dummies(df_1)['iscustomer']#.values

print('test - train split')
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=.25)

print('N-UNIQUE')
print(df_1.nunique())


# %% 3 model test
y.nunique()
X.nunique()
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

feat_imp = ada.feature_importances_
for i,f_i in enumerate(np.argsort(feat_imp)[::-1]):
    print(i,f_i,'\t',feat_imp[f_i]*100,'%\t',X.columns[f_i])
X.nunique()
df_1.nunique()
