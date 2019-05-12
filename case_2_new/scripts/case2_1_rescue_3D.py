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
    print('Accuracy:',(tp+tn)/sum([tn, fp, fn, tp]))
    print('Sens:',('%.1f' % sens),'%')
    print('Spec:',('%.1f' % spec),'%')


if 'visitor_aug.csv' not in os.listdir(os.curdir):
    path = './case_2_new/data/'
    os.chdir(path)

clfs_names = [
    # (KNeighborsClassifier(4),'K-NN 4'),
    (GaussianProcessClassifier(1.0 * RBF(1.0)),'GaussP'),
    (DecisionTreeClassifier(),'DeciT'),
    (RandomForestClassifier(n_estimators=300),'RF3'),
    (MLPClassifier(alpha=1),'Neu-N'),
    (AdaBoostClassifier(),'AdaBoo'),
    (GaussianNB(),'NaiveBayes'),
    (QuadraticDiscriminantAnalysis(),'QDA'),
    (SVC(gamma=2, C=1),'RBF-SVM')
    ]

results_3d = []

import warnings
warnings.filterwarnings("ignore")
# %%
bad_list = ['QDA','NaiveBayes','K-NN 4']

for s in range(1000,10000,1000):
    print(s)
    e = "ISO-8859-1"
    # s = 1000
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

    # clf_results =
    for clf,name in clfs_names:
        if name not in bad_list:
            score = np.mean(cross_val_score(clf, X, y, cv=5))
            this_results[2][name] = score
            print(name,'\t',('%.3f' % score).lstrip('0'))

    results_3d.append(this_results)

# %%
for result in results_3d:
    print(result)

# %%
test_list = sorted(results_3d)
nrows = [nrows for nrows,nsites,accdic in test_list]
nsites = [nsites for nrows,nsites,accdic in test_list]
plt.plot(nrows,nsites)
plt.xlabel('Number of Observations')
plt.ylabel('Number of Unique URLs')
plt.show()
#%%
# results_3d
plt.figure(figsize=(8,5))
zook = sorted(results_3d)
for c,name in clfs_names:
    if name not in ['QDA','NaiveBayes','K-NN 4']:
        accs = [accdic[name] for nrows,nsites,accdic in zook if name in accdic.keys()]
        nrows = [nrows for nrows,nsites,accdic in zook][:len(accs)]
        nsites = [nsites for nrows,nsites,accdic in zook][:len(accs)]
        print(accs)
        plt.plot(nrows,accs,label=name)
        # plt.plot(nsites,accs,label=name)

for obs in nrows:
    print(obs)
for c,name in clfs_names:
    print(name)

plt.legend()
plt.xlabel('Number of Observations')
plt.ylabel('Accuracy')
plt.savefig('help.png')
plt.show()


# %%
# test_list = results_3d
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import numpy as np
fig = plt.figure(figsize=(16,10))
ax = fig.add_subplot(111, projection='3d')
# ax = fig.add_subplot()
# ax.scatter(x1, y1, z1, c='r', marker='o')


for c,name in clfs_names:
    if name not in bad_list:
        nrows = sorted([nrows for nrows,nsites,accdic in test_list])
        nsites = [nsites for nrows,nsites,accdic in test_list]
        accs = [accdic[name] for nrows,nsites,accdic in test_list]
        ax.scatter(nrows, nsites, accs,label=name)

ax.set_xlabel('nrows')
ax.set_ylabel('nsites')
ax.set_zlabel('Accuracy')
ax.legend(loc='best')
plt.show()
