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

yeet = []

import warnings
warnings.filterwarnings("ignore")
# %%
for s in range(1000,500000,10000):
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

    yeet.append((df.shape[0],df.pagelocation.nunique()))

# %%
yeet = sorted(yeet)
plt.plot([nr for nr,ns in yeet],[ns for nr,ns in yeet])
for skeeet in yeet:
    print(skeeet[0],',',skeeet[1])
