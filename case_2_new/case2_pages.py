# %%
import os
import pandas as pd
import numpy as np
import random
path = './case_2_new/data/'
os.chdir(path)
# %%
file_list = [c for c in os.listdir('.') if c[-3:]=='csv']
# [c[:-7]+'aug.csv' for c in os.listdir('.') if c[-3:]=='csv']
# %%
s = 100000 #desired sample size
e = "ISO-8859-1"
df_1 = pd.read_csv("page_aug.csv",
                 nrows=s,
                 encoding =e,
                 error_bad_lines = False)

# df_1.referringpageinstanceid.nunique()
# df_1.pagetitle.nunique()
# df_1.referringpageinstanceid.fillna(0,inplace=True)
# df_1.drop(columns=['sessionnumber','eventtimestamp'])
# df_1.describe()
# df_1.pagetitle.nunique()
# df_1.pagetitle
# df_1.pagelocationdomain.unique()
X = pd.get_dummies(df_1).drop(columns=['sessionnumber','eventtimestamp','iscustomer'])
y = np.array(pd.get_dummies(df_1)['iscustomer'])
X.shape
y.shape
# Use numpy to convert to arrays
# Labels are the values we want to predict
labels= y
# Remove the labels from the features
# axis 1 refers to the columns
features= X
# Saving feature names for later use
feature_list = list(features.columns)
# Convert to numpy array
features = np.array(features)

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
subsets = train_test_split(features, labels, test_size = 0.25, random_state = 42)
train_features, test_features, train_labels, test_labels = subsets

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators=1000, random_state=42)

# Train the model on training data
rf.fit(train_features, train_labels)
pred_labels = rf.predict(test_features)
pred_labels = [int(l) for l in pred_labels]
test_labels = [int(l) for l in test_labels]

from sklearn.metrics import confusion_matrix
tn, fp, fn, tp = confusion_matrix(test_labels,pred_labels).ravel()
print('accuracy:',(tp+tn)/sum([tn, fp, fn, tp]))
confusion_matrix(test_labels,pred_labels)
