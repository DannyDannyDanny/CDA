# %%
import os
import pandas as pd
import numpy as np
import random
path = './case_2_new/data/'
os.chdir(path)
# %%
file_list = [c for c in os.listdir('.') if c[-3:]=='csv']
s = 10000 #desired sample size
e = "ISO-8859-1"
df_1 = pd.read_csv("page_aug.csv",
                 nrows=s,
                 encoding =e,
                 error_bad_lines = False)

df_1.referringpageinstanceid.fillna(0,inplace=True)
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

# %%
df_1.iscustomer.value_counts()
# %%
# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import confusion_matrix
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators=1000, random_state=42)
rf.fit(train_features, train_labels)
pred_labels = rf.predict(test_features)
pred_labels.round()


tn, fp, fn, tp = confusion_matrix(test_labels,pred_labels.round()).ravel()
# %%
print('accuracy:',(tp+tn)/sum([tn, fp, fn, tp]))
confusion_matrix(test_labels,pred_labels.round())

sel = SelectFromModel(RandomForestRegressor(n_estimators=1000, random_state=42))
sel.fit(train_features, train_labels)
sel.get_support()


# %%
