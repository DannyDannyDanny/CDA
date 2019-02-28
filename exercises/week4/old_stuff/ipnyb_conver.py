# %%
#from sklearn.linear_model import LogisticRegression
# %%
#from sklearn.linear_model import LogisticRegressionCV
# %%
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

# Loading data
dataTest = np.loadtxt('GolubGXtest.csv' ,delimiter=',')
dataTrain = np.loadtxt('GolubGXtrain.csv' ,delimiter=',')

# Training Data split
X_train = dataTest[:,1:]
y_train = dataTest[:,0]


# Number of Observations, Features and Classes
#N,M = X_train.shape
#C = 2

# Test Data split
X_test = dataTrain[:,1:]
y_test = dataTrain[:,0]
# %%
len(LogReg.coef_[LogReg.coef_!=0])
# %%
# Instanciate
# l1 decreases accuracy and number of parameters
# l2 increases accuracy and number of paramaters
# It's good to have few features to draw conclusions about data:
# You can see which features are important
LogReg = LogisticRegression(penalty='l1')

# Train
#LogReg.fit(X_train, y_train, sample_weight=None)
LogReg.fit(X_train, y_train)

# Confusion matrix
#print confusion_matrix(y_true=y_train,y_pred=y_predict)
print 'Conf matrix with training data:'
print confusion_matrix(y_true=y_train,y_pred=LogReg.predict(X_train))

print '\nConf matrix with test data:'
print confusion_matrix(y_true=y_test,y_pred=LogReg.predict(X_test))
#LogReg.predict
# %%
# Test
y_predict = LogReg.predict(X_test)

# Count number of matches
nMatches = len([i for i in y_test-y_predict if int(i) == 0])
nData = len(y_test)
accuracy = float(nMatches) / float(nData)

print accuracy
# %%

# %%
