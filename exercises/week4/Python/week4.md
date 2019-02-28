# Ex.2


# 2. а.
```Python
import os
import pandas as pd
import sklearn
import numpy as np
from sklearn.linear_model import LogisticRegression

path = 'exercises/week4/Data'
os.listdir(path)

# load data
trainDataPath = path + '/GolubGXtrain.csv'
testDataPath = path + '/GolubGXtest.csv'
df_train = pd.read_csv(trainDataPath)
df_test = pd.read_csv(testDataPath)

# split into test and train
y_test = df_test.iloc[:,0]
x_test = df_test.iloc[:,1:]
y_train = df_train.iloc[:,0]
x_train = df_train.iloc[:,1:]

# logistic regression
model = LogisticRegression(solver='lbfgs')
# training
model.fit(x_train, y_train)
# testing
y_hat = model.predict(x_test)

n_mistakes = sum([int(np.abs(y_hat[i]-y_test[i])) for i in range(len(y_hat))])

sum([int(np.abs(y_hat[i]-y_test[i])) for i in range(len(y_hat))])

print(n_mistakes,'mistake(s) out of',len(y_hat),'rows')
```

# 2. b. Build a classifier for training data in GolubGXtrain.csv. What regularization method do you prefer if you want to have few genes in the biomarker?

```Python
from sklearn.linear_model import Ridge
>>> import numpy as np
>>> n_samples, n_features = 10, 5
>>> np.random.seed(0)
>>> y = np.random.randn(n_samples)
>>> X = np.random.randn(n_samples, n_features)
>>> clf = Ridge(alpha=1.0)
>>> clf.fit(X, y)
Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
      normalize=False, random_state=None, solver='auto', tol=0.001)
```
