# Multivariate Linear Regression

# Import packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

# Import dataset
# No missing/null data points
# No need to normalize as the regressor object normalizes it automatically
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 4].values # or [:, -1]

"""
X[:, 4] is an independent categorical feature/variable.
We can use the LabelEncoder and OneHotEncoder to split the feature into dummy
variables. With this dataset if we only look at California and New York we 
are encoding the column identifying the state of either New York or California. 
Having only split into 2 encoded columns (dummy variables) we can use just one
of them. When then column we use shows 0, we will know that it means that the 
other column, that we are not using is 1. Inversely, when the column we use 
shows 1, we know that the other column is 0.

Below is what the output model would look like mathematically, using only 
California and New York
y = b0 + b1*x1 + b2*x2 + b3*x3 + b4*D1
or
y = b0 + b1*x1 + b2*x2 + b3*x3 + b4*D2 
Recognize that (D2 = 1 - D1). The phenomenom when one or more features 
(independent variables) can be used to predict another is called 
multi-collinearity. This leads to something called the dummy variable trap
When building a model, always exclude one dummy from each set (column of 
categorical data)
"""

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder = LabelEncoder()
X[:, 3] = labelEncoder.fit_transform(X[:, 3])

oneHotEncoder = OneHotEncoder(categorical_features = [3])
X = oneHotEncoder.fit_transform(X).toarray()

# To take care of the multi-collinearity (dummy variable trap)
# Note that this may not be necessary as the python library should take care of it
X = X[:, 1:]

# Split the data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Fitting the regressor to the training set
# By default, the regressor implements all features
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Here we can see the coefficients in the model: 
# y = b0 + b1*x1 + b2*x2 + b3*x3 + b4*D1
print(regressor.coef_)

# Test set predictions
Y_pred = regressor.predict(X_test)

# Since we have 5 features, we can't really visualize this on a graph

# To build an optimal model with backward elimination
import statsmodels.formula.api as sm 
# statsmodel library  does not consider the intercept b0, unlike the sklearn library
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)

# Iterate through the model, elimating predictors/features with P-value of >5% on each iteration
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit() # OLS: ordinary least squares
regressor_OLS.summary()

X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit() # OLS: ordinary least squares
regressor_OLS.summary()

X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit() # OLS: ordinary least squares
regressor_OLS.summary()

X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit() # OLS: ordinary least squares
regressor_OLS.summary()
