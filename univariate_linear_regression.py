# Import packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# For better console printing
np.set_printoptions(threshold = np.nan)

# Import dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values # If you use [:, 0], X will become a vector and not a matrix
Y = dataset.iloc[:, -1].values

# No missing data in this set
# No labels for encoding

# Split training and test data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

# There is only one feature (independent variable, so no need to scale)
# y = b0 + b1*x1
# Using/fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train) # fits regressor object to the training data

# Here we can see the coefficients in the model y = b0 + b1*x1
print(regressor.coef_)

# Predicting the test set results from the regressor
Y_pred = regressor.predict(X_test)

# Visualizing the training set results 
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs. Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualizing the test set results 
plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs. Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()