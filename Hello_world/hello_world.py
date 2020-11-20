'''
Author - Abhishek Maheshwarappa

This is simple linear Regression using Boston 
housing dataset for training for the Discovery 
cluster by AI Skunkworks

'''

# Importing Libraries
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

print("**********************************************************************")

print("**********************************************************************")

print("**********************************************************************")

print(" \n \n \n ")
from sklearn.datasets import load_boston
boston = load_boston()

print("****** Importing data from Pandas *****")
data = pd.DataFrame(boston.data)
data.columns = boston.feature_names


# Adding 'Price' (target) column to the data
print(' \n \n Shape of the dataset -', boston.data.shape)

# Input Data
x = boston.data

# Output Data
y = boston.target


# splitting data to training and testing dataset.
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2,
                                                random_state=0)

print("\n xtrain shape : ", xtrain.shape)
print("\n xtest shape  : ", xtest.shape)
print("\n ytrain shape : ", ytrain.shape)
print("\n ytest shape  : ", ytest.shape)

# Applying Linear Regression Model to the dataset and predicting the prices.


# Fitting Multi Linear regression model to training model
regressor = LinearRegression()


print("\n \n ******Training the model*******\n \n")

regressor.fit(xtrain, ytrain)

print("\n \n *******Training finished*******\n \n ")
# predicting the test set results
y_pred = regressor.predict(xtest)


time.sleep(3)

# Results of Linear Regression.

print('\n \n ***Mean Square Error***  \n\n')
mse = mean_squared_error(ytest, y_pred)
print(mse,'\n\n')
