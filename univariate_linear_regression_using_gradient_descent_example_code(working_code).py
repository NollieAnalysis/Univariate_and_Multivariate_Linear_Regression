#Univariate Linear Regression using Gradient Descent | Adarsh Menon
    #https://towardsdatascience.com/linear-regression-using-gradient-descent-97a6c8700931

#packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
plt.rcParams['figure.figsize'] = (12.0, 9.0)

#import data
data = pd.read_csv("/data_Linear_Regression_using_Gradient_Descent.csv") # change csv path here
data.head()

# Preprocessing Input data
X = data.iloc[:, 0]
Y = data.iloc[:, 1]
plt.scatter(X, Y)
plt.show()
# Building the model
m = 0
c = 0

L = 0.0001  # The learning Rate
epochs = 1000  # The number of iterations to perform gradient descent

n = float(len(X)) # Number of elements in X

# Performing Gradient Descent 
for i in range(epochs): 
    Y_pred = m*X + c  # The current predicted value of Y
    D_m = (-2/n) * sum(X * (Y - Y_pred))  # Derivative wrt m
    D_c = (-2/n) * sum(Y - Y_pred)  # Derivative wrt c
    m = m - L * D_m  # Update m
    c = c - L * D_c  # Update c
    
print (m, c)

# Making predictions
Y_pred = m*X + c

plt.scatter(X, Y) 
plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red')  # regression line
plt.show()

#checking coefficient and intercept
x_2 = data.loc[:,['0']].values.reshape(-1,1)
y_2 = data['1'].values.reshape(-1,1)
regr = LinearRegression()
regr.fit(x_2,y_2)
print(regr.coef_)
print(regr.intercept_)