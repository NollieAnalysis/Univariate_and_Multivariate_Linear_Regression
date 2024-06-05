# Univariate_and_Multivariate_Linear_Regression_Template_Code

Sources:

Check out Adarsh Menon's "Liear Regression using Gradient Descent" article on Medium (https://towardsdatascience.com/linear-regression-using-gradient-descent-97a6c8700931
). Univariate linear regression code came from Menon.

Check out Module 2, Regression, of IBM's Machine Learning with Python Coursera course (https://www.coursera.org/learn/machine-learning-with-python#modules). Multivariate linear regression code came from IBM.

# Project deliverables:

Univariate linear regression with gradient descent pyhton code (also saved as a .py file):

```python

#packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
plt.rcParams['figure.figsize'] = (12.0, 9.0)

#import data
data = pd.read_csv("C/data_Linear_Regression_using_Gradient_Descent.csv") # change csv path here
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
```

Multivariate linear regression with sklearn linear_model python code (also saved as a .py file):

```python

#packages
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
%matplotlib inline
from sklearn import linear_model

df = pd.read_csv("\FuelConsumptionCo2.csv") # change csv path here
df.head()
df.reset_index
df.head()

#create  a seperate data set for regression
    #choose numerical columns
filtered_df = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
filtered_df.head()

#plot co2emissions (y value) and engine size (x value)
plt.scatter(filtered_df['ENGINESIZE'], filtered_df['CO2EMISSIONS'], color = 'blue')
plt.xlabel("Engine Size")
plt.ylabel("CO2 Emissions")
plt.show()

#creating train and test datasets
msk = np.random.rand(len(df)) < 0.8
train = filtered_df[msk]
test = filtered_df[~msk]

#train data distribution
plt.scatter(train['ENGINESIZE'], train['CO2EMISSIONS'], color = 'blue')
plt.xlabel("Engine Size")
plt.ylabel("CO2 Emissions")
plt.show()

#multiple regression model
regr= linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(x,y)

#the coefficients and intercept
print('Coefficients: ', regr.coef_)
print('Intercept: ', regr.intercept_)

#prediction
y_hat = regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(test[['CO2EMISSIONS']])
print("Mean Squared Error (MSE) : %.2f"
    % np.mean((y_hat - y) **2))

# explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(x,y))

```
