#coursera IBM machine learning with python course

#fuel consumption co2 data

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