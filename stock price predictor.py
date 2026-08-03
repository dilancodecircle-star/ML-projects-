import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , r2_score
from sklearn.linear_model import Lasso , Ridge
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

stocks  = input("Enter the code of stock : ")
data = yf.download(stocks , start = "2020-01-01" , end = "2026-01-01" , auto_adjust = True)
print(data.head())
print(data.info())
print(data.shape)
print(data.describe())

data.Close.plot(figsize = (10 ,7) , color = 'r')
plt.ylabel("{}price".format(stocks))
plt.title("{} price series".format(stocks))
#plt.show()
sns.distplot(data['Close'] )
#plt.show()
sns.distplot(data['Open'] )
#plt.show()
sns.distplot(data['High'] )
#plt.show()

x = data.drop(['Close'] , axis = 1)
y  = data['Close']

x_train , x_test , y_train , y_test = train_test_split(x , y, test_size = 0.2 , random_state = 0)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

model = LinearRegression()
model.fit(x_train , y_train)
predict1 = model.predict(x_test)
print(predict1)


def calculate_metrics(y_test , y_pred):
    #y_test = truth valu and y_pred = model predicted value
    mse = mean_squared_error(y_test , y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test , y_pred)
    print("Mean Squared Error (MSE):", mse)
    print("Root Mean Squared Error (RMSE):", rmse)
    print("R-squared (R2):", r2)

calculate_metrics(y_test , predict1)
ls = Lasso().fit(x_train , y_train)
ridge = Ridge().fit(x_train , y_train)

la_predict = ls.predict(x_test)
ri_predict = ridge.predict(x_test)
print("Lasso Regression Predictions:", la_predict)
print("Ridge Regression Predictions:", ri_predict)

calculate_metrics(y_test , la_predict)

print("ridge errors")
calculate_metrics(y_test , ri_predict)

svr = SVR()
param_grid = {'C' :[0.1 , 1 , 10 , 1000]  , 
              'gamma' : [1 , 0.1 , 0.01 , 0.001 , 0.0001] , 
              'kernel' : ['rbf']}

grid = GridSearchCV(SVR() , param_grid , refit = True , verbose = 3)
grid.fit(x_train , y_train)

svr  = SVR(C = 10 , gamma = 0.01 , kernel = 'rbf')
svr.fit(x_train , y_train)
svr_pred = svr.predict(x_test)