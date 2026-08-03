import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
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

