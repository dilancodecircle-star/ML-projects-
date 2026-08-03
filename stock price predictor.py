import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import yfinance as yf

stocks  = input("Enter the code of stock : ")
data = yf.download(stocks , start = "2020-01-01" , end = "2026-01-01" , auto_adjust = True)
print(data.head())
print(data.info())
print(data.shape)
print(data.describe())

data.Close.plot(figsize = (10 ,7) , color = 'r')
plt.ylabel("{}price".format(stocks))
plt.title("{} price series".format(stocks))
plt.show()
sns.distplot(data['Close'] )
plt.show()
sns.distplot(data['Open'] )
plt.show()
sns.distplot(data['High'] )
plt.show()