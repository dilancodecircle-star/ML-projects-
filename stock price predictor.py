import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import yfinance as yf

stocks  = input("Enter the code of stock : ")
data = yf.download(stocks , start = "2020-01-01" , end = "2026-01-01")
data.head()



