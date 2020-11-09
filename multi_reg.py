import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import statsmodels.api as sm

pd.options.display.max_columns
pd.options.display.max_rows
df = pd.read_csv("/Users/yuji/Desktop/espalhar/suumo_tokyo_data.csv", sep=",")

x = df[['Building Age','Time[min]', 'Deposit', 'Square Meters']]
y = df[['Price']]

#run multiple regresion
sscaler = preprocessing.StandardScaler()
sscaler.fit(x)
xss_sk = sscaler.transform(x) 
sscaler.fit(y)
yss_sk = sscaler.transform(y)


#show summary plots
x_add_const = sm.add_constant(xss_sk)
model_sm = sm.OLS(yss_sk, x_add_const).fit()

results = model_sm.summary(yname = 'Price', xname=['Const', 'Building Age', 'Time to Station', 'Deposit', 'Square Meters'])
print(results)

#show correlation coefficient
sscaler = preprocessing.StandardScaler()
sscaler.fit(x)
xss_sk = sscaler.transform(x) 
sscaler.fit(y)
yss_sk = sscaler.transform(y)

print(xss_sk)
print(yss_sk)