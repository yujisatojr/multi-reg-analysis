import numpy as np
import pandas as pd

from mpl_toolkits.mplot3d import Axes3D  #3Dplot
import matplotlib.pyplot as plt
import seaborn as sns

pd.get_option("display.max_columns")
pd.get_option("display.max_rows")
df = pd.read_csv("/Users/yuji/Desktop/espalhar/rawnumbers.csv", sep=",")
df.head()

x = df[['building_age', 'stories_tall', 'floor', 'administrative_cost', 'deposit', 'gratuity', 'rooms', 'sq_meters']]
y = df[['price']]


x1 = df[['sq_meters']]
x2 = df[['rooms']]
x3 = df[['deposit']]

print(x.shape)
print(y.shape)

fig=plt.figure()
ax=Axes3D(fig)

ax.scatter3D(x1, x2, y)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("y")

plt.show()


#run multiple regresion
from sklearn import preprocessing

sscaler = preprocessing.StandardScaler()
sscaler.fit(x)
xss_sk = sscaler.transform(x) 
sscaler.fit(y)
yss_sk = sscaler.transform(y)


#show summary plots
import statsmodels.api as sm

x_add_const = sm.add_constant(xss_sk)
model_sm = sm.OLS(yss_sk, x_add_const).fit()

results = model_sm.summary(yname = 'Price', xname=['Const', 'Building Age', 'Stories Tall', 'Floor', 'Administrative Cost', 'Deposit', 'Gratuity', 'Number of Rooms', 'Square Meters'])
print(results)

#show correlation coefficient
from sklearn import preprocessing

sscaler = preprocessing.StandardScaler()
sscaler.fit(x)
xss_sk = sscaler.transform(x) 
sscaler.fit(y)
yss_sk = sscaler.transform(y)

print(xss_sk)
print(yss_sk)